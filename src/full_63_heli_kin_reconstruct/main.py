"""
Full-63-heliostat kinematic reconstruction experiment.

63 heliostats, 100 train / 50 val / 50 test samples per heliostat.
Corrected pipeline: dataset generated from perturbed scenario; KR starts clean.

Usage
-----
    python main.py
    python main.py --output-dir outputs/my_run
    python main.py --daic
    python main.py --smoke-test
    python main.py --daic --dataset-type synthetic --loss-type focal_spot --stage1-epochs 100 --stage2-epochs 300
    python main.py --daic --dataset-type real       --loss-type focal_spot --stage1-epochs 100 --stage2-epochs 300
    python main.py --daic --dataset-type synthetic --loss-type pixel       --stage1-epochs 100 --stage2-epochs 500
    python main.py --daic --dataset-type real       --loss-type pixel       --stage1-epochs 100 --stage2-epochs 500
"""
import argparse
import datetime
import gc
import json
import logging
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")

import h5py
import torch
from artist.scenario.scenario import Scenario
from artist.util import constants as config_dictionary, set_logger_config
from artist.util import get_device, setup_distributed_environment

_here = pathlib.Path(__file__).resolve().parent
_src  = _here.parent
sys.path.insert(0, str(_src))

import config as cfg
from train import run
from utils.evaluation import build_heliostat_data_mapping
from utils.synth_data import SyntheticDatasetParser, _equalize_mapping, perturbations_to_json
from utils.synth_reporting import (
    plot_convergence,
    plot_param_recovery,
    plot_stage_convergence,
    plot_per_heliostat_accuracy_table,
    plot_per_heliostat_accuracy_histogram,
    write_summary,
)
from artist.io.paint_calibration_parser import PaintCalibrationDataParser
from reporting import (
    plot_field_accuracy_map,
    render_summary_table,
)


def _build_paint_mapping(split: str) -> list:
    return build_heliostat_data_mapping(
        benchmark_csv=cfg.BENCHMARK_CSV,
        calibration_properties_dir=cfg.CALIBRATION_PROPERTIES_DIR,
        flux_image_dir=cfg.FLUX_IMAGE_DIR,
        split=split,
    )


def _split_dir(split: str) -> pathlib.Path:
    return cfg.SYNTHETIC_DATA_DIR / split


def _split_complete(split: str, heliostat_ids: list, min_samples: int) -> bool:
    base = _split_dir(split)
    for hid in heliostat_ids:
        hel_dir = base / hid
        if not hel_dir.exists() or len(sorted(hel_dir.iterdir())) < min_samples:
            return False
    return True


def _run_reporting(results: dict, perturbations_json: dict | None, heliostat_ids: list, output_dir: pathlib.Path) -> None:
    history_file = output_dir / "convergence_history.json"
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
        plot_convergence(
            history, output_dir,
            pre_perturbation_m=results.get("pre_training",  {}).get("mean_m"),
            post_perturbation_m=None,
            post_training_m=results.get("post_training", {}).get("mean_m"),
            pre_perturbation_mrad=results.get("pre_training",  {}).get("mean_mrad"),
            post_perturbation_mrad=None,
            post_training_mrad=results.get("post_training", {}).get("mean_mrad"),
        )

    stage1_file = output_dir / "convergence_history_stage1.json"
    if stage1_file.exists():
        with open(stage1_file) as f:
            stage1_history = json.load(f)
        plot_stage_convergence(
            stage1_history, output_dir,
            stage_name="Stage 1 — AlignmentLoss (motor-position MSE, no ray tracing)",
            loss_label="AlignmentLoss (rad²)",
            filename="convergence_stage1.png",
        )

    stage2_file = output_dir / "convergence_history_stage2.json"
    if stage2_file.exists():
        with open(stage2_file) as f:
            stage2_history = json.load(f)
        loss_label_map = {
            "focal_spot": "FocalSpotLoss (m)",
            "pixel":      "PixelLoss (MSE)",
            "alignment":  "AlignmentLoss (rad²)",
        }
        loss_label = loss_label_map.get(results.get("loss_type", "focal_spot"), "Stage-2 Loss")
        plot_stage_convergence(
            stage2_history, output_dir,
            stage_name=f"Stage 2 — {loss_label}",
            loss_label=loss_label,
            filename="convergence_stage2.png",
        )

    plot_per_heliostat_accuracy_table(results, heliostat_ids, output_dir)
    with open(output_dir / "per_heliostat_accuracy.json") as _f:
        _rows = json.load(_f)
    plot_per_heliostat_accuracy_histogram(_rows, output_dir)

    if results.get("param_recovery"):
        plot_param_recovery(results["param_recovery"], output_dir)

    # Field accuracy map
    plot_field_accuracy_map(
        field_positions_path=output_dir / "field_positions.json",
        per_heliostat_mrad=results.get("post_training", {}).get("per_heliostat", {}),
        output_dir=output_dir,
    )

    # Summary table (val + test)
    render_summary_table(
        val_eval=results.get("post_training_val"),
        test_eval=results.get("post_training", {}),
        output_dir=output_dir,
    )

    write_summary(results, perturbations_json, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-63-heliostat kinematic reconstruction experiment."
    )
    parser.add_argument("--output-dir",    type=pathlib.Path, default=None)
    parser.add_argument("--loss-type",     choices=["focal_spot", "pixel", "alignment"], default=None)
    parser.add_argument("--dataset-type",  choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--stage1-epochs", type=int, default=None,
                        help="Override Stage-1 (AlignmentLoss) epoch count.")
    parser.add_argument("--stage2-epochs", type=int, default=None,
                        help="Override Stage-2 epoch count.")
    parser.add_argument("--daic", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 5 epochs with 1 train ray for a quick end-to-end check.")
    args = parser.parse_args()

    if args.loss_type is not None:
        cfg.LOSS_TYPE = args.loss_type
    if args.stage1_epochs is not None:
        cfg.STAGE1_EPOCHS = args.stage1_epochs
    if args.stage2_epochs is not None:
        cfg.STAGE2_EPOCHS = args.stage2_epochs

    if args.daic:
        cfg.IS_ON_DAIC = True
        cfg.BASE_DIR   = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
        cfg.PAINT_DIR  = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")
        cfg.SCENARIO_PATH              = cfg.BASE_DIR / "scenarios" / "full_field_200_samples_scenario" / "scenario.h5"
        cfg.SYNTHETIC_DATA_DIR         = cfg.BASE_DIR / "scenarios" / "full_63_heli_kin_reconstruct" / "synthetic_data"
        cfg.BENCHMARK_CSV              = cfg.PAINT_DIR / "splits" / f"{cfg.BENCHMARK_NAME}.csv"
        cfg.CALIBRATION_PROPERTIES_DIR = cfg.PAINT_DIR / cfg.BENCHMARK_NAME / "calibration_properties"
        cfg.FLUX_IMAGE_DIR             = cfg.PAINT_DIR / cfg.BENCHMARK_NAME / "flux_image"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_type = args.dataset_type
    if args.output_dir:
        run_dir = args.output_dir
    elif cfg.IS_ON_DAIC:
        run_dir = cfg.BASE_DIR / "outputs" / f"full_63_{dataset_type}_{cfg.LOSS_TYPE}_{timestamp}"
    else:
        if args.smoke_test:
            run_dir = cfg.BASE_DIR / "outputs" / "local_runs" / "smoke_tests" / f"full_63_{timestamp}"
        else:
            run_dir = cfg.BASE_DIR / "outputs" / "local_runs" / f"full_63_{dataset_type}_{cfg.LOSS_TYPE}_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)

    _config_snapshot = {
        "benchmark_name":           cfg.BENCHMARK_NAME,
        "scenario_path":            str(cfg.SCENARIO_PATH),
        "synthetic_data_dir":       str(cfg.SYNTHETIC_DATA_DIR),
        "dataset_type":             dataset_type,
        "loss_type":                cfg.LOSS_TYPE,
        "stage1_epochs":            cfg.STAGE1_EPOCHS,
        "stage2_epochs":            cfg.STAGE2_EPOCHS,
        "train_samples":            cfg.TRAIN_SAMPLES,
        "val_samples":              cfg.VAL_SAMPLES,
        "test_samples":             cfg.TEST_SAMPLES,
        "train_rays":               cfg.TRAIN_RAYS,
        "synth_gen_rays":           cfg.SYNTH_GEN_RAYS,
        "surface_points_per_facet": cfg.SURFACE_POINTS_PER_FACET,
        "perturbation_seed":        cfg.PERTURBATION_SEED,
        "perturbation_ranges":      cfg.PERTURBATION_RANGES,
        "optimization_config":      {str(k): v for k, v in cfg.OPTIMIZATION_CONFIG.items()},
        "smoke_test":               args.smoke_test,
        "output_dir":               str(run_dir),
        "pipeline":                 "corrected",
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(_config_snapshot, f, indent=2)

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for _h in logging.getLogger().handlers:
        if isinstance(_h, logging.StreamHandler) and _h.stream is sys.stderr:
            _h.stream = sys.stdout
    log = logging.getLogger(__name__)

    fh = logging.FileHandler(run_dir / "run.log")
    fh.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info(f"Experiment     : full_63_heli_kin_reconstruct (corrected pipeline)")
    log.info(f"Dataset type   : {dataset_type}")
    log.info(f"Loss type      : {cfg.LOSS_TYPE}")
    log.info(f"Epochs         : stage1={cfg.STAGE1_EPOCHS}  stage2={cfg.STAGE2_EPOCHS}")
    log.info(f"Scenario       : {cfg.SCENARIO_PATH}")
    log.info(f"Synthetic data : {cfg.SYNTHETIC_DATA_DIR}")
    log.info(f"Output dir     : {run_dir}")

    torch.manual_seed(0)
    device = get_device()
    log.info(f"Device: {device}")

    # Load heliostat IDs from scenario
    with h5py.File(cfg.SCENARIO_PATH, "r") as f:
        tmp = Scenario.load_scenario_from_hdf5(
            scenario_file=f, device=device,
            number_of_surface_points_per_facet=torch.tensor(
                [cfg.SURFACE_POINTS_PER_FACET, cfg.SURFACE_POINTS_PER_FACET]
            ),
        )
    heliostat_ids = list(tmp.heliostat_field.heliostat_groups[0].names)
    del tmp
    gc.collect()
    log.info(f"Heliostats in scenario: {len(heliostat_ids)}")

    # Perturbations are only meaningful for the synthetic pipeline.
    perturbations_json = None
    if dataset_type == "synthetic":
        pert_path = cfg.SYNTHETIC_DATA_DIR / "perturbations.json"
        if pert_path.exists():
            with open(pert_path) as f:
                perturbations_json = json.load(f)
            log.info("Loaded perturbations.json from synthetic data dir.")
        else:
            log.warning(
                f"perturbations.json not found at {pert_path}. "
                "Run generate_dataset.py first. Param recovery reporting will be skipped."
            )

    # Build data mappings
    log.info("Building data mappings …")
    scenario_hids = set(heliostat_ids)

    def _equalized(paint_split, n_samples):
        raw = _build_paint_mapping(paint_split)
        return [e for e in _equalize_mapping(raw, n_samples) if e[0] in scenario_hids]

    train_map = _equalized("train",      cfg.TRAIN_SAMPLES)
    val_map   = _equalized("validation", cfg.VAL_SAMPLES)
    test_map  = _equalized("test",       cfg.TEST_SAMPLES)
    log.info(f"Mapping sizes — train: {len(train_map)}, val: {len(val_map)}, test: {len(test_map)}")

    n_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(cfg.SCENARIO_PATH)

    with setup_distributed_environment(
        number_of_heliostat_groups=n_groups, device=device
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        if dataset_type == "synthetic":
            for split, n in [("val", cfg.VAL_SAMPLES), ("test", cfg.TEST_SAMPLES), ("train", cfg.TRAIN_SAMPLES)]:
                min_ok = int(n * 0.90)
                if not _split_complete(split, heliostat_ids, min_ok):
                    raise FileNotFoundError(
                        f"Synthetic dataset incomplete for split '{split}' "
                        f"(expected ≥{min_ok} samples for {len(heliostat_ids)} heliostats).\n"
                        "Run:  python generate_dataset.py --daic"
                    )
            train_parser = SyntheticDatasetParser(_split_dir("train"))
            val_parser   = SyntheticDatasetParser(_split_dir("val"))
            test_parser  = SyntheticDatasetParser(_split_dir("test"))
            log.info("Synthetic parsers ready.")
        else:
            train_parser = PaintCalibrationDataParser(
                sample_limit=cfg.TRAIN_SAMPLES,
                centroid_extraction_method=cfg.CENTROID_METHOD,
            )
            val_parser = PaintCalibrationDataParser(
                sample_limit=cfg.VAL_SAMPLES,
                centroid_extraction_method=cfg.CENTROID_METHOD,
            )
            test_parser = PaintCalibrationDataParser(
                sample_limit=cfg.TEST_SAMPLES,
                centroid_extraction_method=cfg.CENTROID_METHOD,
            )
            log.info("PAINT calibration parsers ready (real data).")

        stage1_epochs = cfg.STAGE1_EPOCHS
        stage2_epochs = cfg.STAGE2_EPOCHS
        train_rays    = cfg.TRAIN_RAYS
        if args.smoke_test:
            stage1_epochs = 2
            stage2_epochs = 3
            train_rays    = 1
            log.info("SMOKE TEST: stage1=2, stage2=3, 1 train ray.")

        results = run(
            scenario_path=cfg.SCENARIO_PATH,
            device=device,
            ddp_setup=ddp_setup,
            train_mapping=train_map,
            val_mapping=val_map,
            test_mapping=test_map,
            train_parser=train_parser,
            val_parser=val_parser,
            test_parser=test_parser,
            optimization_config=cfg.OPTIMIZATION_CONFIG,
            output_dir=run_dir,
            loss_type=cfg.LOSS_TYPE,
            dataset_type=dataset_type,
            n_surface_pts=cfg.SURFACE_POINTS_PER_FACET,
            train_rays=train_rays,
            perturbations=perturbations_json,
            heliostat_ids=heliostat_ids,
            stage1_epochs=stage1_epochs,
            stage2_epochs=stage2_epochs,
        )

        gc.collect()
        torch.cuda.empty_cache()

    log.info("Generating reports …")
    _run_reporting(results, perturbations_json, heliostat_ids, run_dir)

    log.info(f"\nDone. Results in: {run_dir}")
    log.info(f"  pre-training : {results['pre_training']['mean_mrad']:.3f} mrad")
    log.info(f"  post-training: {results['post_training']['mean_mrad']:.3f} mrad  "
             f"(trained in {results['train_time_min']:.1f} min)")


if __name__ == "__main__":
    main()
