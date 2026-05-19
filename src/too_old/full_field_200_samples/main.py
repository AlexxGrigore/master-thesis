"""
Full-field 200-samples synthetic perturbation experiment.

63 heliostats, 100 train / 50 val / 50 test samples per heliostat.
Loss: BlurredPixelLoss (Gaussian blur σ=1 → peak-normalize → pixel-wise MSE).

Three evaluation checkpoints:
  1. pre_perturbation  — clean scenario vs clean synthetic test  (~0 mrad)
  2. post_perturbation — perturbed scenario vs same clean test  (high mrad)
  3. post_training     — trained (recovered) scenario vs same clean test (low mrad)

Usage
-----
    python main.py
    python main.py --output-dir outputs/my_run
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
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment

_here = pathlib.Path(__file__).resolve().parent
_src  = _here.parent
sys.path.insert(0, str(_src))

import config as cfg
from generate_dataset import _synthetic_data_dir, _split_complete
from train import run
from utils.evaluation import build_heliostat_data_mapping
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from five_heliostats_synth.data import (
    _equalize_mapping,
    sample_perturbations,
    perturbations_to_json,
    SyntheticDatasetParser,
)
from five_heliostats_synth.reporting import (
    plot_convergence,
    plot_param_recovery,
    plot_stage_convergence,
    plot_per_heliostat_accuracy_table,
    plot_per_heliostat_accuracy_histogram,
    write_summary,
)


def _build_mapping(split: str) -> list:
    return build_heliostat_data_mapping(
        benchmark_csv=cfg.BENCHMARK_CSV,
        calibration_properties_dir=cfg.CALIBRATION_PROPERTIES_DIR,
        flux_image_dir=cfg.FLUX_IMAGE_DIR,
        split=split,
    )


def _split_dir(split: str) -> pathlib.Path:
    return _synthetic_data_dir() / split


def _run_reporting(results: dict, perturbations_json: dict | None, heliostat_ids: list, output_dir: pathlib.Path) -> None:
    # Combined convergence (all epochs, with reference lines).
    history_file = output_dir / "convergence_history.json"
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
        plot_convergence(
            history, output_dir,
            pre_perturbation_m=results.get("pre_perturbation",  {}).get("mean_m"),
            post_perturbation_m=results.get("post_perturbation", {}).get("mean_m"),
            post_training_m=results.get("post_training",     {}).get("mean_m"),
            pre_perturbation_mrad=results.get("pre_perturbation",  {}).get("mean_mrad"),
            post_perturbation_mrad=results.get("post_perturbation", {}).get("mean_mrad"),
            post_training_mrad=results.get("post_training",     {}).get("mean_mrad"),
        )

    # Per-stage convergence plots.
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

    # Per-heliostat accuracy table + histogram.
    plot_per_heliostat_accuracy_table(results, heliostat_ids, output_dir)
    with open(output_dir / "per_heliostat_accuracy.json") as _f:
        _rows = json.load(_f)
    plot_per_heliostat_accuracy_histogram(_rows, output_dir)

    if results.get("param_recovery"):
        plot_param_recovery(results["param_recovery"], output_dir)

    write_summary(results, perturbations_json, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-field 200-samples synthetic perturbation experiment."
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--dataset-type", choices=["synthetic", "real"], default=None,
                        help="Override cfg.DATASET_TYPE for this run.")
    parser.add_argument("--daic", action="store_true",
                        help="Use DAIC cluster paths (overrides IS_ON_DAIC in config.py).")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 5 epochs with 1 train ray for a quick end-to-end check.")
    parser.add_argument("--loss-type", choices=["focal_spot", "pixel", "alignment"], default=None,
                        help="Override cfg.LOSS_TYPE for this run.")
    parser.add_argument("--backlash-mrad", type=float, default=None,
                        help="Enable backlash perturbation with this amplitude in mrad (0 = off). "
                             "Overrides cfg.BACKLASH_PERTURBATION.")
    parser.add_argument("--gravity-sag-mrad", type=float, default=None,
                        help="Enable gravity-sag perturbation with this amplitude in mrad (0 = off). "
                             "Overrides cfg.GRAVITY_SAG_PERTURBATION.")
    args = parser.parse_args()

    if args.loss_type is not None:
        cfg.LOSS_TYPE = args.loss_type

    # CLI overrides for perturbation flags.
    if args.backlash_mrad is not None:
        cfg.BACKLASH_PERTURBATION = {"enabled": args.backlash_mrad > 0, "amplitude_mrad": args.backlash_mrad}
    if args.gravity_sag_mrad is not None:
        cfg.GRAVITY_SAG_PERTURBATION = {"enabled": args.gravity_sag_mrad > 0, "amplitude_mrad": args.gravity_sag_mrad}

    if args.dataset_type is not None:
        cfg.DATASET_TYPE = args.dataset_type

    if args.daic:
        cfg.IS_ON_DAIC = True
        cfg.BASE_DIR  = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
        cfg.PAINT_DIR = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")
        cfg.SCENARIO_PATH              = cfg.BASE_DIR / "scenarios" / "full_field_200_samples_scenario" / "scenario.h5"
        cfg.BENCHMARK_CSV              = cfg.PAINT_DIR / "splits" / f"{cfg.BENCHMARK_NAME}.csv"
        cfg.CALIBRATION_PROPERTIES_DIR = cfg.PAINT_DIR / cfg.BENCHMARK_NAME / "calibration_properties"
        cfg.FLUX_IMAGE_DIR             = cfg.PAINT_DIR / cfg.BENCHMARK_NAME / "flux_image"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        run_dir = args.output_dir
    elif cfg.IS_ON_DAIC:
        run_dir = cfg.BASE_DIR / "outputs" / f"full_field_200_{timestamp}"
    else:
        suffix = "smoke" if args.smoke_test else f"full_field_200_{timestamp}"
        run_dir = cfg.BASE_DIR / "outputs" / "local_runs" / suffix

    run_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration snapshot immediately so it's always present even if the run crashes.
    _config_snapshot = {
        "benchmark_name":           cfg.BENCHMARK_NAME,
        "scenario_path":            str(cfg.SCENARIO_PATH),
        "dataset_type":             cfg.DATASET_TYPE,
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
        "backlash_perturbation":    cfg.BACKLASH_PERTURBATION,
        "gravity_sag_perturbation": cfg.GRAVITY_SAG_PERTURBATION,
        "optimization_config":      {str(k): v for k, v in cfg.OPTIMIZATION_CONFIG.items()},
        "smoke_test":               args.smoke_test,
        "output_dir":               str(run_dir),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(_config_snapshot, f, indent=2)

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    log = logging.getLogger(__name__)

    fh = logging.FileHandler(run_dir / "run.log")
    fh.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info(f"Benchmark  : {cfg.BENCHMARK_NAME}")
    log.info(f"Scenario   : {cfg.SCENARIO_PATH}")
    log.info(f"Output dir : {run_dir}")
    log.info(f"Train/val/test samples: {cfg.TRAIN_SAMPLES}/{cfg.VAL_SAMPLES}/{cfg.TEST_SAMPLES}")
    log.info(f"Surface pts/facet: {cfg.SURFACE_POINTS_PER_FACET}×{cfg.SURFACE_POINTS_PER_FACET}")
    log.info(f"Train rays: {cfg.TRAIN_RAYS}")

    torch.manual_seed(0)
    device = get_device()
    log.info(f"Device: {device}")

    # ------------------------------------------------------------------ load heliostat IDs from scenario
    with h5py.File(cfg.SCENARIO_PATH, "r") as f:
        tmp = Scenario.load_scenario_from_hdf5(
            scenario_file=f,
            device=device,
            number_of_surface_points_per_facet=torch.tensor(
                [cfg.SURFACE_POINTS_PER_FACET, cfg.SURFACE_POINTS_PER_FACET]
            ),
        )
    heliostat_ids = list(tmp.heliostat_field.heliostat_groups[0].names)
    del tmp
    gc.collect()
    log.info(f"Heliostats in scenario: {len(heliostat_ids)}")

    # ------------------------------------------------------------------ perturbations
    if cfg.DATASET_TYPE == "synthetic":
        perturbations = sample_perturbations(
            n_heliostats=len(heliostat_ids),
            ranges=cfg.PERTURBATION_RANGES,
            seed=cfg.PERTURBATION_SEED,
        )
        perturbations_json = perturbations_to_json(perturbations, heliostat_ids)
        with open(run_dir / "perturbations.json", "w") as f:
            json.dump(perturbations_json, f, indent=2)
        log.info(f"Perturbations sampled for {len(heliostat_ids)} heliostats and saved.")
    else:
        perturbations = None
        perturbations_json = None
        log.info("Real dataset — skipping kinematic perturbations.")

    # ------------------------------------------------------------------ data mappings
    log.info("Building data mappings …")
    scenario_hids = set(heliostat_ids)
    train_map = [e for e in _equalize_mapping(_build_mapping("train"),      cfg.TRAIN_SAMPLES) if e[0] in scenario_hids]
    val_map   = [e for e in _equalize_mapping(_build_mapping("validation"), cfg.VAL_SAMPLES)   if e[0] in scenario_hids]
    test_map  = [e for e in _equalize_mapping(_build_mapping("test"),       cfg.TEST_SAMPLES)  if e[0] in scenario_hids]
    log.info(
        f"Mapping sizes — train: {len(train_map)}, val: {len(val_map)}, test: {len(test_map)}"
    )

    n_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(cfg.SCENARIO_PATH)

    with setup_distributed_environment(
        number_of_heliostat_groups=n_groups, device=device
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        if cfg.DATASET_TYPE == "synthetic":
            for split, n in [("val", cfg.VAL_SAMPLES), ("test", cfg.TEST_SAMPLES), ("train", cfg.TRAIN_SAMPLES)]:
                min_acceptable = int(n * 0.90)
                if not _split_complete(split, heliostat_ids, min_acceptable):
                    raise FileNotFoundError(
                        f"Synthetic dataset incomplete for split '{split}' "
                        f"(expected ≥{min_acceptable} samples for {len(heliostat_ids)} heliostats).\n"
                        "Run:  python generate_dataset.py"
                    )
            backlash_mrad = (
                cfg.BACKLASH_PERTURBATION["amplitude_mrad"]
                if cfg.BACKLASH_PERTURBATION.get("enabled") else 0.0
            )
            gravity_sag_mrad = (
                cfg.GRAVITY_SAG_PERTURBATION["amplitude_mrad"]
                if cfg.GRAVITY_SAG_PERTURBATION.get("enabled") else 0.0
            )
            train_parser = SyntheticDatasetParser(
                _split_dir("train"),
                backlash_amplitude_mrad=backlash_mrad,
                gravity_sag_amplitude_mrad=gravity_sag_mrad,
            )
            val_parser = SyntheticDatasetParser(
                _split_dir("val"),
                backlash_amplitude_mrad=backlash_mrad,
                gravity_sag_amplitude_mrad=gravity_sag_mrad,
            )
            test_parser = SyntheticDatasetParser(_split_dir("test"))  # clean — no perturbations
            if backlash_mrad > 0:
                log.info(f"Backlash perturbation: ±{backlash_mrad:.1f} mrad (train/val).")
            if gravity_sag_mrad > 0:
                log.info(f"Gravity-sag perturbation: {gravity_sag_mrad:.1f} mrad peak (train/val).")
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
            log.info("Real PAINT parsers ready.")

        # ------------------------------------------------------------------ run
        optimization_config = cfg.OPTIMIZATION_CONFIG
        train_rays = cfg.TRAIN_RAYS
        stage1_epochs = cfg.STAGE1_EPOCHS
        stage2_epochs = cfg.STAGE2_EPOCHS
        if args.smoke_test:
            stage1_epochs = 2
            stage2_epochs = 3
            train_rays = 1
            log.info("SMOKE TEST: stage1=2 epochs, stage2=3 epochs, 1 train ray.")

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
            optimization_config=optimization_config,
            output_dir=run_dir,
            loss_type=cfg.LOSS_TYPE,
            dataset_type=cfg.DATASET_TYPE,
            n_surface_pts=cfg.SURFACE_POINTS_PER_FACET,
            train_rays=train_rays,
            perturbations=perturbations,
            heliostat_ids=heliostat_ids,
            stage1_epochs=stage1_epochs,
            stage2_epochs=stage2_epochs,
        )

        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ reporting
    log.info("Generating reports …")
    _run_reporting(results, perturbations_json, heliostat_ids, run_dir)

    log.info(f"\nDone. Results in: {run_dir}")
    log.info(
        f"  pre-perturb : {results['pre_perturbation']['mean_mrad']:.3f} mrad"
    )
    log.info(
        f"  post-perturb: {results['post_perturbation']['mean_mrad']:.3f} mrad"
    )
    log.info(
        f"  post-train  : {results['post_training']['mean_mrad']:.3f} mrad  "
        f"(trained in {results['train_time_min']:.1f} min)"
    )


if __name__ == "__main__":
    main()
