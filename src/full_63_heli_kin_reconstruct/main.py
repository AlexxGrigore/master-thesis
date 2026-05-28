"""
Full-63-heliostat kinematic reconstruction experiment — per-heliostat training.

Each heliostat is trained independently using its own scenario file from
scenarios/one_heliostat_scenarios/{hid}/scenario.h5. Results are saved per
heliostat and then aggregated into a combined summary.

Usage
-----
    python main.py
    python main.py --output-dir outputs/my_run
    python main.py --daic
    python main.py --smoke-test
    python main.py --daic --dataset-type synthetic --loss-type focal_spot --stage1-epochs 100 --stage2-epochs 300
    python main.py --daic --dataset-type real       --loss-type focal_spot --stage1-epochs 100 --stage2-epochs 300
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

import torch
from artist.util import constants as config_dictionary, set_logger_config
from artist.util import get_device, setup_distributed_environment

_here = pathlib.Path(__file__).resolve().parent
_src  = _here.parent
sys.path.insert(0, str(_src))

import config as cfg
from train import run, _filter_flux_map, _normalize_mapping
from utils.evaluation import build_heliostat_data_mapping
from utils.synth_data import SyntheticDatasetParser, _equalize_mapping
from utils.synth_reporting import (
    plot_stage_convergence,
    plot_per_heliostat_accuracy_table,
    plot_per_heliostat_accuracy_histogram,
    write_summary,
)
from artist.io.paint_calibration_parser import PaintCalibrationDataParser
from reporting import (
    plot_field_accuracy_map,
    plot_contour_loss_components,
    plot_filter_stats_table,
    plot_unified_mrad,
    render_summary_table,
)
from aggregate import aggregate_results


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
            "contour":    "ContourLoss",
        }
        loss_label = loss_label_map.get(results.get("loss_type", "focal_spot"), "Stage-2 Loss")
        plot_stage_convergence(
            stage2_history, output_dir,
            stage_name=f"Stage 2 — {loss_label}",
            loss_label=loss_label,
            filename="convergence_stage2.png",
        )
        if results.get("loss_type") == "contour":
            plot_contour_loss_components(stage2_history, output_dir, split="train")
            plot_contour_loss_components(stage2_history, output_dir, split="val")

    plot_unified_mrad(
        mrad_trajectory_path=output_dir / "mrad_trajectory.json",
        output_dir=output_dir,
    )

    if not results.get("stage2_skipped"):
        plot_per_heliostat_accuracy_table(results, heliostat_ids, output_dir)
        if (output_dir / "per_heliostat_accuracy.json").exists():
            with open(output_dir / "per_heliostat_accuracy.json") as _f:
                _rows = json.load(_f)
            plot_per_heliostat_accuracy_histogram(_rows, output_dir)

        plot_field_accuracy_map(
            field_positions_path=output_dir / "field_positions.json",
            per_heliostat_mrad=results.get("post_training", {}).get("per_heliostat", {}),
            output_dir=output_dir,
        )

        render_summary_table(
            val_eval=results.get("post_training_val"),
            test_eval=results.get("post_training", {}),
            output_dir=output_dir,
        )

    write_summary(results, perturbations_json, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-63-heliostat kinematic reconstruction experiment (per-heliostat)."
    )
    parser.add_argument("--output-dir",    type=pathlib.Path, default=None)
    parser.add_argument("--loss-type",     choices=["focal_spot", "pixel", "alignment", "contour"], default=None)
    parser.add_argument("--dataset-type",  choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--daic", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 3 heliostats with minimal epochs for a quick end-to-end check.")
    parser.add_argument(
        "--no-deflectometry", dest="ideal_scenario", action="store_true",
        help="Train using the ideal (flat) scenario instead of the deflectometry one.",
    )
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
        cfg.SCENARIO_PATH               = cfg.BASE_DIR / "scenarios" / "full_63_heli_kin_reconstruct" / "scenario.h5"
        cfg.ONE_HELIOSTAT_SCENARIOS_DIR = cfg.BASE_DIR / "scenarios" / "one_heliostat_scenarios"
        cfg.SYNTHETIC_DATA_DIR          = cfg.BASE_DIR / "scenarios" / "full_63_heli_kin_reconstruct" / "synthetic_data"
        cfg.BENCHMARK_CSV               = cfg.PAINT_DIR / "splits" / f"{cfg.BENCHMARK_NAME}.csv"
        cfg.CALIBRATION_PROPERTIES_DIR  = cfg.PAINT_DIR / cfg.BENCHMARK_NAME / "calibration_properties"
        cfg.FLUX_IMAGE_DIR              = cfg.PAINT_DIR / cfg.BENCHMARK_NAME / "flux_image"

    if args.ideal_scenario:
        cfg.SCENARIO_PATH = cfg.BASE_DIR / "scenarios" / "full_63_heli_kin_reconstruct" / "scenario_ideal.h5"

    scenario_label = "ideal" if args.ideal_scenario else "deflectometry"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_type = args.dataset_type
    if args.output_dir:
        run_dir = args.output_dir
    elif cfg.IS_ON_DAIC:
        run_dir = cfg.BASE_DIR / "outputs" / f"full_63_{dataset_type}_{cfg.LOSS_TYPE}_{scenario_label}_{timestamp}"
    else:
        if args.smoke_test:
            run_dir = cfg.BASE_DIR / "outputs" / "local_runs" / "smoke_tests" / f"full_63_{timestamp}"
        else:
            run_dir = cfg.BASE_DIR / "outputs" / "local_runs" / f"full_63_{dataset_type}_{cfg.LOSS_TYPE}_{scenario_label}_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for _h in logging.getLogger().handlers:
        if isinstance(_h, logging.StreamHandler) and _h.stream is sys.stderr:
            _h.stream = sys.stdout
    log = logging.getLogger(__name__)

    fh = logging.FileHandler(run_dir / "run.log")
    fh.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
    logging.getLogger().addHandler(fh)

    _config_snapshot = {
        "benchmark_name":               cfg.BENCHMARK_NAME,
        "one_heliostat_scenarios_dir":  str(cfg.ONE_HELIOSTAT_SCENARIOS_DIR),
        "synthetic_data_dir":           str(cfg.SYNTHETIC_DATA_DIR),
        "dataset_type":                 dataset_type,
        "loss_type":                    cfg.LOSS_TYPE,
        "stage1_epochs":                cfg.STAGE1_EPOCHS,
        "stage2_epochs":                cfg.STAGE2_EPOCHS,
        "train_samples":                cfg.TRAIN_SAMPLES,
        "val_samples":                  cfg.VAL_SAMPLES,
        "test_samples":                 cfg.TEST_SAMPLES,
        "train_rays":                   cfg.TRAIN_RAYS,
        "synth_gen_rays":               cfg.SYNTH_GEN_RAYS,
        "train_surface_points":         cfg.TRAIN_SURFACE_POINTS,
        "synth_gen_surface_points":     cfg.SYNTH_GEN_SURFACE_POINTS,
        "perturbation_seed":            cfg.PERTURBATION_SEED,
        "perturbation_ranges":          cfg.PERTURBATION_RANGES,
        "optimization_config":          {str(k): v for k, v in cfg.OPTIMIZATION_CONFIG.items()},
        "contour_params":               cfg.CONTOUR_PARAMS,
        "min_focal_spot_train_samples": cfg.MIN_FOCAL_SPOT_TRAIN_SAMPLES,
        "min_val_samples":              cfg.MIN_VAL_SAMPLES,
        "min_test_samples":             cfg.MIN_TEST_SAMPLES,
        "blur_sigma":                   cfg.BLUR_SIGMA,
        "smoke_test":                   args.smoke_test,
        "output_dir":                   str(run_dir),
        "pipeline":                     "per_heliostat",
        "scenario_label":               scenario_label,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(_config_snapshot, f, indent=2)

    log.info(f"Experiment     : full_63_heli_kin_reconstruct (per-heliostat)")
    log.info(f"Dataset type   : {dataset_type}")
    log.info(f"Loss type      : {cfg.LOSS_TYPE}")
    log.info(f"Epochs         : stage1={cfg.STAGE1_EPOCHS}  stage2={cfg.STAGE2_EPOCHS}")
    log.info(f"Scenarios dir  : {cfg.ONE_HELIOSTAT_SCENARIOS_DIR}")
    log.info(f"Synthetic data : {cfg.SYNTHETIC_DATA_DIR}")
    log.info(f"Output dir     : {run_dir}")

    torch.manual_seed(0)
    device = get_device()
    log.info(f"Device: {device}")

    # Collect heliostat IDs from the per-heliostat scenarios directory.
    hel_scenario_dir = cfg.ONE_HELIOSTAT_SCENARIOS_DIR
    heliostat_ids = sorted(
        p.name for p in hel_scenario_dir.iterdir()
        if p.is_dir() and (p / "scenario.h5").exists()
    )
    log.info(f"Heliostat scenarios found: {len(heliostat_ids)}")

    # Load perturbations once (synthetic pipeline only).
    perturbations_json = None
    if dataset_type == "synthetic":
        pert_path = cfg.SYNTHETIC_DATA_DIR / "perturbations.json"
        if pert_path.exists():
            with open(pert_path) as f:
                perturbations_json = json.load(f)
            log.info("Loaded perturbations.json.")
        else:
            log.warning(
                f"perturbations.json not found at {pert_path}. "
                "Run generate_dataset.py first. Param recovery reporting will be skipped."
            )

    if perturbations_json is not None:
        with open(run_dir / "perturbations.json", "w") as f:
            json.dump(perturbations_json, f, indent=2)

    # Build full raw mappings once (all heliostats, all splits).
    log.info("Building raw data mappings …")

    def _equalized(paint_split, n_samples):
        raw = _build_paint_mapping(paint_split)
        scenario_hids = set(heliostat_ids)
        return [e for e in _equalize_mapping(raw, n_samples) if e[0] in scenario_hids]

    raw_train = _equalized("train",      cfg.TRAIN_SAMPLES)
    raw_val   = _equalized("validation", cfg.VAL_SAMPLES)
    raw_test  = _equalized("test",       cfg.TEST_SAMPLES)

    raw_train_by_hid = {hid: (cal, flux) for hid, cal, flux in raw_train}
    raw_val_by_hid   = {hid: (cal, flux) for hid, cal, flux in raw_val}
    raw_test_by_hid  = {hid: (cal, flux) for hid, cal, flux in raw_test}

    # DDP setup — n_groups=1 for all per-heliostat scenarios.
    n_groups = 1
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
                        "Run:  python generate_dataset.py"
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
            log.info("PAINT calibration parsers ready.")

        stage1_epochs = cfg.STAGE1_EPOCHS
        stage2_epochs = cfg.STAGE2_EPOCHS
        train_rays    = cfg.TRAIN_RAYS
        smoke_test_n_hels = len(heliostat_ids)
        if args.smoke_test:
            stage1_epochs     = 2
            stage2_epochs     = 3
            train_rays        = 1
            smoke_test_n_hels = 3
            log.info(f"SMOKE TEST: stage1=2, stage2=3, 1 train ray, {smoke_test_n_hels} heliostats.")

        hel_results: dict[str, dict] = {}
        skipped: list[str] = []

        for hel_idx, hid in enumerate(heliostat_ids):
            if args.smoke_test and hel_idx >= smoke_test_n_hels:
                break

            scenario_path = hel_scenario_dir / hid / "scenario.h5"
            if not scenario_path.exists():
                log.warning(f"{hid}: scenario not found at {scenario_path} — skipping.")
                skipped.append(hid)
                continue

            # Build and filter per-heliostat mappings.
            def _hel_map(by_hid, hid, dataset_type, split_name):
                if hid not in by_hid:
                    return []
                cal, flux = by_hid[hid]
                raw = [(hid, cal, flux)]
                filtered = _filter_flux_map(
                    raw, dataset_type, cfg.MIN_ACTIVE_PIXEL_PCT,
                    synth_data_dir=cfg.SYNTHETIC_DATA_DIR, split_name=split_name,
                )
                return _normalize_mapping(filtered)

            hel_train = _hel_map(raw_train_by_hid, hid, dataset_type, "train")
            hel_val   = _hel_map(raw_val_by_hid,   hid, dataset_type, "val")
            hel_test  = _hel_map(raw_test_by_hid,  hid, dataset_type, "test")

            n_train = len(hel_train[0][1]) if hel_train else 0
            n_val   = len(hel_val[0][1])   if hel_val   else 0
            n_test  = len(hel_test[0][1])  if hel_test  else 0

            # Hard skip: need at least some val and test signal for meaningful evals.
            if n_val < cfg.MIN_VAL_SAMPLES or n_test < cfg.MIN_TEST_SAMPLES:
                log.warning(
                    f"{hid}: skipped entirely "
                    f"(train={n_train}, val={n_val}, test={n_test} — "
                    f"need val≥{cfg.MIN_VAL_SAMPLES}, test≥{cfg.MIN_TEST_SAMPLES})"
                )
                skipped.append(hid)
                continue

            # Soft skip (low train samples): run() will still do Stage 1 + eval,
            # but skip Stage 2 FocalSpotLoss.
            log.info(
                f"[{hel_idx+1}/{len(heliostat_ids)}] {hid} — "
                f"train={n_train}, val={n_val}, test={n_test}"
            )

            hel_output_dir = run_dir / hid
            hel_output_dir.mkdir(parents=True, exist_ok=True)

            hel_pert = {hid: perturbations_json[hid]} if perturbations_json and hid in perturbations_json else None

            results = run(
                scenario_path=scenario_path,
                device=device,
                ddp_setup=ddp_setup,
                train_mapping=hel_train,
                val_mapping=hel_val,
                test_mapping=hel_test,
                train_parser=train_parser,
                val_parser=val_parser,
                test_parser=test_parser,
                optimization_config=cfg.OPTIMIZATION_CONFIG,
                output_dir=hel_output_dir,
                loss_type=cfg.LOSS_TYPE,
                dataset_type=dataset_type,
                n_surface_pts=cfg.TRAIN_SURFACE_POINTS,
                train_rays=train_rays,
                perturbations=hel_pert,
                heliostat_ids=[hid],
                stage1_epochs=stage1_epochs,
                stage2_epochs=stage2_epochs,
                contour_params=cfg.CONTOUR_PARAMS,
                trail_stride=cfg.CENTROID_TRAIL_STRIDE,
                trail_n_disp=cfg.CENTROID_TRAIL_N_DISP,
                min_focal_spot_samples=cfg.MIN_FOCAL_SPOT_TRAIN_SAMPLES,
                blur_sigma=cfg.BLUR_SIGMA,
            )

            hel_results[hid] = results

            _run_reporting(results, hel_pert, [hid], hel_output_dir)

            gc.collect()
            torch.cuda.empty_cache()

        log.info(
            f"Per-heliostat loop complete: "
            f"{len(hel_results)} trained, {len(skipped)} skipped."
        )
        if skipped:
            log.info(f"Skipped: {skipped}")

    log.info("Aggregating results …")
    combined = aggregate_results(hel_results, run_dir)
    log.info(f"\nDone. Results in: {run_dir}")
    if combined:
        pt = combined.get("post_training", {})
        ps = combined.get("post_stage1", {})
        log.info(f"  post-stage1  : {ps.get('mean_mrad', float('nan')):.3f} mrad (mean)")
        log.info(f"  post-training: {pt.get('mean_mrad', float('nan')):.3f} mrad (mean)  "
                 f"median={pt.get('median_mrad', float('nan')):.3f} mrad")


if __name__ == "__main__":
    main()
