"""
Full-63-heliostat kinematic reconstruction — per-heliostat training.

Uses the same training pipeline as one_heliostat_demo (balanced dataset, pool +
DatasetSplitter, val↔test swap, custom mini-batch loop). Results are directly
comparable to one_heliostat_demo/run_all.py.

Usage
-----
    python main.py
    python main.py --output-dir outputs/my_run
    python main.py --daic
    python main.py --smoke-test
"""
import argparse
import datetime
import gc
import json
import logging
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")

import torch
from artist.util import constants as config_dictionary, set_logger_config
from artist.util import get_device, setup_distributed_environment

_here = pathlib.Path(__file__).resolve().parent   # full_63_heli_kin_reconstruct/
_src  = _here.parent.parent                        # src/
sys.path.insert(0, str(_src))

import config as cfg
from aggregate import aggregate_results

# Import the one_heliostat_demo training function directly.
from one_heliostat_demo.single_heliostat import train as one_hel_train

log = logging.getLogger(__name__)


def _map_results(results: dict) -> dict:
    """Map one_heliostat_demo result keys to the format expected by aggregate_results().

    one_hel_train.run() returns:
        "pre_training"  / "after_stage1" / "after_stage2"  →  "mrad_mean" / "mrad_median"

    aggregate_results() expects:
        "pre_training"  / "post_stage1"  / "post_training"  →  "mean_mrad" / "median_mrad"
    """
    def _s(src_key):
        s = results.get(src_key) or {}
        return {
            "mean_mrad":   s.get("mrad_mean"),
            "median_mrad": s.get("mrad_median"),
        }

    return {
        "pre_training":  _s("pre_training"),
        "post_stage1":   _s("after_stage1"),
        "post_training": _s("after_stage2"),
        "stage2_skipped": False,
        "train_time_min": results.get("total_time_min"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-63-heliostat kinematic reconstruction (per-heliostat, balanced pipeline)."
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--daic", action="store_true")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run 3 heliostats with minimal epochs for a quick end-to-end check.",
    )
    args = parser.parse_args()

    if args.daic:
        cfg.IS_ON_DAIC = True
        cfg.BASE_DIR   = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
        cfg.PAINT_DIR  = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")
        cfg.ONE_HELIOSTAT_SCENARIOS_DIR = cfg.BASE_DIR / "scenarios" / "one_heliostat_scenarios"
        cfg.SYNTHETIC_DATA_DIR = (
            pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets")
            / "synthetic" / "balanced_dataset" / "dataset"
        )
        cfg.SCENARIO_PATH_TEMPLATE = str(
            cfg.ONE_HELIOSTAT_SCENARIOS_DIR / "{heliostat_id}" / "scenario.h5"
        )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        run_dir = args.output_dir
    elif cfg.IS_ON_DAIC:
        run_dir = cfg.BASE_DIR / "outputs" / f"full_63_balanced_{timestamp}"
    else:
        subdir = "smoke_tests" if args.smoke_test else "local_runs"
        run_dir = cfg.BASE_DIR / "outputs" / subdir / f"full_63_balanced_{timestamp}"

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

    log.info(f"Experiment     : full_63_heli_kin_reconstruct (balanced pipeline)")
    log.info(f"Dataset        : {cfg.SYNTHETIC_DATA_DIR}")
    log.info(f"Scenarios      : {cfg.ONE_HELIOSTAT_SCENARIOS_DIR}")
    log.info(f"Output dir     : {run_dir}")
    log.info(f"Smoke test     : {args.smoke_test}")

    # Collect heliostat IDs from the per-heliostat scenario directory.
    hel_scenario_dir = pathlib.Path(cfg.ONE_HELIOSTAT_SCENARIOS_DIR)
    heliostat_ids = sorted(
        p.name for p in hel_scenario_dir.iterdir()
        if p.is_dir() and (p / "scenario.h5").exists()
    )
    log.info(f"Heliostat scenarios found: {len(heliostat_ids)}")

    if args.smoke_test:
        heliostat_ids    = heliostat_ids[:3]
        cfg.STAGE1_EPOCHS = 2
        cfg.STAGE2_EPOCHS = 3
        cfg.TRAIN_RAYS    = 1
        log.info(f"SMOKE TEST: stage1=2, stage2=3, 1 train ray, {len(heliostat_ids)} heliostats.")

    # Load perturbations.json from the balanced dataset (if present).
    perturbations_json = None
    pert_path = pathlib.Path(cfg.SYNTHETIC_DATA_DIR) / "perturbations.json"
    if pert_path.exists():
        with open(pert_path) as f:
            perturbations_json = json.load(f)
        log.info(f"Loaded perturbations.json ({len(perturbations_json)} heliostats).")
    else:
        log.warning(f"perturbations.json not found at {pert_path} — param recovery skipped.")

    if perturbations_json is not None:
        with open(run_dir / "perturbations.json", "w") as f:
            json.dump(perturbations_json, f, indent=2)

    # Save config snapshot.
    config_snapshot = {
        "synthetic_data_dir":    str(cfg.SYNTHETIC_DATA_DIR),
        "scenarios_dir":         str(cfg.ONE_HELIOSTAT_SCENARIOS_DIR),
        "splitter_type":         cfg.SPLITTER_TYPE,
        "splitter_train_size":   cfg.SPLITTER_TRAIN_SIZE,
        "splitter_val_size":     cfg.SPLITTER_VAL_SIZE,
        "swap_val_test":         cfg.SWAP_VAL_TEST,
        "stage1_epochs":         cfg.STAGE1_EPOCHS,
        "stage2_epochs":         cfg.STAGE2_EPOCHS,
        "mini_batch_size":       cfg.MINI_BATCH_SIZE,
        "base_lr":               cfg.BASE_LR,
        "train_rays":            cfg.TRAIN_RAYS,
        "surface_points":        cfg.SURFACE_POINTS_PER_FACET,
        "smoke_test":            args.smoke_test,
        "output_dir":            str(run_dir),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)

    torch.manual_seed(0)
    device = get_device()
    log.info(f"Device: {device}")

    n_groups = 1
    with setup_distributed_environment(
        number_of_heliostat_groups=n_groups, device=device
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        experiment_start = time.time()
        hel_results: dict[str, dict] = {}
        skipped: list[str] = []

        for hel_idx, hid in enumerate(heliostat_ids):
            scenario_path = hel_scenario_dir / hid / "scenario.h5"
            if not scenario_path.exists():
                log.warning(f"{hid}: scenario not found — skipping.")
                skipped.append(hid)
                continue

            hel_data_dir = pathlib.Path(cfg.SYNTHETIC_DATA_DIR)
            train_dir = hel_data_dir / "train" / hid
            if not train_dir.exists() or not any(train_dir.iterdir()):
                log.warning(f"{hid}: no training data in {train_dir} — skipping.")
                skipped.append(hid)
                continue

            log.info(f"[{hel_idx + 1}/{len(heliostat_ids)}] Training {hid} …")
            hel_output_dir = run_dir / hid
            hel_output_dir.mkdir(parents=True, exist_ok=True)

            t_hel = time.time()
            try:
                results = one_hel_train.run(
                    heliostat_id=hid,
                    dataset_dir=cfg.SYNTHETIC_DATA_DIR,
                    output_dir=hel_output_dir,
                    cfg=cfg,
                    device=device,
                )
                elapsed_min = (time.time() - t_hel) / 60.0
                log.info(
                    f"  {hid} done in {elapsed_min:.1f} min  "
                    f"pre={results['pre_training']['mrad_mean']:.3f}  "
                    f"s1={results['after_stage1']['mrad_mean']:.3f}  "
                    f"s2={results['after_stage2']['mrad_mean']:.3f} mrad"
                )
                hel_results[hid] = _map_results(results)

            except Exception as exc:
                elapsed_min = (time.time() - t_hel) / 60.0
                log.error(f"  {hid} FAILED ({elapsed_min:.1f} min): {exc}", exc_info=True)
                skipped.append(hid)

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

    elapsed_s = time.time() - experiment_start
    elapsed_h, rem = divmod(int(elapsed_s), 3600)
    elapsed_m, elapsed_s2 = divmod(rem, 60)
    log.info(f"\nDone. Results in: {run_dir}")
    log.info(f"Total experiment time: {elapsed_h}h {elapsed_m}m {elapsed_s2}s")
    if combined:
        pt = combined.get("post_training", {})
        ps = combined.get("post_stage1",   {})
        log.info(f"  post-stage1  : {ps.get('mean_mrad', float('nan')):.3f} mrad (mean)")
        log.info(
            f"  post-training: {pt.get('mean_mrad', float('nan')):.3f} mrad (mean)  "
            f"median={pt.get('median_mrad', float('nan')):.3f} mrad"
        )


if __name__ == "__main__":
    main()
