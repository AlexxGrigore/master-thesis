"""
One-heliostat kinematic reconstruction demo — main entry point.

Orchestrates:
  1. Generate a perturbed synthetic dataset (generate_dataset.py)
  2. Train the two-stage kinematic reconstructor (train.py)
  3. Print a summary metrics table

Usage
-----
  python main.py                                # use config.py defaults
  python main.py --heliostat-id AC33
  python main.py --heliostat-id AC33 --output-dir /tmp/demo
  python main.py --skip-dataset-gen --output-dir /tmp/demo  # reuse existing dataset
  python main.py --smoke-test                  # tiny run for CI / local sanity check
"""

import argparse
import json
import logging
import pathlib
import sys
import time
from datetime import datetime

_here = pathlib.Path(__file__).resolve().parent   # single_heliostat/
_src  = _here.parent.parent                         # src/
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_here))

import config as cfg  # noqa: E402 (local config.py)

from artist.scenario.scenario import Scenario
from artist.util import constants as _const, get_device, set_logger_config
from artist.util import setup_distributed_environment

import generate_dataset as gd  # noqa: E402
import train as tr              # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-heliostat KR demo: generate + train.")
    p.add_argument("--heliostat-id",     default=None,
                   help="Heliostat ID (overrides config.HELIOSTAT_ID)")
    p.add_argument("--output-dir",       type=pathlib.Path, default=None,
                   help="Output directory (default: outputs/one_heliostat_demo_<timestamp>/)")
    p.add_argument("--skip-dataset-gen", action="store_true",
                   help="Skip generate_dataset step; reuse dataset already in output-dir")
    p.add_argument("--smoke-test",       action="store_true",
                   help="Tiny run: 2 Stage-1 epochs, 5 Stage-2 epochs, 10 rays")
    p.add_argument("--daic",             action="store_true",
                   help="Adjust paths for DAIC cluster (not yet implemented)")

    # Dataset splitter
    p.add_argument("--split-type", choices=["azimuth", "balanced"], default=None,
                   help="DatasetSplitter strategy (overrides config.SPLITTER_TYPE)")
    p.add_argument("--train-size", type=int, default=None,
                   help="Training samples drawn from the pool (overrides config.SPLITTER_TRAIN_SIZE)")
    p.add_argument("--val-size",   type=int, default=None,
                   help="Val/test samples reserved by the splitter (overrides config.SPLITTER_VAL_SIZE)")
    swap_grp = p.add_mutually_exclusive_group()
    swap_grp.add_argument("--swap-val-test",    dest="swap_val_test", action="store_true",  default=None,
                          help="test_flux = VALIDATION_INDEX, val_flux = TEST_INDEX (overrides config)")
    swap_grp.add_argument("--no-swap-val-test", dest="swap_val_test", action="store_false",
                          help="Keep DatasetSplitter assignment as-is")

    # Data generation mode
    p.add_argument("--auto-motor-offset", action="store_true",
                   help="Estimate and remove a per-axis motor-encoder-zero offset "
                        "from the training split before training (real data).")
    p.add_argument("--data-mode", choices=["random_synthetic", "synthetic", "real"], default=None,
                   help="'random_synthetic': random perturbations each run; "
                        "'synthetic': use CUSTOM_PERTURBATIONS_SPEC from config; "
                        "'real': load the PAINT benchmark directly "
                        "(overrides config.DATA_MODE)")

    # Stage control
    p.add_argument("--skip-stage2", action="store_true",
                   help="Run only Stage 1 (AlignmentLoss); skip Stage 2 (FocalSpotLoss)")
    p.add_argument("--stage1-epochs", type=int, default=None,
                   help="Override cfg.STAGE1_EPOCHS (default: 20)")

    return p.parse_args()


def _print_summary(results: dict) -> None:
    hid      = results["heliostat_id"]
    n_test   = results["n_test"]
    dist_m   = results["hel_dist_m"]
    tot_min  = results["total_time_min"]

    print()
    print("=" * 70)
    print(f"  RESULTS  —  {hid}  |  {n_test} test samples  |  "
          f"dist={dist_m:.0f} m  |  {tot_min:.1f} min")
    print("=" * 70)
    print(f"  {'Stage':<22} {'Mean [mrad]':>12} {'Median [mrad]':>14}")
    print("  " + "-" * 50)
    for key, label in [
        ("pre_training", "Pre-training"),
        ("after_stage1", "After Stage 1"),
        ("after_stage2", "After Stage 2"),
    ]:
        ev = results[key]
        print(f"  {label:<22} {ev['mrad_mean']:12.4f} {ev['mrad_median']:14.4f}")
    print("=" * 70)
    print()


def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------ #
    # Apply overrides                                                      #
    # ------------------------------------------------------------------ #
    heliostat_id = args.heliostat_id or cfg.HELIOSTAT_ID

    if args.split_type is not None:
        cfg.SPLITTER_TYPE = args.split_type
    if args.train_size is not None:
        cfg.SPLITTER_TRAIN_SIZE = args.train_size
    if args.val_size is not None:
        cfg.SPLITTER_VAL_SIZE = args.val_size
    if args.swap_val_test is not None:
        cfg.SWAP_VAL_TEST = args.swap_val_test
    if args.data_mode is not None:
        cfg.DATA_MODE = args.data_mode
    if args.stage1_epochs is not None:
        cfg.STAGE1_EPOCHS = args.stage1_epochs
    if args.auto_motor_offset:
        cfg.AUTO_MOTOR_OFFSET = True

    # Real-data mode never needs a generation step.
    if cfg.DATA_MODE == "real":
        args.skip_dataset_gen = True

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        or cfg.BASE_DIR / "outputs" / f"one_heliostat_demo_{heliostat_id}_{timestamp}"
    )
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        cfg.STAGE1_EPOCHS    = 2
        cfg.STAGE2_EPOCHS    = 5
        cfg.TRAIN_RAYS       = 5
        cfg.GENERATE_RAYS    = 10
        cfg.DISPLAY_RAYS     = 10
        cfg.PLOT_EVERY       = 1
        cfg.MIN_TRAIN_SAMPLES = 5
        cfg.MIN_VAL_SAMPLES   = 3
        cfg.MIN_TEST_SAMPLES  = 3

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #
    set_logger_config()
    log = logging.getLogger(__name__)
    logging.getLogger().setLevel(logging.INFO)

    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info(f"Heliostat : {heliostat_id}")
    log.info(f"Output    : {output_dir}")
    log.info(f"Smoke test: {args.smoke_test}")
    log.info(f"Skip gen  : {args.skip_dataset_gen}")

    # ------------------------------------------------------------------ #
    # Save config snapshot                                                 #
    # ------------------------------------------------------------------ #
    config_snap = {
        k: str(v) if isinstance(v, pathlib.Path) else v
        for k, v in vars(cfg).items()
        if not k.startswith("__")
    }
    config_snap["heliostat_id_runtime"] = heliostat_id
    config_snap["smoke_test"]           = args.smoke_test
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_snap, f, indent=2, default=str)

    # ------------------------------------------------------------------ #
    # Determine how many heliostat groups the scenario has (for DDP)      #
    # ------------------------------------------------------------------ #
    scenario_path = pathlib.Path(cfg.SCENARIO_PATH_TEMPLATE.format(heliostat_id=heliostat_id))
    if not scenario_path.exists():
        sys.exit(
            f"Scenario not found: {scenario_path}\n"
            "Run create_scenarios.py first."
        )
    n_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(scenario_path)
    device   = get_device()

    with setup_distributed_environment(
        number_of_heliostat_groups=n_groups, device=device
    ) as ddp:
        device = ddp[_const.device]
        log.info(f"Device: {device}")

        # ---------------------------------------------------------------- #
        # Step 1: generate dataset                                          #
        # ---------------------------------------------------------------- #
        if args.skip_dataset_gen:
            if cfg.DATA_MODE == "real":
                dataset_dir = pathlib.Path(cfg.SYNTHETIC_DATASET_DIR)  # unused by train.py in real mode
                log.info("Data mode: real — skipping dataset generation (loading PAINT benchmark directly)")
            else:
                dataset_dir = pathlib.Path(cfg.SYNTHETIC_DATASET_DIR)
                if not dataset_dir.exists():
                    sys.exit(
                        f"--skip-dataset-gen was set but SYNTHETIC_DATASET_DIR does not exist:\n"
                        f"  {dataset_dir}\n"
                        "Set cfg.SYNTHETIC_DATASET_DIR or generate the dataset first."
                    )
                log.info(f"Skipping dataset generation. Using: {dataset_dir}")
        else:
            log.info("Generating dataset...")
            t0 = time.time()
            gen_result  = gd.generate(heliostat_id, output_dir, cfg, device)
            dataset_dir = gen_result["dataset_dir"]
            log.info(
                f"Dataset generated in {(time.time() - t0) / 60:.1f} min  "
                f"(attempt {gen_result['attempt_used'] + 1})"
            )

        # ---------------------------------------------------------------- #
        # Step 2: train                                                     #
        # ---------------------------------------------------------------- #
        log.info("Starting training...")
        results = tr.run(
            heliostat_id=heliostat_id,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            cfg=cfg,
            device=device,
            skip_stage2=args.skip_stage2,
        )

    _print_summary(results)
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
