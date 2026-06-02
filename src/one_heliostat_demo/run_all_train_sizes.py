"""
Sweep training sample counts for selected heliostats, reusing single_heliostat/train.py.

For each heliostat × each train size: runs the same two-stage training pipeline as run_all.py,
but limits the training set to `train_size` samples (uniform sampling without replacement,
nested subsets across sizes for the same heliostat).

After all runs, calls aggregate_train_sizes.aggregate() to produce comparison plots and tables.

Output layout
-------------
    outputs/one_hel_demo_train_sizes_{timestamp}/
        {hid}/
            train_size_1/
                results.json
                convergence_history.csv
                kinematic_parameters.json
                plots/
            train_size_5/
            ...
            summary.json          (per-heliostat; train_sizes + results keyed by size)
        comparison/
            mrad_vs_train_size.png
            comparison_table.txt
        run.log

Usage
-----
    python run_all_train_sizes.py
    python run_all_train_sizes.py --heliostats AC36 BE35
    python run_all_train_sizes.py --train-sizes 1 5 10 20 50 100
    python run_all_train_sizes.py --smoke-test
    python run_all_train_sizes.py --output-dir outputs/my_run
"""

import argparse
import gc
import json
import logging
import pathlib
import sys
import time
from datetime import datetime

_here = pathlib.Path(__file__).resolve().parent   # one_heliostat_demo/
_src  = _here.parent                               # src/
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_here))

from single_heliostat import config as cfg         # noqa: E402
from single_heliostat import train as tr           # noqa: E402
import aggregate_train_sizes as ats                # noqa: E402

import torch

from artist.util import constants as _const, get_device, set_logger_config
from artist.util import setup_distributed_environment

log = logging.getLogger(__name__)

DEFAULT_HELIOSTATS = ["AC36", "AG33", "AO34", "AW36", "BE35"]
DEFAULT_TRAIN_SIZES = [1, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50]

ALL_HELIOSTAT_IDS = [
    "AA23", "AA24", "AA25", "AA49",
    "AB26", "AB33", "AB43", "AB50",
    "AC24", "AC25", "AC27", "AC33", "AC35", "AC36", "AC39", "AC41", "AC47", "AC48",
    "AD39", "AD40",
    "AE23", "AE24", "AE29", "AE30", "AE32",
    "AF37", "AF38", "AF40", "AF44",
    "AG25", "AG27", "AG31", "AG33",
    "AH30",
    "AI36",
    "AJ37",
    "AK29", "AK32",
    "AM25", "AM38",
    "AN35",
    "AO32", "AO34",
    "AP29", "AP43",
    "AQ24",
    "AW36",
    "AX39",
    "AY36", "AY37", "AY39", "AY42", "AY43", "AY44",
    "AZ27", "AZ41",
    "BA28", "BA35", "BA42",
    "BD39",
    "BE25", "BE35",
    "BF39",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train-size sensitivity sweep using the one_heliostat_demo pipeline."
    )
    p.add_argument(
        "--heliostats", nargs="+", default=None, metavar="ID",
        help=f"Heliostat IDs to run (default: {DEFAULT_HELIOSTATS})",
    )
    p.add_argument(
        "--train-sizes", nargs="+", type=int, default=None, metavar="N",
        help=f"Training sample counts to sweep (default: {DEFAULT_TRAIN_SIZES})",
    )
    p.add_argument(
        "--output-dir", type=pathlib.Path, default=None,
        help="Output root (default: outputs/one_hel_demo_train_sizes_<timestamp>/)",
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Quick sanity check: first 2 heliostats, sizes [1, 5], minimal epochs.",
    )
    p.add_argument(
        "--skip-aggregation", action="store_true",
        help="Skip calling aggregate_train_sizes after all runs.",
    )
    p.add_argument(
        "--sampling-seed", type=int, default=42,
        help="Random seed for uniform sampling without replacement (default: 42).",
    )
    p.add_argument(
        "--split-type",
        choices=["balanced", "azimuth", "solstice", "high_variance"],
        default="balanced",
        help="Which synthetic dataset to use (default: balanced).",
    )
    p.add_argument(
        "--daic", action="store_true",
        help="Use DAIC cluster paths instead of local paths.",
    )
    p.add_argument(
        "--all-heliostats", action="store_true",
        help="Run on all 63 field heliostats instead of the default 5.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    _daic_paint_dir = pathlib.Path(
        "/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint"
    )
    if args.daic:
        cfg.BASE_DIR = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
        cfg.PAINT_DIR = _daic_paint_dir
        cfg.SCENARIO_PATH_TEMPLATE = str(
            cfg.BASE_DIR / "scenarios" / "one_heliostat_scenarios" / "{heliostat_id}" / "scenario.h5"
        )
        _synth_root = _daic_paint_dir / "synthetic"
    else:
        _synth_root = cfg.BASE_DIR / "datasets" / "synthetic"

    heliostats  = args.heliostats or (ALL_HELIOSTAT_IDS if args.all_heliostats else DEFAULT_HELIOSTATS)
    train_sizes = args.train_sizes or DEFAULT_TRAIN_SIZES

    if args.smoke_test:
        heliostats  = heliostats[:2]
        train_sizes = [1, 5]
        cfg.STAGE1_EPOCHS     = 2
        cfg.STAGE2_EPOCHS     = 2
        cfg.TRAIN_RAYS        = 5
        cfg.DISPLAY_RAYS      = 10
        cfg.MINI_BATCH_SIZE   = 5
        cfg.PLOT_EVERY        = 1

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        or cfg.BASE_DIR / "outputs" / f"one_hel_demo_train_sizes_{args.split_type}_{timestamp}"
    )
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info(f"Output dir    : {output_dir}")
    log.info(f"Split type    : {args.split_type}")
    log.info(f"Heliostats    : {heliostats}")
    log.info(f"Train sizes   : {train_sizes}")
    log.info(f"Sampling seed : {args.sampling_seed}")
    log.info(f"Smoke test    : {args.smoke_test}")

    cfg.SYNTHETIC_DATASET_DIR = _synth_root / f"{args.split_type}_dataset" / "dataset"
    dataset_dir = pathlib.Path(cfg.SYNTHETIC_DATASET_DIR)
    if not dataset_dir.exists():
        log.error(
            f"Synthetic dataset not found at {dataset_dir}.\n"
            "Run generate_all.py or run_all.py first to create it, "
            "then point cfg.SYNTHETIC_DATASET_DIR at the dataset/ folder."
        )
        sys.exit(1)
    log.info(f"Dataset       : {dataset_dir}")

    valid_ids:   list[str] = []
    skipped_ids: list[str] = []
    for hid in heliostats:
        spath = pathlib.Path(cfg.SCENARIO_PATH_TEMPLATE.format(heliostat_id=hid))
        if spath.exists():
            valid_ids.append(hid)
        else:
            skipped_ids.append(hid)
            log.warning(f"No scenario for {hid} — skipped ({spath})")

    if not valid_ids:
        log.error("No valid heliostats found. Exiting.")
        sys.exit(1)

    device = get_device()
    succeeded_ids: list[str] = []
    t_total = time.time()

    with setup_distributed_environment(
        number_of_heliostat_groups=1, device=device
    ) as ddp:
        device = ddp[_const.device]
        log.info(f"Device: {device}")

        for hid in valid_ids:
            log.info(f"\n{'=' * 60}")
            log.info(f"  Heliostat: {hid}")
            log.info(f"{'=' * 60}")

            hid_dir = output_dir / hid
            hid_dir.mkdir(parents=True, exist_ok=True)
            hid_results: dict[int, dict] = {}
            hid_failed = False

            for n in train_sizes:
                log.info(f"  train_size={n} ...")
                subdir = hid_dir / f"train_size_{n}"
                t0 = time.time()
                try:
                    results = tr.run(
                        heliostat_id=hid,
                        dataset_dir=dataset_dir,
                        output_dir=subdir,
                        cfg=cfg,
                        device=device,
                        train_size=n,
                        sampling_seed=args.sampling_seed,
                    )
                    elapsed = (time.time() - t0) / 60.0
                    log.info(
                        f"    train_size={n:3d} | "
                        f"pre={results['pre_training']['mrad_mean']:.3f}  "
                        f"s2={results['after_stage2']['mrad_mean']:.3f} mrad  "
                        f"({elapsed:.1f} min)"
                    )
                    hid_results[n] = results
                except Exception as exc:
                    elapsed = (time.time() - t0) / 60.0
                    log.error(f"    train_size={n} FAILED: {exc}", exc_info=True)
                    hid_failed = True
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()

            summary = {
                "heliostat_id": hid,
                "split_type":   args.split_type,
                "train_sizes":  train_sizes,
                "sampling_seed": args.sampling_seed,
                "results": {str(n): r for n, r in hid_results.items()},
            }
            with open(hid_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            if hid_results:
                succeeded_ids.append(hid)
                if hid_failed:
                    log.warning(f"  {hid}: some train sizes failed, but partial results saved.")

    total_min = (time.time() - t_total) / 60.0
    log.info(f"\nAll done in {total_min:.1f} min.")
    log.info(f"Succeeded: {succeeded_ids}")
    if skipped_ids:
        log.info(f"Skipped (no scenario): {skipped_ids}")

    if succeeded_ids and not args.skip_aggregation:
        log.info("Running aggregation ...")
        comparison_dir = output_dir / "comparison"
        comparison_dir.mkdir(exist_ok=True)
        ats.aggregate(output_dir=output_dir, heliostat_ids=succeeded_ids, out_dir=comparison_dir)
    elif not succeeded_ids:
        log.warning("No heliostats succeeded — skipping aggregation.")


if __name__ == "__main__":
    main()
