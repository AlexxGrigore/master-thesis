"""
Run the full one-heliostat pipeline (generate + train) for all (or a subset of) heliostats,
then aggregate the results.

For each heliostat with a valid single-heliostat scenario:
  1. generate_dataset.generate()  — sample random perturbations, ray-trace, save dataset
  2. train.run()                  — two-stage kinematic reconstruction training

Per-heliostat outputs are written to output_dir/{hid}/.
After the loop, combined perturbations and kinematic parameters are saved to output_dir/.
aggregate_results.aggregate() is then called automatically.

Output layout
-------------
    outputs/one_hel_demo_run_all_{timestamp}/
        {hid}/
            dataset/
                perturbations.json      (GT perturbations applied during generation)
                train/{idx:04d}/...
                val/{idx:04d}/...
                test/{idx:04d}/...
            results.json                (pre/s1/s2 mrad metrics)
            convergence_history.csv
            kinematic_parameters.json   (final optimized kinematic parameters)
            kinematic_history.json
            metrics_table.txt
            plots/...
        all_perturbations.json          (GT perturbations for all heliostats combined)
        all_kinematic_parameters.json   (final optimized params for all heliostats combined)
        summary.json
        run.log
        aggregated/
            field_mrad_convergence.png
            field_loss_curves.png
            accuracy_histogram.png
            summary_table.txt
            summary_table.csv

Usage
-----
    python run_all.py
    python run_all.py --output-dir /path/to/dir
    python run_all.py --heliostat-ids AA23 AB26 AC33   # subset
    python run_all.py --smoke-test                      # 3 heliostats, fast settings
    python run_all.py --skip-dataset-gen --output-dir /existing/dir
    python run_all.py --skip-aggregation
"""

import argparse
import json
import logging
import pathlib
import sys
import time
from datetime import datetime

from tqdm import tqdm

_here = pathlib.Path(__file__).resolve().parent   # one_heliostat_demo/
_src  = _here.parent                               # src/
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_here))

from single_heliostat import config as cfg         # noqa: E402
from single_heliostat import generate_dataset as gd  # noqa: E402
from single_heliostat import train as tr           # noqa: E402
import aggregate_results as ar                     # noqa: E402

from artist.util import constants as _const, get_device, set_logger_config
from artist.util import setup_distributed_environment

log = logging.getLogger(__name__)


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
        description="Generate + train for all (or a subset of) heliostats, then aggregate."
    )
    p.add_argument(
        "--heliostat-ids", nargs="+", default=None, metavar="ID",
        help="Subset of heliostat IDs to process (default: all 63)",
    )
    p.add_argument(
        "--output-dir", type=pathlib.Path, default=None,
        help="Output root (default: outputs/one_hel_demo_run_all_<timestamp>/)",
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Quick sanity check: first 3 heliostats, minimal epochs/rays",
    )
    p.add_argument(
        "--skip-dataset-gen", action="store_true",
        help="Skip generation; use the shared SYNTHETIC_DATASET_DIR from config",
    )
    p.add_argument(
        "--skip-aggregation", action="store_true",
        help="Skip calling aggregate_results after training",
    )
    p.add_argument(
        "--daic", action="store_true",
        help="Use DAIC cluster paths instead of local paths.",
    )

    # Dataset splitter
    p.add_argument("--split-type", choices=["azimuth", "balanced"], default=None,
                   help="DatasetSplitter strategy (overrides config.SPLITTER_TYPE)")
    p.add_argument("--train-size", type=int, default=None,
                   help="Training samples drawn from the pool (overrides config.SPLITTER_TRAIN_SIZE)")
    p.add_argument("--val-size",   type=int, default=None,
                   help="Val/test samples reserved by the splitter (overrides config.SPLITTER_VAL_SIZE)")
    p.add_argument("--target", default=None,
                   help="Restrict real data to a single aim target_name "
                        "(e.g. solar_tower_juelich_upper). Real mode only. "
                        "Overrides config.TARGET_FILTER.")
    swap_grp = p.add_mutually_exclusive_group()
    swap_grp.add_argument("--swap-val-test",    dest="swap_val_test", action="store_true",  default=None,
                          help="test_flux = VALIDATION_INDEX, val_flux = TEST_INDEX (overrides config)")
    swap_grp.add_argument("--no-swap-val-test", dest="swap_val_test", action="store_false",
                          help="Keep DatasetSplitter assignment as-is")

    # Data mode
    p.add_argument("--data-mode", choices=["synthetic", "random_synthetic", "real"], default=None,
                   help="'synthetic'/'random_synthetic': load from SYNTHETIC_DATASET_DIR; "
                        "'real': load actual PAINT calibration images (implies --skip-dataset-gen)")

    # Stage control
    p.add_argument("--skip-stage1", action="store_true",
                   help="Run only Stage 2 (FocalSpotLoss); skip Stage 1 (AlignmentLoss)")
    p.add_argument("--skip-stage2", action="store_true",
                   help="Run only Stage 1 (AlignmentLoss); skip Stage 2 (FocalSpotLoss)")
    p.add_argument("--stage1-epochs", type=int, default=None,
                   help="Override cfg.STAGE1_EPOCHS (default: 20)")
    p.add_argument("--stage2-epochs", type=int, default=None,
                   help="Override cfg.STAGE2_EPOCHS (default: 100)")
    p.add_argument("--stage1-loss", choices=["motor_steps", "motor_mse", "normal_mrad"], default=None,
                   help="Stage 1 loss: 'motor_steps' (increment-normalized motor steps, "
                        "no angle conversion, default), 'motor_mse' (AlignmentLoss, angle space), "
                        "or 'normal_mrad' (NormalAlignmentLoss)")

    return p.parse_args()


def _print_table(
    summary: list[dict],
    skipped_ids: list[str],
    total_min: float,
    output_dir: pathlib.Path,
) -> None:
    n_ok     = sum(1 for s in summary if s["status"] == "ok")
    n_failed = len(summary) - n_ok

    print()
    print("=" * 80)
    print(
        f"  RUN ALL COMPLETE  |  {n_ok}/{len(summary)} succeeded  |  "
        f"{n_failed} failed  |  {total_min:.1f} min"
    )
    print("=" * 80)
    print(
        f"  {'Heliostat':<10} {'Status':<8} {'Pre mrad':>9} "
        f"{'S1 mrad':>8} {'S2 mrad':>8} {'Improv%':>8} {'Min':>6}"
    )
    print("  " + "-" * 65)
    for s in summary:
        if s["status"] == "ok":
            pre = s["pre_training"]["mrad_mean"]
            s1  = s["after_stage1"]["mrad_mean"]
            s2  = s["after_stage2"]["mrad_mean"]
            imp = (pre - s2) / pre * 100 if pre > 0 else 0.0
            print(
                f"  {s['heliostat_id']:<10} {'ok':<8} {pre:>9.4f} "
                f"{s1:>8.4f} {s2:>8.4f} {imp:>7.1f}% {s['elapsed_min']:>6.1f}"
            )
        else:
            print(
                f"  {s['heliostat_id']:<10} {'FAILED':<8}  "
                f"{s.get('error', '')[:50]}"
            )

    ok_rows = [s for s in summary if s["status"] == "ok"]
    if ok_rows:
        import numpy as np
        s2_means = [s["after_stage2"]["mrad_mean"] for s in ok_rows]
        print("  " + "-" * 65)
        print(f"  {'MEAN':<10} {'':>8} {'':>9} {'':>8} {float(np.mean(s2_means)):>8.4f}")
        print(f"  {'MEDIAN':<10} {'':>8} {'':>9} {'':>8} {float(np.median(s2_means)):>8.4f}")

    if skipped_ids:
        print(f"\n  Skipped (no scenario): {', '.join(skipped_ids)}")
    print("=" * 80)
    print(f"\n  Output: {output_dir}")
    print()


def main() -> None:
    args = _parse_args()

    if args.daic:
        cfg.BASE_DIR = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
        cfg.PAINT_DIR = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")
        cfg.SCENARIO_PATH_TEMPLATE = str(
            cfg.BASE_DIR / "scenarios" / "one_heliostat_scenarios" / "{heliostat_id}" / "scenario.h5"
        )
        cfg.SYNTHETIC_DATASET_DIR = cfg.PAINT_DIR / "synthetic" / "balanced_dataset" / "dataset"

    if args.split_type is not None:
        cfg.SPLITTER_TYPE = args.split_type
    if args.train_size is not None:
        cfg.SPLITTER_TRAIN_SIZE = args.train_size
    if args.val_size is not None:
        cfg.SPLITTER_VAL_SIZE = args.val_size
    if args.target is not None:
        cfg.TARGET_FILTER = args.target
    if args.swap_val_test is not None:
        cfg.SWAP_VAL_TEST = args.swap_val_test
    if args.data_mode is not None:
        cfg.DATA_MODE = args.data_mode
    if args.stage1_epochs is not None:
        cfg.STAGE1_EPOCHS = args.stage1_epochs
    if args.stage2_epochs is not None:
        cfg.STAGE2_EPOCHS = args.stage2_epochs
    if args.stage1_loss is not None:
        cfg.STAGE1_LOSS = args.stage1_loss

    # Real-data mode never needs a generation step.
    if cfg.DATA_MODE == "real":
        args.skip_dataset_gen = True

    heliostat_ids = args.heliostat_ids or ALL_HELIOSTAT_IDS
    if args.smoke_test:
        heliostat_ids             = heliostat_ids[:5]
        cfg.STAGE1_EPOCHS         = 2
        cfg.STAGE2_EPOCHS         = 2
        cfg.TRAIN_RAYS            = 5
        cfg.GENERATE_RAYS         = 10
        cfg.DISPLAY_RAYS          = 10
        cfg.PLOT_EVERY            = 1
        cfg.MIN_TRAIN_SAMPLES     = 5
        cfg.MIN_VAL_SAMPLES       = 3
        cfg.MIN_TEST_SAMPLES      = 3
        cfg.MAX_RESAMPLE_ATTEMPTS = 3

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        or cfg.BASE_DIR / "outputs" / f"one_hel_demo_run_all_{timestamp}"
    )
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info(f"Output dir       : {output_dir}")
    log.info(f"Heliostats       : {len(heliostat_ids)}")
    log.info(f"Smoke test       : {args.smoke_test}")
    log.info(f"Skip dataset gen : {args.skip_dataset_gen}")
    log.info(f"Skip aggregation : {args.skip_aggregation}")
    log.info(f"Splitter type    : {cfg.SPLITTER_TYPE}")
    log.info(f"Train size       : {cfg.SPLITTER_TRAIN_SIZE}")
    log.info(f"Val/test size    : {cfg.SPLITTER_VAL_SIZE}")
    log.info(f"Swap val/test    : {cfg.SWAP_VAL_TEST}")
    if args.skip_dataset_gen:
        log.info(f"Dataset dir      : {cfg.SYNTHETIC_DATASET_DIR}")

    valid_ids: list[str]   = []
    skipped_ids: list[str] = []
    for hid in heliostat_ids:
        spath = pathlib.Path(cfg.SCENARIO_PATH_TEMPLATE.format(heliostat_id=hid))
        if spath.exists():
            valid_ids.append(hid)
        else:
            skipped_ids.append(hid)
            log.warning(f"No scenario for {hid} — skipped ({spath})")

    log.info(f"Valid heliostats: {len(valid_ids)}  |  skipped: {len(skipped_ids)}")
    if not valid_ids:
        log.error("No valid heliostats found. Exiting.")
        return

    device = get_device()
    summary: list[dict]         = []
    succeeded_ids: list[str]    = []
    all_perturbations: dict     = {}
    all_kinematic_params: dict  = {}
    t_total_start = time.time()

    with setup_distributed_environment(
        number_of_heliostat_groups=1, device=device
    ) as ddp:
        device = ddp[_const.device]
        log.info(f"Device: {device}")

        pbar = tqdm(valid_ids, desc="Processing", unit="hel", dynamic_ncols=True)
        for i, hid in enumerate(pbar):
            pbar.set_postfix(hel=hid, status="running")
            log.info(f"[{i + 1}/{len(valid_ids)}] Processing {hid} ...")
            t_hel  = time.time()
            hid_dir = output_dir / hid
            hid_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Step 1: generate (or reuse) dataset
                if args.skip_dataset_gen:
                    dataset_dir = pathlib.Path(cfg.SYNTHETIC_DATASET_DIR)
                    if not dataset_dir.exists():
                        raise FileNotFoundError(
                            f"--skip-dataset-gen set but {dataset_dir} does not exist"
                        )
                    log.info(f"  Using shared dataset: {dataset_dir}")
                else:
                    gen_result  = gd.generate(hid, hid_dir, cfg, device)
                    dataset_dir = gen_result["dataset_dir"]
                    log.info(f"  Dataset ready (attempt {gen_result['attempt_used'] + 1})")

                # Step 2: train
                results = tr.run(
                    heliostat_id=hid,
                    dataset_dir=dataset_dir,
                    output_dir=hid_dir,
                    cfg=cfg,
                    device=device,
                    skip_stage1=args.skip_stage1,
                    skip_stage2=args.skip_stage2,
                )

                elapsed_min = (time.time() - t_hel) / 60.0
                pre = results["pre_training"]
                s1  = results["after_stage1"]
                s2  = results["after_stage2"]
                log.info(
                    f"  {hid} done  pre={pre['mrad_mean']:.3f}  "
                    f"s1={s1['mrad_mean']:.3f}  s2={s2['mrad_mean']:.3f} mrad  "
                    f"({elapsed_min:.1f} min)"
                )
                pbar.set_postfix(hel=hid, status="ok", s2=f"{s2['mrad_mean']:.3f}mrad")

                summary.append({
                    "heliostat_id": hid,
                    "status":       "ok",
                    "elapsed_min":  round(elapsed_min, 2),
                    "hel_dist_m":   results["hel_dist_m"],
                    "pre_training": pre,
                    "after_stage1": s1,
                    "after_stage2": s2,
                })
                succeeded_ids.append(hid)

                # Collect per-heliostat data for combined output files.
                pfile = dataset_dir / "perturbations.json"
                if pfile.exists():
                    with open(pfile) as f:
                        all_perturbations.update(json.load(f))

                kfile = hid_dir / "kinematic_parameters.json"
                if kfile.exists():
                    with open(kfile) as f:
                        all_kinematic_params[hid] = json.load(f)

            except Exception as exc:
                elapsed_min = (time.time() - t_hel) / 60.0
                log.error(f"  {hid} FAILED: {exc}", exc_info=True)
                pbar.set_postfix(hel=hid, status="FAILED")
                summary.append({
                    "heliostat_id": hid,
                    "status":       "error",
                    "error":        str(exc),
                    "elapsed_min":  round(elapsed_min, 2),
                })

    # --------------------------------------------------------------------- #
    # Save combined files                                                     #
    # --------------------------------------------------------------------- #
    all_pert_path = output_dir / "all_perturbations.json"
    with open(all_pert_path, "w") as f:
        json.dump(all_perturbations, f, indent=2)
    log.info(
        f"Combined perturbations    → {all_pert_path} "
        f"({len(all_perturbations)} heliostats)"
    )

    all_kin_path = output_dir / "all_kinematic_parameters.json"
    with open(all_kin_path, "w") as f:
        json.dump(all_kinematic_params, f, indent=2)
    log.info(
        f"Combined kinematic params → {all_kin_path} "
        f"({len(all_kinematic_params)} heliostats)"
    )

    # --------------------------------------------------------------------- #
    # Write summary.json                                                      #
    # --------------------------------------------------------------------- #
    total_min = (time.time() - t_total_start) / 60.0
    n_ok      = sum(1 for s in summary if s["status"] == "ok")
    summary_doc = {
        "timestamp":   timestamp,
        "output_dir":  str(output_dir),
        "n_ok":        n_ok,
        "n_failed":    len(summary) - n_ok,
        "n_skipped":   len(skipped_ids),
        "total_min":   round(total_min, 2),
        "heliostats":  summary,
        "skipped_ids": skipped_ids,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary_doc, f, indent=2)

    _print_table(summary, skipped_ids, total_min, output_dir)

    # --------------------------------------------------------------------- #
    # Aggregate                                                               #
    # --------------------------------------------------------------------- #
    if succeeded_ids and not args.skip_aggregation:
        log.info("Running aggregation ...")
        ar.aggregate(output_dir=output_dir, heliostat_ids=succeeded_ids)
    elif not succeeded_ids:
        log.warning("No heliostats succeeded — skipping aggregation.")


if __name__ == "__main__":
    main()
