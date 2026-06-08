"""
Generate perturbed synthetic datasets for all 63 heliostats.

For each heliostat that has a valid single-heliostat scenario, this script:
  - Samples random perturbations (resampling if needed to meet minimum counts)
  - Ray-traces synthetic flux + centroids
  - Saves to a shared output directory under dataset/{split}/{hid}/{idx:04d}/

Output layout
-------------
    outputs/one_hel_demo_dataset_all_{timestamp}/
        dataset/
            perturbations.json     (all 63 heliostats combined)
            train/{hid}/{idx:04d}/
                calibration_properties.json
                flux_image.png
            val/{hid}/{idx:04d}/...
            test/{hid}/{idx:04d}/...
        summary.json               (per-heliostat stats: attempt used, counts, time)
        run.log

Usage
-----
    python generate_all.py
    python generate_all.py --output-dir /path/to/dir
    python generate_all.py --heliostat-ids AA23 AB26 AC33   # subset
    python generate_all.py --smoke-test                      # 3 heliostats, fast settings
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

from single_heliostat import config as cfg  # noqa: E402
from single_heliostat import generate_dataset as gd  # noqa: E402

from artist.scenario.scenario import Scenario
from artist.util import constants as _const, get_device, set_logger_config
from artist.util import setup_distributed_environment

log = logging.getLogger(__name__)


# All 63 heliostats in the full_63_heli_kin_reconstruct dataset.
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


_SPLIT_BENCHMARK = {
    "balanced":      "benchmark_split-balanced_train-100_validation-50_deflectometry",
    "azimuth":       "benchmark_split-azimuth_train-100_validation-50_deflectometry",
    "solstice":      "benchmark_split-solstice_train-100_validation-50_deflectometry",
    "high_variance": "benchmark_split-high_variance_train-100_validation-50_deflectometry",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate perturbed synthetic datasets for all (or a subset of) heliostats."
    )
    p.add_argument(
        "--heliostat-ids", nargs="+", default=None, metavar="ID",
        help="Subset of heliostat IDs to process (default: all 63)",
    )
    p.add_argument(
        "--output-dir", type=pathlib.Path, default=None,
        help=(
            "Output root. Defaults to datasets/synthetic/{split_type}_dataset/ "
            "so the generated data lands at datasets/synthetic/{split_type}_dataset/dataset/."
        ),
    )
    p.add_argument(
        "--split-type",
        choices=list(_SPLIT_BENCHMARK),
        default="balanced",
        help="Which PAINT benchmark split to use for ray directions (default: balanced).",
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Quick test: first 3 heliostats, low ray counts, minimal sample counts",
    )
    p.add_argument(
        "--daic", action="store_true",
        help="Use DAIC cluster paths instead of local paths.",
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

    heliostat_ids = args.heliostat_ids or ALL_HELIOSTAT_IDS
    if args.smoke_test:
        heliostat_ids              = heliostat_ids[:3]
        cfg.GENERATE_RAYS          = 10
        cfg.MIN_TRAIN_SAMPLES      = 5
        cfg.MIN_VAL_SAMPLES        = 3
        cfg.MIN_TEST_SAMPLES       = 3
        cfg.MAX_RESAMPLE_ATTEMPTS  = 3

    # Bulk generation always uses random perturbations, one per heliostat.
    cfg.DATA_MODE = "random_synthetic"

    # Override PAINT paths for the selected split type.
    benchmark_name      = _SPLIT_BENCHMARK[args.split_type]
    cfg.BENCHMARK_CSV   = cfg.PAINT_DIR / "splits" / f"{benchmark_name}.csv"
    cfg.CALIBRATION_DIR = cfg.PAINT_DIR / benchmark_name / "calibration_properties"
    cfg.REAL_FLUX_DIR   = cfg.PAINT_DIR / benchmark_name / "flux_image"

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or _synth_root / f"{args.split_type}_dataset"
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info(f"Output dir    : {output_dir}")
    log.info(f"Split type    : {args.split_type}  ({benchmark_name})")
    log.info(f"Heliostats    : {len(heliostat_ids)}")
    log.info(f"Smoke test    : {args.smoke_test}")
    log.info(f"GENERATE_RAYS : {cfg.GENERATE_RAYS}")

    # Find which heliostats actually have a scenario on disk.
    valid_ids: list[str] = []
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

    # All single-heliostat scenarios have exactly 1 heliostat group.
    n_groups = 1
    device   = get_device()

    combined_perturbations: dict = {}
    summary: list[dict]          = []
    t_total_start = time.time()

    with setup_distributed_environment(
        number_of_heliostat_groups=n_groups, device=device
    ) as ddp:
        device = ddp[_const.device]
        log.info(f"Device: {device}")

        for i, hid in enumerate(tqdm(valid_ids, desc="Generating", unit="hel", dynamic_ncols=True)):
            log.info(
                f"[{i + 1}/{len(valid_ids)}] Generating dataset for {hid} ..."
            )
            t_hel = time.time()

            try:
                result = gd.generate(
                    heliostat_id=hid,
                    output_dir=output_dir,
                    cfg=cfg,
                    device=device,
                    seed_offset=i,
                )
                elapsed_min = (time.time() - t_hel) / 60.0

                if not result.get("succeeded", True):
                    log.warning(f"  {hid} no valid seed found — skipped.")
                    summary.append({
                        "heliostat_id": hid,
                        "status":       "no_valid_seed",
                        "attempt_used": result["attempt_used"],
                        "elapsed_min":  round(elapsed_min, 2),
                        "split_counts": {},
                    })
                    continue

                # Read per-split sample counts from the saved dataset.
                split_counts: dict[str, int] = {}
                for split in ("train", "val", "test"):
                    split_dir = result["dataset_dir"] / split / hid
                    if split_dir.exists():
                        split_counts[split] = sum(
                            1 for d in split_dir.iterdir()
                            if d.is_dir() and d.name.isdigit()
                        )

                summary.append({
                    "heliostat_id": hid,
                    "status":       "ok",
                    "attempt_used": result["attempt_used"] + 1,
                    "elapsed_min":  round(elapsed_min, 2),
                    "split_counts": split_counts,
                })

                # Accumulate perturbations from the saved file.
                pfile = result["dataset_dir"] / "perturbations.json"
                if pfile.exists():
                    with open(pfile) as f:
                        data = json.load(f)
                    combined_perturbations.update(data)

                log.info(
                    f"  {hid} done in {elapsed_min:.1f} min  "
                    f"(attempt {result['attempt_used'] + 1})  "
                    + "  ".join(f"{sp}={n}" for sp, n in split_counts.items())
                )

            except Exception as exc:
                elapsed_min = (time.time() - t_hel) / 60.0
                log.error(f"  {hid} FAILED: {exc}")
                summary.append({
                    "heliostat_id": hid,
                    "status":       "error",
                    "error":        str(exc),
                    "elapsed_min":  round(elapsed_min, 2),
                    "split_counts": {},
                })

    # Write the combined perturbations.json (all heliostats in one file).
    combined_pfile = output_dir / "dataset" / "perturbations.json"
    combined_pfile.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_pfile, "w") as f:
        json.dump(combined_perturbations, f, indent=2)
    log.info(f"Combined perturbations.json → {combined_pfile} ({len(combined_perturbations)} heliostats)")

    # Write summary.json
    n_ok           = sum(1 for s in summary if s["status"] == "ok")
    n_no_seed      = sum(1 for s in summary if s["status"] == "no_valid_seed")
    n_error        = sum(1 for s in summary if s["status"] == "error")
    total_min      = (time.time() - t_total_start) / 60.0

    summary_doc = {
        "timestamp":   timestamp,
        "output_dir":  str(output_dir),
        "n_ok":        n_ok,
        "n_no_seed":   n_no_seed,
        "n_error":     n_error,
        "n_skipped":   len(skipped_ids),
        "total_min":   round(total_min, 2),
        "heliostats":  summary,
        "skipped_ids": skipped_ids,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary_doc, f, indent=2)

    # Print final table.
    print()
    print("=" * 70)
    print(
        f"  Dataset generation complete  |  {n_ok}/{len(valid_ids)} succeeded  "
        f"|  {total_min:.1f} min"
    )
    print("=" * 70)
    print(f"  {'Heliostat':<10} {'Status':<14} {'Attempt':>7} {'Train':>6} {'Val':>5} {'Test':>5} {'Min':>6}")
    print("  " + "-" * 62)
    for s in summary:
        sc = s.get("split_counts", {})
        if s["status"] == "ok":
            print(
                f"  {s['heliostat_id']:<10} {'ok':<14} {s['attempt_used']:>7} "
                f"{sc.get('train', 0):>6} {sc.get('val', 0):>5} {sc.get('test', 0):>5} "
                f"{s['elapsed_min']:>6.1f}"
            )
        elif s["status"] == "no_valid_seed":
            print(
                f"  {s['heliostat_id']:<10} {'no valid seed':<14} {s['attempt_used']:>7} "
                f"{'—':>6} {'—':>5} {'—':>5} {s['elapsed_min']:>6.1f}"
            )
        else:
            print(
                f"  {s['heliostat_id']:<10} {'error':<14} {'—':>7}  "
                f"{s.get('error', '')[:35]}"
            )

    failed_ids = [
        s["heliostat_id"] for s in summary if s["status"] in ("no_valid_seed", "error")
    ]
    if failed_ids:
        print(f"\n  Failed ({len(failed_ids)}): {', '.join(failed_ids)}")
    if skipped_ids:
        print(f"  Skipped — no scenario ({len(skipped_ids)}): {', '.join(skipped_ids)}")
    print("=" * 70)
    print(f"\n  Output: {output_dir}")
    print()


if __name__ == "__main__":
    main()
