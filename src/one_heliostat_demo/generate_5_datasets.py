"""
Generate 5 perturbed synthetic datasets for all 63 heliostats, each with a
different perturbation seed block, for robustness evaluation.

Seed scheme
-----------
Dataset d (1-based) uses seed_offset = (d - 1) * N_HELIOSTATS + heliostat_index,
where N_HELIOSTATS = 63.  This keeps each dataset's seed space non-overlapping
with the formula in generate_dataset.py:
    seed = cfg.RANDOM_SEED + seed_offset * (max_attempts + 1) + attempt

Output layout
-------------
    datasets/synthetic/5_datasets/
        dataset_1/
            dataset/
                perturbations.json
                train/{hid}/{idx:04d}/...
                val/{hid}/{idx:04d}/...
                test/{hid}/{idx:04d}/...
            summary.json
            run.log
        dataset_2/ ...
        dataset_5/

Usage
-----
    python generate_5_datasets.py
    python generate_5_datasets.py --n-datasets 3
    python generate_5_datasets.py --heliostat-ids AA23 AB26
    python generate_5_datasets.py --smoke-test
    python generate_5_datasets.py --daic
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

_SPLIT_BENCHMARK = {
    "balanced":      "benchmark_split-balanced_train-100_validation-50_deflectometry",
    "azimuth":       "benchmark_split-azimuth_train-100_validation-50_deflectometry",
    "solstice":      "benchmark_split-solstice_train-100_validation-50_deflectometry",
    "high_variance": "benchmark_split-high_variance_train-100_validation-50_deflectometry",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate N synthetic datasets with non-overlapping perturbation seeds."
    )
    p.add_argument(
        "--n-datasets", type=int, default=5,
        help="Number of datasets to generate (default: 5)",
    )
    p.add_argument(
        "--heliostat-ids", nargs="+", default=None, metavar="ID",
        help="Subset of heliostat IDs (default: all 63)",
    )
    p.add_argument(
        "--output-dir", type=pathlib.Path, default=None,
        help="Root output directory (default: datasets/synthetic/5_datasets/)",
    )
    p.add_argument(
        "--split-type", choices=list(_SPLIT_BENCHMARK), default="balanced",
        help="PAINT benchmark split for ray directions (default: balanced)",
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Quick test: first 3 heliostats, minimal counts",
    )
    p.add_argument(
        "--daic", action="store_true",
        help="Use DAIC cluster paths instead of local paths.",
    )
    return p.parse_args()


def _generate_one_dataset(
    dataset_idx: int,
    valid_ids: list[str],
    n_total_heliostats: int,
    output_dir: pathlib.Path,
    device,
) -> None:
    """Generate one dataset for all valid_ids with the correct seed block."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"))
    root_log = logging.getLogger()
    root_log.addHandler(fh)

    log.info(f"=== Dataset {dataset_idx} → {output_dir} ===")
    log.info(f"  Seed block: offsets {(dataset_idx - 1) * n_total_heliostats} "
             f"to {dataset_idx * n_total_heliostats - 1}")

    combined_perturbations: dict = {}
    summary: list[dict] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    t_start = time.time()

    for i, hid in enumerate(tqdm(valid_ids, desc=f"Dataset {dataset_idx}", unit="hel", dynamic_ncols=True)):
        # Non-overlapping seed_offset: each dataset shifts by n_total_heliostats
        seed_offset = (dataset_idx - 1) * n_total_heliostats + i

        log.info(f"  [{i + 1}/{len(valid_ids)}] {hid}  seed_offset={seed_offset}")
        t_hel = time.time()

        try:
            result = gd.generate(
                heliostat_id=hid,
                output_dir=output_dir,
                cfg=cfg,
                device=device,
                seed_offset=seed_offset,
            )
            elapsed_min = (time.time() - t_hel) / 60.0

            if not result.get("succeeded", True):
                log.warning(f"  {hid} no valid seed — skipped.")
                summary.append({
                    "heliostat_id": hid,
                    "status": "no_valid_seed",
                    "attempt_used": result["attempt_used"],
                    "elapsed_min": round(elapsed_min, 2),
                    "split_counts": {},
                })
                continue

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
                "status": "ok",
                "attempt_used": result["attempt_used"] + 1,
                "elapsed_min": round(elapsed_min, 2),
                "split_counts": split_counts,
                "seed_offset": seed_offset,
            })

            pfile = result["dataset_dir"] / "perturbations.json"
            if pfile.exists():
                with open(pfile) as f:
                    combined_perturbations.update(json.load(f))

            log.info(
                f"  {hid} done in {elapsed_min:.1f} min  "
                + "  ".join(f"{sp}={n}" for sp, n in split_counts.items())
            )

        except Exception as exc:
            elapsed_min = (time.time() - t_hel) / 60.0
            log.error(f"  {hid} FAILED: {exc}")
            summary.append({
                "heliostat_id": hid,
                "status": "error",
                "error": str(exc),
                "elapsed_min": round(elapsed_min, 2),
                "split_counts": {},
            })

    # Save combined perturbations
    combined_pfile = output_dir / "dataset" / "perturbations.json"
    combined_pfile.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_pfile, "w") as f:
        json.dump(combined_perturbations, f, indent=2)

    # Save summary
    n_ok = sum(1 for s in summary if s["status"] == "ok")
    total_min = (time.time() - t_start) / 60.0
    with open(output_dir / "summary.json", "w") as f:
        json.dump({
            "dataset_idx": dataset_idx,
            "timestamp": timestamp,
            "seed_offset_start": (dataset_idx - 1) * n_total_heliostats,
            "n_ok": n_ok,
            "n_error": len(summary) - n_ok,
            "total_min": round(total_min, 2),
            "heliostats": summary,
        }, f, indent=2)

    log.info(f"Dataset {dataset_idx} done: {n_ok}/{len(valid_ids)} ok  |  {total_min:.1f} min")
    root_log.removeHandler(fh)
    fh.close()


def main() -> None:
    args = _parse_args()

    if args.daic:
        cfg.BASE_DIR = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
        cfg.PAINT_DIR = pathlib.Path(
            "/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint"
        )
        cfg.SCENARIO_PATH_TEMPLATE = str(
            cfg.BASE_DIR / "scenarios" / "one_heliostat_scenarios" / "{heliostat_id}" / "scenario.h5"
        )
        _synth_root = cfg.PAINT_DIR / "synthetic"
    else:
        _synth_root = cfg.BASE_DIR / "datasets" / "synthetic"

    heliostat_ids = args.heliostat_ids or ALL_HELIOSTAT_IDS
    if args.smoke_test:
        heliostat_ids             = heliostat_ids[:3]
        cfg.GENERATE_RAYS         = 10
        cfg.MIN_TRAIN_SAMPLES     = 5
        cfg.MIN_VAL_SAMPLES       = 3
        cfg.MIN_TEST_SAMPLES      = 3
        cfg.MAX_RESAMPLE_ATTEMPTS = 3

    cfg.DATA_MODE = "random_synthetic"

    benchmark_name      = _SPLIT_BENCHMARK[args.split_type]
    cfg.BENCHMARK_CSV   = cfg.PAINT_DIR / "splits" / f"{benchmark_name}.csv"
    cfg.CALIBRATION_DIR = cfg.PAINT_DIR / benchmark_name / "calibration_properties"
    cfg.REAL_FLUX_DIR   = cfg.PAINT_DIR / benchmark_name / "flux_image"

    root_dir = args.output_dir or (_synth_root / "5_datasets")
    root_dir = pathlib.Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)

    # Validate which heliostats have scenarios on disk
    valid_ids: list[str] = []
    for hid in heliostat_ids:
        spath = pathlib.Path(cfg.SCENARIO_PATH_TEMPLATE.format(heliostat_id=hid))
        if spath.exists():
            valid_ids.append(hid)
        else:
            log.warning(f"No scenario for {hid} — skipped")

    n_total = len(ALL_HELIOSTAT_IDS)  # always 63, for seed spacing
    log.info(f"Valid heliostats: {len(valid_ids)} / {len(heliostat_ids)}")
    log.info(f"Datasets to generate: {args.n_datasets}")
    log.info(f"Seed block size (n_total_heliostats): {n_total}")
    log.info(f"Output root: {root_dir}")

    device = get_device()

    with setup_distributed_environment(number_of_heliostat_groups=1, device=device) as ddp:
        device = ddp[_const.device]
        log.info(f"Device: {device}")

        for d in range(1, args.n_datasets + 1):
            dataset_dir = root_dir / f"dataset_{d}"
            _generate_one_dataset(
                dataset_idx=d,
                valid_ids=valid_ids,
                n_total_heliostats=n_total,
                output_dir=dataset_dir,
                device=device,
            )

    print()
    print("=" * 60)
    print(f"  All {args.n_datasets} datasets generated → {root_dir}")
    print("=" * 60)
    for d in range(1, args.n_datasets + 1):
        sfile = root_dir / f"dataset_{d}" / "summary.json"
        if sfile.exists():
            with open(sfile) as f:
                s = json.load(f)
            print(f"  dataset_{d}: {s['n_ok']} ok, {s['n_error']} failed  ({s['total_min']:.1f} min)")
    print()


if __name__ == "__main__":
    main()
