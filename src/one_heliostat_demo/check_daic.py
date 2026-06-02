"""
Pre-flight check for one_heliostat_demo on DAIC.

Run on the DAIC login node before submitting any sbatch job to confirm that
every required file and directory is in the expected place.

Checks
------
  - Apptainer SIF + project base dir
  - PAINT benchmark CSV + calibration_properties + flux_image  (per split type)
  - Synthetic dataset train/val/test sample counts              (per split type, per heliostat)
  - Single-heliostat scenario.h5                                (per heliostat)

Usage
-----
    python one_heliostat_demo/check_daic.py
    python one_heliostat_demo/check_daic.py --split-types azimuth
    python one_heliostat_demo/check_daic.py --heliostat-ids AC36 AG33 BE35
    python one_heliostat_demo/check_daic.py --all-heliostats
"""

import argparse
import pathlib
import sys

# ---------------------------------------------------------------------------
# DAIC paths
# ---------------------------------------------------------------------------

BASE_DIR  = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
PAINT_DIR = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")
SIF_PATH  = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif")

SCENARIOS_DIR = BASE_DIR / "scenarios" / "one_heliostat_scenarios"
LOGS_DIR      = BASE_DIR / "logs"

SPLIT_BENCHMARKS = {
    "balanced":      "benchmark_split-balanced_train-100_validation-50_deflectometry",
    "azimuth":       "benchmark_split-azimuth_train-100_validation-50_deflectometry",
    "solstice":      "benchmark_split-solstice_train-100_validation-50_deflectometry",
    "high_variance": "benchmark_split-high_variance_train-100_validation-50_deflectometry",
}

# Minimum expected sample counts per split (from config.py)
MIN_SAMPLES = {"train": 50, "val": 20, "test": 20}

DEFAULT_SPLIT_TYPES = ["balanced", "azimuth"]

DEFAULT_HELIOSTATS = ["AC36", "AG33", "AO34", "AW36", "BE35"]

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(label: str, path: pathlib.Path, is_dir: bool = False) -> bool:
    exists  = path.is_dir() if is_dir else path.is_file()
    status  = "OK     " if exists else "MISSING"
    print(f"  [{status}] {label}")
    if not exists:
        print(f"            {path}")
    return exists


def _count_samples(split_dir: pathlib.Path) -> int:
    if not split_dir.is_dir():
        return 0
    return sum(1 for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit())


# ---------------------------------------------------------------------------
# Section checks
# ---------------------------------------------------------------------------

def check_infrastructure() -> int:
    print("\n=== Infrastructure ===")
    failures = 0
    if not _ok("Apptainer SIF",    SIF_PATH):               failures += 1
    if not _ok("Project base dir", BASE_DIR,  is_dir=True): failures += 1
    return failures


def check_paint_benchmark(split_type: str) -> int:
    benchmark = SPLIT_BENCHMARKS[split_type]
    print(f"\n=== PAINT benchmark — {split_type} ===")
    failures = 0
    if not _ok("splits CSV",               PAINT_DIR / "splits" / f"{benchmark}.csv"):              failures += 1
    if not _ok("calibration_properties/",  PAINT_DIR / benchmark / "calibration_properties", True): failures += 1
    if not _ok("flux_image/",              PAINT_DIR / benchmark / "flux_image",             True): failures += 1
    return failures


def check_synthetic_dataset(split_type: str, heliostat_ids: list[str]) -> int:
    dataset_dir = PAINT_DIR / "synthetic" / f"{split_type}_dataset" / "dataset"
    print(f"\n=== Synthetic dataset — {split_type} ({dataset_dir}) ===")
    failures = 0

    if not _ok("dataset root", dataset_dir, is_dir=True):
        print(f"     → Run: generate_all.py --split-type {split_type} --daic")
        return failures + 1

    if not _ok("perturbations.json", dataset_dir / "perturbations.json"):
        failures += 1

    warn_ids = []
    missing_ids = []
    for hid in heliostat_ids:
        hid_ok = True
        for split in ("train", "val", "test"):
            hid_dir = dataset_dir / split / hid
            n = _count_samples(hid_dir)
            if n < MIN_SAMPLES[split]:
                hid_ok = False
                break
        if not (dataset_dir / "train" / hid).is_dir():
            missing_ids.append(hid)
        elif not hid_ok:
            warn_ids.append(hid)

    ok_count = len(heliostat_ids) - len(missing_ids) - len(warn_ids)
    print(f"  [{'OK     ' if not missing_ids and not warn_ids else 'WARN   '}] "
          f"heliostat data: {ok_count}/{len(heliostat_ids)} fully ready")

    if missing_ids:
        print(f"  [MISSING] no data at all : {missing_ids}")
        failures += 1
    if warn_ids:
        print(f"  [WARN   ] low sample count: {warn_ids}")
        failures += 1

    # Per-heliostat sample count summary
    print()
    print(f"  {'Heliostat':<10} {'train':>6} {'val':>5} {'test':>5}")
    print(f"  {'-'*28}")
    for hid in heliostat_ids:
        row = f"  {hid:<10}"
        for split in ("train", "val", "test"):
            n = _count_samples(dataset_dir / split / hid)
            flag = "" if n >= MIN_SAMPLES[split] else " !"
            row += f" {n:>4}{flag:1}"
        print(row)

    return failures


def check_scenarios(heliostat_ids: list[str]) -> int:
    print(f"\n=== Single-heliostat scenarios ({len(heliostat_ids)} heliostats) ===")
    missing = []
    for hid in heliostat_ids:
        path = SCENARIOS_DIR / hid / "scenario.h5"
        if not path.is_file():
            missing.append(hid)

    ok_count = len(heliostat_ids) - len(missing)
    print(f"  [{'OK     ' if not missing else 'MISSING'}] scenario.h5: {ok_count}/{len(heliostat_ids)} present")
    if missing:
        print(f"            missing: {missing}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Pre-flight check for one_heliostat_demo on DAIC."
    )
    p.add_argument(
        "--split-types", nargs="+",
        choices=list(SPLIT_BENCHMARKS),
        default=DEFAULT_SPLIT_TYPES,
        help=f"Split types to check (default: {DEFAULT_SPLIT_TYPES}).",
    )
    p.add_argument(
        "--heliostat-ids", nargs="+", default=None, metavar="ID",
        help="Subset of heliostat IDs to check (default: 5 representative heliostats).",
    )
    p.add_argument(
        "--all-heliostats", action="store_true",
        help="Check all 63 field heliostats.",
    )
    args = p.parse_args()

    heliostat_ids = (
        args.heliostat_ids
        or (ALL_HELIOSTAT_IDS if args.all_heliostats else DEFAULT_HELIOSTATS)
    )

    print(f"\nChecking {len(heliostat_ids)} heliostat(s), split type(s): {args.split_types}")

    failures = 0
    failures += check_infrastructure()

    for split_type in args.split_types:
        failures += check_paint_benchmark(split_type)
        failures += check_synthetic_dataset(split_type, heliostat_ids)

    failures += check_scenarios(heliostat_ids)

    print(f"\n=== Output / logs dir ===")
    _ok("logs/", LOGS_DIR, is_dir=True)   # warning only, created automatically

    print()
    if failures:
        print(f"RESULT: {failures} check(s) FAILED — fix the above before submitting.\n")
        sys.exit(1)
    else:
        print("RESULT: all checks passed — safe to submit.\n")


if __name__ == "__main__":
    main()
