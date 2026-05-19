"""
Pre-flight check for one_heliostat_train_sizes on DAIC.

Run this on the DAIC login node before submitting run_one_hel_all.sh to confirm
that every required file and directory is in the expected location.

Usage
-----
    python one_heliostat_train_sizes/check_daic.py
    python one_heliostat_train_sizes/check_daic.py --heliostat-ids AC36 AG33
"""
import argparse
import pathlib
import sys

# ---------------------------------------------------------------------------
# DAIC paths (mirrors the --daic overrides in main.py)
# ---------------------------------------------------------------------------

BASE_DIR  = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
PAINT_DIR = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")
SIF_PATH  = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif")

BENCHMARK_NAME              = "benchmark_split-balanced_train-100_validation-50_deflectometry"
ONE_HELIOSTAT_SCENARIOS_DIR = BASE_DIR / "scenarios" / "one_heliostat_scenarios"
SYNTHETIC_DATA_DIR          = BASE_DIR / "scenarios" / "full_63_heli_kin_reconstruct" / "synthetic_data"
BENCHMARK_CSV               = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
CALIBRATION_PROPERTIES_DIR  = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
FLUX_IMAGE_DIR               = PAINT_DIR / BENCHMARK_NAME / "flux_image"
LOGS_DIR                    = BASE_DIR / "logs"

DEFAULT_HELIOSTATS = ["AC36", "AG33", "AO34", "AW36", "BE35"]
EXPECTED_SYNTH_SPLITS = ["train", "val", "test"]
MIN_SAMPLES = 100  # need at least 100 train samples per heliostat


def check(label: str, path: pathlib.Path, is_dir: bool = False) -> bool:
    exists = path.is_dir() if is_dir else path.is_file()
    status = "OK      " if exists else "MISSING "
    print(f"  [{status}] {label}")
    print(f"           {path}")
    return exists


def check_heliostat_scenario(hid: str) -> bool:
    path = ONE_HELIOSTAT_SCENARIOS_DIR / hid / "scenario.h5"
    return check(f"scenario  {hid}", path)


def check_heliostat_synth(hid: str) -> bool:
    ok = True
    for split in EXPECTED_SYNTH_SPLITS:
        hel_dir = SYNTHETIC_DATA_DIR / split / hid
        if not hel_dir.is_dir():
            print(f"  [MISSING ] synthetic/{split}/{hid}/")
            print(f"             {hel_dir}")
            ok = False
            continue
        n_samples = sum(1 for p in hel_dir.iterdir() if p.is_dir())
        enough = n_samples >= MIN_SAMPLES if split == "train" else n_samples >= 50
        status = "OK      " if enough else "WARN    "
        print(f"  [{status}] synthetic/{split}/{hid}/  ({n_samples} samples)")
        if not enough:
            ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-flight check for one_heliostat_train_sizes on DAIC."
    )
    parser.add_argument(
        "--heliostat-ids", nargs="+", default=DEFAULT_HELIOSTATS,
        help=f"Heliostat IDs to check (default: {DEFAULT_HELIOSTATS}).",
    )
    args = parser.parse_args()

    failures = 0

    print("\n=== Infrastructure ===")
    if not check("Apptainer SIF", SIF_PATH):                failures += 1
    if not check("project base dir", BASE_DIR, is_dir=True): failures += 1

    print("\n=== Synthetic data (shared with full_63) ===")
    if not check("synthetic_data/", SYNTHETIC_DATA_DIR, is_dir=True): failures += 1
    if not check("perturbations.json", SYNTHETIC_DATA_DIR / "perturbations.json"): failures += 1

    for hid in args.heliostat_ids:
        print(f"\n=== Heliostat {hid} ===")
        if not check_heliostat_scenario(hid): failures += 1
        if not check_heliostat_synth(hid):   failures += 1

    print("\n=== Output dirs (will be created by the job if missing) ===")
    check("logs/", LOGS_DIR, is_dir=True)

    print()
    if failures:
        print(f"RESULT: {failures} check(s) FAILED — fix the above before submitting.\n")
        sys.exit(1)
    else:
        print("RESULT: all checks passed — safe to submit.\n")


if __name__ == "__main__":
    main()
