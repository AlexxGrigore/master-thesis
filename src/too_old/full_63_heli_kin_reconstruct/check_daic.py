"""
Pre-flight check for full_63_heli_kin_reconstruct on DAIC.

Run this on the DAIC login node before submitting the sbatch job to confirm
that every required file and directory is in the expected location.

Usage
-----
    python full_63_heli_kin_reconstruct/check_daic.py
    python full_63_heli_kin_reconstruct/check_daic.py --dataset-type real
    python full_63_heli_kin_reconstruct/check_daic.py --dataset-type synthetic
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

BENCHMARK_NAME             = "benchmark_split-balanced_train-100_validation-50_deflectometry"
SCENARIO_PATH              = BASE_DIR / "scenarios" / "full_63_heli_kin_reconstruct" / "scenario.h5"
SYNTHETIC_DATA_DIR         = BASE_DIR / "scenarios" / "full_63_heli_kin_reconstruct" / "synthetic_data"
BENCHMARK_CSV              = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
CALIBRATION_PROPERTIES_DIR = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
FLUX_IMAGE_DIR             = PAINT_DIR / BENCHMARK_NAME / "flux_image"
LOGS_DIR                   = BASE_DIR / "logs"
OUTPUTS_DIR                = BASE_DIR / "outputs"

EXPECTED_SYNTH_SPLITS  = ["train", "val", "test"]
MIN_HELIOSTATS_PER_SPLIT = 60  # expect at least 60 of 63 heliostats present


def check(label: str, path: pathlib.Path, is_dir: bool = False) -> bool:
    exists = path.is_dir() if is_dir else path.is_file()
    status = "OK      " if exists else "MISSING "
    print(f"  [{status}] {label}")
    print(f"           {path}")
    return exists


def check_synthetic_data() -> bool:
    ok = True
    for split in EXPECTED_SYNTH_SPLITS:
        split_dir = SYNTHETIC_DATA_DIR / split
        if not split_dir.is_dir():
            print(f"  [MISSING ] synthetic/{split}/")
            print(f"             {split_dir}")
            ok = False
            continue
        n_hels = sum(1 for p in split_dir.iterdir() if p.is_dir())
        status = "OK      " if n_hels >= MIN_HELIOSTATS_PER_SPLIT else "WARN    "
        print(f"  [{status}] synthetic/{split}/  ({n_hels} heliostat dirs)")
        print(f"             {split_dir}")
        if n_hels < MIN_HELIOSTATS_PER_SPLIT:
            ok = False
    return ok


def check_real_data() -> bool:
    ok = True
    ok &= check("benchmark CSV", BENCHMARK_CSV)
    ok &= check("calibration_properties/", CALIBRATION_PROPERTIES_DIR, is_dir=True)
    ok &= check("flux_image/", FLUX_IMAGE_DIR, is_dir=True)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-flight check for full_63_heli_kin_reconstruct on DAIC."
    )
    parser.add_argument(
        "--dataset-type", choices=["synthetic", "real", "both"], default="both",
        help="Which dataset type to check (default: both).",
    )
    args = parser.parse_args()

    failures = 0

    print("\n=== Infrastructure ===")
    if not check("Apptainer SIF", SIF_PATH):               failures += 1
    if not check("project base dir", BASE_DIR, is_dir=True): failures += 1

    print("\n=== Scenario ===")
    if not check("full-field scenario.h5", SCENARIO_PATH): failures += 1

    if args.dataset_type in ("synthetic", "both"):
        print("\n=== Synthetic data ===")
        if not check("synthetic_data/", SYNTHETIC_DATA_DIR, is_dir=True): failures += 1
        if not check("perturbations.json", SYNTHETIC_DATA_DIR / "perturbations.json"): failures += 1
        if not check_synthetic_data(): failures += 1

    if args.dataset_type in ("real", "both"):
        print("\n=== Real (PAINT) data ===")
        if not check_real_data(): failures += 1

    print("\n=== Output dirs (will be created by the job if missing) ===")
    check("logs/", LOGS_DIR, is_dir=True)
    check("outputs/", OUTPUTS_DIR, is_dir=True)

    print()
    if failures:
        print(f"RESULT: {failures} check(s) FAILED — fix the above before submitting.\n")
        sys.exit(1)
    else:
        print("RESULT: all checks passed — safe to submit.\n")


if __name__ == "__main__":
    main()
