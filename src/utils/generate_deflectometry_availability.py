"""
Script to generate a JSON file mapping heliostat IDs to whether
deflectometry data is available locally and whether the heliostat
appears in the balanced benchmark dataset.

Run this locally (where the full PAINT dataset is present), then
upload the resulting JSON to DAIC alongside the benchmark data.

Usage:
    python src/utils/generate_deflectometry_availability.py
"""
import csv
import json
import pathlib

_HERE = pathlib.Path(__file__).parent          # src/utils/
_BASE = _HERE.parent.parent                    # master-thesis/

PAINT_DIR = _BASE / "datasets" / "paint" / "heliostats"
BENCHMARK_CSV = (
    _BASE
    / "datasets"
    / "paint"
    / "splits"
    / "benchmark_split-balanced_train-10_validation-30.csv"
)
OUTPUT_PATH = _HERE / "deflectometry_availability.json"


def _load_benchmark_heliostats(benchmark_csv: pathlib.Path) -> set[str]:
    """Return the set of heliostat IDs present in the benchmark CSV."""
    if not benchmark_csv.exists():
        print(f"WARNING: Benchmark CSV not found at {benchmark_csv}. "
              "in_benchmark will be False for all heliostats.")
        return set()
    with open(benchmark_csv, newline="") as f:
        reader = csv.DictReader(f)
        return {row["HeliostatId"] for row in reader}


def main() -> None:
    if not PAINT_DIR.exists():
        raise FileNotFoundError(f"PAINT dataset directory not found: {PAINT_DIR}")

    benchmark_heliostats = _load_benchmark_heliostats(BENCHMARK_CSV)

    availability: dict[str, dict] = {}
    for heliostat_dir in sorted(PAINT_DIR.iterdir()):
        if not heliostat_dir.is_dir():
            continue
        defl_dir = heliostat_dir / "Deflectometry"
        has_deflectometry = (
            defl_dir.is_dir()
            and any(f.suffix == ".h5" for f in defl_dir.iterdir())
        )
        availability[heliostat_dir.name] = {
            "has_deflectometry": has_deflectometry,
            "in_benchmark": heliostat_dir.name in benchmark_heliostats,
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(availability, f, indent=2, sort_keys=True)

    total = len(availability)
    with_defl = sum(v["has_deflectometry"] for v in availability.values())
    in_bench = sum(v["in_benchmark"] for v in availability.values())
    both = sum(v["has_deflectometry"] and v["in_benchmark"] for v in availability.values())
    print(f"Written to: {OUTPUT_PATH}")
    print(f"Total heliostats          : {total}")
    print(f"With deflectometry        : {with_defl}")
    print(f"In benchmark              : {in_bench}")
    print(f"With deflectometry AND in benchmark: {both}")


if __name__ == "__main__":
    main()
