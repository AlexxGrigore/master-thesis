"""
Script to generate a JSON file mapping heliostat IDs to whether
deflectometry data is available locally.

Run this locally (where the full PAINT dataset is present), then
upload the resulting JSON to DAIC alongside the benchmark data.

Usage:
    python src/generate_deflectometry_availability.py
"""
import json
import pathlib

PAINT_DIR = pathlib.Path(__file__).parent.parent / "datasets" / "paint_dataset"
OUTPUT_PATH = pathlib.Path(__file__).parent / "utils" / "deflectometry_availability.json"


def main() -> None:
    if not PAINT_DIR.exists():
        raise FileNotFoundError(f"PAINT dataset directory not found: {PAINT_DIR}")

    availability: dict[str, bool] = {}
    for heliostat_dir in sorted(PAINT_DIR.iterdir()):
        if not heliostat_dir.is_dir():
            continue
        defl_dir = heliostat_dir / "Deflectometry"
        has_deflectometry = (
            defl_dir.is_dir()
            and any(f.suffix == ".h5" for f in defl_dir.iterdir())
        )
        availability[heliostat_dir.name] = has_deflectometry

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(availability, f, indent=2, sort_keys=True)

    total = len(availability)
    with_defl = sum(availability.values())
    print(f"Written to: {OUTPUT_PATH}")
    print(f"Total heliostats : {total}")
    print(f"With deflectometry : {with_defl}")
    print(f"Without deflectometry: {total - with_defl}")


if __name__ == "__main__":
    main()
