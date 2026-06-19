"""
Build the sun-position universe from all PAINT calibration data.

Reads every *-calibration-properties.json file found under every benchmark
directory in the PAINT root, extracts (sun_azimuth, sun_elevation) pairs
(both in degrees), deduplicates, and writes the result to a JSON file as a
sorted list of [azimuth, elevation] pairs.

The output file is a static asset committed to the repository and consumed by
generate_full_field_dataset.py.  Re-run this script only when new PAINT data
is added.

Usage
-----
    python build_sun_universe.py
    python build_sun_universe.py --paint-dir /path/to/paint --output sun_universe.json
    python build_sun_universe.py --daic
"""

import argparse
import json
import logging
import pathlib
import sys

from tqdm import tqdm

_here = pathlib.Path(__file__).resolve().parent          # daic_full_field/
_src  = _here.parent.parent                              # src/
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_here.parent))                    # one_heliostat_demo/

from single_heliostat import config as cfg               # noqa: E402

log = logging.getLogger(__name__)


def build_universe(paint_dir: pathlib.Path) -> list[tuple[float, float]]:
    """
    Walk all calibration-properties JSON files under paint_dir and return a
    sorted, deduplicated list of (azimuth_deg, elevation_deg) tuples.
    """
    paint_dir = pathlib.Path(paint_dir)
    if not paint_dir.exists():
        raise FileNotFoundError(f"PAINT directory not found: {paint_dir}")

    # Collect every JSON file whose name ends with -calibration-properties.json
    # across all benchmark subdirectories and all splits.
    json_files = sorted(paint_dir.glob("**/train/*-calibration-properties.json"))
    json_files += sorted(paint_dir.glob("**/validation/*-calibration-properties.json"))
    json_files += sorted(paint_dir.glob("**/test/*-calibration-properties.json"))

    log.info(f"Found {len(json_files)} calibration JSON files under {paint_dir}")

    universe: set[tuple[float, float]] = set()
    skipped = 0

    for path in tqdm(json_files, desc="Reading calibration files", unit="file"):
        try:
            with open(path) as fh:
                data = json.load(fh)
            az  = float(data["sun_azimuth"])
            el  = float(data["sun_elevation"])
            universe.add((az, el))
        except (KeyError, ValueError, json.JSONDecodeError):
            skipped += 1

    log.info(
        f"Unique (azimuth, elevation) pairs : {len(universe)}"
        + (f"  ({skipped} files skipped due to parse errors)" if skipped else "")
    )

    return sorted(universe)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build sun-position universe from all PAINT calibration files."
    )
    p.add_argument(
        "--paint-dir", type=pathlib.Path, default=None,
        help="Path to the PAINT dataset root (default: cfg.PAINT_DIR).",
    )
    p.add_argument(
        "--output", type=pathlib.Path, default=None,
        help="Output JSON file path (default: datasets/sun_universe.json).",
    )
    p.add_argument(
        "--daic", action="store_true",
        help="Use DAIC cluster paths.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    if args.daic:
        cfg.PAINT_DIR = pathlib.Path(
            "/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint"
        )

    paint_dir  = args.paint_dir  or cfg.PAINT_DIR
    output     = args.output     or (cfg.BASE_DIR / "datasets" / "sun_universe.json")

    log.info(f"PAINT dir : {paint_dir}")
    log.info(f"Output    : {output}")

    universe = build_universe(paint_dir)

    output = pathlib.Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fh:
        json.dump([[az, el] for az, el in universe], fh)

    log.info(f"Written {len(universe)} entries → {output}")

    # Print summary statistics.
    azimuths   = [az for az, _ in universe]
    elevations = [el for _, el in universe]
    print(f"\nSun universe summary")
    print(f"  Entries   : {len(universe)}")
    print(f"  Azimuth   : {min(azimuths):.2f}° – {max(azimuths):.2f}°")
    print(f"  Elevation : {min(elevations):.2f}° – {max(elevations):.2f}°")
    print(f"  Output    : {output}")


if __name__ == "__main__":
    main()
