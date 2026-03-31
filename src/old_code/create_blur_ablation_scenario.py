"""
Create a scenario HDF5 containing only the 18 heliostats used in the blur ablation study.

The 18 heliostats are the same ones selected by src/blur_ablation/main.py (random.seed(42),
2 per cell in a 3-distance-band × 3-lateral-column grid).

This script does a direct HDF5 copy — no NURBS re-fitting needed:
  - Copies power_plant, target_areas, lightsources, prototypes unchanged.
  - Copies only the 18 selected heliostat entries under /heliostats/.
  - Keeps number_of_heliostat_groups = 1 (all heliostats share the same kinematic/actuator type).

Run locally:
    cd master-thesis/src
    python create_blur_ablation_scenario.py
"""

import pathlib
import sys

import h5py

# ===================================================================
# Paths
# ===================================================================

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

SOURCE_SCENARIO = (
    BASE_DIR / "scenarios" / "deflectometry_scenario" / "deflectometry_scenario.h5"
)
OUTPUT_SCENARIO = (
    BASE_DIR / "scenarios" / "blur_ablation_scenario" / "blur_ablation_scenario.h5"
)

if not SOURCE_SCENARIO.exists():
    sys.exit(f"ERROR: source scenario not found: {SOURCE_SCENARIO}")

OUTPUT_SCENARIO.parent.mkdir(parents=True, exist_ok=True)

# ===================================================================
# The 18 heliostats selected by the blur ablation (random.seed(42),
# 2 per (distance-band × lateral-column) cell)
# ===================================================================

SELECTED_HELIOSTATS = [
    # near / left
    "AH29", "AA30",
    # near / mid
    "AA33", "AL35",
    # near / right
    "AC37", "AB47",
    # mid / left
    "AQ28", "AQ25",
    # mid / mid
    "BA37", "AW40",
    # mid / right
    "AZ43", "AZ45",
    # far / left
    "BC32", "BF29",
    # far / mid
    "AZ55", "AZ52",
    # far / right
    "AY72", "BA71",
]

# ===================================================================
# Copy
# ===================================================================

print(f"Source:  {SOURCE_SCENARIO}")
print(f"Output:  {OUTPUT_SCENARIO}")
print(f"Copying {len(SELECTED_HELIOSTATS)} heliostats …")

with h5py.File(SOURCE_SCENARIO, "r") as src, h5py.File(OUTPUT_SCENARIO, "w") as dst:
    # Preserve file-level attributes (e.g. version).
    for key, val in src.attrs.items():
        dst.attrs[key] = val

    # Copy all top-level groups/datasets except /heliostats verbatim.
    for key in src.keys():
        if key != "heliostats":
            src.copy(key, dst)
            print(f"  Copied /{key}")

    # Copy only the selected heliostats.
    dst_heliostats = dst.require_group("heliostats")
    missing = []
    for name in SELECTED_HELIOSTATS:
        if name not in src["heliostats"]:
            missing.append(name)
            continue
        src.copy(f"heliostats/{name}", dst_heliostats, name=name)

    if missing:
        print(f"\nWARNING: heliostats not found in source: {missing}")

    print(f"\n  Copied {len(SELECTED_HELIOSTATS) - len(missing)} heliostat entries.")
    print(f"  Total heliostats in output: {len(dst['heliostats'])}")

print(f"\nDone. Scenario saved to:\n  {OUTPUT_SCENARIO}")
