"""
Create a scenario HDF5 file containing the SAME heliostats as the deflectometry
scenario, but with IDEAL (flat) surfaces instead of deflectometry-fitted NURBS.

This is the control condition for the deflectometry ablation: same heliostats,
same kinematics, same benchmark split — only the surface model differs.

  deflectometry_scenario.h5  →  surfaces fitted from measured deflectometry data
  ideal_scenario.h5           →  surfaces are perfect flat planes (canting + translation only)

Run on DAIC:
    cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src
    apptainer exec --nv \
        --bind /tudelft.net:/tudelft.net \
        /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
        python create_ideal_scenario.py

Or submit via sbatch:
    sbatch sbatch_files/create_ideal_scenario.sh
"""

import json
import pathlib
import sys

import torch

from artist.data_parser import paint_scenario_parser
from artist.scenario.configuration_classes import (
    LightSourceConfig,
    LightSourceListConfig,
)
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device

set_logger_config()

torch.manual_seed(7)
torch.cuda.manual_seed(7)

# ===================================================================
# Paths
# ===================================================================

IS_ON_DAIC = False

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent.parent

if IS_ON_DAIC:
    BASE_DIR = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
    PAINT_DATASET_DIR = BASE_DIR / "datasets" / "paint_dataset"
else:
    BASE_DIR = _SRC_DIR.parent
    PAINT_DATASET_DIR = BASE_DIR / "datasets" / "paint_dataset"

TOWER_FILE = PAINT_DATASET_DIR / "WRI1030197-tower-measurements.json"
SCENARIO_PATH = BASE_DIR / "scenarios" / "ideal_scenario" / "ideal_scenario.h5"

DEFLECTOMETRY_AVAILABILITY_JSON = (
    _SRC_DIR
    / "utils"
    / "deflectometry_availability.json"
)

print(f"Running on DAIC: {IS_ON_DAIC}")
print(f"PAINT dataset dir: {PAINT_DATASET_DIR}")
print(f"Tower file:        {TOWER_FILE}")
print(f"Scenario output:   {SCENARIO_PATH}")

if not PAINT_DATASET_DIR.exists():
    sys.exit(f"ERROR: PAINT dataset directory not found: {PAINT_DATASET_DIR}")
if not TOWER_FILE.exists():
    sys.exit(f"ERROR: Tower measurements file not found: {TOWER_FILE}")
if not DEFLECTOMETRY_AVAILABILITY_JSON.exists():
    sys.exit(f"ERROR: Availability JSON not found: {DEFLECTOMETRY_AVAILABILITY_JSON}")


# ===================================================================
# Collect the same heliostat set as the deflectometry scenario
# (heliostats that have deflectometry AND are in the benchmark),
# but only keep (name, properties_path) — no deflectometry files.
# ===================================================================

def build_ideal_heliostat_list(
    paint_dataset_dir: pathlib.Path,
    availability_json: pathlib.Path,
) -> list[tuple[str, pathlib.Path]]:
    """Return (name, properties_path) for every heliostat that has
    deflectometry data AND appears in the benchmark dataset.

    This mirrors build_deflectometry_heliostat_list() in
    create_deflectometry_scenario.py but drops the deflectometry path,
    so the resulting scenario contains the same heliostats with ideal surfaces.
    """
    with open(availability_json) as f:
        availability: dict[str, dict] = json.load(f)

    target_names = {
        name for name, info in availability.items()
        if info["has_deflectometry"] and info["in_benchmark"]
    }
    result = []

    for name in sorted(target_names):
        heliostat_folder = paint_dataset_dir / name
        if not heliostat_folder.is_dir():
            print(f"  [skip] {name}: folder not found in PAINT dataset")
            continue

        properties_folder = heliostat_folder / "Properties"
        if not properties_folder.exists():
            print(f"  [skip] {name}: missing Properties subfolder")
            continue

        properties_files = list(properties_folder.glob(f"{name}-heliostat-properties.json"))
        if not properties_files:
            print(f"  [skip] {name}: properties JSON not found")
            continue

        result.append((name, properties_files[0]))
        print(f"  [ok]   {name}")

    return result


print(f"\nUsing availability list: {DEFLECTOMETRY_AVAILABILITY_JSON}")
print("Collecting heliostats (same set as deflectometry scenario, ideal surfaces)...")
heliostat_files_list = build_ideal_heliostat_list(
    PAINT_DATASET_DIR, DEFLECTOMETRY_AVAILABILITY_JSON
)

if not heliostat_files_list:
    sys.exit("ERROR: No heliostats found. Check PAINT_DATASET_DIR.")

print(f"\nFound {len(heliostat_files_list)} heliostats.")

SCENARIO_PATH.parent.mkdir(parents=True, exist_ok=True)

device = get_device()
print(f"Device: {device}")


# ===================================================================
# Build scenario configuration
# ===================================================================

power_plant_config, target_area_list_config = (
    paint_scenario_parser.extract_paint_tower_measurements(
        tower_measurements_path=TOWER_FILE, device=device
    )
)

light_source_list_config = LightSourceListConfig(
    light_source_list=[
        LightSourceConfig(
            light_source_key="sun_1",
            light_source_type=config_dictionary.sun_key,
            number_of_rays=10,
            distribution_type=config_dictionary.light_source_distribution_is_normal,
            mean=0.0,
            covariance=4.3681e-06,
        )
    ]
)

# Ideal surfaces: no deflectometry fitting, no optimizer needed.
# Use the same 20×20 control point grid as the deflectometry scenario
# so both scenarios are structurally identical (same HDF5 layout).
print("\nGenerating ideal surfaces (no deflectometry fitting)...")
heliostat_list_config, prototype_config = (
    paint_scenario_parser.extract_paint_heliostats_ideal_surface(
        paths=heliostat_files_list,
        power_plant_position=power_plant_config.power_plant_position,
        number_of_nurbs_control_points=torch.tensor([20, 20], device=device),
        device=device,
    )
)


# ===================================================================
# Write the HDF5 scenario file
# ===================================================================

print(f"\nGenerating scenario at: {SCENARIO_PATH}")
H5ScenarioGenerator(
    file_path=SCENARIO_PATH,
    power_plant_config=power_plant_config,
    target_area_list_config=target_area_list_config,
    light_source_list_config=light_source_list_config,
    prototype_config=prototype_config,
    heliostat_list_config=heliostat_list_config,
).generate_scenario()

print(f"\nDone. Ideal scenario saved with {len(heliostat_files_list)} heliostats.")
print(f"Path: {SCENARIO_PATH}")
