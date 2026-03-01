"""
Create a scenario HDF5 file that contains ONLY heliostats with deflectometry data.

Each included heliostat gets an individual NURBS surface fitted from its deflectometry
measurements (unlike the all-heliostats scenario which uses a shared prototype surface).

Run on DAIC:
    cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src
    apptainer exec --nv \
        --bind /tudelft.net:/tudelft.net \
        /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \
        python create_deflectometry_scenario.py

Or submit via sbatch:
    sbatch sbatch_files/create_deflectometry_scenario.sh
"""

import pathlib
import sys
from datetime import datetime

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
# Paths — adjust PAINT_DATASET_DIR if the raw PAINT data lives
# somewhere other than BASE_DIR / "datasets" / "paint_dataset" on DAIC
# ===================================================================

IS_ON_DAIC = False

if IS_ON_DAIC:
    BASE_DIR = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
    PAINT_DATASET_DIR = BASE_DIR / "datasets" / "paint_dataset"
else:
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
    PAINT_DATASET_DIR = BASE_DIR / "datasets" / "paint_dataset"

TOWER_FILE = PAINT_DATASET_DIR / "WRI1030197-tower-measurements.json"
SCENARIO_PATH = BASE_DIR / "scenarios" / "deflectometry_scenario" / "deflectometry_scenario.h5"

print(f"Running on DAIC: {IS_ON_DAIC}")
print(f"PAINT dataset dir: {PAINT_DATASET_DIR}")
print(f"Tower file:        {TOWER_FILE}")
print(f"Scenario output:   {SCENARIO_PATH}")

if not PAINT_DATASET_DIR.exists():
    sys.exit(f"ERROR: PAINT dataset directory not found: {PAINT_DATASET_DIR}")
if not TOWER_FILE.exists():
    sys.exit(f"ERROR: Tower measurements file not found: {TOWER_FILE}")


# ===================================================================
# Collect heliostats that have both Properties and Deflectometry data
# Uses deflectometry_availability.json as the authoritative list so
# that the same set of heliostats is used here and in training.
# ===================================================================

import json

DEFLECTOMETRY_AVAILABILITY_JSON = (
    pathlib.Path(__file__).resolve().parent
    / "utils"
    / "deflectometry_availability.json"
)


def _extract_datetime_from_deflectometry_filename(filepath: pathlib.Path) -> datetime:
    """Parse the timestamp embedded in a deflectometry filename.

    Expected format: {name}-filled-YYYY-MM-DDZHH-MM-SSZ-deflectometry.h5
    Returns datetime.min on parse failure so the file is still usable as a fallback.
    """
    parts = filepath.stem.split("-")
    for i, part in enumerate(parts):
        if len(part) == 4 and part.isdigit():  # year token
            try:
                date_str = f"{parts[i]}-{parts[i+1]}-{parts[i+2].split('Z')[0]}"
                time_str = (
                    f"{parts[i+2].split('Z')[1]}-{parts[i+3]}-{parts[i+4].split('Z')[0]}"
                )
                return datetime.strptime(
                    f"{date_str} {time_str.replace('-', ':')}", "%Y-%m-%d %H:%M:%S"
                )
            except (IndexError, ValueError):
                pass
    return datetime.min


def _find_latest_deflectometry_file(
    heliostat_name: str, deflectometry_folder: pathlib.Path
) -> pathlib.Path | None:
    """Return the most recent deflectometry HDF5 file for a heliostat, or None."""
    files = list(deflectometry_folder.glob(f"{heliostat_name}-filled-*-deflectometry.h5"))
    if not files:
        return None
    return max(files, key=_extract_datetime_from_deflectometry_filename)


def build_deflectometry_heliostat_list(
    paint_dataset_dir: pathlib.Path,
    availability_json: pathlib.Path,
) -> list[tuple[str, pathlib.Path, pathlib.Path]]:
    """Return (name, properties_path, deflectometry_path) for every heliostat
    that has deflectometry data AND appears in the benchmark dataset."""
    with open(availability_json) as f:
        availability: dict[str, dict] = json.load(f)

    deflectometry_names = {
        name for name, info in availability.items()
        if info["has_deflectometry"] and info["in_benchmark"]
    }
    result = []

    for name in sorted(deflectometry_names):
        heliostat_folder = paint_dataset_dir / name
        if not heliostat_folder.is_dir():
            print(f"  [skip] {name}: folder not found in PAINT dataset")
            continue

        properties_folder = heliostat_folder / "Properties"
        deflectometry_folder = heliostat_folder / "Deflectometry"

        if not properties_folder.exists() or not deflectometry_folder.exists():
            print(f"  [skip] {name}: missing Properties or Deflectometry subfolder")
            continue

        properties_files = list(properties_folder.glob(f"{name}-heliostat-properties.json"))
        if not properties_files:
            print(f"  [skip] {name}: properties JSON not found")
            continue

        deflectometry_file = _find_latest_deflectometry_file(name, deflectometry_folder)
        if deflectometry_file is None:
            print(f"  [skip] {name}: no filled deflectometry file found")
            continue

        result.append((name, properties_files[0], deflectometry_file))
        print(f"  [ok]   {name}: {deflectometry_file.name}")

    return result


if not DEFLECTOMETRY_AVAILABILITY_JSON.exists():
    sys.exit(f"ERROR: Availability JSON not found: {DEFLECTOMETRY_AVAILABILITY_JSON}")

print(f"\nUsing availability list: {DEFLECTOMETRY_AVAILABILITY_JSON}")
print("Scanning for heliostats with deflectometry data...")
heliostat_files_list = build_deflectometry_heliostat_list(
    PAINT_DATASET_DIR, DEFLECTOMETRY_AVAILABILITY_JSON
)

if not heliostat_files_list:
    sys.exit("ERROR: No heliostats with deflectometry data found. Check PAINT_DATASET_DIR.")

print(f"\nFound {len(heliostat_files_list)} heliostats with deflectometry data.")

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

# NURBS fitting — each heliostat gets its own surface fitted from deflectometry normals.
nurbs_fit_optimizer = torch.optim.Adam([torch.empty(1, requires_grad=True)], lr=1e-3)
nurbs_fit_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    nurbs_fit_optimizer,
    mode="min",
    factor=0.2,
    patience=50,
    threshold=1e-7,
    threshold_mode="abs",
)

print("\nFitting NURBS surfaces from deflectometry data (this may take a while)...")
heliostat_list_config, prototype_config = (
    paint_scenario_parser.extract_paint_heliostats_fitted_surface(
        paths=heliostat_files_list,
        power_plant_position=power_plant_config.power_plant_position,
        number_of_nurbs_control_points=torch.tensor([20, 20], device=device),
        deflectometry_step_size=100,
        nurbs_fit_method=config_dictionary.fit_nurbs_from_normals,
        nurbs_fit_tolerance=1e-10,
        nurbs_fit_max_epoch=400,
        nurbs_fit_optimizer=nurbs_fit_optimizer,
        nurbs_fit_scheduler=nurbs_fit_scheduler,
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

print(f"\nDone. Scenario saved with {len(heliostat_files_list)} deflectometry heliostats.")
print(f"Path: {SCENARIO_PATH}")
