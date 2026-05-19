"""
Create the full-field scenario using H5ScenarioGenerator + PAINT raw data.

Auto-discovers all heliostats that have a filled deflectometry h5 file and a
heliostat-properties.json under datasets/paint/heliostats/. The latest
filled-deflectometry file is used per heliostat.

Outputs to: scenarios/full_field_scenario/scenario.h5

NOTE: NURBS fitting iterates over all discovered heliostats (typically ~376).
      On CPU this can take several hours. Run on GPU for reasonable speed.

Usage
-----
    python create_full_field_scenario.py
    python create_full_field_scenario.py --force   # overwrite existing
"""
import argparse
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
PAINT_HELIOSTATS_DIR = BASE_DIR / "datasets" / "paint" / "heliostats"

ARTIST_DIR = BASE_DIR.parent / "ARTIST"
TOWER_FILE = ARTIST_DIR / "tutorials" / "data" / "paint" / "tower-measurements.json"

OUTPUT_PATH = BASE_DIR / "scenarios" / "full_field_scenario" / "scenario.h5"

# ---------------------------------------------------------------------------
# NURBS fitting config (same as create_scenario.py)
# ---------------------------------------------------------------------------

NUMBER_OF_NURBS_CONTROL_POINTS = torch.tensor([20, 20])
NURBS_FIT_METHOD = config_dictionary.fit_nurbs_from_normals
NURBS_DEFLECTOMETRY_STEP_SIZE = 100
NURBS_FIT_TOLERANCE = 1e-10
NURBS_FIT_MAX_EPOCH = 400


def _discover_heliostats() -> list[tuple]:
    """
    Scan PAINT_HELIOSTATS_DIR for heliostats with both:
      - Deflectometry/{hid}-filled-*-deflectometry.h5   (latest file chosen)
      - Properties/{hid}-heliostat-properties.json

    Returns list of (hid, props_path, deflectometry_path) sorted by hid.
    """
    result = []
    for hid_dir in sorted(PAINT_HELIOSTATS_DIR.iterdir()):
        if not hid_dir.is_dir():
            continue
        hid = hid_dir.name
        props = hid_dir / "Properties" / f"{hid}-heliostat-properties.json"
        if not props.exists():
            continue
        defl_dir = hid_dir / "Deflectometry"
        if not defl_dir.exists():
            continue
        filled = sorted(defl_dir.glob(f"{hid}-filled-*-deflectometry.h5"))
        if not filled:
            continue
        result.append((hid, props, filled[-1]))  # latest filled file
    return result


def main() -> None:
    set_logger_config()
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    parser = argparse.ArgumentParser(description="Create full-field scenario.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing file.")
    args = parser.parse_args()

    if not TOWER_FILE.exists():
        sys.exit(f"tower-measurements.json not found: {TOWER_FILE}")

    heliostat_files = _discover_heliostats()
    if not heliostat_files:
        sys.exit(f"No heliostats with deflectometry found in {PAINT_HELIOSTATS_DIR}")

    if OUTPUT_PATH.exists() and not args.force:
        print(f"Already exists, skipping: {OUTPUT_PATH}")
        print("Use --force to overwrite.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print(f"Tower file       : {TOWER_FILE}")
    print(f"Output           : {OUTPUT_PATH}")
    print(f"Heliostats found : {len(heliostat_files)}")
    print(f"Device           : {device}")
    print("NOTE: NURBS fitting for all heliostats may take several hours on CPU.")

    # ------------------------------------------------------------------
    # Tower + target areas
    # ------------------------------------------------------------------
    power_plant_config, target_area_list_planar_config, target_area_list_cylindrical_config = (
        paint_scenario_parser.extract_paint_tower_measurements(
            tower_measurements_path=TOWER_FILE, device=device
        )
    )

    # ------------------------------------------------------------------
    # Light source
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Heliostats with fitted deflectometry surfaces
    # ------------------------------------------------------------------
    nurbs_fit_optimizer = torch.optim.Adam(
        [torch.empty(1, requires_grad=True)], lr=1e-3
    )
    nurbs_fit_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        nurbs_fit_optimizer,
        mode="min",
        factor=0.2,
        patience=50,
        threshold=1e-7,
        threshold_mode="abs",
    )

    heliostat_list_config, prototype_config = (
        paint_scenario_parser.extract_paint_heliostats_fitted_surface(
            paths=heliostat_files,
            power_plant_position=power_plant_config.power_plant_position,
            number_of_nurbs_control_points=NUMBER_OF_NURBS_CONTROL_POINTS,
            deflectometry_step_size=NURBS_DEFLECTOMETRY_STEP_SIZE,
            nurbs_fit_method=NURBS_FIT_METHOD,
            nurbs_fit_tolerance=NURBS_FIT_TOLERANCE,
            nurbs_fit_max_epoch=NURBS_FIT_MAX_EPOCH,
            nurbs_fit_optimizer=nurbs_fit_optimizer,
            nurbs_fit_scheduler=nurbs_fit_scheduler,
            device=device,
        )
    )

    # ------------------------------------------------------------------
    # Write HDF5
    # ------------------------------------------------------------------
    scenario_generator = H5ScenarioGenerator(
        file_path=OUTPUT_PATH,
        power_plant_config=power_plant_config,
        target_area_list_planar_config=target_area_list_planar_config,
        target_area_list_cylindrical_config=target_area_list_cylindrical_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    )
    scenario_generator.generate_scenario()
    print(f"Done → {OUTPUT_PATH}")
    print(f"Heliostats in scenario: {len(heliostat_files)}")


if __name__ == "__main__":
    main()
