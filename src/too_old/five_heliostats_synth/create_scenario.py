"""
Create the 5-heliostat scenario using H5ScenarioGenerator + PAINT raw data.

Reads tower-measurements.json and per-heliostat (heliostat-properties.json,
deflectometry.h5) to build a scenario HDF5 from scratch via ARTIST's
paint_scenario_parser. Fits NURBS surfaces from deflectometry normals
(20×20 control points, same as the ARTIST tutorial).

Run once before any training.

Usage
-----
    python create_scenario.py
    python create_scenario.py --force   # overwrite existing
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
# Path configuration
# ---------------------------------------------------------------------------

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]          # master-thesis/
PAINT_HELIOSTATS_DIR = BASE_DIR / "datasets" / "paint" / "heliostats"

# tower-measurements.json is not in the local datasets download;
# use the copy shipped with the ARTIST tutorials.
ARTIST_DIR = BASE_DIR.parent / "ARTIST"
TOWER_FILE = ARTIST_DIR / "tutorials" / "data" / "paint" / "tower-measurements.json"

OUTPUT_PATH = BASE_DIR / "scenarios" / "five_heliostats_scenario" / "scenario.h5"

# Per-heliostat PAINT file paths.
# Deflectometry: use the latest "filled" variant (interpolated gaps) per heliostat.
# Properties: {ID}-heliostat-properties.json in the Properties/ subfolder.
def _h(hid: str, defl_filename: str) -> tuple:
    return (
        hid,
        PAINT_HELIOSTATS_DIR / hid / "Properties" / f"{hid}-heliostat-properties.json",
        PAINT_HELIOSTATS_DIR / hid / "Deflectometry" / defl_filename,
    )

HELIOSTAT_FILES = [
    _h("AA31", "AA31-filled-2023-09-07T21-37-10Z-deflectometry.h5"),
    _h("AQ28", "AQ28-filled-2021-09-09T20-46-53Z-deflectometry.h5"),
    _h("BA37", "BA37-filled-2015-05-07T20-07-34Z-deflectometry.h5"),
    _h("BC33", "BC33-filled-2018-07-26T21-29-53Z-deflectometry.h5"),
    _h("AZ55", "AZ55-filled-2022-03-03T19-08-26Z-deflectometry.h5"),
]

# ---------------------------------------------------------------------------
# NURBS fitting config (mirrors 00_generate_scenario_from_paint_tutorial.py)
# ---------------------------------------------------------------------------

NUMBER_OF_NURBS_CONTROL_POINTS = torch.tensor([20, 20])
NURBS_FIT_METHOD = config_dictionary.fit_nurbs_from_normals
NURBS_DEFLECTOMETRY_STEP_SIZE = 100
NURBS_FIT_TOLERANCE = 1e-10
NURBS_FIT_MAX_EPOCH = 400


def main() -> None:
    set_logger_config()
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    parser = argparse.ArgumentParser(description="Create 5-heliostat scenario.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing file.")
    args = parser.parse_args()

    # Validate inputs.
    if not TOWER_FILE.exists():
        sys.exit(f"tower-measurements.json not found: {TOWER_FILE}")
    missing = [
        hid for hid, props, defl in HELIOSTAT_FILES
        if not props.exists() or not defl.exists()
    ]
    if missing:
        sys.exit(f"Missing PAINT data for heliostats: {missing}")
    if OUTPUT_PATH.exists() and not args.force:
        print(f"Already exists, skipping: {OUTPUT_PATH}")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print(f"Tower file : {TOWER_FILE}")
    print(f"Output     : {OUTPUT_PATH}")
    print(f"Heliostats : {[h for h, _, _ in HELIOSTAT_FILES]}")
    print(f"Device     : {device}")

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
            paths=HELIOSTAT_FILES,
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


if __name__ == "__main__":
    main()
