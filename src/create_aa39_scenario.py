"""Create the per-heliostat scenario for AA39 (deflectometry surfaces).

Output: scenarios/one_heliostat_scenarios/AA39/scenario.h5
"""
import pathlib
import sys

import torch

_SRC = pathlib.Path(__file__).resolve().parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT / "ARTIST"))

from artist.io import paint_scenario_parser
from artist.util.config import LightSourceConfig, LightSourceListConfig
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator
from artist.util import constants as config_dictionary, set_logger_config
from artist.util import get_device

set_logger_config()

PAINT_DIR  = _ROOT / "datasets" / "paint" / "heliostats"
TOWER_FILE = PAINT_DIR / "WRI1030197-tower-measurements.json"
OUT_PATH   = _ROOT / "scenarios" / "one_heliostat_scenarios" / "AA39" / "scenario.h5"

HID = "AA39"
NUMBER_OF_NURBS_CONTROL_POINTS = torch.tensor([20, 20])


def main() -> None:
    device = get_device()

    props = PAINT_DIR / HID / "Properties" / f"{HID}-heliostat-properties.json"
    defl_files = sorted((PAINT_DIR / HID / "Deflectometry").glob(f"{HID}-filled-*-deflectometry.h5"))
    if not defl_files:
        sys.exit(f"No filled deflectometry file found for {HID}")
    defl = defl_files[-1]

    print(f"Heliostat    : {HID}")
    print(f"Properties   : {props}")
    print(f"Deflectometry: {defl}")
    print(f"Output       : {OUT_PATH}")
    print(f"Device       : {device}\n")

    power_plant_config, target_area_list_planar_config, target_area_list_cylindrical_config = (
        paint_scenario_parser.extract_paint_tower_measurements(
            tower_measurements_path=TOWER_FILE, device=device
        )
    )

    optimizer = torch.optim.Adam([torch.empty(1, requires_grad=True)], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=50,
        threshold=1e-7, threshold_mode="abs",
    )

    print("Fitting NURBS surfaces from deflectometry …")
    heliostat_list_config, prototype_config = (
        paint_scenario_parser.extract_paint_heliostats_fitted_surface(
            paths=[(HID, props, defl)],
            power_plant_position=power_plant_config.power_plant_position,
            number_of_nurbs_control_points=NUMBER_OF_NURBS_CONTROL_POINTS,
            deflectometry_step_size=100,
            nurbs_fit_method=config_dictionary.fit_nurbs_from_normals,
            nurbs_fit_tolerance=1e-10,
            nurbs_fit_max_epoch=400,
            nurbs_fit_optimizer=optimizer,
            nurbs_fit_scheduler=scheduler,
            device=device,
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

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    H5ScenarioGenerator(
        file_path=OUT_PATH,
        power_plant_config=power_plant_config,
        target_area_list_planar_config=target_area_list_planar_config,
        target_area_list_cylindrical_config=target_area_list_cylindrical_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    ).generate_scenario()

    print(f"\nDone → {OUT_PATH}")


if __name__ == "__main__":
    main()
