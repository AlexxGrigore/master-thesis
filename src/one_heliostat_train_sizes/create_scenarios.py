"""
Create one single-heliostat scenario HDF5 per target heliostat.

Builds each scenario from scratch using ARTIST's paint_scenario_parser:
tower geometry, light source, and deflectometry-fitted NURBS surface.

Output: scenarios/one_heliostat_scenarios/<ID>/scenario.h5

Usage
-----
    cd src
    python one_heliostat_train_sizes/create_scenarios.py
    python one_heliostat_train_sizes/create_scenarios.py --daic
    python one_heliostat_train_sizes/create_scenarios.py --heliostat-ids AC36 BE35
    python one_heliostat_train_sizes/create_scenarios.py --force
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

_HERE = pathlib.Path(__file__).resolve().parent
_SRC  = _HERE.parent
sys.path.insert(0, str(_SRC))

import config as cfg

DEFAULT_HELIOSTATS = ["AC36", "AG33", "AO34", "AW36", "BE35"]

NUMBER_OF_NURBS_CONTROL_POINTS = torch.tensor([20, 20])
NURBS_FIT_METHOD               = config_dictionary.fit_nurbs_from_normals
NURBS_DEFLECTOMETRY_STEP_SIZE  = 100
NURBS_FIT_TOLERANCE            = 1e-10
NURBS_FIT_MAX_EPOCH            = 400


def _find_deflectometry(hid: str, heliostats_dir: pathlib.Path) -> pathlib.Path:
    defl_dir = heliostats_dir / hid / "Deflectometry"
    filled   = sorted(defl_dir.glob(f"{hid}-filled-*-deflectometry.h5"))
    if not filled:
        sys.exit(f"No filled deflectometry file found for {hid} in {defl_dir}")
    return filled[-1]


def create_scenario(
    hid: str,
    heliostats_dir: pathlib.Path,
    tower_file: pathlib.Path,
    out_path: pathlib.Path,
    device: torch.device,
) -> None:
    props = heliostats_dir / hid / "Properties" / f"{hid}-heliostat-properties.json"
    defl  = _find_deflectometry(hid, heliostats_dir)

    if not props.exists():
        sys.exit(f"Properties file missing for {hid}: {props}")

    print(f"\n  Heliostat    : {hid}")
    print(f"  Properties   : {props}")
    print(f"  Deflectometry: {defl}")
    print(f"  Output       : {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    power_plant_config, target_area_list_planar_config, target_area_list_cylindrical_config = (
        paint_scenario_parser.extract_paint_tower_measurements(
            tower_measurements_path=tower_file, device=device
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
            paths=[(hid, props, defl)],
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

    H5ScenarioGenerator(
        file_path=out_path,
        power_plant_config=power_plant_config,
        target_area_list_planar_config=target_area_list_planar_config,
        target_area_list_cylindrical_config=target_area_list_cylindrical_config,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    ).generate_scenario()

    print(f"  Done → {out_path}")


def main() -> None:
    set_logger_config()
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    parser = argparse.ArgumentParser(
        description="Create per-heliostat scenario HDF5 files from PAINT deflectometry data."
    )
    parser.add_argument("--daic", action="store_true", help="Use DAIC cluster paths.")
    parser.add_argument(
        "--heliostat-ids", nargs="+", default=DEFAULT_HELIOSTATS,
        help=f"Heliostat IDs to create scenarios for (default: {DEFAULT_HELIOSTATS}).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    if args.daic:
        cfg.IS_ON_DAIC = True
        cfg.BASE_DIR   = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")

    artist_dir       = cfg.BASE_DIR.parent / "ARTIST"
    tower_file       = artist_dir / "tutorials" / "data" / "paint" / "tower-measurements.json"
    heliostats_dir   = cfg.BASE_DIR / "datasets" / "paint" / "heliostats"
    out_dir          = cfg.ONE_HELIOSTAT_SCENARIOS_DIR

    if not tower_file.exists():
        sys.exit(f"tower-measurements.json not found: {tower_file}")
    if not heliostats_dir.exists():
        sys.exit(f"PAINT heliostats directory not found: {heliostats_dir}")

    print(f"Tower file       : {tower_file}")
    print(f"Heliostats dir   : {heliostats_dir}")
    print(f"Output directory : {out_dir}")

    device = get_device()

    for hid in args.heliostat_ids:
        out_path = out_dir / hid / "scenario.h5"
        if out_path.exists() and not args.force:
            print(f"\n  [SKIP] {hid} — already exists (use --force to overwrite)")
            continue
        create_scenario(hid, heliostats_dir, tower_file, out_path, device)

    print("\nAll done.")


if __name__ == "__main__":
    main()
