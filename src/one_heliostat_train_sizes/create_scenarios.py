"""
Create one single-heliostat scenario HDF5 per target heliostat.

Builds each scenario from scratch using ARTIST's paint_scenario_parser:
tower geometry, light source, and either deflectometry-fitted or ideal (flat) NURBS surface.

Output paths (matching the notebook's SCENARIO_PATH):
    scenarios/one_heliostat_scenarios/<ID>/scenario.h5          (deflectometry, default)
    scenarios/one_heliostat_scenarios/ideal/<ID>/scenario.h5    (--no-deflectometry)

Usage
-----
    cd src
    # All 63 dataset heliostats (default):
    python one_heliostat_train_sizes/create_scenarios.py

    # Specific heliostats only:
    python one_heliostat_train_sizes/create_scenarios.py --heliostat-ids AC36 BE35

    # Ideal (flat) surfaces:
    python one_heliostat_train_sizes/create_scenarios.py --no-deflectometry

    # DAIC cluster:
    python one_heliostat_train_sizes/create_scenarios.py --daic

    # Overwrite existing files:
    python one_heliostat_train_sizes/create_scenarios.py --force
"""
import argparse
import pathlib
import sys

import torch
from artist.io import paint_scenario_parser
from artist.util.config import (
    LightSourceConfig,
    LightSourceListConfig,
)
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator
from artist.util import constants as config_dictionary, set_logger_config
from artist.util import get_device

_HERE = pathlib.Path(__file__).resolve().parent
_SRC  = _HERE.parent
sys.path.insert(0, str(_SRC))

import config as cfg

# All 63 heliostats present in the full_63_heli_kin_reconstruct synthetic dataset.
DEFAULT_HELIOSTATS = [
    "AA23", "AA24", "AA25", "AA49",
    "AB26", "AB33", "AB43", "AB50",
    "AC24", "AC25", "AC27", "AC33", "AC35", "AC36", "AC39", "AC41", "AC47", "AC48",
    "AD39", "AD40",
    "AE23", "AE24", "AE29", "AE30", "AE32",
    "AF37", "AF38", "AF40", "AF44",
    "AG25", "AG27", "AG31", "AG33",
    "AH30",
    "AI36",
    "AJ37",
    "AK29", "AK32",
    "AM25", "AM38",
    "AN35",
    "AO32", "AO34",
    "AP29", "AP43",
    "AQ24",
    "AW36",
    "AX39",
    "AY36", "AY37", "AY39", "AY42", "AY43", "AY44",
    "AZ27", "AZ41",
    "BA28", "BA35", "BA42",
    "BD39",
    "BE25", "BE35",
    "BF39",
]

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
    with_deflectometry: bool = True,
) -> None:
    props = heliostats_dir / hid / "Properties" / f"{hid}-heliostat-properties.json"

    if not props.exists():
        sys.exit(f"Properties file missing for {hid}: {props}")

    surface_label = "deflectometry" if with_deflectometry else "ideal"
    print(f"\n  Heliostat    : {hid}")
    print(f"  Surface      : {surface_label}")
    print(f"  Properties   : {props}")

    if with_deflectometry:
        defl = _find_deflectometry(hid, heliostats_dir)
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

    if with_deflectometry:
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
    else:
        heliostat_list_config, prototype_config = (
            paint_scenario_parser.extract_paint_heliostats_ideal_surface(
                paths=[(hid, props)],
                power_plant_position=power_plant_config.power_plant_position,
                number_of_nurbs_control_points=NUMBER_OF_NURBS_CONTROL_POINTS,
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
        description="Create per-heliostat scenario HDF5 files from PAINT data."
    )
    parser.add_argument("--daic", action="store_true", help="Use DAIC cluster paths.")
    parser.add_argument(
        "--heliostat-ids", nargs="+", default=DEFAULT_HELIOSTATS,
        help=f"Heliostat IDs to create scenarios for (default: {DEFAULT_HELIOSTATS}).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument(
        "--no-deflectometry", dest="with_deflectometry", action="store_false",
        help="Use ideal (flat) surfaces instead of deflectometry-fitted NURBS.",
    )
    parser.set_defaults(with_deflectometry=True)
    args = parser.parse_args()

    if args.daic:
        cfg.IS_ON_DAIC = True
        cfg.BASE_DIR   = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")

    artist_dir     = cfg.BASE_DIR.parent / "ARTIST"
    tower_file     = artist_dir / "tutorials" / "data" / "paint" / "tower-measurements.json"
    heliostats_dir = cfg.BASE_DIR / "datasets" / "paint" / "heliostats"
    surface_label  = "deflectometry" if args.with_deflectometry else "ideal"
    # Deflectometry scenarios go directly under ONE_HELIOSTAT_SCENARIOS_DIR/<ID>/
    # to match the notebook's SCENARIO_PATH and the existing 5 scenarios.
    # Ideal scenarios go under ONE_HELIOSTAT_SCENARIOS_DIR/ideal/<ID>/.
    if args.with_deflectometry:
        out_dir = cfg.ONE_HELIOSTAT_SCENARIOS_DIR
    else:
        out_dir = cfg.ONE_HELIOSTAT_SCENARIOS_DIR / "ideal"

    if not tower_file.exists():
        sys.exit(f"tower-measurements.json not found: {tower_file}")
    if not heliostats_dir.exists():
        sys.exit(f"PAINT heliostats directory not found: {heliostats_dir}")

    print(f"Surface type     : {surface_label}")
    print(f"Tower file       : {tower_file}")
    print(f"Heliostats dir   : {heliostats_dir}")
    print(f"Output directory : {out_dir}")

    device = get_device()

    for hid in args.heliostat_ids:
        out_path = out_dir / hid / "scenario.h5"
        if out_path.exists() and not args.force:
            print(f"\n  [SKIP] {hid} — already exists (use --force to overwrite)")
            continue
        create_scenario(hid, heliostats_dir, tower_file, out_path, device, args.with_deflectometry)

    print("\nAll done.")


if __name__ == "__main__":
    main()
