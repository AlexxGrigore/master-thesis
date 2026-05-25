"""
Create the 63-heliostat scenario for the full_63_heli_kin_reconstruct experiment.

Reads all unique heliostat IDs from the benchmark CSV, discovers the matching
deflectometry files, and builds a single multi-heliostat scenario HDF5 using the
current ARTIST API (deflectometry-fitted NURBS surfaces, 20×20 control points).

Output: scenarios/full_63_heli_kin_reconstruct/scenario.h5

Usage
-----
    cd src
    python full_63_heli_kin_reconstruct/create_scenario.py
    python full_63_heli_kin_reconstruct/create_scenario.py --daic
    python full_63_heli_kin_reconstruct/create_scenario.py --force
"""
import argparse
import pathlib
import sys

import pandas as pd
import torch
from artist.io import paint_scenario_parser
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator
from artist.util import constants as config_dictionary
from artist.util import get_device, set_logger_config
from artist.util.config import LightSourceConfig, LightSourceListConfig

_HERE = pathlib.Path(__file__).resolve().parent
_SRC  = _HERE.parent
sys.path.insert(0, str(_SRC))

import config as cfg  # noqa: E402  (full_63_heli_kin_reconstruct/config.py)

NUMBER_OF_NURBS_CONTROL_POINTS = torch.tensor([20, 20])
NURBS_FIT_METHOD               = config_dictionary.fit_nurbs_from_normals
NURBS_DEFLECTOMETRY_STEP_SIZE  = 100
NURBS_FIT_TOLERANCE            = 1e-10
NURBS_FIT_MAX_EPOCH            = 400


def _find_deflectometry(hid: str, heliostats_dir: pathlib.Path) -> pathlib.Path | None:
    defl_dir = heliostats_dir / hid / "Deflectometry"
    filled   = sorted(defl_dir.glob(f"{hid}-filled-*-deflectometry.h5"))
    return filled[-1] if filled else None


def _collect_paths(
    benchmark_csv: pathlib.Path,
    heliostats_dir: pathlib.Path,
) -> list[tuple[str, pathlib.Path, pathlib.Path]]:
    """Return (hid, props_path, defl_path) for every unique heliostat in the CSV."""
    df  = pd.read_csv(benchmark_csv)
    ids = sorted(df["HeliostatId"].unique())

    paths: list[tuple[str, pathlib.Path, pathlib.Path]] = []
    skipped: list[str] = []

    for hid in ids:
        props = heliostats_dir / hid / "Properties" / f"{hid}-heliostat-properties.json"
        defl  = _find_deflectometry(hid, heliostats_dir)

        if not props.exists():
            print(f"  [SKIP] {hid} — properties file missing")
            skipped.append(hid)
            continue
        if defl is None:
            print(f"  [SKIP] {hid} — no deflectometry file found")
            skipped.append(hid)
            continue

        paths.append((hid, props, defl))

    if skipped:
        print(f"Skipped {len(skipped)} heliostats: {skipped}")

    return paths


def main() -> None:
    set_logger_config()
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    parser = argparse.ArgumentParser(
        description="Create 63-heliostat scenario for full_63_heli_kin_reconstruct."
    )
    parser.add_argument("--daic",  action="store_true", help="Use DAIC cluster paths.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing file.")
    args = parser.parse_args()

    if args.daic:
        cfg.IS_ON_DAIC = True
        cfg.BASE_DIR   = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")

    artist_dir     = cfg.BASE_DIR.parent / "ARTIST"
    tower_file     = artist_dir / "tutorials" / "data" / "paint" / "tower-measurements.json"
    heliostats_dir = cfg.BASE_DIR / "datasets" / "paint" / "heliostats"
    out_path       = cfg.BASE_DIR / "scenarios" / "full_63_heli_kin_reconstruct" / "scenario.h5"

    for label, p in [
        ("tower-measurements.json", tower_file),
        ("heliostats dir",          heliostats_dir),
        ("benchmark CSV",           cfg.BENCHMARK_CSV),
    ]:
        if not p.exists():
            sys.exit(f"{label} not found: {p}")

    if out_path.exists() and not args.force:
        print(f"Scenario already exists: {out_path}\nUse --force to overwrite.")
        return

    print(f"Tower file     : {tower_file}")
    print(f"Heliostats dir : {heliostats_dir}")
    print(f"Benchmark CSV  : {cfg.BENCHMARK_CSV}")
    print(f"Output path    : {out_path}\n")

    device = get_device()
    paths  = _collect_paths(cfg.BENCHMARK_CSV, heliostats_dir)
    print(f"Building scenario with {len(paths)} heliostats …\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    power_plant_config, target_area_list_planar_config, target_area_list_cylindrical_config = (
        paint_scenario_parser.extract_paint_tower_measurements(
            tower_measurements_path=tower_file,
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
            paths=paths,
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

    print(f"\nDone → {out_path}")


if __name__ == "__main__":
    main()
