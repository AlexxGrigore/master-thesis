"""
Create the single-heliostat scenario for the one-heliostat train-size experiment.

Selects one heliostat from the 63 in the full-field benchmark (the heliostat
specified by HELIOSTAT_ID in config.py, or the first one found if None) and
builds a new one-heliostat scenario.h5 via NURBS fitting.

For a single heliostat this takes ~10-30 seconds on GPU.

Output: scenarios/one_heliostat_scenario/scenario.h5

Usage
-----
    python create_scenario.py
    python create_scenario.py --heliostat-id AA23
    python create_scenario.py --force
"""
import argparse
import pathlib
import sys

import pandas as pd
import torch
from artist.data_parser import paint_scenario_parser
from artist.scenario.configuration_classes import (
    LightSourceConfig,
    LightSourceListConfig,
)
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device

_here = pathlib.Path(__file__).resolve().parent
_src  = _here.parent
sys.path.insert(0, str(_src))

import paint.util.paint_mappings as paint_mappings
import config as cfg

# ---------------------------------------------------------------------------
# Paths (self-contained, match config.py)
# ---------------------------------------------------------------------------

BASE_DIR             = pathlib.Path(__file__).resolve().parents[2]
PAINT_HELIOSTATS_DIR = BASE_DIR / "datasets" / "paint" / "heliostats"
ARTIST_DIR           = BASE_DIR.parent / "ARTIST"
TOWER_FILE           = ARTIST_DIR / "tutorials" / "data" / "paint" / "tower-measurements.json"
OUTPUT_PATH          = BASE_DIR / "scenarios" / "one_heliostat_scenario" / "scenario.h5"

# ---------------------------------------------------------------------------
# NURBS fitting config (same as other scenario scripts)
# ---------------------------------------------------------------------------

NUMBER_OF_NURBS_CONTROL_POINTS = torch.tensor([20, 20])
NURBS_FIT_METHOD               = config_dictionary.fit_nurbs_from_normals
NURBS_DEFLECTOMETRY_STEP_SIZE  = 100
NURBS_FIT_TOLERANCE            = 1e-10
NURBS_FIT_MAX_EPOCH            = 400


def _benchmark_heliostat_ids() -> list[str]:
    """Return heliostat IDs from the benchmark CSV (order preserved)."""
    if not cfg.BENCHMARK_CSV.exists():
        sys.exit(
            f"Benchmark CSV not found: {cfg.BENCHMARK_CSV}\n"
            "Run src/download_paint_benchmark_200.py first."
        )
    df = pd.read_csv(cfg.BENCHMARK_CSV)
    seen = set()
    ordered = []
    for hid in df[paint_mappings.HELIOSTAT_ID]:
        if hid not in seen:
            seen.add(hid)
            ordered.append(hid)
    return ordered


def _find_heliostat(benchmark_ids: list[str], target_id: str | None) -> tuple:
    """
    Return (hid, props_path, deflectometry_path) for the chosen heliostat.
    If target_id is None, picks the first heliostat that has local deflectometry.
    Exits with an error message if the heliostat cannot be found.
    """
    candidates = [target_id] if target_id else benchmark_ids

    for hid in candidates:
        hid_dir = PAINT_HELIOSTATS_DIR / hid
        if not hid_dir.is_dir():
            continue
        props = hid_dir / "Properties" / f"{hid}-heliostat-properties.json"
        if not props.exists():
            continue
        defl_dir = hid_dir / "Deflectometry"
        if not defl_dir.exists():
            continue
        filled = sorted(defl_dir.glob(f"{hid}-filled-*-deflectometry.h5"))
        if not filled:
            continue
        return (hid, props, filled[-1])

    if target_id:
        sys.exit(
            f"Heliostat '{target_id}' not found or missing local deflectometry data.\n"
            f"Checked: {PAINT_HELIOSTATS_DIR / target_id}"
        )
    else:
        sys.exit(
            "No heliostat with local deflectometry data found in the benchmark.\n"
            f"Checked: {PAINT_HELIOSTATS_DIR}"
        )


def main() -> None:
    set_logger_config()
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    parser = argparse.ArgumentParser(
        description="Create one-heliostat scenario for the train-size sensitivity experiment."
    )
    parser.add_argument(
        "--heliostat-id",
        default=None,
        help="Heliostat ID to use (e.g. AA23).  Overrides config.HELIOSTAT_ID.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing file.")
    args = parser.parse_args()

    target_id = args.heliostat_id or cfg.HELIOSTAT_ID

    if not TOWER_FILE.exists():
        sys.exit(f"tower-measurements.json not found: {TOWER_FILE}")

    benchmark_ids   = _benchmark_heliostat_ids()
    hid, props, defl = _find_heliostat(benchmark_ids, target_id)

    if OUTPUT_PATH.exists() and not args.force:
        print(f"Already exists, skipping: {OUTPUT_PATH}")
        print("Use --force to overwrite.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print(f"Heliostat        : {hid}")
    print(f"Properties       : {props}")
    print(f"Deflectometry    : {defl}")
    print(f"Tower file       : {TOWER_FILE}")
    print(f"Output           : {OUTPUT_PATH}")
    print(f"Device           : {device}")

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
    # Single heliostat with fitted deflectometry surface
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
    print(f"Done → {OUTPUT_PATH}  (heliostat: {hid})")


if __name__ == "__main__":
    main()
