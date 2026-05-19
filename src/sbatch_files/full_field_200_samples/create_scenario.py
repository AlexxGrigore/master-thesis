"""
Create the scenario for the full-field 200-samples experiment.

Only the 63 heliostats that appear in the
  benchmark_split-balanced_train-100_validation-50_deflectometry
benchmark CSV are included — these are exactly the heliostats that have
both filled deflectometry data AND at least 200 calibration measurements.

Output: scenarios/full_field_200_samples_scenario/scenario.h5

NOTE: NURBS fitting for ~63 heliostats takes ~10-30 min on GPU, several
      hours on CPU.

Usage
-----
    python create_scenario.py
    python create_scenario.py --force   # overwrite existing
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
sys.path.insert(0, str(_here.parent))
import paint.util.paint_mappings as paint_mappings

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR             = pathlib.Path(__file__).resolve().parents[2]
PAINT_HELIOSTATS_DIR = BASE_DIR / "datasets" / "paint" / "heliostats"
BENCHMARK_CSV        = (
    BASE_DIR / "datasets" / "paint" / "splits"
    / "benchmark_split-balanced_train-100_validation-50_deflectometry.csv"
)

ARTIST_DIR  = BASE_DIR.parent / "ARTIST"
TOWER_FILE  = ARTIST_DIR / "tutorials" / "data" / "paint" / "tower-measurements.json"
OUTPUT_PATH = BASE_DIR / "scenarios" / "full_field_200_samples_scenario" / "scenario.h5"

# ---------------------------------------------------------------------------
# NURBS fitting config (same as other scenario scripts)
# ---------------------------------------------------------------------------

NUMBER_OF_NURBS_CONTROL_POINTS  = torch.tensor([20, 20])
NURBS_FIT_METHOD                = config_dictionary.fit_nurbs_from_normals
NURBS_DEFLECTOMETRY_STEP_SIZE   = 100
NURBS_FIT_TOLERANCE             = 1e-10
NURBS_FIT_MAX_EPOCH             = 400


def _benchmark_heliostat_ids() -> set[str]:
    """Return the set of heliostat IDs in the 200-sample benchmark CSV."""
    if not BENCHMARK_CSV.exists():
        sys.exit(
            f"Benchmark CSV not found: {BENCHMARK_CSV}\n"
            "Run src/download_paint_benchmark_200.py first."
        )
    df = pd.read_csv(BENCHMARK_CSV)
    return set(df[paint_mappings.HELIOSTAT_ID].unique())


def _discover_heliostats(benchmark_ids: set[str]) -> list[tuple]:
    """
    Return (hid, props_path, deflectometry_path) for heliostats that are
    both in the benchmark and have filled deflectometry locally.
    """
    result = []
    for hid_dir in sorted(PAINT_HELIOSTATS_DIR.iterdir()):
        if not hid_dir.is_dir():
            continue
        hid = hid_dir.name
        if hid not in benchmark_ids:
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
        result.append((hid, props, filled[-1]))
    return result


def main() -> None:
    set_logger_config()
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    parser = argparse.ArgumentParser(
        description="Create scenario for full-field 200-samples experiment."
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing file.")
    args = parser.parse_args()

    if not TOWER_FILE.exists():
        sys.exit(f"tower-measurements.json not found: {TOWER_FILE}")

    benchmark_ids    = _benchmark_heliostat_ids()
    heliostat_files  = _discover_heliostats(benchmark_ids)

    if not heliostat_files:
        sys.exit(
            f"No heliostats found with both benchmark membership and local deflectometry.\n"
            f"Benchmark has {len(benchmark_ids)} heliostats; checked {PAINT_HELIOSTATS_DIR}"
        )

    if OUTPUT_PATH.exists() and not args.force:
        print(f"Already exists, skipping: {OUTPUT_PATH}")
        print("Use --force to overwrite.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print(f"Tower file       : {TOWER_FILE}")
    print(f"Output           : {OUTPUT_PATH}")
    print(f"Benchmark hels   : {len(benchmark_ids)}")
    print(f"Heliostats found : {len(heliostat_files)}")
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
