"""
Create all four scenario HDF5 files used in the thesis.

Scenarios
---------
deflectometry   All benchmark heliostats with deflectometry data — NURBS surfaces
                fitted from measured deflectometry (deflectometry_scenario.h5).

ideal           Same heliostat set, same kinematics — but ideal (flat) surfaces,
                i.e. no deflectometry fitting (ideal_scenario.h5).

one_heliostat   Single heliostat AA31 with a fitted deflectometry surface
                (one_heliostat_scenarios/scenario1.h5).

blur_ablation   18 hand-picked heliostats copied directly from the deflectometry
                scenario — no re-fitting (blur_ablation_scenario.h5).
                Requires the deflectometry scenario to exist first.

Usage
-----
    # Build all four (deflectometry → ideal → one_heliostat → blur_ablation)
    python src/create_all_scenarios.py

    # Build only selected scenarios
    python src/create_all_scenarios.py --scenarios deflectometry ideal

    # Overwrite existing .h5 files
    python src/create_all_scenarios.py --force

    # Use DAIC paths
    python src/create_all_scenarios.py --daic

Run on DAIC:
    cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src
    apptainer exec --nv \\
        --bind /tudelft.net:/tudelft.net \\
        /tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif \\
        python create_all_scenarios.py
"""

import argparse
import json
import pathlib
import sys
from datetime import datetime

import h5py
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

# ===========================================================================
# Paths — only thing to update when the dataset location changes.
# ===========================================================================

_SRC_DIR = pathlib.Path(__file__).resolve().parent

LOCAL_BASE_DIR = _SRC_DIR.parent
LOCAL_PAINT_DIR = LOCAL_BASE_DIR / "datasets" / "paint" / "heliostats"

DAIC_BASE_DIR = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
DAIC_PAINT_DIR = DAIC_BASE_DIR / "datasets" / "paint" / "heliostats"

DEFLECTOMETRY_AVAILABILITY_JSON = _SRC_DIR / "utils" / "deflectometry_availability.json"

ONE_HELIOSTAT_NAME = "AA31"

# 18 heliostats for the blur ablation (random.seed(42), 2 per distance-band × lateral-column cell)
BLUR_ABLATION_HELIOSTATS = [
    "AH29", "AA30",   # near / left
    "AA33", "AL35",   # near / mid
    "AC37", "AB47",   # near / right
    "AQ28", "AQ25",   # mid  / left
    "BA37", "AW40",   # mid  / mid
    "AZ43", "AZ45",   # mid  / right
    "BC32", "BF29",   # far  / left
    "AZ55", "AZ52",   # far  / mid
    "AY72", "BA71",   # far  / right
]

# ===========================================================================
# Shared helpers
# ===========================================================================

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
    """Return the most recent filled deflectometry HDF5 file, or None."""
    files = list(deflectometry_folder.glob(f"{heliostat_name}-filled-*-deflectometry.h5"))
    if not files:
        return None
    return max(files, key=_extract_datetime_from_deflectometry_filename)


def _build_fitted_heliostat_list(
    paint_dir: pathlib.Path,
    availability_json: pathlib.Path,
    name_filter: set[str] | None = None,
) -> list[tuple[str, pathlib.Path, pathlib.Path]]:
    """Return (name, properties_path, deflectometry_path) for benchmark heliostats
    that have deflectometry data.  Optionally restrict to name_filter.
    """
    with open(availability_json) as f:
        availability: dict[str, dict] = json.load(f)

    target_names = sorted(
        name for name, info in availability.items()
        if info["has_deflectometry"] and info["in_benchmark"]
    )
    if name_filter is not None:
        target_names = [n for n in target_names if n in name_filter]

    result = []
    for name in target_names:
        heliostat_folder = paint_dir / name
        if not heliostat_folder.is_dir():
            print(f"  [skip] {name}: folder not found")
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
        print(f"  [ok]   {name}")

    return result


def _build_ideal_heliostat_list(
    paint_dir: pathlib.Path,
    availability_json: pathlib.Path,
) -> list[tuple[str, pathlib.Path]]:
    """Return (name, properties_path) for the same benchmark heliostat set
    used in the deflectometry scenario (same filter, no deflectometry path).
    """
    with open(availability_json) as f:
        availability: dict[str, dict] = json.load(f)

    target_names = sorted(
        name for name, info in availability.items()
        if info["has_deflectometry"] and info["in_benchmark"]
    )

    result = []
    for name in target_names:
        heliostat_folder = paint_dir / name
        if not heliostat_folder.is_dir():
            print(f"  [skip] {name}: folder not found")
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


def _make_light_source_list_config() -> LightSourceListConfig:
    return LightSourceListConfig(
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


def _make_nurbs_optimizer_scheduler() -> tuple:
    """Return a fresh (optimizer, scheduler) pair for NURBS fitting."""
    optimizer = torch.optim.Adam([torch.empty(1, requires_grad=True)], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience=50,
        threshold=1e-7,
        threshold_mode="abs",
    )
    return optimizer, scheduler


# ===========================================================================
# Scenario builders
# ===========================================================================

def build_deflectometry_scenario(
    paint_dir: pathlib.Path,
    base_dir: pathlib.Path,
    force: bool,
    device: torch.device,
) -> None:
    """Fit NURBS surfaces from deflectometry and write deflectometry_scenario.h5."""
    output = base_dir / "scenarios" / "deflectometry_scenario" / "deflectometry_scenario.h5"
    if output.exists() and not force:
        print(f"[deflectometry] Already exists, skipping: {output}")
        return

    tower_file = paint_dir / "WRI1030197-tower-measurements.json"
    print(f"\n[deflectometry] Scanning heliostats …")
    heliostat_list = _build_fitted_heliostat_list(paint_dir, DEFLECTOMETRY_AVAILABILITY_JSON)
    if not heliostat_list:
        print("[deflectometry] ERROR: No heliostats found. Aborting.")
        return

    print(f"[deflectometry] Found {len(heliostat_list)} heliostats.")
    power_plant_config, target_area_list_config = (
        paint_scenario_parser.extract_paint_tower_measurements(
            tower_measurements_path=tower_file, device=device
        )
    )

    optimizer, scheduler = _make_nurbs_optimizer_scheduler()
    print("[deflectometry] Fitting NURBS surfaces (this may take a while) …")
    heliostat_list_config, prototype_config = (
        paint_scenario_parser.extract_paint_heliostats_fitted_surface(
            paths=heliostat_list,
            power_plant_position=power_plant_config.power_plant_position,
            number_of_nurbs_control_points=torch.tensor([20, 20], device=device),
            deflectometry_step_size=100,
            nurbs_fit_method=config_dictionary.fit_nurbs_from_normals,
            nurbs_fit_tolerance=1e-10,
            nurbs_fit_max_epoch=400,
            nurbs_fit_optimizer=optimizer,
            nurbs_fit_scheduler=scheduler,
            device=device,
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    H5ScenarioGenerator(
        file_path=output,
        power_plant_config=power_plant_config,
        target_area_list_config=target_area_list_config,
        light_source_list_config=_make_light_source_list_config(),
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    ).generate_scenario()
    print(f"[deflectometry] Done → {output}")


def build_ideal_scenario(
    paint_dir: pathlib.Path,
    base_dir: pathlib.Path,
    force: bool,
    device: torch.device,
) -> None:
    """Generate ideal (flat) surfaces for the same heliostat set and write ideal_scenario.h5."""
    output = base_dir / "scenarios" / "ideal_scenario" / "ideal_scenario.h5"
    if output.exists() and not force:
        print(f"[ideal] Already exists, skipping: {output}")
        return

    tower_file = paint_dir / "WRI1030197-tower-measurements.json"
    print(f"\n[ideal] Scanning heliostats …")
    heliostat_list = _build_ideal_heliostat_list(paint_dir, DEFLECTOMETRY_AVAILABILITY_JSON)
    if not heliostat_list:
        print("[ideal] ERROR: No heliostats found. Aborting.")
        return

    print(f"[ideal] Found {len(heliostat_list)} heliostats.")
    power_plant_config, target_area_list_config = (
        paint_scenario_parser.extract_paint_tower_measurements(
            tower_measurements_path=tower_file, device=device
        )
    )

    print("[ideal] Generating ideal surfaces …")
    heliostat_list_config, prototype_config = (
        paint_scenario_parser.extract_paint_heliostats_ideal_surface(
            paths=heliostat_list,
            power_plant_position=power_plant_config.power_plant_position,
            number_of_nurbs_control_points=torch.tensor([20, 20], device=device),
            device=device,
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    H5ScenarioGenerator(
        file_path=output,
        power_plant_config=power_plant_config,
        target_area_list_config=target_area_list_config,
        light_source_list_config=_make_light_source_list_config(),
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    ).generate_scenario()
    print(f"[ideal] Done → {output}")


def build_one_heliostat_scenario(
    paint_dir: pathlib.Path,
    base_dir: pathlib.Path,
    force: bool,
    device: torch.device,
) -> None:
    """Fit NURBS for AA31 only and write one_heliostat_scenarios/scenario1.h5."""
    output = base_dir / "scenarios" / "one_heliostat_scenarios" / "scenario1.h5"
    if output.exists() and not force:
        print(f"[one_heliostat] Already exists, skipping: {output}")
        return

    tower_file = paint_dir / "WRI1030197-tower-measurements.json"
    print(f"\n[one_heliostat] Scanning for {ONE_HELIOSTAT_NAME} …")
    heliostat_list = _build_fitted_heliostat_list(
        paint_dir, DEFLECTOMETRY_AVAILABILITY_JSON, name_filter={ONE_HELIOSTAT_NAME}
    )
    if not heliostat_list:
        print(f"[one_heliostat] ERROR: {ONE_HELIOSTAT_NAME} not found. Aborting.")
        return

    power_plant_config, target_area_list_config = (
        paint_scenario_parser.extract_paint_tower_measurements(
            tower_measurements_path=tower_file, device=device
        )
    )

    optimizer, scheduler = _make_nurbs_optimizer_scheduler()
    print(f"[one_heliostat] Fitting NURBS surface for {ONE_HELIOSTAT_NAME} …")
    heliostat_list_config, prototype_config = (
        paint_scenario_parser.extract_paint_heliostats_fitted_surface(
            paths=heliostat_list,
            power_plant_position=power_plant_config.power_plant_position,
            number_of_nurbs_control_points=torch.tensor([20, 20], device=device),
            deflectometry_step_size=100,
            nurbs_fit_method=config_dictionary.fit_nurbs_from_normals,
            nurbs_fit_tolerance=1e-10,
            nurbs_fit_max_epoch=400,
            nurbs_fit_optimizer=optimizer,
            nurbs_fit_scheduler=scheduler,
            device=device,
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    H5ScenarioGenerator(
        file_path=output,
        power_plant_config=power_plant_config,
        target_area_list_config=target_area_list_config,
        light_source_list_config=_make_light_source_list_config(),
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    ).generate_scenario()
    print(f"[one_heliostat] Done → {output}")


def build_blur_ablation_scenario(
    base_dir: pathlib.Path,
    force: bool,
) -> None:
    """Copy 18 heliostats from deflectometry_scenario.h5 — no re-fitting."""
    source = base_dir / "scenarios" / "deflectometry_scenario" / "deflectometry_scenario.h5"
    output = base_dir / "scenarios" / "blur_ablation_scenario" / "blur_ablation_scenario.h5"

    if not source.exists():
        print(
            f"[blur_ablation] ERROR: source scenario not found: {source}\n"
            "  Build the deflectometry scenario first."
        )
        return

    if output.exists() and not force:
        print(f"[blur_ablation] Already exists, skipping: {output}")
        return

    print(f"\n[blur_ablation] Copying {len(BLUR_ABLATION_HELIOSTATS)} heliostats from deflectometry scenario …")
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as src, h5py.File(output, "w") as dst:
        for key, val in src.attrs.items():
            dst.attrs[key] = val

        for key in src.keys():
            if key != "heliostats":
                src.copy(key, dst)

        dst_heliostats = dst.require_group("heliostats")
        missing = []
        for name in BLUR_ABLATION_HELIOSTATS:
            if name not in src["heliostats"]:
                missing.append(name)
                continue
            src.copy(f"heliostats/{name}", dst_heliostats, name=name)

        if missing:
            print(f"[blur_ablation] WARNING: not found in source: {missing}")
        print(f"[blur_ablation] Copied {len(BLUR_ABLATION_HELIOSTATS) - len(missing)} heliostats.")

    print(f"[blur_ablation] Done → {output}")


# ===========================================================================
# Entry point
# ===========================================================================

ALL_SCENARIOS = ["deflectometry", "ideal", "one_heliostat", "blur_ablation"]
# blur_ablation must run after deflectometry — enforced in main().
_BUILD_ORDER = ["deflectometry", "ideal", "one_heliostat", "blur_ablation"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create all thesis scenario HDF5 files.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=ALL_SCENARIOS,
        default=ALL_SCENARIOS,
        metavar="SCENARIO",
        help="Which scenarios to build (default: all). Choices: " + ", ".join(ALL_SCENARIOS),
    )
    parser.add_argument(
        "--daic",
        action="store_true",
        help="Use DAIC paths instead of local paths.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .h5 files.",
    )
    args = parser.parse_args()

    base_dir = DAIC_BASE_DIR if args.daic else LOCAL_BASE_DIR
    paint_dir = DAIC_PAINT_DIR if args.daic else LOCAL_PAINT_DIR

    print(f"Base dir:   {base_dir}")
    print(f"PAINT dir:  {paint_dir}")
    print(f"Scenarios:  {', '.join(args.scenarios)}")
    print(f"Force:      {args.force}")

    if not paint_dir.exists():
        sys.exit(f"ERROR: PAINT dataset directory not found: {paint_dir}")
    if not (paint_dir / "WRI1030197-tower-measurements.json").exists():
        sys.exit(f"ERROR: Tower measurements file not found in {paint_dir}")
    if not DEFLECTOMETRY_AVAILABILITY_JSON.exists():
        sys.exit(f"ERROR: Availability JSON not found: {DEFLECTOMETRY_AVAILABILITY_JSON}")

    to_build = [s for s in _BUILD_ORDER if s in args.scenarios]

    # blur_ablation depends on deflectometry; add it to the run if not already selected.
    if "blur_ablation" in to_build and "deflectometry" not in to_build:
        defl_output = (
            base_dir / "scenarios" / "deflectometry_scenario" / "deflectometry_scenario.h5"
        )
        if not defl_output.exists():
            print(
                "[main] WARNING: blur_ablation requires deflectometry_scenario.h5 "
                "which does not exist. Building deflectometry first."
            )
            to_build.insert(0, "deflectometry")

    device = get_device()
    print(f"Device:     {device}\n")

    for scenario in to_build:
        if scenario == "deflectometry":
            build_deflectometry_scenario(paint_dir, base_dir, args.force, device)
        elif scenario == "ideal":
            build_ideal_scenario(paint_dir, base_dir, args.force, device)
        elif scenario == "one_heliostat":
            build_one_heliostat_scenario(paint_dir, base_dir, args.force, device)
        elif scenario == "blur_ablation":
            build_blur_ablation_scenario(base_dir, args.force)

    print("\nAll requested scenarios finished.")


if __name__ == "__main__":
    main()
