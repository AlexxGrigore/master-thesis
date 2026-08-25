"""Build a small multi-heliostat ARTIST scenario: one trained heliostat plus its blockers.

ARTIST can only block with heliostats that are present in the scenario, so the trained
heliostat and every candidate blocker have to live in the same file (and, so that one
active mask addresses them all, in the same heliostat group).

The candidate list comes from `blockers.select_blockers`, a conservative orientation free
screen, so the scenario is valid for every neighbour-state hypothesis. Typical size is
2 to 7 heliostats.

Surfaces: the trained heliostat gets its measured deflectometry surface when one exists,
because it decides where the beam actually goes. Blockers always get ideal surfaces, since
only their rectangular silhouette is used (`create_blocking_primitives_rectangles_by_index`
reduces every heliostat to four corner points).

Construction mirrors `src/field_batch_training/build_batch_scenarios.py` and
`src/one_heliostat_demo/daic_full_field/create_full_field_scenarios.py`.

Usage
-----
    python build_neighbourhood_scenario.py BE25 AA27
    python build_neighbourhood_scenario.py BE25 --daic
    python build_neighbourhood_scenario.py BE25 --ideal-only     # fast, for the gate
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time

import torch

_here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

from artist.io import paint_scenario_parser  # noqa: E402
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator  # noqa: E402
from artist.util import constants as config_dictionary  # noqa: E402
from artist.util import get_device, set_logger_config  # noqa: E402
from artist.util.config import LightSourceConfig, LightSourceListConfig  # noqa: E402

from blockers import load_field, select_blockers  # noqa: E402

log = logging.getLogger(__name__)

# NURBS fitting parameters, identical to the other scenario creators in this repo.
_N_CONTROL_POINTS = torch.tensor([20, 20])
_FIT_METHOD = config_dictionary.fit_nurbs_from_normals
_DEFL_STEP_SIZE = 100
_FIT_TOLERANCE = 1e-10
_FIT_MAX_EPOCH = 400

_DAIC_ROOT = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
_DAIC_PAINT = pathlib.Path(
    "/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint"
)


def _paths(daic: bool) -> tuple[pathlib.Path, pathlib.Path]:
    """Return (paint_dir, scenario_root) for the local or the DAIC layout."""
    if daic:
        return _DAIC_PAINT, _DAIC_ROOT / "scenarios" / "neighbourhoods"
    root = _here.parents[2]
    return root / "datasets" / "paint", root / "scenarios" / "neighbourhoods"


def _light_source_config() -> LightSourceListConfig:
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


def _nurbs_optimizer_scheduler():
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


def _properties_path(heliostats_dir: pathlib.Path, hid: str) -> pathlib.Path:
    return heliostats_dir / hid / "Properties" / f"{hid}-heliostat-properties.json"


def _deflectometry_path(heliostats_dir: pathlib.Path, hid: str) -> pathlib.Path | None:
    """Most recent deflectometry HDF5 for `hid`, preferring the gap-filled variant."""
    directory = heliostats_dir / hid / "Deflectometry"
    if not directory.exists():
        return None
    filled = sorted(directory.glob(f"{hid}-filled-*-deflectometry.h5"))
    if filled:
        return filled[-1]
    plain = sorted(directory.glob(f"{hid}-*-deflectometry.h5"))
    return plain[-1] if plain else None


def build(
    heliostat_id: str,
    paint_dir: pathlib.Path,
    scenario_root: pathlib.Path,
    ideal_only: bool,
    forward_m: float,
    lateral_m: float,
    force: bool,
    device: torch.device,
) -> pathlib.Path:
    """Build one neighbourhood scenario and return the path of the HDF5 file."""
    heliostats_dir = paint_dir / "heliostats"
    out_dir = scenario_root / heliostat_id
    out_path = out_dir / ("scenario_ideal.h5" if ideal_only else "scenario.h5")

    field = load_field(heliostats_dir, device=torch.device("cpu"))
    screen = select_blockers(
        heliostat_id, field, forward_m=forward_m, lateral_m=lateral_m
    )
    blockers = [h for h in screen["union"] if _properties_path(heliostats_dir, h).exists()]

    # The trained heliostat comes first so its group row index is 0 by construction, but
    # nothing downstream may rely on that: always resolve it through `hg.names`.
    members = [heliostat_id] + blockers
    log.info(f"{heliostat_id}: {len(blockers)} candidate blockers {blockers}")

    if out_path.exists() and not force:
        log.info(f"  [SKIP] exists: {out_path}")
        return out_path
    out_dir.mkdir(parents=True, exist_ok=True)

    tower_file = heliostats_dir / "WRI1030197-tower-measurements.json"
    (
        power_plant_config,
        target_area_list_planar_config,
        target_area_list_cylindrical_config,
    ) = paint_scenario_parser.extract_paint_tower_measurements(
        tower_measurements_path=tower_file, device=device
    )

    deflectometry = (
        None if ideal_only else _deflectometry_path(heliostats_dir, heliostat_id)
    )
    if deflectometry is not None:
        # Fitted surface for the trained heliostat, ideal for the blockers, then merge the
        # two heliostat lists into one group.
        optimizer, scheduler = _nurbs_optimizer_scheduler()
        fitted_list, prototype_config = (
            paint_scenario_parser.extract_paint_heliostats_fitted_surface(
                paths=[
                    (
                        heliostat_id,
                        _properties_path(heliostats_dir, heliostat_id),
                        deflectometry,
                    )
                ],
                power_plant_position=power_plant_config.power_plant_position,
                number_of_nurbs_control_points=_N_CONTROL_POINTS,
                deflectometry_step_size=_DEFL_STEP_SIZE,
                nurbs_fit_method=_FIT_METHOD,
                nurbs_fit_tolerance=_FIT_TOLERANCE,
                nurbs_fit_max_epoch=_FIT_MAX_EPOCH,
                nurbs_fit_optimizer=optimizer,
                nurbs_fit_scheduler=scheduler,
                device=device,
            )
        )
        ideal_list, _ = paint_scenario_parser.extract_paint_heliostats_ideal_surface(
            paths=[(h, _properties_path(heliostats_dir, h)) for h in blockers],
            power_plant_position=power_plant_config.power_plant_position,
            number_of_nurbs_control_points=_N_CONTROL_POINTS,
            device=device,
        )
        heliostat_list_config = fitted_list
        heliostat_list_config.heliostat_list = (
            fitted_list.heliostat_list + ideal_list.heliostat_list
        )
        surface_note = f"deflectometry ({deflectometry.name}) + {len(blockers)} ideal"
    else:
        heliostat_list_config, prototype_config = (
            paint_scenario_parser.extract_paint_heliostats_ideal_surface(
                paths=[(h, _properties_path(heliostats_dir, h)) for h in members],
                power_plant_position=power_plant_config.power_plant_position,
                number_of_nurbs_control_points=_N_CONTROL_POINTS,
                device=device,
            )
        )
        surface_note = f"{len(members)} ideal"

    log.info(f"  building {out_path.name} with {len(members)} heliostats ({surface_note})")
    H5ScenarioGenerator(
        file_path=out_path,
        power_plant_config=power_plant_config,
        target_area_list_planar_config=target_area_list_planar_config,
        target_area_list_cylindrical_config=target_area_list_cylindrical_config,
        light_source_list_config=_light_source_config(),
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    ).generate_scenario()

    (out_dir / "blockers.json").write_text(
        json.dumps(
            {
                "heliostat_id": heliostat_id,
                "members": members,
                "blockers": blockers,
                "per_target": {k: v for k, v in screen.items() if k != "union"},
                "forward_m": forward_m,
                "lateral_m": lateral_m,
                "surfaces": surface_note,
            },
            indent=2,
        )
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build neighbourhood scenarios for the blocking study."
    )
    parser.add_argument("heliostat_ids", nargs="+")
    parser.add_argument("--daic", action="store_true", help="Use DAIC paths.")
    parser.add_argument(
        "--ideal-only",
        action="store_true",
        help="Ideal surfaces for every heliostat (fast; enough for the blocking gate).",
    )
    parser.add_argument("--forward", type=float, default=30.0,
                        help="Box depth along the beam [m].")
    parser.add_argument("--lateral", type=float, default=6.0,
                        help="Box half-width sideways [m].")
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    torch.manual_seed(7)

    paint_dir, scenario_root = _paths(args.daic)
    scenario_root = args.output_dir or scenario_root
    device = get_device()
    log.info(f"PAINT dir: {paint_dir}")
    log.info(f"Scenarios: {scenario_root}")
    log.info(f"Device   : {device}")

    started = time.time()
    for hid in args.heliostat_ids:
        path = build(
            heliostat_id=hid,
            paint_dir=paint_dir,
            scenario_root=scenario_root,
            ideal_only=args.ideal_only,
            forward_m=args.forward,
            lateral_m=args.lateral,
            force=args.force,
            device=device,
        )
        log.info(f"  -> {path}")
    log.info(f"done in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
