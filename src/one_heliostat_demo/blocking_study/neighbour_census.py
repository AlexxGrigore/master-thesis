"""How many neighbours actually block a heliostat? Measure it, do not assume it.

The coarse screen in `blockers.py` is a conservative superset: it keeps any heliostat whose
bounding sphere comes near the beam line in ANY pose. That is the right way to choose
candidates, but it over-predicts badly (AP43 passed the screen and has literally zero real
blocking). This module answers the operational question instead:

    Given a generous candidate set, which neighbours does the RAYTRACER actually
    intercept rays on, how much does each one contribute, and therefore how large does a
    blocking scenario have to be?

Method. Build one scenario holding every generous candidate. For each calibration sample,
trace with all candidates present to get the total blocked fraction, then trace once per
candidate with every OTHER candidate removed, to get that candidate's individual
contribution. Candidates are removed by sinking their blocking plane 1 km underground,
which is the only way to neutralise a row without breaking ARTIST's self-exclusion
indexing (`ray_to_heliostat_mapping` indexes the group, so rows cannot be dropped).

Neighbours are posed with an AIMED hypothesis, since stowed neighbours block almost
nothing (see GATE_RESULTS.md) and the scenario has to be sized for the worst case.

Usage
-----
    python neighbour_census.py BE25 AY43 AY44 --max-samples 10
    python neighbour_census.py BE25 --candidate-margin 8.0
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import statistics
import sys

import h5py
import torch

_here = pathlib.Path(__file__).resolve().parent
_src = _here.parents[1]
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_here))

from artist.io.paint_calibration_parser import PaintCalibrationDataParser  # noqa: E402
from artist.scenario.scenario import Scenario  # noqa: E402
from artist.util import set_logger_config  # noqa: E402

from utils.evaluation import build_heliostat_data_mapping  # noqa: E402

from blockers import load_field, select_blockers  # noqa: E402
from gate import BENCHMARK, _paths, blocker_surfaces, trace_one_sample  # noqa: E402

log = logging.getLogger(__name__)

SINK_DEPTH_M = 1000.0
"""Neutralise a blocker by sinking its plane this far below the field."""


def _sink_all_except(
    surfaces: torch.Tensor, keep_indices: list[int], trained_index: int
) -> torch.Tensor:
    """Copy of `surfaces` with every row except `keep_indices` sunk far underground.

    The trained heliostat's own row is always kept as-is: it is excluded from blocking by
    ARTIST's self-intersection test, and moving it would corrupt that test.
    """
    out = surfaces.clone()
    keep = set(keep_indices) | {trained_index}
    for row in range(out.shape[0]):
        if row not in keep:
            out[row, :, 2] -= SINK_DEPTH_M
    return out


def census(
    heliostat_id: str,
    hypothesis: str,
    candidate_margin_m: float,
    paint_dir: pathlib.Path,
    scenario_root: pathlib.Path,
    output_root: pathlib.Path,
    max_samples: int,
    surface_points_per_facet: int,
    number_of_rays: int,
    device: torch.device,
) -> dict:
    """Measure each candidate neighbour's individual blocking contribution."""
    scenario_dir = scenario_root / heliostat_id
    scenario_path = scenario_dir / "scenario.h5"
    if not scenario_path.exists():
        scenario_path = scenario_dir / "scenario_ideal.h5"
    if not scenario_path.exists():
        raise FileNotFoundError(
            f"No scenario for {heliostat_id}; run build_neighbourhood_scenario.py first."
        )

    with h5py.File(scenario_path) as file_handle:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=file_handle,
            device=device,
            number_of_surface_points_per_facet=torch.tensor(
                [surface_points_per_facet, surface_points_per_facet]
            ),
        )
    scenario.set_number_of_rays(number_of_rays)
    group = scenario.heliostat_field.heliostat_groups[0]
    trained_index = group.names.index(heliostat_id)
    candidate_indices = [i for i in range(group.number_of_heliostats) if i != trained_index]

    # Geometry of each candidate relative to the trained heliostat, for reporting.
    field = load_field(paint_dir / "heliostats", device=torch.device("cpu"))
    origin = field.positions[heliostat_id]
    geometry = {}
    for index in candidate_indices:
        name = group.names[index]
        if name in field.positions:
            offset = field.positions[name] - origin
            geometry[name] = {
                "distance_m": float(torch.norm(offset[:2])),
                "delta_east_m": float(offset[0]),
                "delta_north_m": float(offset[1]),
            }

    mapping = [
        entry
        for entry in build_heliostat_data_mapping(
            paint_dir / "splits" / f"{BENCHMARK}.csv",
            paint_dir / BENCHMARK / "calibration_properties",
            paint_dir / BENCHMARK / "flux_image",
            "test",
        )
        if entry[0] == heliostat_id
    ]
    _, centroids, rays, motor_positions, _, target_mask = PaintCalibrationDataParser(
        centroid_extraction_method="UTIS"
    ).parse_data_for_reconstruction(
        heliostat_data_mapping=mapping,
        heliostat_group=group,
        scenario=scenario,
        device=device,
    )

    number_of_samples = min(max_samples, rays.shape[0])
    log.info(
        f"{heliostat_id}: {len(candidate_indices)} candidates, {number_of_samples} samples, "
        f"hypothesis '{hypothesis}'"
    )

    totals: list[float] = []
    individual: dict[str, list[float]] = {group.names[i]: [] for i in candidate_indices}

    for sample in range(number_of_samples):
        surfaces = blocker_surfaces(group, hypothesis, scenario, rays[sample], device)

        def blocked_with(kept: list[int]) -> float:
            posed = _sink_all_except(surfaces, kept, trained_index)
            _, _, on_target, blocking_factor = trace_one_sample(
                scenario, group, trained_index,
                motor_positions[sample], rays[sample], target_mask[sample],
                posed, device, centroid=centroids[sample],
            )
            if on_target.item() < 1e-6:
                raise RuntimeError(
                    f"{heliostat_id} sample {sample}: beam misses the target entirely."
                )
            return 1.0 - float(blocking_factor.item())

        totals.append(blocked_with(candidate_indices))
        for index in candidate_indices:
            individual[group.names[index]].append(blocked_with([index]))

    ranked = sorted(
        (
            {
                "name": name,
                "blocked_fraction_median": statistics.median(values),
                "blocked_fraction_max": max(values),
                **geometry.get(name, {}),
            }
            for name, values in individual.items()
        ),
        key=lambda row: row["blocked_fraction_median"],
        reverse=True,
    )
    total_median = statistics.median(totals)
    contributing = [r for r in ranked if r["blocked_fraction_max"] > 1e-4]

    summary = {
        "heliostat_id": heliostat_id,
        "hypothesis": hypothesis,
        "candidate_margin_m": candidate_margin_m,
        "number_of_samples": number_of_samples,
        "number_of_candidates": len(candidate_indices),
        "total_blocked_fraction_median": total_median,
        "number_contributing": len(contributing),
        "max_contributing_distance_m": (
            max(r.get("distance_m", 0.0) for r in contributing) if contributing else 0.0
        ),
        "per_neighbour": ranked,
    }

    out_dir = output_root / heliostat_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"neighbour_census_{hypothesis}.json").write_text(
        json.dumps(summary, indent=2)
    )

    print()
    print(f"  {heliostat_id}  hypothesis={hypothesis}  n={number_of_samples}  "
          f"total blocked median {total_median * 100:.2f}%")
    print(f"    {'neighbour':10s} {'dist_m':>7s} {'dE':>7s} {'dN':>7s} "
          f"{'blocked_med':>12s} {'blocked_max':>12s}")
    for row in ranked:
        mark = "  <-" if row["blocked_fraction_max"] > 1e-4 else ""
        print(f"    {row['name']:10s} {row.get('distance_m', float('nan')):7.1f} "
              f"{row.get('delta_east_m', float('nan')):7.1f} "
              f"{row.get('delta_north_m', float('nan')):7.1f} "
              f"{row['blocked_fraction_median'] * 100:11.2f}% "
              f"{row['blocked_fraction_max'] * 100:11.2f}%{mark}")
    print(f"    -> {len(contributing)} of {len(candidate_indices)} candidates contribute; "
          f"furthest contributor {summary['max_contributing_distance_m']:.1f} m")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Census of real blocking neighbours.")
    parser.add_argument("heliostat_ids", nargs="+")
    parser.add_argument("--daic", action="store_true")
    parser.add_argument("--hypothesis", default="lower")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--surface-points", type=int, default=25)
    parser.add_argument("--rays", type=int, default=10)
    parser.add_argument("--scenario-root", type=pathlib.Path, default=None)
    parser.add_argument(
        "--candidate-margin",
        type=float,
        default=1.0,
        help="Lateral margin used when the scenario was built (reported, not re-screened).",
    )
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.WARNING)
    log.setLevel(logging.INFO)
    torch.manual_seed(7)

    paint_dir, scenario_root, output_root = _paths(args.daic)
    if args.scenario_root is not None:
        scenario_root = args.scenario_root
    for hid in args.heliostat_ids:
        census(
            heliostat_id=hid,
            hypothesis=args.hypothesis,
            candidate_margin_m=args.candidate_margin,
            paint_dir=paint_dir,
            scenario_root=scenario_root,
            output_root=output_root,
            max_samples=args.max_samples,
            surface_points_per_facet=args.surface_points,
            number_of_rays=args.rays,
            device=torch.device("cpu"),
        )


if __name__ == "__main__":
    main()
