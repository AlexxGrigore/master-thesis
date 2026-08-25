"""How far in front can a heliostat still block? An analytic bound, checked by ray tracing.

Motivation
----------
Every blocking scenario has to contain the neighbours that can intercept the beam. Guessing
that number is unsafe (too few and blocking is under-reported, too many and the scenario is
needlessly expensive). This module derives the reach from geometry and then verifies it by
measuring, for every candidate neighbour individually, whether it actually blocks anything.

The formula
-----------
Put the studied heliostat at horizontal distance `D` from a target of height `H` (both
measured in the field's own vertical datum), with hub height `h` and mirror height `a`.

The lowest ray leaves the bottom edge of the mirror, at height `h - a/2`, and climbs linearly
toward the target. A neighbour of the same hub height presents a top edge at `h + a/2` at
worst, when its mirror stands vertical. The beam is therefore intercepted only while it has
climbed less than `a`, one full mirror height:

    beam height at horizontal distance x:   z(x) = (h - a/2) + (H - h + a/2) * x / D
    blocked while                           z(x) < h + a/2

        =>   x_beam  =  a * D / (H - h + a/2)

`x_beam` is an EDGE-to-EDGE distance. Heliostat positions are centres, and the emitting
bottom edge sits up to `a/2` forward of its centre while the neighbour's intercepting near
edge sits up to `a/2` short of its own, so the centre-to-centre bound carries one extra
mirror height:

        x_max  =  a * D / (H - h + a/2)  +  a

Measured: with the edge margin no blocker anywhere in the field lies beyond `x_max`
(0 of 88 blocked cases), while the bare climb term is exceeded by up to 1.6 m.

Three properties follow directly, and they are exactly what the field screen measured:

- `x_max` grows with `D`, so distant heliostats are blocked and near ones are not.
- `x_max` shrinks with target height `H`, so a high target blocks less than a low one.
- Whether any blocking happens at all depends on `x_max` against the local row spacing: if
  the nearest in-beam neighbour is further away than `x_max`, nothing can block.

The bound is deliberately conservative. It assumes the neighbour stands vertical, which is
the largest silhouette any pose can present, so no heliostat beyond `x_max` can block in ANY
pose. The measured reach therefore comes out well below it.

Verification
------------
For each studied heliostat and target, every candidate neighbour is neutralised in turn (its
blocking plane sunk far underground) so that its individual contribution can be read off.
The furthest neighbour that contributes anything, over the whole screened set, is the
empirical reach.

Outputs, under `outputs/new_mapping_function/blocking_study/reach/`:
    blocking_reach_<label>.png   field view for the near / middle / far example heliostat
    blocking_reach.csv           per heliostat and target: x_max, nearest neighbour,
                                 measured furthest blocker, blocked fraction
    blocking_reach.json          the same plus the analytic parameters

Usage
-----
    python blocking_reach.py
    python blocking_reach.py --radius 30 --rays 10
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pathlib
import statistics
import sys

import h5py
import torch

_here = pathlib.Path(__file__).resolve().parent
_src = _here.parents[1]
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_here))

from artist.scenario.scenario import Scenario  # noqa: E402
from artist.util import set_logger_config  # noqa: E402

from field_blocking_map import (  # noqa: E402
    SUN_POSITIONS,
    TARGET_LABEL,
    TARGETS,
    blocked_fraction,
    pose_whole_field,
    standard_heliostats,
    sun_incident_direction,
)

log = logging.getLogger(__name__)

SINK_DEPTH_M = 1000.0
"""Neutralise a neighbour by sinking its blocking plane this far below the field."""

_DAIC_ROOT = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")


def analytic_reach(
    distance_to_target_m: float,
    target_height_m: float,
    hub_height_m: float,
    mirror_height_m: float,
    centre_to_centre: bool = True,
) -> float:
    """Maximum distance in front at which a neighbour can still block.

    The physical core is the climb term: the beam has to gain one mirror height `a` before it
    clears the neighbouring row, which takes

        x_beam = a * D / (H - h + a/2)

    That distance is measured from the emitting EDGE to the intercepting EDGE. Heliostat
    positions, however, are centres. The bottom edge of the studied mirror sits up to `a/2`
    forward of its own centre, and the neighbour's near edge sits up to `a/2` short of its
    centre, so a centre-to-centre bound needs one extra mirror height:

        x_centres = x_beam + a

    Verified: with the edge margin included no measured blocker anywhere in the field lies
    beyond the bound, whereas the bare climb term is exceeded by up to 1.6 m.

    Conservative in every other respect too: it assumes the neighbour stands vertical, the
    largest silhouette any pose can present, so nothing beyond `x_centres` can block in ANY
    pose.
    """
    denominator = target_height_m - hub_height_m + 0.5 * mirror_height_m
    if denominator <= 0.0:
        return float("inf")
    climb = mirror_height_m * distance_to_target_m / denominator
    return climb + mirror_height_m if centre_to_centre else climb


def candidates_in_corridor(
    positions: torch.Tensor,
    heliostat_index: int,
    target_centre: torch.Tensor,
    radius_m: float,
    lateral_m: float,
) -> list[tuple[int, float, float]]:
    """Neighbours in front of the heliostat: (index, distance along beam, lateral offset)."""
    origin = positions[heliostat_index, :3]
    beam = (target_centre[:3] - origin)[:2]
    length = torch.norm(beam)
    if length < 1e-6:
        return []
    direction = beam / length

    offsets = positions[:, :2] - origin[:2]
    along = offsets @ direction
    lateral = (offsets - along.unsqueeze(1) * direction.unsqueeze(0)).norm(dim=1)

    keep = (along > 0.5) & (along < radius_m) & (lateral < lateral_m)
    keep[heliostat_index] = False
    return [
        (int(i), float(along[i]), float(lateral[i]))
        for i in torch.nonzero(keep, as_tuple=True)[0].tolist()
    ]


def _sunk_except(surfaces: torch.Tensor, keep_row: int | None) -> torch.Tensor:
    """Copy of `surfaces` where every row except `keep_row` is sunk underground."""
    out = surfaces.clone()
    if keep_row is None:
        out[:, :, 2] -= SINK_DEPTH_M
        return out
    mask = torch.ones(out.shape[0], dtype=torch.bool)
    mask[keep_row] = False
    out[mask, :, 2] -= SINK_DEPTH_M
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analytic and measured blocking reach.")
    parser.add_argument("--daic", action="store_true")
    parser.add_argument("--surface-points", type=int, default=25)
    parser.add_argument("--rays", type=int, default=10)
    parser.add_argument("--radius", type=float, default=30.0,
                        help="Search radius for candidate neighbours [m].")
    parser.add_argument("--lateral", type=float, default=6.0,
                        help="Lateral half-width of the search corridor [m].")
    parser.add_argument("--sun-index", type=int, default=2,
                        help="Which representative sun to use (default: the median one).")
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("artist.field.kinematics_rigid_body").setLevel(logging.ERROR)
    torch.manual_seed(7)

    root = _DAIC_ROOT if args.daic else _here.parents[2]
    scenario_path = root / "scenarios" / "full_benchmark_ideal" / "ideal_1277.h5"
    output_dir = (
        root / "outputs" / "new_mapping_function" / "blocking_study" / "reach"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    with h5py.File(scenario_path) as handle:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=handle,
            device=device,
            number_of_surface_points_per_facet=torch.tensor(
                [args.surface_points, args.surface_points]
            ),
        )
    scenario.set_number_of_rays(args.rays)
    group = scenario.heliostat_field.heliostat_groups[0]
    print(f"Scenario: {group.number_of_heliostats} heliostats, ideal surfaces")

    # Mirror dimensions from the local surface-point cloud: local e is width, local n height.
    local = group.surface_points[0, :, :3]
    mirror_width = float(local[:, 0].max() - local[:, 0].min())
    mirror_height = float(local[:, 1].max() - local[:, 1].min())
    print(f"Mirror: {mirror_width:.2f} m wide x {mirror_height:.2f} m high")

    azimuth, elevation = SUN_POSITIONS[args.sun_index]
    incident = sun_incident_direction(azimuth, elevation, device)
    print(f"Sun: azimuth {azimuth:.1f} deg, elevation {elevation:.1f} deg")

    present = [h for h in standard_heliostats(root) if h in group.names]
    print(f"Screening {len(present)} heliostats x {len(TARGETS)} targets, "
          f"candidates within {args.radius:.0f} m\n")

    rows: list[dict] = []
    detail: dict[tuple[str, str], list[dict]] = {}

    for target_name in TARGETS:
        target_index = scenario.solar_tower.target_name_to_index[target_name]
        target_centre = scenario.solar_tower.get_centers_of_target_areas(
            target_area_indices=torch.tensor([target_index], device=device), device=device
        )[0]
        target_height = float(target_centre[2])
        field_surfaces = pose_whole_field(group, target_centre, incident, device)

        print(f"{TARGET_LABEL[target_name]}  (target {target_height:.1f} m)")
        reaches: list[float] = []
        measured: list[float] = []

        for heliostat_id in present:
            index = group.names.index(heliostat_id)
            hub_height = float(group.positions[index, 2])
            distance = float(torch.norm((target_centre[:3] - group.positions[index, :3])[:2]))
            x_max = analytic_reach(distance, target_height, hub_height, mirror_height)
            x_climb = analytic_reach(
                distance, target_height, hub_height, mirror_height, centre_to_centre=False
            )
            reaches.append(x_max)

            candidates = candidates_in_corridor(
                group.positions, index, target_centre, args.radius, args.lateral
            )
            nearest = min((d for _, d, _ in candidates), default=float("nan"))

            # Total, then each candidate alone, so its own contribution is isolated.
            neighbour_rows = [i for i, _, _ in candidates]
            total, on_target = blocked_fraction(
                scenario, group, index, neighbour_rows, target_index,
                target_centre, incident, device, field_surfaces,
            )
            per_neighbour = []
            if on_target > 1e-6:
                for position_in_list, (i, along, lateral) in enumerate(candidates):
                    single = _sunk_except(field_surfaces[neighbour_rows], position_in_list)
                    fraction, _ = blocked_fraction(
                        scenario, group, index, list(range(len(neighbour_rows))),
                        target_index, target_centre, incident, device, single,
                    )
                    per_neighbour.append(
                        {
                            "name": group.names[i],
                            "row": i,
                            "along_m": along,
                            "lateral_m": lateral,
                            "blocked": fraction,
                        }
                    )
            detail[(heliostat_id, target_name)] = per_neighbour
            blockers = [n for n in per_neighbour if n["blocked"] > 1e-4]
            furthest = max((n["along_m"] for n in blockers), default=float("nan"))
            if furthest == furthest:
                measured.append(furthest)

            rows.append(
                {
                    "heliostat": heliostat_id,
                    "target": target_name,
                    "east_m": float(group.positions[index, 0]),
                    "north_m": float(group.positions[index, 1]),
                    "distance_to_target_m": distance,
                    "target_height_m": target_height,
                    "analytic_x_max_m": x_max,
                    "analytic_climb_term_m": x_climb,
                    "nearest_candidate_m": nearest,
                    "n_candidates": len(candidates),
                    "n_actual_blockers": len(blockers),
                    "furthest_actual_blocker_m": furthest,
                    "blocked_fraction_total": total if on_target > 1e-6 else float("nan"),
                    "beam_reaches_target": on_target > 1e-6,
                }
            )

        print(f"  analytic x_max : {min(reaches):.1f} to {max(reaches):.1f} m "
              f"(median {statistics.median(reaches):.1f})")
        if measured:
            print(f"  measured reach : furthest actual blocker {max(measured):.1f} m, "
                  f"median {statistics.median(measured):.1f} m")
        else:
            print("  measured reach : no blockers found")
        print()

    # ------------------------------------------------------------------ summary
    valid = [r for r in rows if r["beam_reaches_target"]]
    all_blockers = [
        r["furthest_actual_blocker_m"] for r in valid
        if r["furthest_actual_blocker_m"] == r["furthest_actual_blocker_m"]
    ]
    bound_ok = [
        r for r in valid
        if r["furthest_actual_blocker_m"] == r["furthest_actual_blocker_m"]
        and r["furthest_actual_blocker_m"] > r["analytic_x_max_m"]
    ]
    print("=" * 72)
    print(f"Analytic bound  x_max = a*D/(H - h + a/2) + a,  a = {mirror_height:.2f} m")
    print(f"  range over all heliostats and targets : "
          f"{min(r['analytic_x_max_m'] for r in valid):.1f} to "
          f"{max(r['analytic_x_max_m'] for r in valid):.1f} m")
    if all_blockers:
        print(f"Measured furthest actual blocker, anywhere : {max(all_blockers):.1f} m")
    print(f"Cases where a blocker sat BEYOND the analytic bound : {len(bound_ok)} "
          f"(expected 0)")
    n_one_row = sum(
        1 for r in valid
        if r["n_actual_blockers"] > 0
        and r["furthest_actual_blocker_m"] < 1.6 * r["nearest_candidate_m"]
    )
    n_blocked = sum(1 for r in valid if r["n_actual_blockers"] > 0)
    print(f"Blocked cases whose blockers all sit in the nearest row : "
          f"{n_one_row} of {n_blocked}")
    print("=" * 72)

    with open(output_dir / "blocking_reach.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "blocking_reach.json").write_text(
        json.dumps(
            {
                "formula": "x_max = a * D / (H - h + a/2)",
                "mirror_width_m": mirror_width,
                "mirror_height_m": mirror_height,
                "sun_azimuth_deg": azimuth,
                "sun_elevation_deg": elevation,
                "search_radius_m": args.radius,
                "search_lateral_m": args.lateral,
                "rows": rows,
                "per_neighbour": {
                    f"{h}|{t}": v for (h, t), v in detail.items()
                },
            },
            indent=2,
        )
    )
    print(f"\nWrote {output_dir / 'blocking_reach.csv'}")

    _plot_examples(rows, detail, group, scenario, mirror_height, output_dir, device)


def _plot_examples(rows, detail, group, scenario, mirror_height, output_dir, device) -> None:
    """Field views for a near, a middle and a far heliostat, on the worst-case target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # The lower target has the largest reach, so it is the informative case.
    target_name = "solar_tower_juelich_lower"
    subset = [
        r for r in rows if r["target"] == target_name and r["beam_reaches_target"]
    ]
    subset.sort(key=lambda r: r["distance_to_target_m"])
    picks = [
        ("near", subset[0]),
        ("middle", subset[len(subset) // 2]),
        ("far", subset[-1]),
    ]

    target_index = scenario.solar_tower.target_name_to_index[target_name]
    target_centre = scenario.solar_tower.get_centers_of_target_areas(
        target_area_indices=torch.tensor([target_index], device=device), device=device
    )[0]

    all_east = group.positions[:, 0].tolist()
    all_north = group.positions[:, 1].tolist()

    for label, row in picks:
        heliostat_id = row["heliostat"]
        neighbours = detail[(heliostat_id, target_name)]
        origin = torch.tensor([row["east_m"], row["north_m"]])
        beam = (target_centre[:2] - origin)
        direction = beam / torch.norm(beam)
        x_max = row["analytic_x_max_m"]

        figure, axis = plt.subplots(figsize=(9.0, 8.6))
        axis.scatter(all_east, all_north, s=6, c="0.88", linewidths=0, zorder=1,
                     label=f"field ({group.number_of_heliostats} heliostats)")

        # Beam direction and the analytic cut-off, drawn in the plan view.
        tip = origin + direction * min(x_max * 2.4, 60.0)
        axis.annotate("", xy=(float(tip[0]), float(tip[1])),
                      xytext=(float(origin[0]), float(origin[1])),
                      arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=1.6,
                                      alpha=0.75), zorder=4)
        cut = origin + direction * x_max
        perpendicular = torch.tensor([-direction[1], direction[0]])
        end_a = cut + perpendicular * 7.0
        end_b = cut - perpendicular * 7.0
        axis.plot([float(end_a[0]), float(end_b[0])], [float(end_a[1]), float(end_b[1])],
                  color="#1f77b4", lw=2.0, ls="--", zorder=4,
                  label=f"analytic reach $x_{{max}}$ = {x_max:.1f} m")

        blockers = [n for n in neighbours if n["blocked"] > 1e-4]
        quiet = [n for n in neighbours if n["blocked"] <= 1e-4]
        if quiet:
            axis.scatter(
                [float(group.positions[n["row"], 0]) for n in quiet],
                [float(group.positions[n["row"], 1]) for n in quiet],
                s=70, facecolors="none", edgecolors="0.45", linewidths=1.1, zorder=5,
                label=f"in corridor, blocks nothing ({len(quiet)})",
            )
        if blockers:
            scatter = axis.scatter(
                [float(group.positions[n["row"], 0]) for n in blockers],
                [float(group.positions[n["row"], 1]) for n in blockers],
                c=[n["blocked"] * 100 for n in blockers], s=190, cmap="inferno_r",
                vmin=0.0, edgecolors="0.15", linewidths=0.8, zorder=6, marker="s",
                label=f"actually blocks ({len(blockers)})",
            )
            bar = figure.colorbar(scatter, ax=axis, pad=0.02, shrink=0.82)
            bar.set_label("rays this one neighbour blocks [%]")
            for n in blockers:
                axis.annotate(
                    f"{n['name']}\n{n['along_m']:.1f} m, {n['blocked'] * 100:.1f}%",
                    (float(group.positions[n["row"], 0]),
                     float(group.positions[n["row"], 1])),
                    textcoords="offset points", xytext=(11, -4), fontsize=8.0,
                    zorder=7,
                )

        axis.scatter([row["east_m"]], [row["north_m"]], marker="*", s=440, c="#2ca02c",
                     edgecolors="white", linewidths=1.1, zorder=8,
                     label=f"{heliostat_id} (studied)")

        span = max(x_max * 2.6, 34.0)
        centre = origin + direction * (span * 0.30)
        axis.set_xlim(float(centre[0]) - span, float(centre[0]) + span)
        axis.set_ylim(float(centre[1]) - span, float(centre[1]) + span)

        furthest = row["furthest_actual_blocker_m"]
        furthest_text = f"{furthest:.1f} m" if furthest == furthest else "none"
        axis.set_title(
            f"{heliostat_id}   ({label} case, {row['distance_to_target_m']:.0f} m to the "
            f"lower target)\n"
            f"analytic reach $x_{{max}}$ = {x_max:.1f} m (a conservative upper bound), "
            f"furthest neighbour that actually blocks: {furthest_text}\n"
            f"{row['n_actual_blockers']} of {row['n_candidates']} neighbours in the "
            f"corridor block anything; total {row['blocked_fraction_total'] * 100:.1f}% "
            f"of the beam",
            fontsize=10.5,
        )
        axis.set_xlabel("east [m]")
        axis.set_ylabel("north [m]")
        axis.set_aspect("equal")
        axis.grid(alpha=0.25, linewidth=0.5)
        handles, labels = axis.get_legend_handles_labels()
        handles.append(Line2D([], [], color="#1f77b4", lw=1.6, marker=">",
                              markersize=5, label="beam direction to target"))
        labels.append("beam direction to target")
        axis.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.93)
        figure.tight_layout()
        out = output_dir / f"blocking_reach_{label}_{heliostat_id}.png"
        figure.savefig(out, dpi=170)
        plt.close(figure)
        print(f"Wrote {out.name}")


if __name__ == "__main__":
    main()
