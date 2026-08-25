"""Row-by-row test: does only the front row block, or does the row behind it matter too?

The earlier reach study concluded "only the immediately adjacent row blocks", but it was run
on the 63 standard heliostats, which reach at most 224 m from the tower. The full field
extends to roughly 290 m, where the beam is shallower and the analytic reach

    x_max = a*D/(H - h + a/2) + a

is correspondingly larger, so it can in principle admit a second row. This module tests that
directly, on the full 1277-heliostat field.

Method. For a studied heliostat, group the in-beam neighbours into rows by their distance
along the beam, then take the three most central members of ROW 1 (immediately in front) and
the three most central of ROW 2 (in front of row 1). Each of these six is then traced ALONE as
the only blocker in the scene, so its individual contribution is isolated with no
interference from the others.

Studied heliostats span the field from the nearest to the farthest, because the whole question
is whether the conclusion changes with distance.

Usage
-----
    python row_by_row.py
    python row_by_row.py --target solar_tower_juelich_lower --rays 10
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import sys

import h5py
import torch

_here = pathlib.Path(__file__).resolve().parent
_src = _here.parents[1]
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_here))

from artist.scenario.scenario import Scenario  # noqa: E402
from artist.util import set_logger_config  # noqa: E402

from blocking_reach import analytic_reach  # noqa: E402
from field_blocking_map import (  # noqa: E402
    SUN_POSITIONS,
    TARGET_LABEL,
    blocked_fraction,
    pose_whole_field,
    sun_incident_direction,
)

log = logging.getLogger(__name__)

_DAIC_ROOT = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")

ROW_TOLERANCE_M = 3.5
"""Neighbours within this distance of each other along the beam count as the same row."""


def rows_in_front(
    positions: torch.Tensor,
    heliostat_index: int,
    target_centre: torch.Tensor,
    search_radius_m: float,
    lateral_m: float,
) -> list[list[dict]]:
    """Group the in-beam neighbours into rows ordered by distance along the beam."""
    origin = positions[heliostat_index, :3]
    beam = (target_centre[:3] - origin)[:2]
    length = torch.norm(beam)
    if length < 1e-6:
        return []
    direction = beam / length

    offsets = positions[:, :2] - origin[:2]
    along = offsets @ direction
    lateral = (offsets - along.unsqueeze(1) * direction.unsqueeze(0)) @ torch.tensor(
        [-direction[1], direction[0]]
    )

    keep = (along > 0.5) & (along < search_radius_m) & (lateral.abs() < lateral_m)
    keep[heliostat_index] = False
    candidates = [
        {"row_index": int(i), "along_m": float(along[i]), "lateral_m": float(lateral[i])}
        for i in torch.nonzero(keep, as_tuple=True)[0].tolist()
    ]
    candidates.sort(key=lambda c: c["along_m"])

    rows: list[list[dict]] = []
    for candidate in candidates:
        if rows and candidate["along_m"] - rows[-1][0]["along_m"] <= ROW_TOLERANCE_M:
            rows[-1].append(candidate)
        else:
            rows.append([candidate])
    return rows


def most_central(row: list[dict], count: int) -> list[dict]:
    """The `count` members of a row closest to the beam axis."""
    return sorted(row, key=lambda c: abs(c["lateral_m"]))[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Row-by-row blocking contributions.")
    parser.add_argument("--daic", action="store_true")
    parser.add_argument("--target", default="solar_tower_juelich_lower")
    parser.add_argument("--surface-points", type=int, default=25)
    parser.add_argument("--rays", type=int, default=10)
    parser.add_argument("--per-row", type=int, default=3)
    parser.add_argument("--rows", type=int, default=2, help="How many rows to test.")
    parser.add_argument("--search-radius", type=float, default=45.0)
    parser.add_argument("--lateral", type=float, default=6.0)
    parser.add_argument("--sun-index", type=int, default=2)
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("artist.field.kinematics_rigid_body").setLevel(logging.ERROR)
    torch.manual_seed(7)

    root = _DAIC_ROOT if args.daic else _here.parents[2]
    scenario_path = root / "scenarios" / "full_benchmark_ideal" / "ideal_1277.h5"
    output_dir = root / "outputs" / "new_mapping_function" / "blocking_study" / "reach"
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

    local = group.surface_points[0, :, :3]
    mirror_height = float(local[:, 1].max() - local[:, 1].min())

    target_index = scenario.solar_tower.target_name_to_index[args.target]
    target_centre = scenario.solar_tower.get_centers_of_target_areas(
        target_area_indices=torch.tensor([target_index], device=device), device=device
    )[0]
    target_height = float(target_centre[2])

    azimuth, elevation = SUN_POSITIONS[args.sun_index]
    incident = sun_incident_direction(azimuth, elevation, device)
    field_surfaces = pose_whole_field(group, target_centre, incident, device)

    print(f"Field: {group.number_of_heliostats} heliostats, ideal surfaces")
    print(f"Target: {TARGET_LABEL[args.target]} at {target_height:.1f} m")
    print(f"Sun: azimuth {azimuth:.1f}, elevation {elevation:.1f} deg")
    print(f"Mirror height a = {mirror_height:.2f} m\n")

    # Distance of every heliostat to this target, used both to pick the examples and to
    # report how the answer changes across the field.
    distances = torch.norm(
        (target_centre[:3].unsqueeze(0) - group.positions[:, :3])[:, :2], dim=1
    )
    order = torch.argsort(distances)
    # Nearest, two intermediate, and the three FARTHEST, since the far end is where the
    # analytic reach is largest and a second row could plausibly matter.
    picks = [
        int(order[0]),
        int(order[len(order) // 3]),
        int(order[2 * len(order) // 3]),
        int(order[-3]),
        int(order[-2]),
        int(order[-1]),
    ]

    records: list[dict] = []
    for heliostat_index in picks:
        name = group.names[heliostat_index]
        distance = float(distances[heliostat_index])
        hub = float(group.positions[heliostat_index, 2])
        x_max = analytic_reach(distance, target_height, hub, mirror_height)

        rows = rows_in_front(
            group.positions, heliostat_index, target_centre,
            args.search_radius, args.lateral,
        )
        if not rows:
            print(f"{name}: no neighbours in front, skipped\n")
            continue

        print(f"{'=' * 78}")
        print(f"{name}   {distance:.0f} m to the target   analytic reach x_max = {x_max:.1f} m")
        print(f"{'row':>4s} {'heliostat':10s} {'along_m':>8s} {'lateral_m':>10s} "
              f"{'within x_max':>13s} {'BLOCKS':>9s}")

        for row_number, row in enumerate(rows[: args.rows], start=1):
            for candidate in most_central(row, args.per_row):
                fraction, on_target = blocked_fraction(
                    scenario, group, heliostat_index, [candidate["row_index"]],
                    target_index, target_centre, incident, device, field_surfaces,
                )
                if on_target < 1e-6:
                    print(f"  {name}: beam does not reach the target, skipped")
                    break
                inside = candidate["along_m"] <= x_max
                print(f"{row_number:>4d} {group.names[candidate['row_index']]:10s} "
                      f"{candidate['along_m']:8.1f} {candidate['lateral_m']:10.1f} "
                      f"{('yes' if inside else 'no'):>13s} "
                      f"{fraction * 100:8.2f}%")
                records.append(
                    {
                        "studied": name,
                        "studied_distance_m": distance,
                        "analytic_x_max_m": x_max,
                        "row": row_number,
                        "neighbour": group.names[candidate["row_index"]],
                        "along_m": candidate["along_m"],
                        "lateral_m": candidate["lateral_m"],
                        "within_x_max": inside,
                        "blocked_fraction": fraction,
                    }
                )
        print()

    # ------------------------------------------------------------------ verdict
    print("=" * 78)
    by_row: dict[int, list[float]] = {}
    for record in records:
        by_row.setdefault(record["row"], []).append(record["blocked_fraction"])
    for row_number in sorted(by_row):
        values = by_row[row_number]
        blocking = [v for v in values if v > 1e-4]
        print(f"Row {row_number}: {len(blocking)} of {len(values)} tested neighbours block "
              f"anything" + (f", up to {max(blocking) * 100:.2f}% of the beam"
                             if blocking else ""))
    second_row_blockers = [
        r for r in records if r["row"] >= 2 and r["blocked_fraction"] > 1e-4
    ]
    print()
    if second_row_blockers:
        print("ANSWER: no, not only the front row. Second-row heliostats that block:")
        for record in second_row_blockers:
            print(f"  {record['studied']} (at {record['studied_distance_m']:.0f} m) "
                  f"is blocked by {record['neighbour']} at {record['along_m']:.1f} m "
                  f"by {record['blocked_fraction'] * 100:.2f}%")
    else:
        print("ANSWER: yes, only the front row blocks. No row-2 neighbour blocked anything.")
    print("=" * 78)

    with open(output_dir / "row_by_row.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "row_by_row.json").write_text(
        json.dumps(
            {
                "target": args.target,
                "target_height_m": target_height,
                "sun_azimuth_deg": azimuth,
                "sun_elevation_deg": elevation,
                "mirror_height_m": mirror_height,
                "row_tolerance_m": ROW_TOLERANCE_M,
                "records": records,
            },
            indent=2,
        )
    )
    print(f"\nWrote {output_dir / 'row_by_row.csv'}")


if __name__ == "__main__":
    main()
