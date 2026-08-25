"""Field-wide reach sweep: the largest neighbourhood radius any heliostat could ever need.

Every earlier reach measurement (blocking_reach.py, row_by_row.py) was run on a hand-picked
subset of heliostats. This module answers the sizing question for ALL 1277, so that a
neighbourhood-scenario builder can be given one number (or, better, a rule) that is safe for
any heliostat someone later decides to study.

Two stages.

STAGE 1, analytic, exact, no ray tracing (fast: 1277 heliostats x 3 targets is pure
geometry). For every heliostat and every calibration target, evaluate the verified bound

    x_max = a*D/(H - h + a/2) + a

using its actual field position and hub height. This gives the full distribution of "how
far out would I need to look" across the real field, not just the 63-heliostat subset.

STAGE 2, raytraced validation on a STRATIFIED SAMPLE. Checking all 1277 with the raytracer
would cost roughly 20x the earlier 63-heliostat screen for no new information away from the
sample; the earlier row_by_row study already showed the bound can be tight exactly at the
field's edge, which is precisely where stage 1 identifies the worst cases. So stage 2 draws a
sample spanning every distance decile PLUS every one of the single worst analytic cases, and
confirms, neighbour by neighbour, that nothing measured ever exceeds its own analytic bound.

Targets are the three actually used for calibration in the benchmark (verified from real
calibration_properties JSONs: lower, upper, multi focus tower; the receiver does not occur).

Outputs, under `outputs/new_mapping_function/blocking_study/field_wide/`:
    field_wide_reach.csv       x_max, distance, target height for all 1277 x 3 targets
    field_wide_reach_hist.png  distribution of x_max across the whole field
    field_wide_reach_vs_distance.png   x_max against distance, coloured by target
    validation.csv             stage-2 raytraced check: per-neighbour contribution vs bound

Usage
-----
    python field_wide_reach.py
    python field_wide_reach.py --validate-n 60 --rays 5
"""

from __future__ import annotations

import argparse
import csv
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

from artist.scenario.scenario import Scenario  # noqa: E402
from artist.util import set_logger_config  # noqa: E402

from blocking_reach import analytic_reach, candidates_in_corridor  # noqa: E402
from field_blocking_map import (  # noqa: E402
    SUN_POSITIONS,
    TARGET_LABEL,
    TARGETS,
    blocked_fraction,
    pose_whole_field,
    sun_incident_direction,
)

log = logging.getLogger(__name__)

_DAIC_ROOT = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")


def main() -> None:
    parser = argparse.ArgumentParser(description="Field-wide blocking-reach sweep.")
    parser.add_argument("--daic", action="store_true")
    parser.add_argument("--surface-points", type=int, default=25)
    parser.add_argument("--rays", type=int, default=5)
    parser.add_argument("--sun-index", type=int, default=2)
    parser.add_argument("--search-radius", type=float, default=35.0,
                        help="Validation corridor radius [m] (must exceed the global x_max).")
    parser.add_argument("--lateral", type=float, default=6.0)
    parser.add_argument("--validate-n", type=int, default=60,
                        help="Heliostats to raytrace in stage 2, spread across distance "
                             "deciles plus the single worst analytic case per target.")
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("artist.field.kinematics_rigid_body").setLevel(logging.ERROR)
    torch.manual_seed(7)

    root = _DAIC_ROOT if args.daic else _here.parents[2]
    scenario_path = root / "scenarios" / "full_benchmark_ideal" / "ideal_1277.h5"
    output_dir = root / "outputs" / "new_mapping_function" / "blocking_study" / "field_wide"
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
    print(f"Field: {group.number_of_heliostats} heliostats, ideal surfaces")

    local = group.surface_points[0, :, :3]
    mirror_height = float(local[:, 1].max() - local[:, 1].min())
    print(f"Mirror height a = {mirror_height:.2f} m\n")

    # ============================================================ STAGE 1: analytic sweep
    rows: list[dict] = []
    target_centres: dict[str, torch.Tensor] = {}
    for target_name in TARGETS:
        target_index = scenario.solar_tower.target_name_to_index[target_name]
        centre = scenario.solar_tower.get_centers_of_target_areas(
            target_area_indices=torch.tensor([target_index], device=device), device=device
        )[0]
        target_centres[target_name] = centre
        target_height = float(centre[2])

        distances = torch.norm(
            (centre[:3].unsqueeze(0) - group.positions[:, :3])[:, :2], dim=1
        )
        for i in range(group.number_of_heliostats):
            hub = float(group.positions[i, 2])
            distance = float(distances[i])
            x_max = analytic_reach(distance, target_height, hub, mirror_height)
            rows.append(
                {
                    "heliostat": group.names[i],
                    "row_index": i,
                    "target": target_name,
                    "distance_m": distance,
                    "target_height_m": target_height,
                    "hub_height_m": hub,
                    "analytic_x_max_m": x_max,
                }
            )

    with open(output_dir / "field_wide_reach.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    values = [r["analytic_x_max_m"] for r in rows]
    values_sorted = sorted(values)

    def pct(p: float) -> float:
        return values_sorted[int(p * (len(values_sorted) - 1))]

    print(f"Stage 1: analytic x_max over {len(rows)} heliostat-target combinations "
          f"({group.number_of_heliostats} heliostats x {len(TARGETS)} targets)")
    print(f"  min    : {min(values):.1f} m")
    print(f"  median : {pct(0.50):.1f} m")
    print(f"  p90    : {pct(0.90):.1f} m")
    print(f"  p99    : {pct(0.99):.1f} m")
    print(f"  max    : {max(values):.1f} m   <-- the single global bound")

    worst_overall = max(rows, key=lambda r: r["analytic_x_max_m"])
    print(f"  worst case: {worst_overall['heliostat']} vs {TARGET_LABEL[worst_overall['target']]}, "
          f"{worst_overall['distance_m']:.0f} m away, x_max = "
          f"{worst_overall['analytic_x_max_m']:.1f} m\n")

    print("  by target:")
    for target_name in TARGETS:
        target_values = sorted(r["analytic_x_max_m"] for r in rows if r["target"] == target_name)
        worst_t = max(
            (r for r in rows if r["target"] == target_name),
            key=lambda r: r["analytic_x_max_m"],
        )
        print(f"    {TARGET_LABEL[target_name]:28s} median {statistics.median(target_values):5.1f} m  "
              f"max {max(target_values):5.1f} m  (worst: {worst_t['heliostat']}, "
              f"{worst_t['distance_m']:.0f} m away)")
    print()

    _plot_stage1(rows, output_dir)

    # ============================================================ STAGE 2: raytraced check
    print(f"Stage 2: raytraced validation on a stratified sample (target validate-n = "
          f"{args.validate_n})")
    azimuth, elevation = SUN_POSITIONS[args.sun_index]
    incident = sun_incident_direction(azimuth, elevation, device)

    validation_records: list[dict] = []
    violations: list[dict] = []

    for target_name in TARGETS:
        target_index = scenario.solar_tower.target_name_to_index[target_name]
        target_centre = target_centres[target_name]
        target_height = float(target_centre[2])
        field_surfaces = pose_whole_field(group, target_centre, incident, device)

        by_target = [r for r in rows if r["target"] == target_name]
        by_target.sort(key=lambda r: r["distance_m"])

        # Stratified sample: spread across deciles of distance, plus the single worst
        # analytic case for this target explicitly (it is the most likely to reveal a
        # second-row effect, per the earlier row_by_row finding).
        n_per_target = max(4, args.validate_n // len(TARGETS))
        n_deciles = min(10, n_per_target)
        picks: set[int] = set()
        for k in range(n_deciles):
            idx = min(len(by_target) - 1, int(k / max(1, n_deciles - 1) * (len(by_target) - 1)))
            picks.add(idx)
        worst = max(range(len(by_target)), key=lambda i: by_target[i]["analytic_x_max_m"])
        picks.add(worst)
        while len(picks) < n_per_target:
            picks.add(torch.randint(0, len(by_target), (1,)).item())

        print(f"  {TARGET_LABEL[target_name]}: validating {len(picks)} heliostats")

        for idx in sorted(picks):
            record = by_target[idx]
            heliostat_index = record["row_index"]
            x_max = record["analytic_x_max_m"]

            candidates = candidates_in_corridor(
                group.positions, heliostat_index, target_centre,
                args.search_radius, args.lateral,
            )
            for candidate_index, along, lateral in candidates:
                fraction, on_target = blocked_fraction(
                    scenario, group, heliostat_index, [candidate_index], target_index,
                    target_centre, incident, device, field_surfaces,
                )
                if on_target < 1e-6:
                    continue
                if fraction < 1e-4:
                    continue
                exceeds = along > x_max
                validation_records.append(
                    {
                        "studied": record["heliostat"],
                        "studied_distance_m": record["distance_m"],
                        "target": target_name,
                        "analytic_x_max_m": x_max,
                        "neighbour": group.names[candidate_index],
                        "along_m": along,
                        "lateral_m": lateral,
                        "blocked_fraction": fraction,
                        "exceeds_bound": exceeds,
                    }
                )
                if exceeds:
                    violations.append(validation_records[-1])

    with open(output_dir / "validation.csv", "w", newline="") as handle:
        if validation_records:
            writer = csv.DictWriter(handle, fieldnames=list(validation_records[0].keys()))
            writer.writeheader()
            writer.writerows(validation_records)

    print(f"\n  {len(validation_records)} actual blocking contributions found in the sample")
    if violations:
        print(f"  VIOLATIONS: {len(violations)} exceeded their own analytic bound:")
        for v in violations:
            print(f"    {v['studied']} vs {v['target']}: {v['neighbour']} at {v['along_m']:.1f} m "
                  f"> x_max {v['analytic_x_max_m']:.1f} m, blocks {v['blocked_fraction']*100:.1f}%")
    else:
        print("  0 violations: every measured blocker stayed within its own analytic bound.")

    print(f"\nWrote {output_dir / 'field_wide_reach.csv'}")
    print(f"Wrote {output_dir / 'validation.csv'}")

    print("\n" + "=" * 72)
    print(f"RECOMMENDED single global radius (safe for ANY heliostat in the field): "
          f"{max(values):.1f} m")
    print(f"Better: use analytic_reach(distance, target_height, hub, mirror_height) "
          f"per heliostat.")
    print(f"  p90 across the field is only {pct(0.90):.1f} m: the tail near the field's "
          f"edge is what drives the maximum, not the typical case.")
    print("=" * 72)


def _plot_stage1(rows: list[dict], output_dir: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [r["analytic_x_max_m"] for r in rows]

    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.hist(values, bins=60, color="#c1440e", alpha=0.85)
    for p, style in ((0.90, "--"), (0.99, ":")):
        v = sorted(values)[int(p * (len(values) - 1))]
        axis.axvline(v, color="0.2", ls=style, lw=1.3)
        axis.text(v, axis.get_ylim()[1] * 0.92, f" p{int(p*100)} = {v:.1f} m",
                  fontsize=9, color="0.2")
    axis.axvline(max(values), color="#1f77b4", lw=1.8)
    axis.text(max(values), axis.get_ylim()[1] * 0.98, f" max = {max(values):.1f} m",
              fontsize=9.5, color="#1f77b4", fontweight="bold")
    axis.set_xlabel("analytic reach $x_{max}$ [m]")
    axis.set_ylabel("count (heliostat x target)")
    axis.set_title("Distribution of blocking reach across the whole 1277-heliostat field\n"
                    "3 targets x 1277 heliostats")
    axis.grid(alpha=0.25, linewidth=0.5)
    figure.tight_layout()
    out = output_dir / "field_wide_reach_hist.png"
    figure.savefig(out, dpi=170)
    plt.close(figure)
    print(f"Wrote {out.name}")

    figure, axis = plt.subplots(figsize=(9.0, 6.0))
    colours = {"solar_tower_juelich_lower": "#c1440e",
               "solar_tower_juelich_upper": "#2b6cb0",
               "multi_focus_tower": "#2ca02c"}
    for target_name, colour in colours.items():
        subset = [r for r in rows if r["target"] == target_name]
        axis.scatter(
            [r["distance_m"] for r in subset], [r["analytic_x_max_m"] for r in subset],
            s=6, alpha=0.5, c=colour, label=TARGET_LABEL[target_name],
        )
    axis.set_xlabel("distance to target [m]")
    axis.set_ylabel("analytic reach $x_{max}$ [m]")
    axis.set_title("Blocking reach grows with distance, exactly as the formula predicts")
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(fontsize=9)
    figure.tight_layout()
    out = output_dir / "field_wide_reach_vs_distance.png"
    figure.savefig(out, dpi=170)
    plt.close(figure)
    print(f"Wrote {out.name}")


if __name__ == "__main__":
    main()
