"""Field-wide blocking screen: which heliostats lose beam to their neighbours, per target?

Blocking is a geometric property of the field layout and the aim direction, so this needs no
calibration data and no kinematic perturbation. Each heliostat is aimed at a target centre
with its NOMINAL (ideal) parameters, its neighbours are aimed at the same target, and the
raytracer reports what fraction of the reflected rays never reach the target because a
neighbour is in the way.

Sun dependence. The reflected beam always travels from the heliostat toward the target
regardless of where the sun is, so which neighbours sit in the path is essentially
sun-independent. What the sun does change is the ATTITUDE of those neighbours, hence the
silhouette they present. The screen therefore evaluates several representative sun positions
drawn from the benchmark's own distribution and reports the median and the maximum.

Neighbours are posed AIMED, not stowed. A stowed neighbour is nearly edge-on to a shallow
beam and blocks almost nothing, so aimed is both the worst case and the realistic one during
a calibration campaign. Which target they aim at barely matters: from a few hundred metres
the three targets and the receiver lie within about a degree of each other.

Only neighbours within `--neighbour-radius` metres are considered. That is not an
approximation for convenience: the census in DESIGN.md measured that only the immediately
adjacent row, inside roughly 12 m, contributes anything at all, because a beam climbing at
9 degrees clears a 2.8 m mirror after about 20 m.

Outputs, under `outputs/new_mapping_function/blocking_study/field_map/`:
    field_blocking_<target>.png    one field view per target, colour = fraction blocked
    field_blocking.csv             per heliostat and target, median and max over suns
    field_blocking.json            same, plus the sun positions used

Usage
-----
    python field_blocking_map.py
    python field_blocking_map.py --suns 5 --rays 10 --neighbour-radius 25
"""

from __future__ import annotations

import argparse
import csv
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

from artist.geometry import coordinates  # noqa: E402
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer  # noqa: E402
from artist.scenario.scenario import Scenario  # noqa: E402
from artist.util import set_logger_config  # noqa: E402

from brute_blocking import exact_blocking  # noqa: E402

log = logging.getLogger(__name__)

TARGETS = ("solar_tower_juelich_upper", "solar_tower_juelich_lower", "multi_focus_tower")
TARGET_LABEL = {
    "solar_tower_juelich_upper": "Solar tower, upper target",
    "solar_tower_juelich_lower": "Solar tower, lower target",
    "multi_focus_tower": "Multi focus tower",
}

# Representative sun positions, taken as quantiles of the 50-20-20 benchmark's own
# distribution (elevation p10 17, p25 25, median 37, p75 49, p90 57 degrees;
# azimuth p10 -67, median -5, p90 +61, south-oriented convention).
SUN_POSITIONS = (
    (-66.6, 16.9),
    (-41.2, 25.5),
    (-4.5, 37.4),
    (36.0, 49.4),
    (60.5, 57.2),
)

_DAIC_ROOT = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")


def _paths(daic: bool) -> tuple[pathlib.Path, pathlib.Path]:
    """Return (scenario_path, output_dir)."""
    root = _DAIC_ROOT if daic else _here.parents[2]
    return (
        root / "scenarios" / "full_benchmark_ideal" / "ideal_1277.h5",
        root / "outputs" / "new_mapping_function" / "blocking_study" / "field_map",
    )


def sun_incident_direction(
    azimuth_deg: float, elevation_deg: float, device: torch.device
) -> torch.Tensor:
    """Incident ray direction (sun to heliostat) for one sun position, ARTIST convention."""
    enu = coordinates.azimuth_elevation_to_enu(
        torch.tensor([azimuth_deg], device=device),
        torch.tensor([elevation_deg], device=device),
        degree=True,
        device=device,
    )
    position = coordinates.convert_3d_points_to_4d_format(enu, device=device)
    return torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - position[0]


def standard_heliostats(root: pathlib.Path) -> list[str]:
    """The standard evaluation set, read from the standard-results table."""
    path = (
        root
        / "outputs"
        / "new_mapping_function"
        / "standard_results"
        / "standard_results.csv"
    )
    with open(path) as handle:
        return [row["heliostat"] for row in csv.DictReader(handle)]


def neighbour_indices(
    positions: torch.Tensor,
    heliostat_index: int,
    target_centre: torch.Tensor,
    radius_m: float,
) -> list[int]:
    """Indices of heliostats that could intercept the beam to the target.

    Kept if they lie between the heliostat and the target along the beam's horizontal
    direction, within `radius_m`, and within a generous lateral corridor. The raytracer makes
    the final decision; this only bounds the cost.
    """
    origin = positions[heliostat_index, :3]
    beam = target_centre[:3] - origin
    beam_horizontal = beam[:2]
    beam_length = torch.norm(beam_horizontal)
    if beam_length < 1e-6:
        return []
    direction = beam_horizontal / beam_length

    offsets = positions[:, :2] - origin[:2]
    along = offsets @ direction
    lateral = (offsets - along.unsqueeze(1) * direction.unsqueeze(0)).norm(dim=1)

    keep = (along > 0.5) & (along < radius_m) & (lateral < 6.0)
    keep[heliostat_index] = False
    return torch.nonzero(keep, as_tuple=True)[0].tolist()


def pose_whole_field(
    group,
    target_centre: torch.Tensor,
    incident_ray_direction: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Aim every heliostat in the field at one target under one sun; return its surfaces.

    Computed once per (target, sun) and then sliced per studied heliostat, which is the
    difference between 15 field-wide alignments and roughly 950 of them.

    ARTIST warns "no valid motor position combination" for heliostats whose actuators cannot
    reach this aim point. That is expected for a field-wide sweep and harmless here: such a
    heliostat is only ever used as a blocker silhouette, and the studied heliostat is
    separately rejected if its own beam fails to reach the target.
    """
    number_of_heliostats = group.number_of_heliostats
    all_mask = torch.ones(number_of_heliostats, dtype=torch.int32, device=device)
    with torch.no_grad():
        group.activate_heliostats(active_heliostats_mask=all_mask, device=device)
        group.align_surfaces_with_incident_ray_directions(
            aim_points=target_centre.expand(number_of_heliostats, -1),
            incident_ray_directions=incident_ray_direction.expand(
                number_of_heliostats, -1
            ),
            active_heliostats_mask=all_mask,
            device=device,
        )
        return group.active_surface_points.detach().clone()


def blocked_fraction(
    scenario,
    group,
    heliostat_index: int,
    neighbours: list[int],
    target_index: int,
    target_centre: torch.Tensor,
    incident_ray_direction: torch.Tensor,
    device: torch.device,
    field_surfaces: torch.Tensor,
) -> tuple[float, float]:
    """Return (fraction of rays blocked, fraction reaching the target unblocked).

    The blocker set deliberately EXCLUDES the trained heliostat, so no self-exclusion is
    needed and its own mirror can never block itself.
    """
    number_of_heliostats = group.number_of_heliostats
    blockers = field_surfaces[neighbours]

    # Activate only the studied heliostat and aim it at the target.
    mask = torch.zeros(number_of_heliostats, dtype=torch.int32, device=device)
    mask[heliostat_index] = 1
    group.activate_heliostats(active_heliostats_mask=mask, device=device)
    group.align_surfaces_with_incident_ray_directions(
        aim_points=target_centre.unsqueeze(0),
        incident_ray_directions=incident_ray_direction.unsqueeze(0),
        active_heliostats_mask=mask,
        device=device,
    )

    ray_tracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=group,
        blocking_active=False,
        world_size=1,
        rank=0,
        batch_size=1,
        random_seed=7,
    )
    if len(neighbours) > 0:
        ray_tracer.blocking_active = True
        ray_tracer.blocking_heliostat_surfaces_active = blockers

    # ARTIST's LBVH filter silently drops blockers (see brute_blocking.py).
    with exact_blocking():
        _, _, on_target, blocking_factor = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_direction.unsqueeze(0),
            active_heliostats_mask=mask,
            target_area_indices=torch.tensor([target_index], device=device),
            device=device,
        )
    return 1.0 - float(blocking_factor.item()), float(on_target.item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Field-wide blocking screen.")
    parser.add_argument("--daic", action="store_true")
    parser.add_argument("--surface-points", type=int, default=25)
    parser.add_argument("--rays", type=int, default=10)
    parser.add_argument("--neighbour-radius", type=float, default=25.0)
    parser.add_argument(
        "--suns", type=int, default=len(SUN_POSITIONS),
        help="How many of the representative sun positions to evaluate.",
    )
    parser.add_argument(
        "--heliostat-ids", nargs="+", default=None,
        help="Override the heliostat set (default: the standard evaluation set).",
    )
    parser.add_argument(
        "--full-field", action="store_true",
        help="Screen every heliostat in the scenario, not just the standard 63.",
    )
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.WARNING)
    # Expected and very noisy during a field-wide aim sweep; see pose_whole_field.
    logging.getLogger("artist.field.kinematics_rigid_body").setLevel(logging.ERROR)
    log.setLevel(logging.INFO)
    torch.manual_seed(7)

    root = _DAIC_ROOT if args.daic else _here.parents[2]
    scenario_path, output_dir = _paths(args.daic)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    log.info(f"Loading {scenario_path} ...")
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

    wanted = (
        list(group.names) if args.full_field
        else args.heliostat_ids or standard_heliostats(root)
    )
    present = [h for h in wanted if h in group.names]
    missing = [h for h in wanted if h not in group.names]
    if missing:
        log.warning(f"{len(missing)} requested heliostats absent from the scenario: {missing}")
    print(f"Screening {len(present)} heliostats x {len(TARGETS)} targets x {args.suns} suns "
          f"(neighbours within {args.neighbour_radius:.0f} m, aimed)")

    suns = [
        (azimuth, elevation, sun_incident_direction(azimuth, elevation, device))
        for azimuth, elevation in SUN_POSITIONS[: args.suns]
    ]

    results: dict[str, dict] = {}
    for target_name in TARGETS:
        target_index = scenario.solar_tower.target_name_to_index[target_name]
        target_centre = scenario.solar_tower.get_centers_of_target_areas(
            target_area_indices=torch.tensor([target_index], device=device), device=device
        )[0]
        print(f"\n{TARGET_LABEL[target_name]}")

        # One field-wide posing per sun, reused by every studied heliostat.
        posed = [
            pose_whole_field(group, target_centre, incident, device)
            for _, _, incident in suns
        ]

        for heliostat_id in present:
            index = group.names.index(heliostat_id)
            neighbours = neighbour_indices(
                group.positions, index, target_centre, args.neighbour_radius
            )
            per_sun = []
            skipped = 0
            for (_, _, incident), field_surfaces in zip(suns, posed):
                fraction, on_target = blocked_fraction(
                    scenario, group, index, neighbours, target_index,
                    target_centre, incident, device, field_surfaces,
                )
                # A beam that never reaches the target cannot be meaningfully "unblocked".
                if on_target < 1e-6:
                    skipped += 1
                    continue
                per_sun.append(fraction)
            entry = results.setdefault(
                heliostat_id,
                {
                    "heliostat": heliostat_id,
                    "east_m": float(group.positions[index, 0]),
                    "north_m": float(group.positions[index, 1]),
                },
            )
            entry[f"{target_name}__n_neighbours"] = len(neighbours)
            entry[f"{target_name}__n_suns_used"] = len(per_sun)
            entry[f"{target_name}__median"] = (
                statistics.median(per_sun) if per_sun else float("nan")
            )
            entry[f"{target_name}__max"] = max(per_sun) if per_sun else float("nan")
            if skipped:
                log.warning(f"  {heliostat_id}: {skipped} sun(s) gave no on-target beam")

        medians = [
            results[h][f"{target_name}__median"]
            for h in present
            if results[h][f"{target_name}__median"] == results[h][f"{target_name}__median"]
        ]
        affected = sum(1 for v in medians if v > 0.01)
        ranked = sorted(
            ((results[h][f"{target_name}__median"], h) for h in present
             if results[h][f"{target_name}__median"] == results[h][f"{target_name}__median"]),
            reverse=True,
        )
        print(f"  above 1% blocked : {affected} of {len(medians)} heliostats")
        print(f"  median / worst   : {statistics.median(medians) * 100:.2f}% / "
              f"{max(medians) * 100:.2f}%")
        print("  most affected    : " + ", ".join(
            f"{name} {value * 100:.1f}%" for value, name in ranked[:6]))

    # ---------------------------------------------------------------- outputs
    rows = [results[h] for h in present]
    fieldnames = ["heliostat", "east_m", "north_m"] + [
        f"{t}__{suffix}"
        for t in TARGETS
        for suffix in ("median", "max", "n_neighbours", "n_suns_used")
    ]
    with open(output_dir / "field_blocking.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "field_blocking.json").write_text(
        json.dumps(
            {
                "scenario": str(scenario_path),
                "sun_positions_azimuth_elevation_deg": [
                    [a, e] for a, e, _ in suns
                ],
                "neighbour_radius_m": args.neighbour_radius,
                "rays_per_surface_point": args.rays,
                "surface_points_per_facet": args.surface_points,
                "neighbour_pose": "aimed at the same target",
                "heliostats": rows,
            },
            indent=2,
        )
    )
    print(f"\nWrote {output_dir / 'field_blocking.csv'}")

    _plot(rows, group, scenario, output_dir, device)


def _points_per_data_unit(figure, axis) -> float:
    """Exact points-per-data-unit along x, after the axes are drawn and limits fixed.

    Marker sizes in matplotlib are areas in POINTS^2, a fixed physical size, while the field
    is laid out in METRES. The two only line up if this conversion is measured from the
    actual rendered axes box, not guessed from the figure size — guessing is exactly what
    produced the overlapping circles in the first version of this plot (real minimum
    heliostat spacing is 4.03 m, and the marker diameters were picked without checking that).
    """
    figure.canvas.draw()
    bbox = axis.get_window_extent()
    xmin, xmax = axis.get_xlim()
    pixels_per_unit = bbox.width / (xmax - xmin)
    return pixels_per_unit * 72.0 / figure.dpi


def _plot(rows, group, scenario, output_dir: pathlib.Path, device) -> None:
    """One field view per target: all heliostats grey, screened ones coloured by blocking.

    Marker sizes are derived from the real minimum heliostat spacing (see
    `_points_per_data_unit`) so that circles never touch, and the legend is placed entirely
    below the axes so it can never overlap plotted heliostats regardless of field shape.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    all_positions = group.positions[:, :2]
    all_east = all_positions[:, 0].tolist()
    all_north = all_positions[:, 1].tolist()

    # Real minimum spacing between any two heliostats in the field: this is the hard
    # constraint the marker sizes must respect.
    pairwise = torch.cdist(all_positions, all_positions)
    pairwise.fill_diagonal_(float("inf"))
    min_spacing_m = float(pairwise.min())

    values = [
        r[f"{t}__median"] * 100
        for t in TARGETS
        for r in rows
        if r[f"{t}__median"] == r[f"{t}__median"]
    ]
    vmax = max(values) if values else 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)

    # Axis limits must cover the target too, not just the heliostats: the tower sits at
    # north ~ -3 m, well outside the field's own north range (24.4 to 243.5 m), so it was
    # being silently clipped out of frame in an earlier version of this plot.
    target_centres = []
    for target_name in TARGETS:
        target_index = scenario.solar_tower.target_name_to_index[target_name]
        target_centres.append(
            scenario.solar_tower.get_centers_of_target_areas(
                target_area_indices=torch.tensor([target_index], device=device),
                device=device,
            )[0]
        )
    east_extent = all_east + [float(c[0]) for c in target_centres]
    north_extent = all_north + [float(c[1]) for c in target_centres]
    margin = 0.06 * max(
        max(east_extent) - min(east_extent), max(north_extent) - min(north_extent)
    )
    xlim = (min(east_extent) - margin, max(east_extent) + margin)
    ylim = (min(north_extent) - margin, max(north_extent) + margin)

    for target_name, centre in zip(TARGETS, target_centres):
        figure, axis = plt.subplots(figsize=(9.5, 9.5))
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_aspect("equal")

        # Convert the real 4.03 m minimum spacing into a marker size that leaves a visible
        # gap: diameter = 70% of that spacing, so touching markers are structurally
        # impossible even for two heliostats standing at the tightest real distance.
        points_per_m = _points_per_data_unit(figure, axis)
        blocked_diameter_pt = 0.70 * min_spacing_m * points_per_m
        clean_diameter_pt = 0.55 * min_spacing_m * points_per_m
        background_diameter_pt = 0.20 * min_spacing_m * points_per_m
        s_blocked = blocked_diameter_pt ** 2
        s_clean = clean_diameter_pt ** 2
        s_background = background_diameter_pt ** 2

        axis.scatter(all_east, all_north, s=s_background, c="0.87", linewidths=0, zorder=1,
                     label=f"field ({group.number_of_heliostats} heliostats)")

        east = [r["east_m"] for r in rows]
        north = [r["north_m"] for r in rows]
        blocked = [r[f"{target_name}__median"] * 100 for r in rows]
        clean = [(e, n) for e, n, b in zip(east, north, blocked) if b <= 1.0 and b == b]
        if clean:
            axis.scatter([e for e, _ in clean], [n for _, n in clean], s=s_clean,
                         facecolors="none", edgecolors="0.45", linewidths=0.8, zorder=2,
                         label="screened, below 1% blocked")
        hit = [(e, n, b) for e, n, b in zip(east, north, blocked) if b > 1.0 and b == b]
        if hit:
            scatter = axis.scatter([e for e, _, _ in hit], [n for _, n, _ in hit],
                                   c=[b for _, _, b in hit], s=s_blocked, cmap="inferno_r",
                                   norm=norm, edgecolors="0.2", linewidths=0.5, zorder=3,
                                   label="screened, blocked (colour = amount)")
            bar = figure.colorbar(scatter, ax=axis, pad=0.02, shrink=0.85)
            bar.set_label("reflected rays blocked by neighbours [%]")

        axis.scatter([float(centre[0])], [float(centre[1])], marker="*", s=380,
                     c="#1f77b4", edgecolors="white", linewidths=1.0, zorder=5,
                     label="aim target")

        n_affected = len(hit)
        finite = [b for b in blocked if b == b]
        n_total = len(finite)
        target_height = float(centre[2])
        # Target height is the explanatory variable: a higher target means a steeper beam,
        # which clears the neighbouring row sooner and so suffers less blocking.
        axis.set_title(
            f"{TARGET_LABEL[target_name]}   (target {target_height:.1f} m above the field)\n"
            f"{n_affected} of {n_total} screened heliostats lose more than 1% of the beam"
            f"      median {statistics.median(finite):.2f}%, worst {max(finite):.2f}%\n"
            f"colour scale is shared across all three targets",
            fontsize=10.5,
        )
        axis.set_xlabel("east [m]")
        axis.set_ylabel("north [m]")
        axis.grid(alpha=0.25, linewidth=0.5)
        # Legend sits entirely BELOW the axes, never over plotted heliostats, regardless of
        # the field's shape (the earlier "upper left" placement overlapped real data there).
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=8.3,
                    framealpha=0.95)
        out = output_dir / f"field_blocking_{target_name}.png"
        figure.savefig(out, dpi=170, bbox_inches="tight")
        plt.close(figure)
        print(f"Wrote {out.name}")


if __name__ == "__main__":
    main()
