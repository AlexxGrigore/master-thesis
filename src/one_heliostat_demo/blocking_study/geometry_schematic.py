"""Plan-view geometry schematic for a blocking_validation showcase heliostat.

Two panels, side by side: full-field plan view (context, with a box marking
the zoom region) and a zoomed plan view showing the studied heliostat, its
candidate blockers, the target, and the sun's compass bearing (drawn as a
yellow sun icon, not a text label, to keep the plot uncluttered). Answers
"where is this heliostat relative to its neighbours, the target and the sun"
at a glance.

Blocker names come from `identify_blockers`'s geometric cone test for the
given `--sun-sample` (same method used everywhere else in this study, e.g.
`mirror_target_diagnostics.py`) -- no prior pose-sweep run required, just
`--heliostat-id` (+ optionally `--sun-sample`, default the study's usual
worst-blocking sample "0011").

Usage
-----
    python geometry_schematic.py --heliostat-id BH58
    python geometry_schematic.py --heliostat-id BA72 --sun-sample 0011
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

_here = pathlib.Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from validate_blocking_flux import (  # noqa: E402
    configure,
    identify_blockers,
    load_context,
    load_train_sun_positions,
)


def draw_sun(ax, pos: np.ndarray, radius: float, zorder: int = 8) -> None:
    """A small yellow sun icon (glow halo + disc + rays) at `pos`, in data coords."""
    for i in range(8):
        ang = i * np.pi / 4
        dx, dy = np.cos(ang), np.sin(ang)
        ax.plot([pos[0] + 0.55 * radius * dx, pos[0] + 1.25 * radius * dx],
                 [pos[1] + 0.55 * radius * dy, pos[1] + 1.25 * radius * dy],
                 color="gold", lw=1.8, zorder=zorder, solid_capstyle="round")
    ax.scatter(*pos, s=(radius * 620) ** 1.15, c="gold", edgecolor="orange", linewidths=1.2,
               zorder=zorder, alpha=0.35)
    ax.scatter(*pos, s=(radius * 260) ** 1.15, c="gold", edgecolor="darkorange", linewidths=1.5,
               zorder=zorder + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heliostat-id", default="BH58")
    parser.add_argument("--sun-sample", default="0011")
    parser.add_argument("--zoom-plan-pad-m", type=float, default=12.0,
                        help="padding around heliostat+blockers in the zoomed plan view")
    args = parser.parse_args()

    configure(args.heliostat_id)
    device = torch.device("cpu")
    scenario, hg, hel_idx, target_index, aim_center, hel_dist_m = load_context(
        device, surface_points_per_facet=10, rays_per_surface_point=1
    )

    samples = load_train_sun_positions()
    sample = next(s for s in samples if s["sample_id"] == args.sun_sample)
    sun_az_deg = sample["sun_azimuth_deg"]
    sun_el_deg = sample["sun_elevation_deg"]
    sun = torch.tensor(sample["incident_ray_direction"] + [0.0], dtype=torch.float, device=device)
    blocker_names = [
        str(hg.names[i])
        for i in identify_blockers(scenario, hg, hel_idx, sun, target_index, aim_center, device)
    ]
    print(f"blockers: {blocker_names}")

    names = [str(n) for n in hg.names]
    positions = hg.positions.detach().cpu().numpy()[:, :3]
    blocker_idx = [names.index(n) for n in blocker_names]
    hel = positions[hel_idx]
    blockers = positions[blocker_idx]
    target = aim_center[:3].numpy()

    az_rad = np.radians(sun_az_deg)
    sun_dir_en = np.array([np.sin(az_rad), np.cos(az_rad)])  # (east, north), 0=N/90=E bearing

    horiz_dist = float(np.linalg.norm(target[:2] - hel[:2]))

    labels = {n: str(i + 1) for i, n in enumerate(blocker_names)}
    blocker_legend = "\n".join(f"{labels[n]}: {n}" for n in blocker_names)

    # ---------------------------------------------------------------- figure
    # Extra figure width (vs. the data axes) reserved so the legend and notes
    # box can sit fully OUTSIDE axA2, to its right -- guarantees they never
    # overlap plotted data, at the cost of some blank margin on the right.
    fig, (axA1, axA2) = plt.subplots(1, 2, figsize=(18.5, 7.2), gridspec_kw={"width_ratios": [1, 1.15]})

    # ---- A1: full field context ----
    axA1.scatter(positions[:, 0], positions[:, 1], s=3, c="lightgray", zorder=1)
    axA1.scatter(target[0], target[1], marker="^", s=150, c="black", zorder=5)
    axA1.scatter(hel[0], hel[1], marker="*", s=260, c="gold", edgecolor="k", zorder=6)
    axA1.scatter(blockers[:, 0], blockers[:, 1], marker="o", s=45, c="crimson", edgecolor="k", zorder=6)
    axA1.plot([hel[0], target[0]], [hel[1], target[1]], "--", c="navy", lw=1.2, zorder=4)
    pad0 = args.zoom_plan_pad_m
    zx0, zx1 = min(hel[0], blockers[:, 0].min()) - pad0, max(hel[0], blockers[:, 0].max()) + pad0
    zy0, zy1 = min(hel[1], blockers[:, 1].min()) - pad0, max(hel[1], blockers[:, 1].max()) + pad0
    axA1.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, edgecolor="black", lw=1.5, zorder=7))
    axA1.set_xlabel("east [m]"); axA1.set_ylabel("north [m]")
    axA1.set_title(f"Plan view -- full field ({hg.number_of_heliostats} heliostats)", fontsize=11)
    axA1.set_aspect("equal"); axA1.grid(alpha=0.3)

    # ---- A2: zoomed plan ----
    axA2.scatter(positions[:, 0], positions[:, 1], s=8, c="lightgray", zorder=1, label="field (context)")
    axA2.scatter(target[0], target[1], marker="^", s=250, c="black", zorder=5, label="target")
    axA2.scatter(hel[0], hel[1], marker="*", s=550, c="gold", edgecolor="k", zorder=6,
                  label=f"{args.heliostat_id} (studied)")
    axA2.scatter(blockers[:, 0], blockers[:, 1], marker="o", s=160, c="crimson", edgecolor="k", zorder=6,
                  label="blockers")
    for n, b in zip(blocker_names, blockers):
        axA2.annotate(labels[n], (b[0], b[1]), ha="center", va="center", fontsize=9,
                       color="white", fontweight="bold", zorder=7)
    axA2.annotate(args.heliostat_id, (hel[0], hel[1]), textcoords="offset points", xytext=(10, -16),
                   fontsize=10, fontweight="bold", va="top", zorder=9)
    axA2.plot([hel[0], target[0]], [hel[1], target[1]], "--", c="navy", lw=1.8, zorder=4,
               label=f"reflected beam -> target ({horiz_dist:.0f} m)")

    # Sun icon: placed just outside the heliostat/blocker cluster, along the
    # real compass bearing to the sun, so it stays a genuine (if compressed)
    # direction indicator rather than a fixed decoration.
    pad_diag = np.hypot(zx1 - zx0, zy1 - zy0)
    icon_margin = 0.22 * pad_diag
    sun_pos = np.array([(zx0 + zx1) / 2, (zy0 + zy1) / 2]) + 0.66 * pad_diag * sun_dir_en
    draw_sun(axA2, sun_pos, radius=0.04 * pad_diag)
    axA2.plot([sun_pos[0], hel[0]], [sun_pos[1], hel[1]], ":", c="darkorange", lw=1.3, alpha=0.7, zorder=3)
    axA2.set_xlim(min(zx0, sun_pos[0] - icon_margin), max(zx1, sun_pos[0] + icon_margin))
    axA2.set_ylim(min(zy0, sun_pos[1] - icon_margin), max(zy1, sun_pos[1] + icon_margin))

    # Legend and notes box both live OUTSIDE the data axes (bbox_to_anchor
    # x > 1), stacked top/bottom in the reserved right-hand margin -- they
    # can never overlap a data point because they're not drawn over the data
    # area at all.
    axA2.set_xlabel("east [m]"); axA2.set_ylabel("north [m]")
    axA2.set_title(f"Plan view -- zoom on {args.heliostat_id} + blockers", fontsize=11)
    axA2.set_aspect("equal"); axA2.grid(alpha=0.3)
    axA2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, framealpha=0.9,
                labelspacing=0.9, handletextpad=0.6, borderaxespad=0)

    notes = f"Sun\naz {sun_az_deg:.0f} deg, el {sun_el_deg:.0f} deg\n\nBlockers\n{blocker_legend}"
    axA2.text(1.02, 0.0, notes, transform=axA2.transAxes, fontsize=9.5, va="bottom", ha="left",
               bbox=dict(boxstyle="round", facecolor="white", edgecolor="crimson", alpha=0.92))

    fig.suptitle(f"{args.heliostat_id} blocking geometry: sun, studied heliostat, blockers, target", fontsize=15)
    fig.tight_layout(rect=(0, 0.03, 0.86, 0.94))
    out_dir = (
        _here.parents[2] / "outputs" / "new_mapping_function" / "blocking_study"
        / "blocking_validation" / args.heliostat_id / "plots"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "geometry_schematic.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
