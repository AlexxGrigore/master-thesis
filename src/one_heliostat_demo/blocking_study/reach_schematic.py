"""Draw the blocking-reach geometry: what every symbol in x_max means.

Two panels:

  Top     the whole situation, not to scale, so sun, heliostat, blocking neighbour and target
          all fit in one view. This is where D, H, h and a are defined.
  Bottom  the near zone at TRUE scale, with the real row spacing, which is where the argument
          lives: the beam leaves the bottom edge of the mirror and has to climb one mirror
          height `a` before it passes over the top of the next row.

Usage
-----
    python reach_schematic.py
    python reach_schematic.py --distance 218 --target-height 35.9 --row-spacing 10.2
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

_here = pathlib.Path(__file__).resolve().parent

BEAM = "#c1440e"      # reflected beam
SUNRAY = "#e8a33d"    # incoming sunlight
MIRROR = "#2b6cb0"    # studied heliostat
BLOCKER = "#6f6f6f"   # neighbour
TARGET = "#2ca02c"
DIM = "#3a3a3a"       # dimension lines


def _mirror(axis, x, hub, height, tilt_deg, colour, lw=3.6, zorder=6, pedestal=True):
    """Draw a mirror as a tilted segment centred at (x, hub); return its end points."""
    angle = np.deg2rad(tilt_deg)
    half = 0.5 * height
    dx, dz = half * np.sin(angle), half * np.cos(angle)
    low = (x - dx, hub - dz)
    high = (x + dx, hub + dz)
    axis.plot([low[0], high[0]], [low[1], high[1]], color=colour, lw=lw,
              solid_capstyle="round", zorder=zorder)
    if pedestal:
        axis.plot([x, x], [0.0, hub], color="0.6", lw=1.4, zorder=zorder - 1)
    return low, high


def _dim(axis, p0, p1, text, label_at=None, colour=DIM, fontsize=10.5,
         ha="center", va="center"):
    """Double-headed dimension line with a label at an explicit position."""
    axis.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<->", mutation_scale=11,
                                   color=colour, lw=1.2, shrinkA=0, shrinkB=0, zorder=8))
    if label_at is None:
        label_at = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
    axis.text(label_at[0], label_at[1], text, color=colour, fontsize=fontsize,
              ha=ha, va=va, zorder=9,
              bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.90))


def draw(distance_m: float, target_height_m: float, hub_m: float, mirror_m: float,
         row_spacing_m: float, output: pathlib.Path) -> None:
    climb = mirror_m * distance_m / (target_height_m - hub_m + 0.5 * mirror_m)
    x_max = climb + mirror_m

    figure = plt.figure(figsize=(13.0, 11.2))
    grid = figure.add_gridspec(2, 1, height_ratios=[1.0, 0.86], hspace=0.34)

    # ================================================================= TOP PANEL
    axis = figure.add_subplot(grid[0])
    axis.set_title(
        "What each symbol means:   "
        r"$x_{max} = \dfrac{a\,D}{H - h + a/2} \; + \; a$",
        fontsize=15, pad=18,
    )

    ground = 0.0
    axis.plot([-2.2, 13.2], [ground, ground], color="0.35", lw=1.7, zorder=2)
    axis.fill_between([-2.2, 13.2], -0.95, ground, color="0.94", zorder=1)

    hub_d, mirror_d = 1.15, 1.6
    blocker_x, tower_x, target_z = 3.3, 10.6, 6.5

    low, high = _mirror(axis, 0.0, hub_d, mirror_d, -30.0, MIRROR, lw=4.2)
    axis.text(0.0, -1.25, "studied\nheliostat", color=MIRROR, fontsize=11,
              ha="center", va="top", fontweight="bold")

    b_low, b_high = _mirror(axis, blocker_x, hub_d, mirror_d, 0.0, BLOCKER, lw=4.0)
    axis.text(blocker_x, -1.25, "neighbour\nin front", color=BLOCKER, fontsize=11,
              ha="center", va="top", fontweight="bold")

    # Tower and target.
    axis.plot([tower_x, tower_x], [ground, target_z + 0.8], color="0.55", lw=7.0,
              solid_capstyle="butt", zorder=3)
    axis.plot([tower_x - 0.5, tower_x + 0.5], [target_z, target_z], color=TARGET,
              lw=8.0, solid_capstyle="round", zorder=6)
    axis.text(tower_x, target_z + 1.25, "target", color=TARGET, fontsize=12.5,
              ha="center", fontweight="bold")

    # Sun, upper left, with a few incoming rays onto the mirror.
    sun = (-1.35, 7.35)
    axis.scatter([sun[0]], [sun[1]], s=900, c=SUNRAY, zorder=6, edgecolors="none")
    axis.text(sun[0], sun[1] + 0.85, "sun", color="#9a6206", fontsize=13,
              ha="center", fontweight="bold")
    for shift in (-0.34, 0.0, 0.34):
        axis.annotate(
            "", xy=(shift * 0.4, hub_d + 0.25 + shift * 0.3),
            xytext=(sun[0] + shift * 1.1, sun[1] - 0.62),
            arrowprops=dict(arrowstyle="-|>", color=SUNRAY, lw=1.8, alpha=0.92), zorder=5,
        )
    axis.text(-1.75, 4.3, "incoming\nsunlight", color="#9a6206", fontsize=10.5,
              ha="center")

    # Reflected beam: the low ray that is intercepted, and the rest that gets through.
    axis.annotate("", xy=(b_low[0] - 0.06, b_low[1] + 0.42),
                  xytext=(low[0], low[1]),
                  arrowprops=dict(arrowstyle="-|>", color=BEAM, lw=2.7), zorder=7)
    axis.annotate("", xy=(tower_x - 0.55, target_z),
                  xytext=(high[0], high[1]),
                  arrowprops=dict(arrowstyle="-|>", color=BEAM, lw=2.7), zorder=7)
    axis.text(1.80, 1.46, "lowest ray, hits the neighbour", color=BEAM, fontsize=10.5,
              ha="center", va="center", rotation=21, rotation_mode="anchor")
    axis.text(7.0, 5.15, "beam that reaches the target", color=BEAM, fontsize=11,
              ha="center", rotation=21)

    # Dimensions, each on its own clear line.
    _dim(axis, (0.0, -0.62), (blocker_x, -0.62),
         f"$x_{{max}}$ = {x_max:.1f} m", label_at=(blocker_x / 2, -0.62), colour=BEAM,
         fontsize=11)
    _dim(axis, (0.0, -2.85), (tower_x, -2.85),
         f"$D$  horizontal distance to the target = {distance_m:.0f} m",
         label_at=(tower_x / 2, -2.85))
    _dim(axis, (tower_x + 1.15, ground), (tower_x + 1.15, target_z),
         f"$H$  target height\nabove the field\n= {target_height_m:.1f} m",
         label_at=(tower_x + 1.45, target_z / 2), ha="left")
    _dim(axis, (-0.85, ground), (-0.85, hub_d),
         f"$h$  hub height = {hub_m:.2f} m", label_at=(-1.05, hub_d / 2), ha="right")
    _dim(axis, (blocker_x + 0.75, b_low[1]), (blocker_x + 0.75, b_high[1]),
         f"$a$  mirror height = {mirror_m:.2f} m",
         label_at=(blocker_x + 1.0, hub_d), ha="left")

    axis.set_xlim(-3.0, 14.2)
    axis.set_ylim(-3.7, 8.9)
    axis.axis("off")
    axis.text(0.5, -0.045, "not to scale, so that the whole field fits in one view",
              transform=axis.transAxes, fontsize=10, style="italic",
              color="0.45", ha="center")

    # ============================================================== BOTTOM PANEL
    axis2 = figure.add_subplot(grid[1])
    axis2.set_title(
        r"Near zone at TRUE scale: the beam must climb one mirror height $a$ to clear a row",
        fontsize=13.5, pad=12,
    )

    slope = (target_height_m - hub_m + 0.5 * mirror_m) / distance_m
    z_bottom = hub_m - 0.5 * mirror_m
    z_top = hub_m + 0.5 * mirror_m

    x_view = max(x_max * 1.42, 2.4 * row_spacing_m + 3.0)
    axis2.plot([-3.2, x_view], [0, 0], color="0.35", lw=1.7, zorder=2)
    axis2.fill_between([-3.2, x_view], -1.5, 0, color="0.94", zorder=1)

    _mirror(axis2, 0.0, hub_m, mirror_m, -30.0, MIRROR, lw=4.2)
    axis2.text(0.0, z_top + 0.30, "studied\nheliostat", color=MIRROR, fontsize=10.5,
               ha="center", va="bottom", fontweight="bold")

    # The lowest ray, at the true slope.
    xs = np.linspace(0.0, x_view, 200)
    axis2.plot(xs, z_bottom + slope * xs, color=BEAM, lw=2.7, zorder=6)

    # Real rows at the real spacing. Colour shows whether the beam is still below the top.
    row = row_spacing_m
    while row < x_view - 0.8:
        beam_z = z_bottom + slope * row
        blocks = beam_z < z_top
        _mirror(axis2, row, hub_m, mirror_m, 0.0, BEAM if blocks else "0.66", lw=3.6)
        axis2.text(row, -0.30, f"{row:.1f} m", fontsize=9.2, ha="center", va="top",
                   color="0.35")
        if blocks:
            axis2.text(row, z_top + 0.22, "blocks", color=BEAM, fontsize=10.5,
                       ha="center", va="bottom", fontweight="bold")
        else:
            axis2.text(row, -0.95, "beam passes\nover this row", color="0.45",
                       fontsize=9.4, ha="center", va="top")
        row += row_spacing_m

    # Guides at the mirror's bottom and top edge, and the climb of exactly `a`.
    axis2.plot([-2.6, x_view], [z_top, z_top], color=DIM, lw=0.9, ls=":", zorder=3)
    axis2.plot([-2.6, x_view], [z_bottom, z_bottom], color=DIM, lw=0.9, ls=":", zorder=3)
    _dim(axis2, (-1.75, z_bottom), (-1.75, z_top), f"$a$ = {mirror_m:.2f} m",
         label_at=(-1.95, hub_m), ha="right", fontsize=10)

    axis2.plot([climb, climb], [z_bottom - 0.55, z_top], color=BEAM, lw=1.5, ls="--",
               zorder=5)
    axis2.text(climb, z_bottom - 0.72,
               f"climbed $a$ here\n$aD/(H\\!-\\!h\\!+\\!a/2)$ = {climb:.1f} m",
               color=BEAM, fontsize=9.8, ha="center", va="top")
    axis2.plot([x_max, x_max], [z_bottom - 0.55, z_top + 1.05], color="#1f77b4", lw=1.9,
               ls="--", zorder=5)
    axis2.text(x_max, z_top + 1.15,
               f"$x_{{max}}$ = {x_max:.1f} m\n(plus $a$: positions are centres, not edges)",
               color="#1f77b4", fontsize=9.8, ha="center", va="bottom")

    axis2.annotate("", xy=(x_view * 0.98, z_bottom + slope * x_view * 0.98),
                   xytext=(x_view * 0.86, z_bottom + slope * x_view * 0.86),
                   arrowprops=dict(arrowstyle="-|>", color=BEAM, lw=2.7), zorder=7)
    axis2.text(x_view * 0.90, z_bottom + slope * x_view * 0.90 - 0.62,
               "to the target", color=BEAM, fontsize=10.5, ha="center")

    axis2.set_xlim(-4.6, x_view + 1.2)
    axis2.set_ylim(-2.6, z_top + 2.9)
    axis2.set_xlabel("distance in front of the studied heliostat [m]")
    axis2.set_ylabel("height [m]")
    axis2.set_aspect("equal")
    axis2.grid(alpha=0.20, linewidth=0.5)
    for spine in ("top", "right"):
        axis2.spines[spine].set_visible(False)

    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")
    print(f"  climb term a*D/(H-h+a/2) = {climb:.2f} m")
    print(f"  x_max (centre to centre) = {x_max:.2f} m")
    print(f"  row spacing used         = {row_spacing_m:.2f} m")


def main() -> None:
    parser = argparse.ArgumentParser(description="Schematic of the blocking-reach formula.")
    parser.add_argument("--distance", type=float, default=218.0,
                        help="Horizontal distance to the target D [m] (default: BE25).")
    parser.add_argument("--target-height", type=float, default=35.9,
                        help="Target height H [m] (default: solar tower lower target).")
    parser.add_argument("--hub-height", type=float, default=1.53)
    parser.add_argument("--mirror-height", type=float, default=2.56)
    parser.add_argument("--row-spacing", type=float, default=10.2,
                        help="Spacing between rows along the beam [m] (BE25: 10.2).")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    output = args.output or (
        _here.parents[2]
        / "outputs" / "new_mapping_function" / "blocking_study" / "reach"
        / "reach_formula_schematic.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    draw(args.distance, args.target_height, args.hub_height, args.mirror_height,
         args.row_spacing, output)


if __name__ == "__main__":
    main()
