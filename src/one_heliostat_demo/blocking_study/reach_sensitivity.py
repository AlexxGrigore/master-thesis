"""Sensitivity illustration: what does each variable in x_max actually do?

Three small panels, each varying ONE variable while holding the other two at a fixed,
representative field value (D = 200 m, H = the lower target at 35.9 m, h = 1.6 m,
a = 2.56 m, whichever is not being swept). This is the complement to
`reach_schematic.py` (which shows the geometry of a single case) and
`field_wide_reach_vs_distance.png` (which shows the real field's scatter): here every
line is the pure formula, so the causal direction of each term is visible in isolation.

Usage
-----
    python reach_sensitivity.py
"""

from __future__ import annotations

import pathlib
import sys

_here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from blocking_reach import analytic_reach

# Representative field values, reused as the "held fixed" baseline in each panel.
D_REF, H_REF, H_LOWER, H_UPPER, H_MFT, H_REF_LABEL = 200.0, 35.9, 35.9, 43.1, 52.0, "lower target"
H_HUB, A_MIRROR = 1.6, 2.56

BEAM = "#c1440e"
BLUE = "#2b6cb0"
GREEN = "#2ca02c"


def main() -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    figure.suptitle(
        r"What each variable in $x_{max} = \dfrac{a\,D}{H - h + a/2} + a$ actually does",
        fontsize=14.5, y=1.04,
    )

    # ------------------------------------------------------------ panel 1: vs distance D
    axis = axes[0]
    D = np.linspace(20, 300, 200)
    for H, label, colour in (
        (H_LOWER, "lower target (35.9 m)", "#c1440e"),
        (H_UPPER, "upper target (43.1 m)", "#2b6cb0"),
        (H_MFT, "multi focus tower (52.0 m)", "#2ca02c"),
    ):
        x = [analytic_reach(d, H, H_HUB, A_MIRROR) for d in D]
        axis.plot(D, x, color=colour, lw=2.2, label=label)
    axis.scatter([218], [analytic_reach(218, H_LOWER, 1.53, A_MIRROR)], s=70, zorder=5,
                 facecolors="white", edgecolors="#c1440e", linewidths=1.8)
    axis.annotate("BE25\n218 m -> 18.2 m", (218, analytic_reach(218, H_LOWER, 1.53, A_MIRROR)),
                  textcoords="offset points", xytext=(-70, -8), fontsize=8.7, color="#c1440e")
    axis.set_xlabel("$D$  distance to target [m]")
    axis.set_ylabel("$x_{max}$  reach [m]")
    axis.set_title("Farther heliostat $\\Rightarrow$ shallower beam\n$\\Rightarrow$ more reach "
                    "(linear in $D$)", fontsize=11)
    axis.legend(fontsize=8, loc="upper left")
    axis.grid(alpha=0.25, linewidth=0.5)

    # ------------------------------------------------------------ panel 2: vs target height H
    axis = axes[1]
    H = np.linspace(30, 100, 200)
    x = [analytic_reach(D_REF, h, H_HUB, A_MIRROR) for h in H]
    axis.plot(H, x, color="#7a3ea1", lw=2.4)
    for hh, label, colour in (
        (H_LOWER, "lower", "#c1440e"), (H_UPPER, "upper", "#2b6cb0"), (H_MFT, "mft", "#2ca02c"),
    ):
        val = analytic_reach(D_REF, hh, H_HUB, A_MIRROR)
        axis.scatter([hh], [val], s=60, color=colour, zorder=5)
        axis.annotate(label, (hh, val), textcoords="offset points", xytext=(6, 6),
                      fontsize=9, color=colour)
    axis.set_xlabel("$H$  target height above the field [m]")
    axis.set_ylabel("$x_{max}$  reach [m]")
    axis.set_title(f"Higher target $\\Rightarrow$ steeper beam\n$\\Rightarrow$ less reach   "
                    f"(at fixed $D$={D_REF:.0f} m)", fontsize=11)
    axis.grid(alpha=0.25, linewidth=0.5)

    # ------------------------------------------------------------ panel 3: vs mirror height a
    axis = axes[2]
    a_range = np.linspace(1.0, 4.5, 200)
    x = [analytic_reach(D_REF, H_REF, H_HUB, a) for a in a_range]
    axis.plot(a_range, x, color="#c17c1c", lw=2.4)
    a_actual = 2.56
    val = analytic_reach(D_REF, H_REF, H_HUB, a_actual)
    axis.scatter([a_actual], [val], s=70, color="#c17c1c", zorder=5, edgecolors="0.2")
    axis.annotate(f"real PAINT mirror\na = {a_actual} m", (a_actual, val),
                  textcoords="offset points", xytext=(8, -18), fontsize=9, color="#c17c1c")
    axis.set_xlabel("$a$  mirror height [m]")
    axis.set_ylabel("$x_{max}$  reach [m]")
    axis.set_title(f"Taller mirror $\\Rightarrow$ needs a bigger climb\n$\\Rightarrow$ more "
                    f"reach   (at fixed $D$={D_REF:.0f} m, $H$=35.9 m)", fontsize=11)
    axis.grid(alpha=0.25, linewidth=0.5)

    figure.tight_layout(rect=[0, 0, 1, 0.94])
    output = (
        _here.parents[2]
        / "outputs" / "new_mapping_function" / "blocking_study" / "reach"
        / "reach_sensitivity.png"
    )
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
