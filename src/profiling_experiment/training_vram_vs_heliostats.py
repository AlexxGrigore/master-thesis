"""Training VRAM vs field size (25×25 surface points), from measured runs + extrapolation.

Measured (both 25×25 surface points, 2-stage kinematics training):
  * 1 heliostat   -> 0.33 GB   (one_hel_train_sizes; flat across #samples)
  * 63 heliostats -> 19.68 GB  (full_63_real_focal_spot_deflectometry)

Training VRAM is linear in field size (the retained autograd graph scales with the number
of active heliostat-instances), giving ~0.31 GB/heliostat, so 1000 heliostats extrapolates
to ~312 GB — far past a single GPU. Pure-data plot; no GPU needed.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import paths  # noqa: E402

# Measured peak GPU memory (GB) at 25×25 surface points.
MEASURED = {1: 0.329, 63: 19.678}
EXTRAP_TO = 1000
A40_GB = 44.0


def main() -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = sorted(MEASURED)
    slope = (MEASURED[ns[-1]] - MEASURED[ns[0]]) / (ns[-1] - ns[0])
    intercept = MEASURED[ns[0]] - slope * ns[0]
    extrap = slope * EXTRAP_TO + intercept

    labels = [f"{n} heliostat" + ("s" if n != 1 else "") for n in ns] + [f"{EXTRAP_TO} heliostats"]
    heights = [MEASURED[n] for n in ns] + [extrap]
    # Calm teal for measured, warm terracotta for the extrapolation.
    measured_c, extrap_c = "#2A9D8F", "#E76F51"
    colors = [measured_c] * len(ns) + [extrap_c]
    notes = ["measured", "measured", "extrapolated"]

    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(8, 5.6))
    x = np.arange(len(heights))
    ax.bar(x, heights, color=colors, width=0.62, edgecolor="white", linewidth=1.2, zorder=3)
    for xi, h, note in zip(x, heights, notes):
        txt = f"{h:.2f} GB" if h < 10 else f"{h:.0f} GB"
        if h > A40_GB:
            txt += f"  (~{h / A40_GB:.0f}× A40)"
        ax.annotate(f"{txt}\n[{note}]", (xi, h), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=10.5, fontweight="bold",
                    color="#333333")

    ax.axhline(A40_GB, ls="--", color="#8d99ae", lw=1.6, zorder=2,
               label=f"single A40 = {A40_GB:.0f} GB")
    ax.set_yscale("log")
    ax.set_ylim(0.1, max(heights) * 3.0)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Peak training VRAM (GB, log scale)")
    ax.set_title("Total training memory vs field size (25×25 surface points)\n"
                 f"~{slope:.2f} GB per heliostat → {EXTRAP_TO} heliostats ≈ {extrap:.0f} GB",
                 fontsize=12.5)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.25, which="both", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    out = paths.output_dir() / "training_vram_vs_heliostats.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"slope={slope:.3f} GB/hel  extrap({EXTRAP_TO})={extrap:.0f} GB  -> {out}")


if __name__ == "__main__":
    main()
