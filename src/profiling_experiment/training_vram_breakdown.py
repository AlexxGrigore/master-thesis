"""Training VRAM split into ray-tracing vs the rest, per field size (25×25 surface pts).

The split comes from the measured full_63 run: between epochs it sat at ~3.0 GB (resident:
stored target-flux images + scenario + optimizer state), and peaked at 19.68 GB during
forward+backward. The difference, ~16.7 GB (~85%), is the ray-tracer's autograd graph.
Both parts scale ~linearly with field size, so the same split is applied to 1 and 1000
heliostats.

Shown as grouped bars (not stacked) on a log axis: stacked bars on a log scale don't add
up visually, so grouping is the honest way to compare the two components.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import paths  # noqa: E402

# Measured peak training VRAM (GB) at 25×25 surface points.
TOTAL = {1: 0.329, 63: 19.678}
# Ray-tracer VRAM (single forward pass, 25×25) — the SAME numbers as the ray-tracer plot.
RAYTRACE = {1: 0.018, 63: 0.303}
A40_GB = 44.0


def main() -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = [1, 63]
    raytrace = [RAYTRACE[n] for n in fields]
    totals = [TOTAL[n] for n in fields]
    rest = [t - r for t, r in zip(totals, raytrace)]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    x = np.arange(len(fields)); w = 0.5
    # Stacked: ray tracing on the bottom, the training overhead on top -> bar = total.
    ax.bar(x, raytrace, w, color="#2c6fbb", label="Ray tracing (single forward pass)")
    ax.bar(x, rest, w, bottom=raytrace, color="#e08214",
           label="The rest (100 samples × autograd graph + stored data + optimizer)")

    fmt = lambda v: (f"{v:.3f}" if v < 0.1 else f"{v:.2f}" if v < 10 else f"{v:.1f}")
    floor = 0.008
    for xi, tot, rtv, rsv in zip(x, totals, raytrace, rest):
        ax.annotate(f"total {fmt(tot)} GB", (xi, tot), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=10, fontweight="bold")
        # Segment labels at the geometric midpoint of each segment (log-friendly).
        ax.annotate(f"RT {fmt(rtv)} GB", (xi, (floor * rtv) ** 0.5), ha="center",
                    va="center", fontsize=8, color="white", fontweight="bold")
        ax.annotate(f"rest {fmt(rsv)} GB", (xi, (rtv * tot) ** 0.5), ha="center",
                    va="center", fontsize=8, color="black", fontweight="bold")

    ax.axhline(A40_GB, ls="--", color="gray", lw=1.4, label=f"single A40 = {A40_GB:.0f} GB")
    ax.set_yscale("log")
    ax.set_ylim(0.008, A40_GB * 2.2)
    ax.set_xticks(x)
    ax.set_xticklabels(["1 heliostat", "63 heliostats"])
    ax.set_ylabel("Peak training VRAM (GB, log scale)")
    ax.set_title("Training VRAM: ray tracing vs the rest (25×25 surface points)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()

    out = paths.output_dir() / "training_vram_breakdown.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    for n, rtv, rsv in zip(fields, raytrace, rest):
        print(f"{n:>4} hel: total={TOTAL[n]:.3f} GB  raytrace={rtv:.3f}  rest={rsv:.3f}")
    print("->", out)


if __name__ == "__main__":
    main()
