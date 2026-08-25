"""Bar-chart summary of the BA72 occlusion experiment, using the DIRECTION
(vector/alignment) error -- the pure kinematic normal-vs-normal angle, which
excludes mirror-surface/ray-tracing effects entirely (unlike the centroid
metric, which is influenced by blocking's effect on which rays land where).
This is the metric requested for comparing training results, since it's the
same kind of quantity Stage 1's own ForwardAimLoss optimizes.

Two figures, split by occlusion level (mean blocked ray fraction from each
dataset's own generation report):
  high occlusion: tilt_000 (18.4%), tilt_033 (15.9%)
  low occlusion:  tilt_067 (7.9%),  tilt_100 (0.5%)

Each figure has one group of bars per tilt level: Stage-1-only baseline (that
dataset's own test-split eval), Stage-2 blocking OFF, Stage-2 blocking ON --
mean and median shown as paired bars, with value labels.

Usage
-----
    python plot_ba72_direction_comparison.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_here = pathlib.Path(__file__).resolve().parent
_ROOT = _here.parents[2]
COMPARISON_ROOT = _ROOT / "outputs" / "new_mapping_function" / "ba72_occlusion" / "stage2_blocking_comparison"
OUT_DIR = _ROOT / "outputs" / "new_mapping_function" / "ba72_occlusion" / "plots"

# (tilt label, path fragment, mean blocked % from that dataset's own generation_report.json)
TILT_INFO = {
    "000": (COMPARISON_ROOT, 18.4),
    "033": (COMPARISON_ROOT / "tilt_033", 15.9),
    "067": (COMPARISON_ROOT / "tilt_067", 7.9),
    "100": (COMPARISON_ROOT / "tilt_100", 0.5),
}

COLOR_S1 = "#888888"
COLOR_OFF = "#d95f02"
COLOR_ON = "#1f78b4"


def load_tilt(tilt: str) -> dict:
    root, blocked_pct = TILT_INFO[tilt]
    off = json.load(open(root / "blocking_off" / "arm_results.json"))
    on = json.load(open(root / "blocking_on" / "arm_results.json"))
    return {
        "blocked_pct": blocked_pct,
        "stage1": off["after_stage1"],  # identical model; direction metric doesn't depend on blocking flag
        "off": off["after_stage2"],
        "on": on["after_stage2"],
    }


def make_panel(ax, tilts: list[str], data: dict, metric: str = "direction_mrad") -> None:
    conditions = ["Stage 1\n(baseline)", "Stage 2\nblocking OFF", "Stage 2\nblocking ON"]
    colors = [COLOR_S1, COLOR_OFF, COLOR_ON]
    keys = ["stage1", "off", "on"]

    n_tilt = len(tilts)
    n_cond = len(conditions)
    group_width = 0.8
    bar_width = group_width / (n_cond * 2)
    tilt_gap = 1.3  # spacing between tilt-level groups

    cond_ticks, cond_labels = [], []
    for gi, tilt in enumerate(tilts):
        d = data[tilt]
        group_center = gi * tilt_gap
        for ci, (cond, key, color) in enumerate(zip(conditions, keys, colors)):
            mean_v = d[key][f"{metric}_mean"]
            med_v = d[key][f"{metric}_median"]
            x_mean = group_center - group_width / 2 + (2 * ci + 0.5) * bar_width
            x_med = x_mean + bar_width
            # Mean = solid fill; median = hollow (empty interior, coloured outline
            # only) -- much clearer at a glance than an alpha fade.
            b1 = ax.bar(x_mean, mean_v, width=bar_width * 0.92, color=color,
                         edgecolor="black", linewidth=0.6, zorder=3)
            b2 = ax.bar(x_med, med_v, width=bar_width * 0.92, facecolor="none",
                         edgecolor=color, linewidth=2.0, zorder=3)
            for bars in (b1, b2):
                for rect in bars:
                    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.012,
                             f"{rect.get_height():.2f}", ha="center", va="bottom", fontsize=7.5, rotation=0)
            cond_ticks.append((x_mean + x_med) / 2)
            cond_labels.append(["S1", "S2 OFF", "S2 ON"][ci])

    # Primary x-ticks: the condition labels (S1 / S2 OFF / S2 ON), right under the axis.
    ax.set_xticks(cond_ticks)
    ax.set_xticklabels(cond_labels, fontsize=8, color="0.25")
    ax.tick_params(axis="x", length=0)

    # Secondary row: one tilt-level label per group, well below the condition
    # labels so the two rows never collide.
    for gi, tilt in enumerate(tilts):
        d = data[tilt]
        group_center = gi * tilt_gap
        ax.annotate(f"tilt_{tilt}  ({d['blocked_pct']:.1f}% blocked)",
                     xy=(group_center, 0), xycoords=("data", "axes fraction"),
                     xytext=(0, -34), textcoords="offset points",
                     ha="center", va="top", fontsize=10.5, fontweight="bold")

    ylabel = ("direction (vector/alignment) error [mrad]" if metric == "direction_mrad"
              else "centroid (focal-spot landing) error [mrad]")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_S1, label="Stage 1 baseline"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_OFF, label="Stage 2, blocking OFF"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_ON, label="Stage 2, blocking ON"),
        plt.Rectangle((0, 0), 1, 1, facecolor="0.3", edgecolor="black", label="solid fill = mean"),
        plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="0.3", linewidth=2.0, label="hollow = median"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
               fontsize=8.5, framealpha=0.9, ncol=1, borderaxespad=0)


def main() -> None:
    data = {t: load_tilt(t) for t in TILT_INFO}

    groups = [("high_occlusion", ["000", "033"]), ("low_occlusion", ["067", "100"])]
    metrics = [("direction_mrad", "direction_error", "direction error"),
               ("centroid_mrad", "centroid_error", "centroid error")]
    for metric_key, file_prefix, metric_label in metrics:
        for name, tilts in groups:
            fig, ax = plt.subplots(figsize=(10.8, 6.2))
            make_panel(ax, tilts, data, metric=metric_key)
            title_bits = ", ".join(f"tilt_{t} ({data[t]['blocked_pct']:.1f}%)" for t in tilts)
            occlusion_label = "High occlusion" if name == "high_occlusion" else "Low occlusion"
            ax.set_title(f"BA72: {occlusion_label} -- {title_bits}\n{metric_label} after Stage 1 vs. Stage 2 (blocking off/on)")
            fig.tight_layout()
            out = OUT_DIR / f"{file_prefix}_{name}.png"
            fig.savefig(out, dpi=140, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
