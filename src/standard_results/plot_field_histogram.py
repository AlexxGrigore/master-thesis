"""
Presentation-ready HISTOGRAM of accuracy across a run_all.py field run — built for
the full-field 1277-heliostat Stage-1-only run, but works on any run_all output dir.

Unlike plot_com_accuracy.py (a sorted bar chart, one bar per heliostat — readable
at 63 heliostats), this is a frequency histogram: at ~1277 heliostats, individual
IDs stop being a useful x-axis and the question becomes "how many heliostats land
in each accuracy range" instead. Both metrics (direction, centroid) always shown.

    python src/standard_results/plot_field_histogram.py \
        --run-dir outputs/full_field_1277/stage1_only_ideal_surfaces \
        --title-suffix "1277 heliostats, ideal surfaces"
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# dataviz skill chrome (references/palette.md)
BLUE     = "#2a78d6"
ORANGE   = "#eb6834"
INK      = "#0b0b0b"
INK_2    = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE  = "#fcfcfb"

METRICS = [
    ("direction_mrad", BLUE,   "Direction (kinematic pointing, excl. surface)"),
    ("centroid_mrad",  ORANGE, "Centroid (ray-traced landing, incl. surface)"),
]


def load_results(run_dir: pathlib.Path, stage: str) -> dict:
    """{metric_key: np.array of per-heliostat values}, plus the heliostat count."""
    vals = {k: [] for k, _, _ in METRICS}
    n = 0
    for rf in sorted(run_dir.glob("*/results.json")):
        r = json.loads(rf.read_text())
        block = r.get(stage, {})
        for key, _, _ in METRICS:
            v = block.get(f"{key}_median")
            if v is not None:
                vals[key].append(v)
        n += 1
    return {k: np.array(v) for k, v in vals.items()}, n


def plot_histogram(vals: dict, n: int, out: pathlib.Path, title_suffix: str,
                   clip_at: float | None = None):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 6))
    fig.patch.set_facecolor(SURFACE)

    for ax, (key, color, label) in zip(axes, METRICS):
        ax.set_facecolor(SURFACE)
        v = vals[key]
        v = v[~np.isnan(v)]
        if len(v) == 0:
            ax.set_title(f"{label} — no data")
            continue

        median, mean = float(np.median(v)), float(np.mean(v))
        # Clip the display range so a handful of extreme outliers (the field-wide
        # equivalent of AC33/AW36 — real, but not representative of the bulk) don't
        # crush the histogram into one tall bar and a flat line. Values beyond the
        # clip are counted and reported in the subtitle, never silently dropped.
        hi = clip_at if clip_at is not None else np.percentile(v, 99)
        n_over = int((v > hi).sum())
        v_clipped = np.clip(v, 0, hi)

        bins = np.linspace(0, hi, 51)
        ax.hist(v_clipped, bins=bins, color=color, alpha=0.85, edgecolor=SURFACE,
                linewidth=0.4, zorder=3)

        ax.axvline(median, color=INK_2, lw=1.4, ls="--", zorder=4)
        ax.axvline(mean, color=INK_2, lw=1.4, ls=":", zorder=4)
        ax.text(0.98, 0.96,
                f"median = {median:.2f} mrad\nmean    = {mean:.2f} mrad\n"
                f"n = {len(v)}" + (f"\n({n_over} beyond axis, clipped)" if n_over else ""),
                transform=ax.transAxes, ha="right", va="top", fontsize=10.5,
                color=INK_2, linespacing=1.6,
                bbox=dict(facecolor=SURFACE, edgecolor=BASELINE, linewidth=0.8, pad=6))

        ax.set_xlabel(f"{label}  [mrad]", fontsize=11, color=INK)
        ax.set_ylabel("Heliostats", fontsize=11, color=INK)
        ax.set_title(label, fontsize=12, color=INK, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.grid(axis="y", color=GRID, lw=1.0, zorder=0)
        ax.grid(axis="x", visible=False)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=9.5)

    fig.suptitle(f"Field-wide Stage-1 accuracy — {title_suffix}  (n={n} heliostats)",
                fontsize=15, color=INK, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=pathlib.Path, required=True)
    ap.add_argument("--stage", default="after_stage1",
                   choices=["pre_training", "after_stage1", "after_stage2"])
    ap.add_argument("--title-suffix", default="")
    ap.add_argument("--clip-at", type=float, default=None,
                   help="mrad value to clip the display at (default: 99th percentile)")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    vals, n = load_results(args.run_dir, args.stage)
    out = args.out or args.run_dir / "field_accuracy_histogram.png"
    plot_histogram(vals, n, out, args.title_suffix or args.run_dir.name, args.clip_at)


if __name__ == "__main__":
    main()
