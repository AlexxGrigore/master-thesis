"""
Presentation-ready bar chart: centroid (COM) accuracy per heliostat, after training.

Post-training only (no pre-training comparison), linear y-axis with plain mrad
values — this is a presentation figure, not a diagnostic one.

    python src/standard_results/plot_com_accuracy.py
"""

import pathlib

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_CSV = ROOT / "outputs" / "new_mapping_function" / "standard_results" / "standard_results.csv"
OUT = ROOT / "outputs" / "new_mapping_function" / "standard_results" / "com_accuracy_per_heliostat.png"

# dataviz skill chrome (references/palette.md) + a user-specified 6-tier accuracy
# banding (magnitude tiers, not entity identity — hence the non-standard hue count).
INK       = "#0b0b0b"   # primary ink (title)
INK_2     = "#52514e"   # secondary ink (subtitle, direct labels)
MUTED     = "#898781"   # muted ink (axis ticks)
GRID      = "#e1e0d9"   # hairline gridline
BASELINE  = "#c3c2b7"   # axis baseline
SURFACE   = "#fcfcfb"   # chart surface

# (upper bound exclusive, color, legend label) — first match wins, checked in order.
BANDS = [
    (1,  "#e377c2", "< 1 mrad"),
    (2,  "#2ca02c", "< 2 mrad"),
    (3,  "#f4c430", "< 3 mrad"),
    (5,  "#ff7f0e", "< 5 mrad"),
    (10, "#9467bd", "< 10 mrad"),
    (float("inf"), "#d62728", "≥ 10 mrad"),
]
GRAY = "#a3a3a3"   # documented data fault — overrides the band regardless of value


def _band_color(value: float, fault: bool) -> str:
    if fault:
        return GRAY
    for upper, color, _ in BANDS:
        if value < upper:
            return color
    return BANDS[-1][1]


def main():
    d = pd.read_csv(SRC_CSV).sort_values("centroid_mrad_median").reset_index(drop=True)
    # Lines are drawn at the EXCL.-faults value — consistent with the legend, which
    # already labels the two gray heliostats "excluded from field statistics", and
    # with STANDARD_RESULTS.md. Both numbers are still given in the label text: the
    # incl.-all-63 value moves a lot for the mean (5.2 vs 4.2 mrad — two 29/42 mrad
    # outliers pull hard on a mean of 63) but barely for the median (3.55 vs 3.49),
    # which is exactly why the median is the robust headline statistic elsewhere.
    clean = d[~d["data_fault"]]
    median_excl, median_incl = clean["centroid_mrad_median"].median(), d["centroid_mrad_median"].median()
    mean_excl,   mean_incl   = clean["centroid_mrad_median"].mean(),   d["centroid_mrad_median"].mean()

    fig, ax = plt.subplots(figsize=(20, 8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = range(len(d))
    colors = [_band_color(v, f) for v, f in zip(d["centroid_mrad_median"], d["data_fault"])]
    bars = ax.bar(x, d["centroid_mrad_median"], color=colors, width=0.72, zorder=3)

    # Reference lines: field median AND mean. No on-chart text at all now — nothing
    # to sit over a bar — the line style (dashed vs dotted) is the only thing
    # distinguishing them, matched to the "— —"/"· · · ·" prefixes on the value text
    # next to the legend below, which is where the identification and the numbers
    # both live.
    ax.axhline(median_excl, color=INK_2, lw=1.3, ls="--", zorder=4)
    ax.axhline(mean_excl, color=INK_2, lw=1.3, ls=":", zorder=4)

    # Direct labels only on the two flagged bars — everything else is read off the
    # axis; labeling all 63 would be noise (dataviz: label selectively).
    for i, (val, fault) in enumerate(zip(d["centroid_mrad_median"], d["data_fault"])):
        if fault:
            ax.text(i, val + 0.6, f"{val:.1f}", ha="center", va="bottom",
                    color=INK, fontsize=11, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(d["heliostat"], rotation=90, fontsize=8, color=INK_2)
    ax.set_xlim(-0.7, len(d) - 0.3)

    # Linear axis, plain mrad values — no log, no powers of ten.
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(0, float(d["centroid_mrad_median"].max()) * 1.10)
    ax.set_ylabel("Centroid (COM) error [mrad]", fontsize=12, color=INK)

    ax.grid(axis="y", color=GRID, lw=1.0, zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="y", colors=MUTED, labelsize=10)
    ax.tick_params(axis="x", colors=MUTED, length=0)

    ax.set_title("Centroid (center-of-mass) accuracy per heliostat, after training",
                fontsize=17, color=INK, fontweight="bold", pad=34)
    ax.text(0.5, 1.035,
            "Median ray-traced focal-spot error on held-out test data, 63 heliostats",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11.5, color=INK_2)

    from matplotlib.patches import Patch
    legend = ax.legend(
        handles=[Patch(facecolor=c, label=lbl) for _, c, lbl in BANDS]
        + [Patch(facecolor=GRAY, label="Data fault (excluded from field statistics)")],
        loc="upper left", bbox_to_anchor=(0.005, 0.99), frameon=False, fontsize=10.5,
    )
    ax.add_artist(legend)

    # The actual median/mean values live HERE, next to the colour-band explanation,
    # instead of as floating labels over the bars.
    ax.text(
        0.005, 0.99 - 0.045 * (len(BANDS) + 1) - 0.03,
        "— — median = {:.2f} mrad  ({:.2f} incl. data faults)\n"
        "· · · ·  mean = {:.2f} mrad  ({:.2f} incl. data faults)".format(
            median_excl, median_incl, mean_excl, mean_incl),
        transform=ax.transAxes, va="top", ha="left",
        fontsize=10.5, color=INK_2, linespacing=1.7,
    )

    fig.tight_layout()
    fig.savefig(OUT, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
