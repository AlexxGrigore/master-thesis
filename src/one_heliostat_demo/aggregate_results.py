"""
Aggregate training results across all heliostats in a run_all.py output directory.

Reads per-heliostat results.json and convergence_history.csv and saves:
    aggregated/
        field_mrad_convergence.png   — mean ± 1 std train/val mrad over S1+S2 epochs
        field_loss_curves.png        — mean ± 1 std Stage-1 and Stage-2 loss curves
        accuracy_sorted.png          — sorted bar chart of per-heliostat test mrad
        accuracy_distribution.png    — frequency histogram of test mrad
        field_view.png               — scatter of heliostat positions colored by accuracy
        summary_table.txt            — ASCII table: per-heliostat pre/s1/s2 mrad + mean/median
        summary_table.csv            — same as CSV

Can also be run standalone on any run_all output directory.

Usage
-----
    python aggregate_results.py /path/to/run_all_output_dir
    python aggregate_results.py /path/to/dir --heliostat-ids AA23 AB26 AC33
"""

import argparse
import csv
import json
import logging
import pathlib
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

_here = pathlib.Path(__file__).resolve().parent   # one_heliostat_demo/
_src  = _here.parent                               # src/
sys.path.insert(0, str(_src))

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _safe_float(s) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _load_results(output_dir: pathlib.Path, heliostat_ids: list[str]) -> dict:
    results = {}
    for hid in heliostat_ids:
        rf = output_dir / hid / "results.json"
        if rf.exists():
            with open(rf) as f:
                results[hid] = json.load(f)
        else:
            log.warning(f"results.json not found for {hid} — skipped in aggregation")
    return results


def _load_convergence(output_dir: pathlib.Path, heliostat_ids: list[str]) -> dict:
    """
    Returns {hid: {"pre": row|None, "stage1": [rows], "stage2": [rows]}}
    Each row: {train_loss, val_loss, mrad_train_mean, mrad_val_mean} as floats (NaN if missing).
    """
    data: dict = {}
    for hid in heliostat_ids:
        cf = output_dir / hid / "convergence_history.csv"
        if not cf.exists():
            continue

        pre_row: dict | None = None
        s1_rows: list[dict]  = []
        s2_rows: list[dict]  = []

        with open(cf) as f:
            for row in csv.DictReader(f):
                entry = {
                    "train_loss":      _safe_float(row.get("train_loss")),
                    "val_loss":        _safe_float(row.get("val_loss")),
                    "mrad_train_mean": _safe_float(row.get("mrad_train_mean")),
                    "mrad_val_mean":   _safe_float(row.get("mrad_val_mean")),
                }
                stage = row.get("stage", "")
                if stage == "pre":
                    pre_row = entry
                elif stage == "stage1":
                    s1_rows.append(entry)
                elif stage == "stage2":
                    s2_rows.append(entry)

        data[hid] = {"pre": pre_row, "stage1": s1_rows, "stage2": s2_rows}
    return data


def _align(conv_data: dict, stage: str, metric: str):
    """
    Align per-heliostat metric arrays by within-stage epoch index, padding with NaN.
    Returns (mean, std) arrays of shape (max_len,).
    """
    series = [
        [r[metric] for r in hdata[stage]]
        for hdata in conv_data.values()
        if hdata.get(stage)
    ]
    if not series:
        empty = np.empty(0)
        return empty, empty

    L   = max(len(s) for s in series)
    arr = np.full((len(series), L), np.nan)
    for i, s in enumerate(series):
        arr[i, :len(s)] = s

    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _band(ax, x, mean, std, color, ls="solid", label=None):
    ax.plot(x, mean, color=color, ls=ls, lw=2, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)


# ---------------------------------------------------------------------------
# Plot 1: field-wide mrad convergence
# ---------------------------------------------------------------------------

def _plot_mrad_convergence(conv_data: dict, agg_dir: pathlib.Path) -> None:
    s1_tr_mean, s1_tr_std = _align(conv_data, "stage1", "mrad_train_mean")
    s1_va_mean, s1_va_std = _align(conv_data, "stage1", "mrad_val_mean")
    s2_tr_mean, s2_tr_std = _align(conv_data, "stage2", "mrad_train_mean")
    s2_va_mean, s2_va_std = _align(conv_data, "stage2", "mrad_val_mean")

    n_s1, n_s2 = len(s1_tr_mean), len(s2_tr_mean)
    if n_s1 + n_s2 == 0:
        log.warning("No convergence data — skipping mrad convergence plot")
        return

    # Continuous x-axis: S1 epochs 1..n_s1, S2 epochs n_s1+1..n_s1+n_s2
    x_s1 = np.arange(1, n_s1 + 1)
    x_s2 = np.arange(n_s1 + 1, n_s1 + n_s2 + 1)

    fig, ax = plt.subplots(figsize=(11, 5))

    def _connected_band(x_a, mean_a, std_a, x_b, mean_b, std_b, color, label_a, label_b):
        """Plot S1 and S2 segments as a single continuous solid line."""
        if len(x_a) > 0:
            x_join  = np.concatenate([x_a, [x_b[0]]]) if len(x_b) > 0 else x_a
            m_join  = np.concatenate([mean_a, [mean_b[0]]]) if len(x_b) > 0 else mean_a
            sd_join = np.concatenate([std_a,  [std_b[0]]]) if len(x_b) > 0 else std_a
            _band(ax, x_join, m_join, sd_join, color, "solid", label_a)
        if len(x_b) > 0:
            _band(ax, x_b, mean_b, std_b, color, "solid", label_b)

    has_s1_val = n_s1 > 0 and not np.all(np.isnan(s1_va_mean))
    has_s2_val = n_s2 > 0 and not np.all(np.isnan(s2_va_mean))

    if n_s1 > 0 or n_s2 > 0:
        _connected_band(
            x_s1, s1_tr_mean, s1_tr_std,
            x_s2, s2_tr_mean, s2_tr_std,
            "steelblue", "Train (S1)", "Train (S2)",
        )
    if has_s1_val or has_s2_val:
        _connected_band(
            x_s1 if has_s1_val else np.array([]),
            s1_va_mean if has_s1_val else np.array([]),
            s1_va_std  if has_s1_val else np.array([]),
            x_s2 if has_s2_val else np.array([]),
            s2_va_mean if has_s2_val else np.array([]),
            s2_va_std  if has_s2_val else np.array([]),
            "darkorange", "Val (S1)", "Val (S2)",
        )

    if n_s1 > 0 and n_s2 > 0:
        ax.axvline(n_s1 + 0.5, color="gray", ls=":", lw=1.2)
        total = n_s1 + n_s2
        ax.text(
            n_s1 / (2 * total), 0.97, "Stage 1",
            transform=ax.transAxes, ha="center", va="top", fontsize=10, color="gray",
        )
        ax.text(
            (n_s1 + n_s2 / 2) / total, 0.97, "Stage 2",
            transform=ax.transAxes, ha="center", va="top", fontsize=10, color="gray",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mrad")
    ax.set_title(
        f"Field-wide mrad convergence  ({len(conv_data)} heliostats, mean ± 1 std)"
    )
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = agg_dir / "field_mrad_convergence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"  → {out}")


# ---------------------------------------------------------------------------
# Plot 2: field-wide loss curves (S1 and S2 separately)
# ---------------------------------------------------------------------------

def _plot_loss_curves(conv_data: dict, agg_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    stage_meta = [
        ("stage1", "Stage 1 — AlignmentLoss [mrad]",   axes[0]),
        ("stage2", "Stage 2 — FocalSpotLoss",    axes[1]),
    ]
    for stage, title, ax in stage_meta:
        tr_mean, tr_std = _align(conv_data, stage, "train_loss")
        va_mean, va_std = _align(conv_data, stage, "val_loss")

        if len(tr_mean) == 0:
            ax.set_title(title + "  (no data)")
            continue

        x = np.arange(1, len(tr_mean) + 1)
        _band(ax, x, tr_mean, tr_std, "steelblue",  label="Train")
        if not np.all(np.isnan(va_mean)):
            _band(ax, x, va_mean, va_std, "darkorange", label="Val")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{title}  ({len(conv_data)} heliostats, mean ± 1 std)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = agg_dir / "field_loss_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"  → {out}")


# ---------------------------------------------------------------------------
# Plot 3a: per-heliostat accuracy — sorted bar chart
# ---------------------------------------------------------------------------

def _plot_accuracy_sorted(results: dict, agg_dir: pathlib.Path) -> None:
    if not results:
        return

    hids    = sorted(results, key=lambda h: results[h]["after_stage2"]["mrad_mean"])
    s2_mrad = np.array([results[h]["after_stage2"]["mrad_mean"] for h in hids])

    field_mean   = float(np.mean(s2_mrad))
    field_median = float(np.median(s2_mrad))

    fig, ax = plt.subplots(figsize=(max(10, len(hids) * 0.22), 5))

    x = np.arange(len(hids))
    colors = [
        "#2ca02c" if v < 1.0 else "#ff7f0e" if v < 2.0 else "#d62728"
        for v in s2_mrad
    ]
    ax.bar(x, s2_mrad, color=colors, width=0.7, zorder=2)
    ax.axhline(field_mean,   color="navy",   ls="--", lw=1.5,
               label=f"Mean = {field_mean:.3f} mrad")
    ax.axhline(field_median, color="purple", ls=":",  lw=1.5,
               label=f"Median = {field_median:.3f} mrad")
    ax.set_xticks(x)
    ax.set_xticklabels(hids, rotation=90, fontsize=7)
    ax.set_ylabel("Test mrad (mean per heliostat)")
    ax.set_title("Per-heliostat test accuracy — sorted (after Stage 2)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, zorder=1)

    fig.tight_layout()
    out = agg_dir / "accuracy_sorted.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"  → {out}")


# ---------------------------------------------------------------------------
# Plot 3b: accuracy distribution — frequency histogram
# ---------------------------------------------------------------------------

def _plot_accuracy_distribution(results: dict, agg_dir: pathlib.Path) -> None:
    if not results:
        return

    s2_mrad = np.array([results[h]["after_stage2"]["mrad_mean"] for h in results])

    field_mean   = float(np.mean(s2_mrad))
    field_median = float(np.median(s2_mrad))

    fig, ax = plt.subplots(figsize=(8, 5))

    n_bins = max(8, len(s2_mrad) // 5)
    ax.hist(s2_mrad, bins=n_bins, color="steelblue", edgecolor="white", alpha=0.85, zorder=2)
    ax.axvline(field_mean,   color="navy",   ls="--", lw=1.5,
               label=f"Mean = {field_mean:.3f} mrad")
    ax.axvline(field_median, color="purple", ls=":",  lw=1.5,
               label=f"Median = {field_median:.3f} mrad")
    ax.set_xlabel("Test mrad (mean per heliostat)")
    ax.set_ylabel("Count")
    ax.set_title(f"Accuracy distribution  (N = {len(s2_mrad)} heliostats)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, zorder=1)

    fig.tight_layout()
    out = agg_dir / "accuracy_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"  → {out}")


# ---------------------------------------------------------------------------
# Plot 4: field view — heliostat positions colored by accuracy
# ---------------------------------------------------------------------------

_SCENARIO_H5 = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scenarios" / "full_63_heli_kin_reconstruct" / "scenario.h5"
)


def _plot_field_view(results: dict, agg_dir: pathlib.Path) -> None:
    if not _SCENARIO_H5.exists():
        log.warning(f"Field view skipped — scenario not found: {_SCENARIO_H5}")
        return

    # Load all heliostat ENU positions from the scenario file.
    field_E: dict[str, float] = {}
    field_N: dict[str, float] = {}
    tower_E = tower_N = None

    try:
        with h5py.File(_SCENARIO_H5, "r") as f:
            for hid, hgrp in f["heliostats"].items():
                pos = hgrp["position"][:]   # [E, N, U, 1] or [E, N, U]
                field_E[hid] = float(pos[0])
                field_N[hid] = float(pos[1])
            # Mirror the notebook: use the mean of planar target area centers
            # (solar_tower_juelich_lower/upper) as the tower reference point.
            # power_plant/position is the physical tower structure — at a different ENU.
            ta_centers = []
            if "target_areas_planar" in f:
                for name, grp in f["target_areas_planar"].items():
                    if "solar_tower" in name.lower() and "position_center" in grp:
                        pc = grp["position_center"][()]
                        ta_centers.append((float(pc[0]), float(pc[1])))
            if ta_centers:
                tower_E = float(np.mean([c[0] for c in ta_centers]))
                tower_N = float(np.mean([c[1] for c in ta_centers]))
    except Exception as exc:
        log.warning(f"Field view: failed to read scenario: {exc}")
        return

    # Color each heliostat by Stage-2 accuracy.
    def _color(hid: str) -> str:
        if hid not in results:
            return "#aaaaaa"
        v = results[hid]["after_stage2"]["mrad_mean"]
        if v < 1.0:
            return "#2ca02c"
        if v < 2.0:
            return "#ff7f0e"
        return "#d62728"

    all_hids = sorted(field_E.keys())
    colors   = [_color(h) for h in all_hids]
    xs       = [field_E[h] for h in all_hids]
    ys       = [field_N[h] for h in all_hids]

    fig, ax = plt.subplots(figsize=(9, 9))
    gray_x  = [x for x, c in zip(xs, colors) if c == "#aaaaaa"]
    gray_y  = [y for y, c in zip(ys, colors) if c == "#aaaaaa"]
    color_x = [x for x, c in zip(xs, colors) if c != "#aaaaaa"]
    color_y = [y for y, c in zip(ys, colors) if c != "#aaaaaa"]
    color_c = [c for c in colors if c != "#aaaaaa"]

    if gray_x:
        ax.scatter(gray_x, gray_y, c="#aaaaaa", s=60, zorder=2, label="No result")
    if color_x:
        ax.scatter(color_x, color_y, c=color_c, s=60, zorder=3)

    if tower_E is not None:
        ax.scatter([tower_E], [tower_N], marker="^", c="red", s=200, zorder=4, label="Tower")

    # Legend patches for accuracy bands.
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label="< 1 mrad"),
        Patch(facecolor="#ff7f0e", label="1 – 2 mrad"),
        Patch(facecolor="#d62728", label="> 2 mrad"),
        Patch(facecolor="#aaaaaa", label="No result"),
    ]
    if tower_E is not None:
        from matplotlib.lines import Line2D
        legend_elements.append(
            Line2D([0], [0], marker="^", color="w", markerfacecolor="red",
                   markersize=10, label="Tower")
        )
    ax.legend(handles=legend_elements, fontsize=9, loc="best")

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"Heliostat field — Stage-2 accuracy  ({len(results)} heliostats evaluated)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = agg_dir / "field_view.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"  → {out}")


# ---------------------------------------------------------------------------
# Plot 4b: field view — finer accuracy bands
# ---------------------------------------------------------------------------

_ACCURACY_BANDS = [
    (0.10, "#08519c", "≤ 0.10 mrad"),
    (0.50, "#238b45", "0.10 – 0.50 mrad"),
    (1.00, "#fec44f", "0.50 – 1.00 mrad"),
    (1.50, "#f16913", "1.00 – 1.50 mrad"),
    (2.00, "#cb181d", "1.50 – 2.00 mrad"),
    (float("inf"), "#67000d", "> 2.00 mrad"),
]


def _band_color(v: float) -> str:
    for threshold, color, _ in _ACCURACY_BANDS:
        if v <= threshold:
            return color
    return _ACCURACY_BANDS[-1][1]


def _plot_field_view_detailed(results: dict, agg_dir: pathlib.Path) -> None:
    if not _SCENARIO_H5.exists():
        log.warning(f"Field view (detailed) skipped — scenario not found: {_SCENARIO_H5}")
        return

    field_E: dict[str, float] = {}
    field_N: dict[str, float] = {}
    tower_E = tower_N = None

    try:
        with h5py.File(_SCENARIO_H5, "r") as f:
            for hid, hgrp in f["heliostats"].items():
                pos = hgrp["position"][:]
                field_E[hid] = float(pos[0])
                field_N[hid] = float(pos[1])
            ta_centers = []
            if "target_areas_planar" in f:
                for name, grp in f["target_areas_planar"].items():
                    if "solar_tower" in name.lower() and "position_center" in grp:
                        pc = grp["position_center"][()]
                        ta_centers.append((float(pc[0]), float(pc[1])))
            if ta_centers:
                tower_E = float(np.mean([c[0] for c in ta_centers]))
                tower_N = float(np.mean([c[1] for c in ta_centers]))
    except Exception as exc:
        log.warning(f"Field view (detailed): failed to read scenario: {exc}")
        return

    all_hids = sorted(field_E.keys())
    colors   = [
        _band_color(results[h]["after_stage2"]["mrad_mean"]) if h in results else "#aaaaaa"
        for h in all_hids
    ]
    xs = [field_E[h] for h in all_hids]
    ys = [field_N[h] for h in all_hids]

    fig, ax = plt.subplots(figsize=(9, 9))

    # Draw "no result" heliostats first (bottom layer).
    gray_mask = [c == "#aaaaaa" for c in colors]
    ax.scatter(
        [x for x, m in zip(xs, gray_mask) if m],
        [y for y, m in zip(ys, gray_mask) if m],
        c="#aaaaaa", s=70, zorder=2, label="No result",
    )

    # Draw each accuracy band separately so the legend is ordered correctly.
    for z, (_, band_color, band_label) in enumerate(_ACCURACY_BANDS, start=3):
        mask = [c == band_color for c in colors]
        bx = [x for x, m in zip(xs, mask) if m]
        by = [y for y, m in zip(ys, mask) if m]
        if bx:
            ax.scatter(bx, by, c=band_color, s=70, zorder=z, label=band_label)

    if tower_E is not None:
        ax.scatter([tower_E], [tower_N], marker="^", c="red", s=200, zorder=10, label="Tower")

    # Label each evaluated heliostat with its mrad value.
    for hid, x, y in zip(all_hids, xs, ys):
        if hid in results:
            v = results[hid]["after_stage2"]["mrad_mean"]
            ax.annotate(
                f"{v:.2f}",
                (x, y),
                textcoords="offset points", xytext=(4, 4),
                fontsize=5.5, color="#333333",
            )

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(
        f"Heliostat field — Stage-2 accuracy  ({len(results)} heliostats evaluated)"
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = agg_dir / "field_view_detailed.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"  → {out}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _write_summary_table(results: dict, agg_dir: pathlib.Path) -> None:
    if not results:
        return

    rows = sorted(
        [
            (
                hid,
                r.get("hel_dist_m", float("nan")),
                r["pre_training"]["mrad_mean"],  r["pre_training"]["mrad_median"],
                r["after_stage1"]["mrad_mean"],  r["after_stage1"]["mrad_median"],
                r["after_stage2"]["mrad_mean"],  r["after_stage2"]["mrad_median"],
            )
            for hid, r in results.items()
        ],
        key=lambda t: t[6],
    )

    pre_means = np.array([t[2] for t in rows])
    s1_means  = np.array([t[4] for t in rows])
    s2_means  = np.array([t[6] for t in rows])

    header = (
        f"  {'Heliostat':<10} {'Dist(m)':>7}"
        f"  {'Pre mean':>9} {'Pre med':>8}"
        f"  {'S1 mean':>8} {'S1 med':>7}"
        f"  {'S2 mean':>8} {'S2 med':>7}"
        f"  {'Improv%':>7}"
    )
    sep = "  " + "-" * (len(header) - 2)

    lines = [
        f"  Heliostat accuracy summary  ({len(rows)} heliostats)\n",
        header, sep,
    ]
    for hid, dist, pre_mn, pre_md, s1_mn, s1_md, s2_mn, s2_md in rows:
        improv = (pre_mn - s2_mn) / pre_mn * 100 if pre_mn > 0 else 0.0
        lines.append(
            f"  {hid:<10} {dist:>7.0f}"
            f"  {pre_mn:>9.4f} {pre_md:>8.4f}"
            f"  {s1_mn:>8.4f} {s1_md:>7.4f}"
            f"  {s2_mn:>8.4f} {s2_md:>7.4f}"
            f"  {improv:>6.1f}%"
        )
    lines.append(sep)
    lines.append(
        f"  {'MEAN':<10} {'':>7}"
        f"  {float(np.mean(pre_means)):>9.4f} {'':>8}"
        f"  {float(np.mean(s1_means)):>8.4f} {'':>7}"
        f"  {float(np.mean(s2_means)):>8.4f} {'':>7}"
        f"  {'':>7}"
    )
    lines.append(
        f"  {'MEDIAN':<10} {'':>7}"
        f"  {float(np.median(pre_means)):>9.4f} {'':>8}"
        f"  {float(np.median(s1_means)):>8.4f} {'':>7}"
        f"  {float(np.median(s2_means)):>8.4f} {'':>7}"
        f"  {'':>7}"
    )

    txt = "\n".join(lines) + "\n"

    txt_path = agg_dir / "summary_table.txt"
    with open(txt_path, "w") as f:
        f.write(txt)
    print(txt)

    csv_path = agg_dir / "summary_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "heliostat_id", "dist_m",
            "pre_mrad_mean", "pre_mrad_median",
            "s1_mrad_mean",  "s1_mrad_median",
            "s2_mrad_mean",  "s2_mrad_median",
            "improvement_pct",
        ])
        for hid, dist, pre_mn, pre_md, s1_mn, s1_md, s2_mn, s2_md in rows:
            improv = (pre_mn - s2_mn) / pre_mn * 100 if pre_mn > 0 else 0.0
            w.writerow([hid, round(dist, 1), pre_mn, pre_md, s1_mn, s1_md, s2_mn, s2_md, round(improv, 2)])
        w.writerow(["MEAN",   "", float(np.mean(pre_means)),   "", float(np.mean(s1_means)),   "", float(np.mean(s2_means)),   "", ""])
        w.writerow(["MEDIAN", "", float(np.median(pre_means)), "", float(np.median(s1_means)), "", float(np.median(s2_means)), "", ""])

    log.info(f"  → {txt_path}")
    log.info(f"  → {csv_path}")


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def aggregate(
    output_dir: pathlib.Path,
    heliostat_ids: list[str] | None = None,
) -> None:
    """
    Aggregate results from a run_all.py output directory.

    Parameters
    ----------
    output_dir:
        Root output directory produced by run_all.py.
    heliostat_ids:
        Subset to aggregate. If None, auto-discovers all {hid}/ subdirs with results.json.
    """
    output_dir = pathlib.Path(output_dir)

    if heliostat_ids is None:
        heliostat_ids = sorted(
            d.name for d in output_dir.iterdir()
            if d.is_dir() and (d / "results.json").exists()
        )

    if not heliostat_ids:
        log.warning("aggregate: no heliostat results found in %s", output_dir)
        return

    agg_dir = output_dir / "aggregated"
    agg_dir.mkdir(exist_ok=True)

    log.info(f"Aggregating {len(heliostat_ids)} heliostats → {agg_dir}")

    results   = _load_results(output_dir, heliostat_ids)
    conv_data = _load_convergence(output_dir, heliostat_ids)

    _plot_mrad_convergence(conv_data, agg_dir)
    _plot_loss_curves(conv_data, agg_dir)
    _plot_accuracy_sorted(results, agg_dir)
    _plot_accuracy_distribution(results, agg_dir)
    _plot_field_view(results, agg_dir)
    _plot_field_view_detailed(results, agg_dir)
    _write_summary_table(results, agg_dir)

    log.info("Aggregation complete.")
    print(f"\nAggregated results → {agg_dir}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate run_all.py results into field-wide plots and tables."
    )
    parser.add_argument("output_dir", type=pathlib.Path,
                        help="run_all.py output directory to aggregate")
    parser.add_argument("--heliostat-ids", nargs="+", default=None, metavar="ID",
                        help="Subset of heliostat IDs (default: all with results.json)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s  %(message)s")
    aggregate(output_dir=args.output_dir, heliostat_ids=args.heliostat_ids)


if __name__ == "__main__":
    main()
