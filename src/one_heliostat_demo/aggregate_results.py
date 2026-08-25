"""
Aggregate training results across all heliostats in a run_all.py output directory.

Every accuracy output reports BOTH metrics — direction (kinematic pointing, excludes
mirror-surface spread) and centroid (ray-traced landing, includes it). They answer
different questions and either one alone is misleading; that is also how a metric
mismatch once turned into a phantom accuracy gap.

Reads per-heliostat results.json and convergence_history.csv and saves:
    aggregated/
        field_mrad_convergence.png   — mean ± 1 std train/val mrad over S1+S2 epochs
        field_loss_curves.png        — mean ± 1 std Stage-1 and Stage-2 loss curves
        accuracy_sorted.png          — per-heliostat bars, one panel per metric
        accuracy_distribution.png    — histograms, one panel per metric
        field_view_detailed.png      — field map coloured by centroid error
        summary_table.txt            — ASCII table, both metrics side by side
        summary_table.csv            — per-heliostat data ONLY (no summary rows)
        summary_field.csv            — field mean/median of every column

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


# The two metrics, always reported together. `mrad_mean`/`mrad_median` in results.json
# are LEGACY ALIASES of the centroid metric; asking for them by that name is what let
# every aggregated plot show the centroid error labelled with the neutral word "mrad",
# with the direction error absent entirely. Never read the alias — go through _metric.
METRICS = [
    ("direction", "direction_mrad", "direction — kinematic pointing (excl. surface)",
     "tab:blue"),
    ("centroid",  "centroid_mrad",  "centroid — ray-traced landing (incl. surface)",
     "tab:orange"),
]


def _metric(r: dict, stage: str, key: str, stat: str) -> float:
    """results[hid][stage]["<key>_<stat>"], falling back to the legacy alias.

    Old runs on disk predate the explicit names and only carry `mrad_*`, which is the
    centroid metric — so the fallback is valid for centroid and correctly yields NaN
    for direction rather than silently substituting the wrong quantity.
    """
    block = r.get(stage, {})
    if f"{key}_{stat}" in block:
        return float(block[f"{key}_{stat}"])
    if key == "centroid_mrad" and f"mrad_{stat}" in block:
        return float(block[f"mrad_{stat}"])
    return float("nan")


def _stage2_ran(results: dict) -> bool:
    """Did Stage 2 actually execute for any heliostat in this run?

    When it did not, `after_stage2` is a copy of `after_stage1`, and labelling plots
    "after Stage 2" tells the reader Stage 2 ran and achieved nothing — a materially
    different claim from "Stage 2 was not run". Older runs have no flag, so fall back
    to comparing the two blocks.
    """
    for r in results.values():
        if "stage2_ran" in r:
            if r["stage2_ran"]:
                return True
        elif r.get("after_stage2") != r.get("after_stage1"):
            return True
    return False


def _post_label(results: dict) -> str:
    return "Stage 2" if _stage2_ran(results) else "Stage 1"


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


def _log_actual_ticks(axis, vmin: float, vmax: float) -> None:
    """Label a log axis with actual round values (1, 2, 5, 10, 20, 50, …) instead of 10^k."""
    from matplotlib.ticker import FixedLocator, FuncFormatter
    cand = [0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000]
    ticks = [t for t in cand if vmin <= t <= vmax]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(FixedLocator([]))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _compute_real_pre_mrad(heliostat_ids: list[str]) -> dict:
    """True geometric before-training miss per heliostat (mean, median) in mrad.

    The training pipeline's pre_training metric is the ray-traced focal-spot centroid,
    which SATURATES (clamps to the target-bitmap edge) when the uncalibrated beam misses
    the target — so badly-aimed heliostats look ~90 mrad when they are really hundreds.
    This recomputes the honest, unbounded miss: reflect one ray off the mirror at its
    nominal kinematics, intersect the target plane through the measured centroid, and
    measure the 3-D distance. Real-data only; heliostats that can't be loaded are omitted
    (the caller keeps the stored value).
    """
    import importlib.util

    import torch

    cfg_path = _here / "single_heliostat" / "config.py"
    spec = importlib.util.spec_from_file_location("_sh_cfg", cfg_path)
    cfg = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"  real pre-miss: could not load config ({exc}); keeping stored values")
        return {}

    csv_path = pathlib.Path(cfg.BENCHMARK_CSV)
    if not csv_path.exists():
        log.info("  real pre-miss: benchmark CSV absent; keeping stored pre values")
        return {}

    from artist.scenario.scenario import Scenario
    from artist.io.paint_calibration_parser import PaintCalibrationDataParser
    from utils.evaluation import build_heliostat_data_mapping

    device = torch.device("cpu")
    plane_n = torch.tensor([0.0, 1.0, 0.0])
    cal_dir, flux_dir = pathlib.Path(cfg.CALIBRATION_DIR), pathlib.Path(cfg.REAL_FLUX_DIR)

    maps: dict = {}
    for split in ("train", "validation", "test"):
        try:
            for h, cs, fs in build_heliostat_data_mapping(csv_path, cal_dir, flux_dir, split):
                maps.setdefault(h, [[], []])
                maps[h][0] += list(cs); maps[h][1] += list(fs)
        except Exception:  # noqa: BLE001
            pass

    parser = PaintCalibrationDataParser(
        centroid_extraction_method=getattr(cfg, "CENTROID_METHOD", "UTIS"))
    out: dict = {}
    for hid in heliostat_ids:
        if hid not in maps:
            continue
        sc_path = pathlib.Path(cfg.SCENARIO_PATH_TEMPLATE.format(heliostat_id=hid))
        if not sc_path.exists():
            continue
        try:
            with h5py.File(sc_path, "r") as fh:
                scenario = Scenario.load_scenario_from_hdf5(
                    scenario_file=fh, device=device,
                    number_of_surface_points_per_facet=torch.tensor([2, 2]))
            hg = scenario.heliostat_field.heliostat_groups[0]
            kin = hg.kinematics
            cals, fluxes = maps[hid]
            _, cents, rays, motors, amask, _ = parser.parse_data_for_reconstruction(
                heliostat_data_mapping=[(hid, cals, fluxes)],
                heliostat_group=hg, scenario=scenario, device=device)
            hg.activate_heliostats(active_heliostats_mask=amask, device=device)
            with torch.no_grad():
                orient = kin.motor_positions_to_orientations(motor_positions=motors, device=device)
                n = torch.nn.functional.normalize(
                    (orient @ torch.tensor([0.0, 0.0, 1.0, 0.0]))[:, :3], dim=-1)
                o = (orient @ torch.tensor([0.0, 0.0, 0.0, 1.0]))[:, :3]
                i = torch.nn.functional.normalize(rays[:, :3], dim=-1)
                r = i - 2.0 * (i * n).sum(-1, keepdim=True) * n
                denom = (r * plane_n).sum(-1)
                tt = ((cents[:, :3] - o) * plane_n).sum(-1) / torch.where(
                    denom.abs() < 1e-9, torch.full_like(denom, float("nan")), denom)
                p = o + tt.unsqueeze(-1) * r
                rng = (cents[:, :3] - o).norm(dim=-1)
                mrad = (p - cents[:, :3]).norm(dim=-1) / rng * 1000
                mrad = mrad[torch.isfinite(mrad)]
            if mrad.numel():
                out[hid] = (float(mrad.mean()), float(mrad.median()))
        except Exception as exc:  # noqa: BLE001
            log.warning(f"  real pre-miss failed for {hid}: {exc}")
    return out


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

def _plot_loss_curves(conv_data: dict, agg_dir: pathlib.Path,
                      stage1_loss: str = "Stage-1 loss") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # The Stage-1 loss name comes from the run (results.json["stage1_loss"]). It was
    # hardcoded to "AlignmentLoss", which has not been the default since the
    # inverse-kinematics fix and is documented as theoretically broken — so every
    # field loss plot named a loss that did not run.
    stage_meta = [
        ("stage1", f"Stage 1 — {stage1_loss}",   axes[0]),
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

    post = _post_label(results)
    hids = sorted(results, key=lambda h: _metric(results[h], "after_stage2",
                                                 "centroid_mrad", "mean"))
    x = np.arange(len(hids))
    pre_mrad = np.array([_metric(results[h], "pre_training", "centroid_mrad", "mean")
                         for h in hids])

    # One panel per metric. Both are always drawn: the direction metric excludes
    # mirror-surface spread and the centroid metric includes it, so they answer
    # different questions and either alone is misleading.
    fig, axes = plt.subplots(len(METRICS), 1, sharex=True,
                             figsize=(max(10, len(hids) * 0.22), 4.4 * len(METRICS)))
    for ax, (_, key, desc, _c) in zip(np.atleast_1d(axes), METRICS):
        vals = np.array([_metric(results[h], "after_stage2", key, "mean") for h in hids])
        if np.all(np.isnan(vals)):
            ax.set_title(f"{desc} — not present in these results.json files")
            continue
        field_mean, field_median = float(np.nanmean(vals)), float(np.nanmedian(vals))

        # Pre-training as a translucent grey bar behind each post-training bar; the
        # visible grey above each bar is the improvement training achieved.
        ax.bar(x, pre_mrad, color="grey", alpha=0.35, width=0.7, zorder=1,
               label="Before training (pre, centroid)")
        ax.bar(x, vals, color=[_band_color(v) for v in vals], width=0.7, zorder=2)
        ax.axhline(field_mean, color="navy", ls="--", lw=1.5,
                   label=f"Mean = {field_mean:.2f} mrad")
        ax.axhline(field_median, color="black", ls=":", lw=1.5,
                   label=f"Median = {field_median:.2f} mrad")

        # The real geometric pre-miss spans ~2 to ~1000 mrad; a log axis keeps the
        # post-training bars readable next to the tall before-training ones.
        big_range = float(np.nanmax(pre_mrad)) > 150 and float(np.nanmin(vals)) > 0
        if big_range:
            ax.set_yscale("log")
            ymin = max(0.5, float(np.nanmin(vals)) * 0.6)
            ymax = float(np.nanmax(pre_mrad)) * 1.3
            ax.set_ylim(ymin, ymax)
            _log_actual_ticks(ax.yaxis, ymin, ymax)
        ax.set_ylabel(f"{key.replace('_mrad', '')} [mrad]"
                      + ("  [log]" if big_range else ""))
        ax.set_title(f"{desc}   —   mean per heliostat, after {post}", fontsize=10)

        from matplotlib.patches import Patch
        band_handles = [Patch(facecolor=c, label=lbl) for _, c, lbl in _ACCURACY_BANDS]
        lh, ll = ax.get_legend_handles_labels()
        ax.legend(lh + band_handles, ll + [lbl for _, _, lbl in _ACCURACY_BANDS],
                  fontsize=7.5, ncol=2)
        ax.grid(True, axis="y", alpha=0.3, zorder=1)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(hids, rotation=90, fontsize=7)
    fig.suptitle(f"Per-heliostat test accuracy — sorted by centroid error, after {post}",
                 fontsize=12)
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

    post = _post_label(results)
    hids = list(results)
    pre_mrad = np.array([_metric(results[h], "pre_training", "centroid_mrad", "mean")
                         for h in hids])

    fig, axes = plt.subplots(1, len(METRICS), figsize=(7.5 * len(METRICS), 5.5))
    for ax, (_, key, desc, _c) in zip(np.atleast_1d(axes), METRICS):
        vals = np.array([_metric(results[h], "after_stage2", key, "mean") for h in hids])
        if np.all(np.isnan(vals)):
            ax.set_title(f"{desc} — not present in these results.json files")
            continue

        # Shared bins over the combined range so the two histograms are comparable;
        # log-spaced when the (real, unbounded) pre values span a wide range.
        n_bins = max(8, len(vals) // 5)
        hi = float(np.nanmax([np.nanmax(pre_mrad), np.nanmax(vals)]))
        if hi > 150:
            lo = max(0.5, float(np.nanmin([np.nanmin(pre_mrad), np.nanmin(vals)])))
            bins = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
            ax.set_xscale("log")
            _log_actual_ticks(ax.xaxis, lo, hi)
        else:
            bins = np.linspace(0.0, hi, n_bins + 1)

        ax.hist(pre_mrad, bins=bins, color="firebrick", edgecolor="white", alpha=0.55,
                zorder=2, label="Before training (pre, centroid)")
        ax.hist(vals, bins=bins, color="steelblue", edgecolor="white", alpha=0.75,
                zorder=3, label=f"After training ({post})")
        for v, c, ls, lab in [
            (float(np.nanmean(pre_mrad)),   "firebrick", "--", "Pre mean"),
            (float(np.nanmedian(pre_mrad)), "firebrick", ":",  "Pre median"),
            (float(np.nanmean(vals)),       "navy",      "--", "Post mean"),
            (float(np.nanmedian(vals)),     "purple",    ":",  "Post median"),
        ]:
            ax.axvline(v, color=c, ls=ls, lw=1.5, label=f"{lab} = {v:.2f} mrad")

        ax.set_xlabel(f"{desc}  [mrad], mean per heliostat")
        ax.set_ylabel("Count")
        ax.set_title(desc, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, zorder=1)

    fig.suptitle("Accuracy distribution — before vs after training  "
                 f"(N = {len(hids)} heliostats, after {post})", fontsize=12)
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


# NOTE: the coarse `_plot_field_view` (field_view.png) was removed. It plotted the
# same data as the detailed view but with obsolete <1 / 1-2 / >2 mrad bands, which
# on real data colour almost the whole field red and disagree with _ACCURACY_BANDS
# used everywhere else. Two field maps of one quantity with different thresholds is
# a correctness hazard, not a convenience.


# ---------------------------------------------------------------------------
# Plot 4b: field view — finer accuracy bands
# ---------------------------------------------------------------------------

_ACCURACY_BANDS = [
    (3.0,  "#2ca02c", "< 3 mrad"),
    (5.0,  "#ffd92f", "3 – 5 mrad"),
    (10.0, "#ff7f0e", "5 – 10 mrad"),
    (20.0, "#9467bd", "10 – 20 mrad"),
    (float("inf"), "#d62728", "> 20 mrad"),
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

    post = _post_label(results)
    all_hids = sorted(field_E.keys())
    # Coloured by the CENTROID metric — what actually lands on the target, which is the
    # question a field map answers. Named in the title so it cannot be mistaken for the
    # direction metric; the per-heliostat breakdown of both is in accuracy_sorted.png.
    vals = {h: _metric(results[h], "after_stage2", "centroid_mrad", "mean")
            for h in all_hids if h in results}
    colors = [_band_color(vals[h]) if h in vals else "#aaaaaa" for h in all_hids]
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
        if hid in vals:
            ax.annotate(
                f"{vals[hid]:.2f}",
                (x, y),
                textcoords="offset points", xytext=(4, 4),
                fontsize=5.5, color="#333333",
            )

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(
        f"Heliostat field — CENTROID error after {post}, mean per heliostat [mrad]"
        f"  ({len(results)} heliostats evaluated)"
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
    """Per-heliostat table with BOTH metrics, plus a separate field-summary file.

    Three things this deliberately does differently from the previous version:

    * Both metrics are tabulated. It used to emit only `mrad_*` — the legacy alias of
      the centroid metric — under column names that named neither.
    * MEAN/MEDIAN are NOT appended as extra data rows. They used to occupy the
      `heliostat_id` column, so the CSV had 65 rows for 63 heliostats and any naive
      `.mean()` or merge silently swallowed them. They now go to their own file.
    * Every column is summarised, not just the means. The old summary rows left all
      median columns blank — dropping exactly the robust statistic we rely on.
    """
    if not results:
        return

    post = _post_label(results)
    pre_is_geometric = any("pre_training_saturated" in r for r in results.values())
    # Name each pre column for what it actually holds. The geometric substitution
    # replaces the CENTROID pre-value only (it is the ray-traced one that saturates
    # against the target bitmap); the direction pre-value is untouched. Tagging both
    # `pre_geom` would claim a substitution that never happened to direction.
    def _pre_tag(key: str) -> str:
        return "pre_geom" if (pre_is_geometric and key == "centroid_mrad") else "pre"

    cols = ["heliostat_id", "dist_m"]
    for _, key, _, _ in METRICS:
        short = key.replace("_mrad", "")
        for stage, tag in (("pre_training", _pre_tag(key)), ("after_stage1", "s1"),
                           ("after_stage2", "s2")):
            cols += [f"{tag}_{short}_mean", f"{tag}_{short}_median"]
    cols.append("improvement_pct_centroid")

    rows = []
    for hid, r in results.items():
        row = [hid, round(r.get("hel_dist_m", float("nan")), 1)]
        for _, key, _, _ in METRICS:
            for stage in ("pre_training", "after_stage1", "after_stage2"):
                row += [_metric(r, stage, key, "mean"), _metric(r, stage, key, "median")]
        pre_c = _metric(r, "pre_training", "centroid_mrad", "mean")
        post_c = _metric(r, "after_stage2", "centroid_mrad", "mean")
        row.append(round((pre_c - post_c) / pre_c * 100, 2) if pre_c > 0 else 0.0)
        rows.append(row)
    rows.sort(key=lambda t: (np.isnan(t[cols.index("s2_centroid_mean")]),
                             t[cols.index("s2_centroid_mean")]))

    csv_path = agg_dir / "summary_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)

    # Field summary — its own file, so the per-heliostat table stays pure data.
    arr = np.array([[v if isinstance(v, (int, float)) else np.nan for v in r[2:-1]]
                    for r in rows], dtype=float)
    summ_path = agg_dir / "summary_field.csv"
    with open(summ_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["statistic"] + cols[2:-1])
        with np.errstate(all="ignore"):
            w.writerow(["field_mean"] + [float(np.nanmean(arr[:, i])) for i in range(arr.shape[1])])
            w.writerow(["field_median"] + [float(np.nanmedian(arr[:, i])) for i in range(arr.shape[1])])

    # ASCII view — centroid and direction side by side, sorted by centroid.
    hdr = (f"  {'Heliostat':<10} {'Dist':>5}"
           f"  {'preC mean':>10} {'S1C mean':>9} {'S1C med':>8} {'S2C mean':>9} {'S2C med':>8}"
           f"  {'S1D mean':>9} {'S1D med':>8} {'S2D mean':>9} {'S2D med':>8}")
    sep = "  " + "-" * (len(hdr) - 2)
    lines = [
        f"  Heliostat accuracy summary  ({len(rows)} heliostats, post = after {post})",
        f"  C = centroid (ray-traced, incl. surface)   D = direction (kinematic, excl. surface)",
        f"  preC = {'GEOMETRIC unbounded miss — NOT the same quantity as S1/S2' if pre_is_geometric else 'ray-traced, same quantity as S1/S2'}",
        "", hdr, sep,
    ]
    idx = {c: i for i, c in enumerate(cols)}
    pre_c_key = f'{_pre_tag("centroid_mrad")}_centroid_mean'
    def _g(r, c):
        v = r[idx[c]]
        return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "—"
    for r in rows:
        lines.append(
            f"  {r[0]:<10} {r[1]:>5.0f}"
            f"  {_g(r, pre_c_key):>10} {_g(r, 's1_centroid_mean'):>9}"
            f" {_g(r, 's1_centroid_median'):>8} {_g(r, 's2_centroid_mean'):>9}"
            f" {_g(r, 's2_centroid_median'):>8}"
            f"  {_g(r, 's1_direction_mean'):>9} {_g(r, 's1_direction_median'):>8}"
            f" {_g(r, 's2_direction_mean'):>9} {_g(r, 's2_direction_median'):>8}"
        )
    lines.append(sep)
    for stat, fn in (("MEAN", np.nanmean), ("MEDIAN", np.nanmedian)):
        with np.errstate(all="ignore"):
            g = lambda c: fn(arr[:, idx[c] - 2])  # noqa: E731
            lines.append(
                f"  {stat:<10} {'':>5}"
                f"  {g(pre_c_key):>10.4f} {g('s1_centroid_mean'):>9.4f}"
                f" {g('s1_centroid_median'):>8.4f} {g('s2_centroid_mean'):>9.4f}"
                f" {g('s2_centroid_median'):>8.4f}"
                f"  {g('s1_direction_mean'):>9.4f} {g('s1_direction_median'):>8.4f}"
                f" {g('s2_direction_mean'):>9.4f} {g('s2_direction_median'):>8.4f}"
            )

    txt = "\n".join(lines) + "\n"
    txt_path = agg_dir / "summary_table.txt"
    with open(txt_path, "w") as f:
        f.write(txt)
    print(txt)

    log.info(f"  → {txt_path}")
    log.info(f"  → {csv_path}")
    log.info(f"  → {summ_path}")


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def aggregate(
    output_dir: pathlib.Path,
    heliostat_ids: list[str] | None = None,
    geometric_pre: bool = True,
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

    # Replace the SATURATED pre-training metric (ray-traced centroid, capped when the
    # beam misses the target bitmap) with the TRUE geometric miss. Real-data runs only;
    # falls back to the stored value where the geometric miss can't be computed. Disable
    # with geometric_pre=False for synthetic runs (where the stored pre is already honest).
    if geometric_pre and results:
        real = _compute_real_pre_mrad(list(results.keys()))
        for hid, (mn, md) in real.items():
            results[hid].setdefault("pre_training_saturated", dict(results[hid]["pre_training"]))
            # Substitute under BOTH spellings, so consumers that read the explicit
            # centroid keys see the same value as those still reading the alias.
            for k, v in (("mrad_mean", mn), ("mrad_median", md),
                         ("centroid_mrad_mean", mn), ("centroid_mrad_median", md)):
                results[hid]["pre_training"][k] = v
        if real:
            log.info(f"  real geometric pre-miss applied to {len(real)}/{len(results)} "
                     f"heliostats — pre columns are now a GEOMETRIC miss, not the "
                     f"ray-traced quantity in s1/s2 (column names say so)")

    stage1_loss = next((r["stage1_loss"] for r in results.values() if "stage1_loss" in r),
                       "Stage-1 loss")
    _plot_mrad_convergence(conv_data, agg_dir)
    _plot_loss_curves(conv_data, agg_dir, stage1_loss=stage1_loss)
    _plot_accuracy_sorted(results, agg_dir)
    _plot_accuracy_distribution(results, agg_dir)
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
    parser.add_argument("--no-geometric-pre", action="store_true",
                        help="Keep the stored (saturated) pre-training metric instead of "
                             "recomputing the true geometric miss. Use for synthetic runs.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s  %(message)s")
    aggregate(output_dir=args.output_dir, heliostat_ids=args.heliostat_ids,
              geometric_pre=not args.no_geometric_pre)


if __name__ == "__main__":
    main()
