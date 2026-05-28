"""
Aggregate per-heliostat results into a combined summary.

Called at the end of main.py after all per-heliostat runs complete.
Also usable standalone to regenerate the combined summary from an existing run directory:

    python aggregate.py outputs/local_runs/full_63_...
"""
import collections
import json
import logging
import pathlib
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)

_C_PRE = "#7f8c8d"   # grey   — pre-training
_C_S1  = "#2980b9"   # blue   — post-stage1
_C_S2  = "#e67e22"   # orange — post-training (stage2)


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def aggregate_results(hel_results: dict, run_dir: pathlib.Path) -> dict:
    """Combine per-heliostat results dicts into a single summary.

    Parameters
    ----------
    hel_results : dict[hid -> results_dict]
        Results returned by train.run() for each heliostat.
    run_dir : pathlib.Path
        Root output directory; combined files are written here.

    Returns
    -------
    dict
        The combined results dict (also written to results_combined.json).
    """
    run_dir = pathlib.Path(run_dir)

    if not hel_results:
        log.warning("aggregate_results: no heliostat results to aggregate.")
        return {}

    def _safe_mean(vals):
        vals = [v for v in vals if v is not None and v == v]  # drop None and NaN
        return statistics.mean(vals) if vals else None

    def _safe_median(vals):
        vals = [v for v in vals if v is not None and v == v]
        return statistics.median(vals) if vals else None

    pre_mrads, pre_medians         = [], []
    post_s1_mrads, post_s1_medians = [], []
    post_mrads, post_medians       = [], []
    per_hel_summary = {}

    for hid, r in hel_results.items():
        pre  = r.get("pre_training")  or {}
        ps1  = r.get("post_stage1")   or {}
        pt   = r.get("post_training") or {}
        skipped = r.get("stage2_skipped", False)

        if pre.get("mean_mrad") is not None:
            pre_mrads.append(pre["mean_mrad"])
        if pre.get("median_mrad") is not None:
            pre_medians.append(pre["median_mrad"])

        if ps1.get("mean_mrad") is not None:
            post_s1_mrads.append(ps1["mean_mrad"])
        if ps1.get("median_mrad") is not None:
            post_s1_medians.append(ps1["median_mrad"])

        if not skipped and pt.get("mean_mrad") is not None:
            post_mrads.append(pt["mean_mrad"])
        if not skipped and pt.get("median_mrad") is not None:
            post_medians.append(pt["median_mrad"])

        per_hel_summary[hid] = {
            "pre_mrad":       pre.get("mean_mrad"),
            "post_s1_mrad":   ps1.get("mean_mrad"),
            "post_mrad":      pt.get("mean_mrad") if not skipped else None,
            "stage2_skipped": skipped,
            "train_time_min": r.get("train_time_min"),
        }

    combined = {
        "n_heliostats_trained":  len(hel_results),
        "n_stage2_skipped":      sum(1 for r in hel_results.values() if r.get("stage2_skipped")),
        "pre_training": {
            "mean_mrad":   _safe_mean(pre_mrads),
            "median_mrad": _safe_median(pre_medians),
        },
        "post_stage1": {
            "mean_mrad":   _safe_mean(post_s1_mrads),
            "median_mrad": _safe_median(post_s1_medians),
        },
        "post_training": {
            "mean_mrad":   _safe_mean(post_mrads),
            "median_mrad": _safe_median(post_medians),
            "n_heliostats": len(post_mrads),
        },
        "per_heliostat": per_hel_summary,
    }

    out_path = run_dir / "results_combined.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    log.info(f"Combined results saved → {out_path}")

    _print_summary_table(combined)

    _plot_accuracy_histogram(per_hel_summary, run_dir)
    _plot_accuracy_tables(per_hel_summary, run_dir)
    _plot_aggregated_unified_mrad(hel_results, run_dir)
    _plot_aggregated_stage_loss(hel_results, run_dir, stage=1)
    _plot_aggregated_stage_loss(hel_results, run_dir, stage=2)

    return combined


# ---------------------------------------------------------------------------
# Text summary table
# ---------------------------------------------------------------------------

def _print_summary_table(combined: dict) -> None:
    pre  = combined.get("pre_training",  {})
    ps1  = combined.get("post_stage1",   {})
    pt   = combined.get("post_training", {})
    n    = combined.get("n_heliostats_trained", 0)
    n_s2 = combined.get("n_stage2_skipped", 0)

    def _fmt(v):
        return f"{v:.3f}" if v is not None else "  n/a "

    log.info("=" * 72)
    log.info(f"  {'Checkpoint':<38} {'Mean/hel':>9}  {'Med/hel':>9}  {'n':>4}")
    log.info(f"  {'-'*38} {'-'*9}  {'-'*9}  {'-'*4}")
    log.info(f"  {'Pre-training (clean scenario)':<38} {_fmt(pre.get('mean_mrad')):>9}  {_fmt(pre.get('median_mrad')):>9}  {n:>4}")
    log.info(f"  {'Post-stage1 (alignment)':<38} {_fmt(ps1.get('mean_mrad')):>9}  {_fmt(ps1.get('median_mrad')):>9}  {n:>4}")
    log.info(f"  {'Post-training (stage-2 best-val)':<38} {_fmt(pt.get('mean_mrad')):>9}  {_fmt(pt.get('median_mrad')):>9}  {pt.get('n_heliostats', 0):>4}")
    if n_s2:
        log.info(f"  ({n_s2} heliostat(s) skipped Stage 2 — too few train samples)")
    log.info("=" * 72)


# ---------------------------------------------------------------------------
# Accuracy histogram (distribution across heliostats)
# ---------------------------------------------------------------------------

def _plot_accuracy_histogram(per_hel_summary: dict, run_dir: pathlib.Path) -> None:
    pre_vals  = [d["pre_mrad"]     for d in per_hel_summary.values() if d.get("pre_mrad")     is not None]
    ps1_vals  = [d["post_s1_mrad"] for d in per_hel_summary.values() if d.get("post_s1_mrad") is not None]
    post_vals = [d["post_mrad"]    for d in per_hel_summary.values() if d.get("post_mrad")    is not None]

    all_vals = pre_vals + ps1_vals + post_vals
    if not all_vals:
        return

    lo, hi = min(all_vals), max(all_vals)
    bins = np.linspace(lo, hi, min(30, max(10, len(pre_vals) // 2)))

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    ax.hist(pre_vals,  bins=bins, alpha=0.55, color=_C_PRE, label=f"Pre-training  (n={len(pre_vals)})")
    ax.hist(ps1_vals,  bins=bins, alpha=0.55, color=_C_S1,  label=f"Post-Stage 1  (n={len(ps1_vals)})")
    ax.hist(post_vals, bins=bins, alpha=0.55, color=_C_S2,  label=f"Post-Stage 2  (n={len(post_vals)})")

    for vals, color in [(pre_vals, _C_PRE), (ps1_vals, _C_S1), (post_vals, _C_S2)]:
        if vals:
            ax.axvline(statistics.mean(vals), color=color, linestyle="--", linewidth=1.4, alpha=0.9)

    ax.set_xlabel("Focal-spot error (mrad)", fontsize=11)
    ax.set_ylabel("Number of heliostats", fontsize=11)
    ax.set_title("Field-wide accuracy distribution per stage", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    fig.tight_layout()

    out = run_dir / "results_histogram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Accuracy histogram saved → {out}")


# ---------------------------------------------------------------------------
# Per-heliostat accuracy tables
# ---------------------------------------------------------------------------

_THRESH_GREEN  = 1.5   # mrad
_THRESH_YELLOW = 2.5   # mrad


def _mrad_cell_color(v: float | None) -> str:
    if v is None:
        return "#ecf0f1"
    if v < _THRESH_GREEN:
        return "#a9dfbf"
    if v < _THRESH_YELLOW:
        return "#fad7a0"
    return "#f1948a"


def _render_table_png(rows, col_labels, run_dir, filename, title, mrad_cols=(1, 2, 3)) -> None:
    n_rows = len(rows)
    row_h  = 0.26
    fig_h  = max(3.5, row_h * (n_rows + 2.5))
    fig_w  = max(9.0, len(col_labels) * 1.9)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    cell_text   = [[str(v) for v in row] for row in rows]
    cell_colors = []
    for row in rows:
        row_c = []
        for i, v in enumerate(row):
            if i in mrad_cols:
                try:
                    num = float(v) if v not in ("—", "n/a", "") else None
                except (ValueError, TypeError):
                    num = None
                row_c.append(_mrad_cell_color(num))
            else:
                row_c.append("#f8f9fa")
        cell_colors.append(row_c)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5 if n_rows > 30 else 8.5)
    tbl.scale(1.0, max(1.2, row_h / 0.18))

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#dee2e6")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()
    out = run_dir / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Accuracy table saved → {out}")


def _plot_accuracy_tables(per_hel_summary: dict, run_dir: pathlib.Path) -> None:
    hids = sorted(per_hel_summary)

    def _f(v, dec=2):
        return f"{v:.{dec}f}" if v is not None else "—"

    # Table 1 — all heliostats
    rows_all = []
    for hid in hids:
        d = per_hel_summary[hid]
        rows_all.append([
            hid,
            _f(d.get("pre_mrad")),
            _f(d.get("post_s1_mrad")),
            _f(d.get("post_mrad")),
            "yes" if d.get("stage2_skipped") else "no",
            _f(d.get("train_time_min")),
        ])
    _render_table_png(
        rows_all,
        col_labels=["Heliostat", "Pre (mrad)", "Post-S1 (mrad)", "Post-S2 (mrad)", "S2 skipped", "Time (min)"],
        run_dir=run_dir,
        filename="accuracy_table_all.png",
        title="Per-heliostat accuracy — all heliostats",
        mrad_cols=(1, 2, 3),
    )

    # Table 2 — stage-2 heliostats only
    rows_s2 = []
    for hid in hids:
        d = per_hel_summary[hid]
        if d.get("stage2_skipped"):
            continue
        pre  = d.get("pre_mrad")
        post = d.get("post_mrad")
        impr = f"{(pre - post) / pre * 100:.1f}%" if (pre and post and pre > 0) else "—"
        rows_s2.append([
            hid,
            _f(d.get("pre_mrad")),
            _f(d.get("post_s1_mrad")),
            _f(d.get("post_mrad")),
            impr,
        ])
    if rows_s2:
        _render_table_png(
            rows_s2,
            col_labels=["Heliostat", "Pre (mrad)", "Post-S1 (mrad)", "Post-S2 (mrad)", "Improvement"],
            run_dir=run_dir,
            filename="accuracy_table_stage2.png",
            title="Per-heliostat accuracy — Stage-2 heliostats only",
            mrad_cols=(1, 2, 3),
        )


# ---------------------------------------------------------------------------
# Aggregated mrad trajectory (unified, both stages)
# ---------------------------------------------------------------------------

def _avg_mrad_by_epoch(trajs: list, key: str) -> list:
    """Return sorted [(epoch_int, mean_value)] averaged across trajectory dicts."""
    by_ep = collections.defaultdict(list)
    for traj in trajs:
        for ep_str, v in traj.get(key, {}).items():
            by_ep[int(ep_str)].append(v)
    return sorted((ep, statistics.mean(vs)) for ep, vs in by_ep.items())


def _plot_aggregated_unified_mrad(hel_results: dict, run_dir: pathlib.Path) -> None:
    all_hids = sorted(hel_results)
    s2_hids  = [h for h in all_hids if not hel_results[h].get("stage2_skipped")]

    all_trajs, s2_trajs = [], []
    pre_vals, ps1_vals, post_vals = [], [], []

    for hid in all_hids:
        p = run_dir / hid / "mrad_trajectory.json"
        if not p.exists():
            continue
        with open(p) as f:
            traj = json.load(f)
        all_trajs.append(traj)
        if traj.get("pre_training_mrad") is not None:
            pre_vals.append(traj["pre_training_mrad"])
        if traj.get("post_stage1_mrad") is not None:
            ps1_vals.append(traj["post_stage1_mrad"])

    for hid in s2_hids:
        p = run_dir / hid / "mrad_trajectory.json"
        if not p.exists():
            continue
        with open(p) as f:
            traj = json.load(f)
        s2_trajs.append(traj)
        if traj.get("post_training_mrad") is not None:
            post_vals.append(traj["post_training_mrad"])

    if not all_trajs:
        return

    s1_tr = _avg_mrad_by_epoch(all_trajs, "stage1")
    s1_vl = _avg_mrad_by_epoch(all_trajs, "stage1_val")
    s2_tr = _avg_mrad_by_epoch(s2_trajs,  "stage2")
    s2_vl = _avg_mrad_by_epoch(s2_trajs,  "stage2_val")

    # Stage 2 x-coords are shifted to start right after Stage 1 ends
    s1_len = (s1_tr[-1][0] + 1) if s1_tr else 0

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")

    if s1_tr:
        xs, ys = zip(*s1_tr)
        ax.plot(xs, ys, color=_C_S1, linewidth=1.8, label=f"Stage 1 train (avg, n={len(all_trajs)})")
        ax.scatter(xs, ys, color=_C_S1, s=18, zorder=3)
    if s1_vl:
        xs, ys = zip(*s1_vl)
        ax.plot(xs, ys, color=_C_S1, linewidth=1.2, linestyle="--", alpha=0.7,
                label=f"Stage 1 val (avg, n={len(all_trajs)})")
        ax.scatter(xs, ys, color=_C_S1, s=10, alpha=0.7, zorder=3)

    if s2_tr:
        xs, ys = zip(*s2_tr)
        xs2 = [x + s1_len for x in xs]
        ax.plot(xs2, ys, color=_C_S2, linewidth=1.8, label=f"Stage 2 train (avg, n={len(s2_hids)})")
        ax.scatter(xs2, ys, color=_C_S2, s=18, zorder=3)
    if s2_vl:
        xs, ys = zip(*s2_vl)
        xs2 = [x + s1_len for x in xs]
        ax.plot(xs2, ys, color=_C_S2, linewidth=1.2, linestyle="--", alpha=0.7,
                label=f"Stage 2 val (avg, n={len(s2_hids)})")
        ax.scatter(xs2, ys, color=_C_S2, s=10, alpha=0.7, zorder=3)

    if s1_len > 0:
        ax.axvline(s1_len, color="grey", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.text(s1_len + 0.3, ax.get_ylim()[1] * 0.98, "S1→S2",
                fontsize=7, color="grey", va="top")

    if pre_vals:
        mv = statistics.mean(pre_vals)
        ax.scatter(0, mv, marker="D", s=70, color="black", zorder=5,
                   label=f"Pre-train ({mv:.2f} mrad)")
    if ps1_vals and s1_len > 0:
        mv = statistics.mean(ps1_vals)
        ax.scatter(s1_len, mv, marker="D", s=70, color=_C_S1, zorder=5,
                   label=f"Post-S1 ({mv:.2f} mrad)", edgecolors="black", linewidths=0.8)
    if post_vals and s2_tr:
        mv  = statistics.mean(post_vals)
        xlast = s2_tr[-1][0] + s1_len
        ax.scatter(xlast, mv, marker="D", s=70, color=_C_S2, zorder=5,
                   label=f"Post-train ({mv:.2f} mrad)", edgecolors="black", linewidths=0.8)

    ax.set_xlabel("Epoch (Stage 1 → Stage 2)", fontsize=11)
    ax.set_ylabel("Mean focal-spot error (mrad)", fontsize=11)
    ax.set_title(
        f"Aggregated convergence — unified mrad  "
        f"(S1: {len(all_trajs)} hels, S2: {len(s2_hids)} hels)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    fig.tight_layout()

    out = run_dir / "aggregated_unified_mrad.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Aggregated unified mrad saved → {out}")


# ---------------------------------------------------------------------------
# Aggregated per-stage loss (original units)
# ---------------------------------------------------------------------------

def _avg_loss_by_epoch(hel_histories: list) -> list:
    """Return sorted [(epoch, mean_train_loss, mean_eval_loss)] across heliostat histories."""
    by_ep_train = collections.defaultdict(list)
    by_ep_eval  = collections.defaultdict(list)
    for hist in hel_histories:
        for entry in hist:
            ep = entry["epoch"]
            if entry.get("loss") is not None:
                by_ep_train[ep].append(entry["loss"])
            if entry.get("eval_loss") is not None:
                by_ep_eval[ep].append(entry["eval_loss"])
    all_epochs = sorted(set(by_ep_train) | set(by_ep_eval))
    return [
        (
            ep,
            statistics.mean(by_ep_train[ep]) if by_ep_train.get(ep) else None,
            statistics.mean(by_ep_eval[ep])  if by_ep_eval.get(ep)  else None,
        )
        for ep in all_epochs
    ]


def _plot_aggregated_stage_loss(hel_results: dict, run_dir: pathlib.Path, stage: int) -> None:
    assert stage in (1, 2)
    fname    = f"convergence_history_stage{stage}.json"
    all_hids = sorted(hel_results)
    use_hids = all_hids if stage == 1 else [h for h in all_hids if not hel_results[h].get("stage2_skipped")]

    hel_histories = []
    for hid in use_hids:
        p = run_dir / hid / fname
        if p.exists():
            with open(p) as f:
                hist = json.load(f)
            if hist:
                hel_histories.append(hist)

    if not hel_histories:
        return

    avg    = _avg_loss_by_epoch(hel_histories)
    epochs = [a[0] for a in avg]
    tr     = [a[1] for a in avg]
    vl     = [a[2] for a in avg]

    color      = _C_S1 if stage == 1 else _C_S2
    stage_name = "AlignmentLoss" if stage == 1 else "FocalSpotLoss"
    n          = len(hel_histories)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")

    ax.plot(epochs, tr, color=color, linewidth=1.8, label=f"Stage {stage} train (avg, n={n})")
    ax.plot(epochs, vl, color=color, linewidth=1.2, linestyle="--", alpha=0.8,
            label=f"Stage {stage} val   (avg, n={n})")

    ax.set_xlabel(f"Stage {stage} epoch", fontsize=11)
    ax.set_ylabel(f"{stage_name} (original units)", fontsize=11)
    ax.set_title(
        f"Aggregated Stage {stage} loss — {stage_name}  (n={n} heliostats)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    fig.tight_layout()

    out = run_dir / f"aggregated_stage{stage}_loss.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Aggregated Stage {stage} loss saved → {out}")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python aggregate.py <run_dir>")
        sys.exit(1)

    from artist.util import set_logger_config
    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)

    run_dir = pathlib.Path(sys.argv[1])
    hel_results = {}
    for results_file in sorted(run_dir.glob("*/results.json")):
        hid = results_file.parent.name
        with open(results_file) as f:
            hel_results[hid] = json.load(f)

    if not hel_results:
        print(f"No per-heliostat results.json found under {run_dir}")
        sys.exit(1)

    print(f"Found {len(hel_results)} heliostat results.")
    aggregate_results(hel_results, run_dir)
