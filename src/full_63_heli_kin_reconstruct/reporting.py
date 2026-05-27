"""
Reporting and visualisation for the full-63-heliostat kinematic reconstruction experiment.

Output files
------------
filter_stats.png                  — per-heliostat sample counts after flux filtering (train/val/test)
convergence_unified_mrad.png      — Stage 1 + Stage 2 on a single mrad y-axis
convergence_stage1.png            — Stage 1 AlignmentLoss in rad² over epochs
convergence_stage2.png            — Stage 2 loss in native units over epochs
gt_grids/train.png                — GT measured flux grid for training split (one row per heliostat)
gt_grids/val.png                  — same for validation split
gt_grids/test.png                 — same for test split
centroid_trails/{hid}_trail.png   — Stage-2 training samples with per-epoch centroid trail overlay
per_heliostat_accuracy.png        — table of pre/post mrad for every heliostat
per_heliostat_accuracy_histogram.png — histogram of post-training mrad across heliostats
field_accuracy_map.png            — ENU field scatter plot coloured by post-training accuracy
summary_table.png                 — clean 2-row table: val + test mean/median mrad
contour_components_train.png      — (contour loss only) coarse/fine/gravity per-component train loss
contour_components_val.png        — (contour loss only) same, val split
contour_overlay_best_{hid}.png    — (contour loss only) GT flux + GT/pred contour for best heliostat
contour_overlay_worst_{hid}.png   — same for worst heliostat
pipeline_steps_best_{hid}.png     — (contour loss only) step-by-step contour pipeline, best heliostat
pipeline_steps_worst_{hid}.png    — same for worst heliostat
"""
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Filter stats table  (per-heliostat sample counts after flux filtering)
# ---------------------------------------------------------------------------

_C_FULL    = "#d5e8d4"   # light green  — full count retained
_C_PARTIAL = "#fff2cc"   # light yellow — some samples removed
_C_EMPTY   = "#f8cecc"   # light red    — all samples removed
_C_ID_COL  = "#f0f0f0"   # light grey   — heliostat-ID column


def plot_filter_stats_table(
    hel_counts: "dict[str, dict[str, int]]",
    max_train: int,
    max_val: int,
    max_test: int,
    output_dir: pathlib.Path,
) -> None:
    """Save a per-heliostat sample-count table after flux filtering.

    Parameters
    ----------
    hel_counts : {"train": {hid: n, ...}, "val": {hid: n, ...}, "test": {hid: n, ...}}
    max_train  : original (unfiltered) train sample count per heliostat
    max_val    : same for val
    max_test   : same for test
    output_dir : destination (``filter_stats.png`` written here)
    """
    splits     = ["train", "val", "test"]
    max_counts = {"train": max_train, "val": max_val, "test": max_test}

    all_hids = sorted(set().union(*(set(hel_counts.get(s, {})) for s in splits)))
    if not all_hids:
        return

    col_labels = [
        "Heliostat",
        f"Train  (/{max_train})",
        f"Val  (/{max_val})",
        f"Test  (/{max_test})",
    ]

    cell_text, cell_colors = [], []
    for hid in all_hids:
        row_t = [hid]
        row_c = [_C_ID_COL]
        for split in splits:
            cnt   = hel_counts.get(split, {}).get(hid, 0)
            max_c = max_counts[split]
            row_t.append(str(cnt))
            if cnt == max_c:
                row_c.append(_C_FULL)
            elif cnt > 0:
                row_c.append(_C_PARTIAL)
            else:
                row_c.append(_C_EMPTY)
        cell_text.append(row_t)
        cell_colors.append(row_c)

    n_rows   = len(all_hids)
    cell_h   = 0.22
    header_h = 0.34

    fig, ax = plt.subplots(figsize=(7, 2))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="upper center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)

    for c in range(len(col_labels)):
        tbl[0, c].set_height(header_h)
        tbl[0, c].set_facecolor("#3a3a3a")
        tbl[0, c].set_text_props(color="white", fontweight="bold")

    for r in range(1, n_rows + 1):
        for c in range(len(col_labels)):
            tbl[r, c].set_height(cell_h)
        tbl[r, 0].get_text().set_ha("left")

    ax.set_title(
        "Samples remaining after flux filtering  "
        "(green = full  ·  yellow = partial  ·  red = zero)",
        fontsize=9, fontweight="bold", pad=6,
    )
    fig.tight_layout()

    out_path = output_dir / "filter_stats.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# GT flux grid  (one row per heliostat, one column per sample)
# ---------------------------------------------------------------------------

def plot_gt_flux_grids(
    hel_images: "dict[str, list[np.ndarray]]",
    split_name: str,
    output_dir: pathlib.Path,
    n_cols: int = 10,
) -> None:
    """Save a GT flux grid for one data split.

    Layout: one row per heliostat (labelled on the left), up to ``n_cols``
    sample images per row.  Written to ``output_dir/gt_grids/{split_name}.png``.

    Parameters
    ----------
    hel_images : {hid: [H×W float array (peak-normalised [0,1]), ...]}
    split_name : "train", "val", or "test" — used in title and filename
    output_dir : experiment output directory (``gt_grids/`` subdirectory created inside)
    n_cols     : maximum number of samples shown per heliostat row
    """
    hids = [h for h in hel_images if hel_images[h]]
    if not hids:
        return

    n_rows = len(hids)
    n_cols_actual = min(n_cols, max(len(hel_images[h]) for h in hids))

    cell_w, cell_h = 0.85, 0.75
    fig, axes = plt.subplots(
        n_rows, n_cols_actual,
        figsize=(n_cols_actual * cell_w + 1.2, n_rows * cell_h + 0.5),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for r, hid in enumerate(hids):
        imgs = hel_images[hid]
        for c in range(n_cols_actual):
            ax = axes[r, c]
            if c < len(imgs):
                ax.imshow(imgs[c], cmap="inferno", vmin=0, vmax=1)
            else:
                ax.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(hid, fontsize=5.5, rotation=0,
                               labelpad=4, va="center", ha="right")

    fig.suptitle(
        f"GT measured flux — {split_name} split  "
        f"({n_rows} heliostats, up to {n_cols_actual} samples/row)",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout(rect=[0.05, 0, 1, 0.98], h_pad=0.05, w_pad=0.05)

    grid_dir = output_dir / "gt_grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    out_path = grid_dir / f"{split_name}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Unified mrad convergence  (Stage 1 + Stage 2 on the same axis)
# ---------------------------------------------------------------------------

def plot_unified_mrad(
    mrad_trajectory_path: pathlib.Path,
    output_dir: pathlib.Path,
) -> None:
    """Plot both training stages on a single mrad y-axis.

    Reads ``mrad_trajectory.json`` produced by ``_save_mrad_trajectory()`` in
    train.py.  Trail-recorder samples (one dot every CAPTURE_STRIDE epochs) are
    shown as a curve; the three exact eval-pass reference points
    (pre-training, post-Stage-1, post-training) are shown as larger markers.

    Stage 1 is plotted in blue, Stage 2 in orange.  A vertical dashed line
    marks the stage boundary.

    Parameters
    ----------
    mrad_trajectory_path : path to ``mrad_trajectory.json``
    output_dir           : directory where ``convergence_unified_mrad.png`` is saved
    """
    if not mrad_trajectory_path.exists():
        return

    with open(mrad_trajectory_path) as f:
        data = json.load(f)

    offset     = int(data["stage2_epoch_offset"])
    pre_mrad   = data.get("pre_training_mrad")
    post_s1    = data.get("post_stage1_mrad")
    post_train = data.get("post_training_mrad")

    s1_items = sorted((int(k), v) for k, v in data["stage1"].items())
    s2_items = sorted((int(k), v) for k, v in data["stage2"].items())

    s1_x = [ep         for ep, _ in s1_items]
    s1_y = [v          for _,  v in s1_items]
    s2_x = [ep + offset for ep, _ in s2_items]
    s2_y = [v           for _,  v in s2_items]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")

    _C1 = "#2980b9"   # Stage 1 — blue
    _C2 = "#e67e22"   # Stage 2 — orange

    if s1_x:
        ax.plot(s1_x, s1_y, color=_C1, linewidth=1.5, label="Stage 1 (AlignmentLoss)")
        ax.scatter(s1_x, s1_y, color=_C1, s=18, zorder=3)

    if s2_x:
        ax.plot(s2_x, s2_y, color=_C2, linewidth=1.5, label="Stage 2 (ray-tracing loss)")
        ax.scatter(s2_x, s2_y, color=_C2, s=18, zorder=3)

    # Stage boundary
    if offset > 0:
        ax.axvline(offset, color="grey", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.text(offset + 0.5, ax.get_ylim()[1], "S1→S2", fontsize=7,
                color="grey", va="top")

    # Exact eval reference points
    ref_x_max = (s2_x[-1] if s2_x else (s1_x[-1] if s1_x else offset))

    if pre_mrad is not None:
        ax.scatter(0, pre_mrad, marker="D", s=60, color="black",
                   zorder=5, label=f"Pre-train ({pre_mrad:.2f} mrad)")
    if post_s1 is not None and offset > 0:
        ax.scatter(offset, post_s1, marker="D", s=60, color=_C1,
                   zorder=5, label=f"Post-S1 ({post_s1:.2f} mrad)", edgecolors="black",
                   linewidths=0.8)
    if post_train is not None:
        ax.scatter(ref_x_max, post_train, marker="D", s=60, color=_C2,
                   zorder=5, label=f"Post-train ({post_train:.2f} mrad)",
                   edgecolors="black", linewidths=0.8)

    ax.set_xlabel("Training epoch (Stage 1 → Stage 2)", fontsize=11)
    ax.set_ylabel("Mean mrad (train samples)", fontsize=11)
    ax.set_title("Unified convergence — both stages in mrad", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    fig.tight_layout()

    out_path = output_dir / "convergence_unified_mrad.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Field accuracy map
# ---------------------------------------------------------------------------

_GREEN  = "#2ecc71"
_YELLOW = "#f39c12"
_RED    = "#e74c3c"
_GREY   = "#95a5a6"   # NaN / missing

_THRESH_GREEN  = 1.5   # mrad
_THRESH_YELLOW = 2.5   # mrad


def _accuracy_color(mrad: float | None) -> str:
    if mrad is None or np.isnan(mrad):
        return _GREY
    if mrad < _THRESH_GREEN:
        return _GREEN
    if mrad < _THRESH_YELLOW:
        return _YELLOW
    return _RED


def plot_field_accuracy_map(
    field_positions_path: pathlib.Path,
    per_heliostat_mrad: dict,
    output_dir: pathlib.Path,
) -> None:
    """
    Scatter plot of the heliostat field in the East-North plane.

    Dots are coloured by post-training focal-spot error:
      green  < 1.5 mrad
      yellow  1.5 – 2.5 mrad
      red    > 2.5 mrad

    Parameters
    ----------
    field_positions_path : path to field_positions.json written by train.py
    per_heliostat_mrad   : dict  {hid: mrad_float}  from results["post_training"]["per_heliostat"]
    output_dir           : destination directory
    """
    if not field_positions_path.exists():
        print(f"WARNING: {field_positions_path} not found — skipping field accuracy map.")
        return

    with open(field_positions_path) as f:
        field_data = json.load(f)

    heliostat_ids    = field_data["heliostat_ids"]
    positions_enu    = np.array(field_data["positions_enu"])   # [N, 3]
    tower_enu        = np.array(field_data.get("tower_enu", [0.0, 0.0, 0.0]))

    east_rel  = positions_enu[:, 0] - tower_enu[0]
    north_rel = positions_enu[:, 1] - tower_enu[1]

    colors = [
        _accuracy_color(per_heliostat_mrad.get(hid, {}).get("focal_spot_error_mrad"))
        for hid in heliostat_ids
    ]

    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor("white")

    ax.scatter(east_rel, north_rel, c=colors, s=55, edgecolors="black",
               linewidths=0.3, zorder=3)

    # Tower marker
    ax.scatter(0, 0, marker="^", s=180, c="navy", zorder=5, label="Tower")

    # Annotate heliostat IDs (small font, offset slightly so they don't overlap dot)
    for hid, e, n in zip(heliostat_ids, east_rel, north_rel):
        ax.text(e + 1.5, n, hid, fontsize=4.5, va="center", color="#333333", zorder=4)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_GREEN,  edgecolor="black", linewidth=0.5, label=f"< {_THRESH_GREEN} mrad"),
        Patch(facecolor=_YELLOW, edgecolor="black", linewidth=0.5, label=f"{_THRESH_GREEN}–{_THRESH_YELLOW} mrad"),
        Patch(facecolor=_RED,    edgecolor="black", linewidth=0.5, label=f"> {_THRESH_YELLOW} mrad"),
        Patch(facecolor=_GREY,   edgecolor="black", linewidth=0.5, label="N/A"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="navy",
                   markersize=9, label="Tower"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, framealpha=0.9, loc="upper left")

    ax.set_xlabel("East offset from tower (m)", fontsize=11)
    ax.set_ylabel("North offset from tower (m)", fontsize=11)
    ax.set_title("Field accuracy map — post-training focal-spot error", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_aspect("equal")
    fig.tight_layout()

    out_path = output_dir / "field_accuracy_map.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def render_summary_table(
    val_eval: dict | None,
    test_eval: dict,
    output_dir: pathlib.Path,
) -> None:
    """
    Render a clean PNG table showing val + test accuracy.

    Columns: Split | Mean/sample | Median/sample | Mean/hel | Median/hel
    Rows   : Validation, Test

    The per-heliostat columns aggregate each heliostat's mean error first, then
    take the mean/median over heliostats — matching the histogram statistics.
    The per-sample columns are the mean/median over all individual samples.

    Parameters
    ----------
    val_eval  : results["post_training_val"]  (may be None if not computed)
    test_eval : results["post_training"]
    output_dir: destination directory
    """
    def _fmt(d: dict | None, key: str) -> str:
        if d is None:
            return "—"
        v = d.get(key)
        return f"{v:.3f}" if v is not None else "—"

    def _ph_stats(d: dict | None) -> tuple[str, str]:
        if d is None:
            return "—", "—"
        ph = d.get("per_heliostat", {})
        vals = [
            v.get("focal_spot_error_mrad")
            for v in ph.values()
            if isinstance(v, dict) and v.get("focal_spot_error_mrad") is not None
        ]
        if not vals:
            return "—", "—"
        arr = np.array(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return "—", "—"
        return f"{np.mean(arr):.3f}", f"{np.median(arr):.3f}"

    val_ph_mean,  val_ph_median  = _ph_stats(val_eval)
    test_ph_mean, test_ph_median = _ph_stats(test_eval)

    rows = [
        ["Validation", val_ph_mean,  val_ph_median],
        ["Test",       test_ph_mean, test_ph_median],
    ]
    col_headers = ["Split", "Mean/hel (mrad)", "Median/hel (mrad)"]

    fig, ax = plt.subplots(figsize=(5, 1.8))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    header_h = 0.38
    row_h    = 0.30
    for c in range(len(col_headers)):
        tbl[0, c].set_height(header_h)
        tbl[0, c].set_facecolor("#3a3a3a")
        tbl[0, c].set_text_props(color="white", fontweight="bold")
    for r in range(1, len(rows) + 1):
        for c in range(len(col_headers)):
            tbl[r, c].set_height(row_h)
            tbl[r, c].set_facecolor("#f5f5f5" if r % 2 == 0 else "white")

    ax.set_title(
        "Post-training accuracy  (mrad)",
        fontsize=11, fontweight="bold", pad=8,
    )
    fig.tight_layout()
    out_path = output_dir / "summary_table.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Contour-loss component convergence
# ---------------------------------------------------------------------------

def plot_contour_loss_components(
    history: list[dict],
    output_dir: pathlib.Path,
    split: str = "train",
) -> None:
    """
    Plot coarse / fine / gravity loss components over epochs.

    For split="train" reads  loss_coarse / loss_fine / loss_gravity + loss.
    For split="val"   reads  eval_loss_coarse / eval_loss_fine / eval_loss_gravity + eval_loss.

    Parameters
    ----------
    history    : convergence history list (convergence_history_stage2.json)
    output_dir : destination directory
    split      : "train" or "val"
    """
    if split == "val":
        prefix, total_key = "eval_loss_", "eval_loss"
    else:
        prefix, total_key = "loss_", "loss"

    entries = [e for e in history if prefix + "coarse" in e]
    if not entries:
        return  # no component data — loss type was not contour

    epochs  = [e["epoch"]                  for e in entries]
    coarse  = [e[prefix + "coarse"]        for e in entries]
    fine    = [e[prefix + "fine"]          for e in entries]
    gravity = [e[prefix + "gravity"]       for e in entries]
    total   = [e.get(total_key, float("nan")) for e in entries]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    ax.plot(epochs, coarse,  label=f"Coarse (β={0.3:.1f})",  color="#e74c3c", linewidth=1.5)
    ax.plot(epochs, fine,    label=f"Fine (1−β−γ={0.5:.1f})", color="#2980b9", linewidth=1.5)
    ax.plot(epochs, gravity, label=f"Gravity (γ={0.2:.1f})", color="#27ae60", linewidth=1.5)
    ax.plot(epochs, total,   label="Total (weighted)",        color="black",   linewidth=1.0,
            linestyle="--", alpha=0.6)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (unweighted per-term mean)", fontsize=11)
    split_label = split.capitalize()
    ax.set_title(f"ContourLoss components — {split_label}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    fig.tight_layout()

    out_path = output_dir / f"contour_components_{split}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Contour overlay (GT flux + GT contour + predicted contour)
# ---------------------------------------------------------------------------

def plot_contour_overlay(
    measured_tensors: list,
    predicted_tensors: list,
    heliostat_id: str,
    mean_mrad: float,
    role: str,
    output_dir: pathlib.Path,
    contour_params: dict | None = None,
    n_show: int = 6,
) -> None:
    """
    For each of n_show samples show four panels side-by-side:
      measured flux | GT contour on measured | predicted flux | pred contour on predicted

    Parameters
    ----------
    measured_tensors  : list of [1, H, W] float32 tensors (raw measured flux)
    predicted_tensors : list of [1, H, W] float32 tensors (raw predicted flux)
    heliostat_id      : used for title and filename
    mean_mrad         : post-training FSE in mrad
    role              : "best" or "worst"
    output_dir        : destination directory
    contour_params    : kwargs for ContourLoss (uses defaults if None)
    n_show            : number of samples to plot (rows)
    """
    import sys, pathlib as _pl
    _src = _pl.Path(__file__).resolve().parents[1]
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from artist_extensions.loss_functions_ext import ContourLoss

    loss_fn = ContourLoss(**(contour_params or {}))
    n_show  = min(n_show, len(measured_tensors), len(predicted_tensors))

    fig, axes = plt.subplots(n_show, 4, figsize=(12, n_show * 2.8))
    fig.patch.set_facecolor("white")
    if n_show == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "Measured flux",
        "GT contour\n(on measured)",
        "Predicted flux",
        "Pred contour\n(on predicted)",
    ]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=9, fontweight="bold")

    def _norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / max(mx - mn, 1e-12)

    def _overlay(ax, flux_np, contour_np, color_rgb):
        """Show flux as grayscale background with contour overlaid in color_rgb."""
        ax.imshow(_norm(flux_np), cmap="gray", vmin=0, vmax=1)
        H, W = contour_np.shape
        rgba = np.zeros((H, W, 4), dtype=np.float32)
        rgba[..., 0] = color_rgb[0]
        rgba[..., 1] = color_rgb[1]
        rgba[..., 2] = color_rgb[2]
        rgba[..., 3] = np.clip(contour_np, 0, 1)
        ax.imshow(rgba)

    for i in range(n_show):
        m_t = measured_tensors[i]   # [1, H, W]
        p_t = predicted_tensors[i]  # [1, H, W]

        c_gt   = loss_fn._to_contour(m_t)[0].cpu().numpy()
        c_pred = loss_fn._to_contour(p_t)[0].cpu().numpy()

        m_np = m_t[0].cpu().numpy()
        p_np = p_t[0].cpu().numpy()

        axes[i, 0].imshow(_norm(m_np), cmap="inferno", vmin=0, vmax=1)
        _overlay(axes[i, 1], m_np, c_gt,   color_rgb=(0.0, 1.0, 1.0))  # cyan  — GT contour
        axes[i, 2].imshow(_norm(p_np), cmap="inferno", vmin=0, vmax=1)
        _overlay(axes[i, 3], p_np, c_pred, color_rgb=(1.0, 0.3, 0.0))  # orange-red — pred contour

        axes[i, 0].set_ylabel(f"#{i+1}", fontsize=8, rotation=0, labelpad=16, va="center")
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    mrad_str = f"{mean_mrad:.3f}" if not np.isnan(mean_mrad) else "NaN"
    fig.suptitle(
        f"{role.upper()} heliostat — {heliostat_id}   |   FSE = {mrad_str} mrad   |   "
        f"Contour overlay ({n_show} samples)",
        fontsize=10, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out_path = output_dir / f"contour_overlay_{role}_{heliostat_id}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step-by-step pipeline visualization
# ---------------------------------------------------------------------------

def plot_pipeline_steps(
    measured_tensor,
    predicted_tensor,
    heliostat_id: str,
    mean_mrad: float,
    role: str,
    output_dir: pathlib.Path,
    contour_params: dict | None = None,
) -> None:
    """
    2-row × 6-column grid showing each preprocessing step for one sample.

    Row 0 = measured flux pipeline.
    Row 1 = predicted flux pipeline.
    Columns: Raw → Normalized → Smoothed → Thresholded → Eroded → Contour.

    Parameters
    ----------
    measured_tensor  : [1, H, W] float32 tensor (one sample)
    predicted_tensor : [1, H, W] float32 tensor (one sample)
    heliostat_id     : used for title and filename
    mean_mrad        : post-training FSE in mrad
    role             : "best" or "worst"
    output_dir       : destination directory
    contour_params   : kwargs for ContourLoss (uses defaults if None)
    """
    import sys, pathlib as _pl
    _src = _pl.Path(__file__).resolve().parents[1]
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from artist_extensions.loss_functions_ext import ContourLoss

    loss_fn = ContourLoss(**(contour_params or {}))
    meas_steps = loss_fn.get_intermediate_steps(measured_tensor)
    pred_steps = loss_fn.get_intermediate_steps(predicted_tensor)

    n_cols = len(meas_steps)
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 2.2, 5.2))
    fig.patch.set_facecolor("white")

    row_labels = ["Measured", "Predicted"]
    for row, (row_label, steps) in enumerate([(row_labels[0], meas_steps), (row_labels[1], pred_steps)]):
        for col, (step_name, img) in enumerate(steps):
            ax = axes[row, col]
            # Last column (contour) use a hot colormap to make the edge pop
            cmap = "hot" if col == n_cols - 1 else "inferno"
            mn, mx = img.min(), img.max()
            img_norm = (img - mn) / max(mx - mn, 1e-12)
            ax.imshow(img_norm, cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(step_name, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(row_label, fontsize=9, fontweight="bold")

    mrad_str = f"{mean_mrad:.3f}" if not np.isnan(mean_mrad) else "NaN"
    fig.suptitle(
        f"Contour pipeline — {role.upper()} heliostat {heliostat_id}   |   FSE = {mrad_str} mrad",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    out_path = output_dir / f"pipeline_steps_{role}_{heliostat_id}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Centroid trail grid  (Stage-2 training samples + per-epoch predicted centroid)
# ---------------------------------------------------------------------------

def _enu_to_pixel(
    enu: "list[float]",
    center: "list[float]",
    width: float,
    height: float,
    bitmap_w: int = 256,
    bitmap_h: int = 256,
) -> "tuple[float, float]":
    """Convert a planar-target ENU point to bitmap pixel (col, row).

    Inverse of ARTIST's ``bitmap_coordinates_to_target_coordinates`` for planar areas:
        ENU = center + (0.5 - e_norm) * width * [1,0,0]
                     + (0.5 - u_norm) * height * [0,0,1]
    so:
        e_norm = 0.5 - (ENU[0] - center[0]) / width
        u_norm = 0.5 - (ENU[2] - center[2]) / height
        col    = e_norm * bitmap_w - 0.5
        row    = u_norm * bitmap_h - 0.5
    """
    e_norm = 0.5 - (enu[0] - center[0]) / width
    u_norm = 0.5 - (enu[2] - center[2]) / height
    col    = e_norm * bitmap_w - 0.5
    row    = u_norm * bitmap_h - 0.5
    return col, row


def plot_centroid_trail_grids(
    trail_dir: pathlib.Path,
    hid: str,
    flux_images: "list[np.ndarray]",
    gt_centroids_enu: "list[list[float]]",
    trail_epochs: "list[int]",
    trail_centroids_enu: "dict[int, list[list[float]]]",
    target_centers: "list[list[float]]",
    target_dims_list: "list[list[float]]",
    dist_m: float = 1.0,
    bitmap_res: int = 256,
) -> None:
    """Save a grid of training flux images with Stage-2 centroid trail overlaid.

    Each subplot shows one training sample as background (inferno colormap).
    Scatter dots show where the predicted centroid was at each captured epoch,
    coloured by epoch (RdYlGn: red = early, green = late).
    A white star marks the GT centroid position.
    A small text label in the bottom-right shows the mrad FSE at the final epoch.

    Parameters
    ----------
    trail_dir          : directory to write ``{hid}_trail.png``
    hid                : heliostat identifier (used in title and filename)
    flux_images        : list of H×W float32 arrays in [0, 1]
    gt_centroids_enu   : list of [E, N, U] GT centroid positions per sample
    trail_epochs       : sorted list of epochs at which centroids were captured
    trail_centroids_enu: {epoch: [[E,N,U], ...]} per captured epoch
    target_centers     : per-sample list of [E, N, U] target area centres
    target_dims_list   : per-sample list of [width_m, height_m]
    dist_m             : heliostat-to-tower distance in metres (for mrad computation)
    bitmap_res         : edge length of the square flux bitmap in pixels (default 256)
    """
    n_samples = len(flux_images)
    if n_samples == 0 or not trail_epochs:
        return

    # Grid layout: up to 5 columns, enough rows to fit all samples
    n_cols = min(5, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.0, n_rows * 2.0 + 0.6),
    )
    fig.patch.set_facecolor("white")

    # Normalise axes to always be 2-D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # Colormap: red = epoch 0, green = last epoch
    cmap      = plt.cm.RdYlGn
    n_epochs  = len(trail_epochs)
    epoch_min = trail_epochs[0]
    epoch_max = trail_epochs[-1]
    epoch_range = max(epoch_max - epoch_min, 1)

    for i in range(n_rows * n_cols):
        row = i // n_cols
        col = i  % n_cols
        ax  = axes[row, col]

        if i >= n_samples:
            ax.axis("off")
            continue

        # Background: GT training flux image
        ax.imshow(flux_images[i], cmap="inferno", vmin=0, vmax=1,
                  extent=[0, bitmap_res, bitmap_res, 0])

        # Per-sample target area geometry
        t_center = target_centers[i]
        t_w, t_h = target_dims_list[i][0], target_dims_list[i][1]

        # Trail dots: one per captured epoch
        for t_idx, ep in enumerate(trail_epochs):
            cents = trail_centroids_enu.get(ep, [])
            if i >= len(cents):
                continue
            px_col, px_row = _enu_to_pixel(
                cents[i], t_center, t_w, t_h, bitmap_res, bitmap_res
            )
            t_norm = (ep - epoch_min) / epoch_range
            ax.scatter(
                px_col, px_row,
                c=[cmap(t_norm)],
                s=12,
                linewidths=0,
                zorder=3,
            )

        # GT centroid: white star
        if i < len(gt_centroids_enu):
            gt_col, gt_row = _enu_to_pixel(
                gt_centroids_enu[i], t_center, t_w, t_h, bitmap_res, bitmap_res
            )
            ax.scatter(gt_col, gt_row, marker="*", c="white", s=40,
                       linewidths=0.5, edgecolors="black", zorder=4)

        # mrad label: FSE at the final captured epoch
        final_ep = trail_epochs[-1]
        final_cents = trail_centroids_enu.get(final_ep, [])
        if i < len(final_cents) and i < len(gt_centroids_enu):
            pred_c = np.array(final_cents[i][:3], dtype=np.float64)
            gt_c   = np.array(gt_centroids_enu[i][:3], dtype=np.float64)
            mrad_val = float(np.linalg.norm(pred_c - gt_c) / max(dist_m, 1e-6) * 1000)
            ax.text(
                bitmap_res - 2, bitmap_res - 2,
                f"{mrad_val:.4f} mrad",
                ha="right", va="bottom",
                fontsize=4.5, color="white",
                bbox=dict(facecolor="black", alpha=0.55, pad=1, edgecolor="none"),
                zorder=5,
            )

        ax.set_xlim(0, bitmap_res)
        ax.set_ylim(bitmap_res, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    # Colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=epoch_min, vmax=epoch_max),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02)
    cbar.set_label("Stage-2 epoch", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Centroid trail — {hid}   "
        f"(red = epoch {epoch_min}, green = epoch {epoch_max} | ★ = GT)",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])

    out_path = trail_dir / f"{hid}_trail.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
