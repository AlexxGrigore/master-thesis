"""
Reporting and visualisation for the full-63-heliostat kinematic reconstruction experiment.

Output files
------------
flux_grid_best_{hid}.png          — 10×5-pair grid of measured|predicted for best heliostat
flux_grid_worst_{hid}.png         — same for worst heliostat
field_accuracy_map.png            — ENU field scatter plot coloured by post-training accuracy
summary_table.png                 — clean table: val + test mean/median mrad
contour_components_train.png      — per-component train loss over epochs (contour loss only)
contour_components_val.png        — per-component val loss over epochs (contour loss only)
contour_overlay_best_{hid}.png    — GT flux + GT/pred contour overlaid for best heliostat
contour_overlay_worst_{hid}.png   — same for worst heliostat
pipeline_steps_best_{hid}.png     — step-by-step contour pipeline for best heliostat
pipeline_steps_worst_{hid}.png    — same for worst heliostat
flux_grids_all/{hid}.png          — 50×2 GT|predicted grid for every heliostat
"""
import json
import pathlib

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ---------------------------------------------------------------------------
# Flux grid  (10 rows × 5 pairs, each pair = measured | predicted)
# ---------------------------------------------------------------------------

def plot_flux_grid(
    measured: "list[np.ndarray]",
    predicted: "list[np.ndarray]",
    heliostat_id: str,
    mean_mrad: float,
    role: str,
    output_dir: pathlib.Path,
) -> None:
    """
    Save a grid of flux image pairs for a single heliostat.

    Layout: 10 rows × 5 pairs.  Each pair occupies two adjacent subplot columns
    (left = measured, right = predicted).  Up to 50 samples are shown.

    Parameters
    ----------
    measured     : list of H×W float arrays (peak-normalised [0, 1])
    predicted    : list of H×W float arrays (peak-normalised [0, 1])
    heliostat_id : heliostat name for the title and filename
    mean_mrad    : post-training focal-spot error in mrad
    role         : "best" or "worst"
    output_dir   : directory where the PNG is saved
    """
    n_samples   = min(len(measured), len(predicted), 50)
    n_pairs_col = 5    # pairs per row
    n_rows      = 10   # rows of pairs
    n_img_cols  = n_pairs_col * 2   # each pair = 2 image columns

    fig = plt.figure(figsize=(n_img_cols * 1.1, n_rows * 1.15))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        n_rows, n_img_cols,
        figure=fig,
        wspace=0.04,
        hspace=0.08,
        left=0.02, right=0.98,
        top=0.90, bottom=0.02,
    )

    for i in range(n_samples):
        row     = i // n_pairs_col
        pair    = i  % n_pairs_col
        col_m   = pair * 2
        col_p   = pair * 2 + 1

        ax_m = fig.add_subplot(gs[row, col_m])
        ax_p = fig.add_subplot(gs[row, col_p])

        ax_m.imshow(measured[i],  cmap="inferno", vmin=0, vmax=1)
        ax_p.imshow(predicted[i], cmap="inferno", vmin=0, vmax=1)

        for ax in (ax_m, ax_p):
            ax.set_xticks([])
            ax.set_yticks([])

        # Label the sample index on the left column of each pair
        ax_m.set_ylabel(f"{i+1}", fontsize=6, rotation=0, labelpad=8,
                        va="center", ha="right")

    # Column-header labels: "M / P" above each pair
    for pair in range(n_pairs_col):
        col_m = pair * 2
        col_p = pair * 2 + 1
        # Use the first-row axes positions to place text via figure coords
        ax_m0 = fig.add_subplot(gs[0, col_m])
        ax_p0 = fig.add_subplot(gs[0, col_p])
        ax_m0.set_xticks([])
        ax_m0.set_yticks([])
        ax_p0.set_xticks([])
        ax_p0.set_yticks([])

    # Re-draw sample 0 (first row already plotted above — GridSpec reuse is fine)
    # Title
    role_str = role.upper()
    mrad_str = f"{mean_mrad:.3f}" if not np.isnan(mean_mrad) else "NaN"
    fig.suptitle(
        f"{role_str} heliostat — {heliostat_id}   |   "
        f"Post-training FSE = {mrad_str} mrad   |   "
        f"Left = measured, Right = predicted   (50 test samples, 10 rows × 5 pairs)",
        fontsize=9,
        fontweight="bold",
        y=0.97,
    )

    out_path = output_dir / f"flux_grid_{role}_{heliostat_id}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
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

    Columns: Split | Mean (mrad) | Median (mrad)
    Rows   : Validation, Test

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

    rows = [
        ["Validation",
         _fmt(val_eval,  "mean_mrad"),
         _fmt(val_eval,  "median_mrad")],
        ["Test",
         _fmt(test_eval, "mean_mrad"),
         _fmt(test_eval, "median_mrad")],
    ]
    col_headers = ["Split", "Mean (mrad)", "Median (mrad)"]

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
        "Post-training accuracy",
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
# Per-heliostat 50×2 flux grid (all heliostats)
# ---------------------------------------------------------------------------

def plot_all_heliostats_flux_grid(
    hel_data: dict,
    output_dir: pathlib.Path,
) -> None:
    """
    Save one PNG per heliostat: 50 rows × 2 columns (measured | predicted).

    Files are written to output_dir/flux_grids_all/{hid}.png.

    Parameters
    ----------
    hel_data   : dict  {hid: {"measured": list[H×W array], "predicted": list[H×W array],
                               "mean_mrad": float}}
    output_dir : experiment output directory
    """
    grid_dir = output_dir / "flux_grids_all"
    grid_dir.mkdir(parents=True, exist_ok=True)

    for hid, data in hel_data.items():
        measured  = data["measured"]
        predicted = data["predicted"]
        mean_mrad = data["mean_mrad"]

        n = min(len(measured), len(predicted), 50)
        if n == 0:
            continue

        fig, axes = plt.subplots(n, 2, figsize=(3.0, n * 0.75))
        fig.patch.set_facecolor("white")
        if n == 1:
            axes = axes[np.newaxis, :]

        for i in range(n):
            axes[i, 0].imshow(measured[i],  cmap="inferno", vmin=0, vmax=1)
            axes[i, 1].imshow(predicted[i], cmap="inferno", vmin=0, vmax=1)
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])
            axes[i, 0].set_ylabel(f"{i+1}", fontsize=5, rotation=0,
                                  labelpad=8, va="center", ha="right")

        axes[0, 0].set_title("Measured",  fontsize=8, fontweight="bold")
        axes[0, 1].set_title("Predicted", fontsize=8, fontweight="bold")

        mrad_str = f"{mean_mrad:.3f}" if not np.isnan(mean_mrad) else "NaN"
        fig.suptitle(f"{hid}   |   FSE = {mrad_str} mrad", fontsize=8,
                     fontweight="bold", y=1.002)
        fig.tight_layout(h_pad=0.05, w_pad=0.05)

        fig.savefig(grid_dir / f"{hid}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
