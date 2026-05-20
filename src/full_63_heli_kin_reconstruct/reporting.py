"""
Reporting and visualisation for the full-63-heliostat kinematic reconstruction experiment.

Output files
------------
flux_grid_best_{hid}.png     — 10×5-pair grid of measured|predicted for the best heliostat
flux_grid_worst_{hid}.png    — same for the worst heliostat
field_accuracy_map.png       — ENU field scatter plot coloured by post-training accuracy
summary_table.png            — clean table: val + test mean/median mrad
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
