from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Feature layout constants (must match model.py INPUT_DIM = 42)
# ---------------------------------------------------------------------------

_FEATURE_NAMES: list[str] = (
    ["helpos_E", "helpos_N", "helpos_U"]
    + [f"kin_{i}" for i in range(20)]
    + ["mean_cen_E", "mean_cen_N", "mean_cen_U"]
    + ["std_cen_E",  "std_cen_N",  "std_cen_U"]
    + ["rng_cen_E",  "rng_cen_N",  "rng_cen_U"]
    + ["mean_sun_x", "mean_sun_y", "mean_sun_z"]
    + ["std_sun_x",  "std_sun_y",  "std_sun_z"]
    + ["mean_mtr_0", "mean_mtr_1"]
    + ["std_mtr_0",  "std_mtr_1"]
)

# (group_label, start_dim, end_dim, color)
_FEATURE_GROUPS: list[tuple[str, int, int, str]] = [
    ("helpos",    0,  3,  "#1f77b4"),
    ("kinematic", 3,  23, "#ff7f0e"),
    ("mean_cen",  23, 26, "#2ca02c"),
    ("std_cen",   26, 29, "#d62728"),
    ("rng_cen",   29, 32, "#9467bd"),
    ("mean_sun",  32, 35, "#8c564b"),
    ("std_sun",   35, 38, "#e377c2"),
    ("mean_mtr",  38, 40, "#7f7f7f"),
    ("std_mtr",   40, 42, "#bcbd22"),
]

# Input feature dimensions to sweep in response curve plots.
_RESPONSE_FEATURE_DIMS: list[tuple[int, str]] = [
    (0,  "Heliostat E [norm]"),
    (1,  "Heliostat N [norm]"),
    (23, "Mean centroid E [norm]"),
    (24, "Mean centroid N [norm]"),
    (35, "Std sun x [norm]"),
    (36, "Std sun y [norm]"),
]

# One representative output per Wortberg parameter group.
_RESPONSE_OUTPUT_DIMS: list[int] = [0, 9, 13, 15, 17]


# ---------------------------------------------------------------------------
# Shared model forward helper
# ---------------------------------------------------------------------------

def _model_forward_from_features(
    model: torch.nn.Module,
    features: torch.Tensor,
) -> torch.Tensor:
    """Forward pass directly from raw 42-D base features (handles linear, poly, and snn)."""
    device = next(model.parameters()).device
    f = features.to(device)
    if hasattr(model, "net"):  # SNN
        return torch.tanh(model.net(f)) * model.residual_bounds
    if hasattr(model, "degree"):  # poly
        f = torch.cat([f ** k for k in range(1, model.degree + 1)], dim=-1)
    return torch.tanh(model.linear(f)) * model.residual_bounds


def plot_loss_curves(
    *,
    history: list[dict[str, float]],
    test_loss_m: float,
    output_path: pathlib.Path,
) -> None:
    epochs = [int(record["epoch"]) for record in history]
    train_loss_m = [record["train_loss_m"] for record in history]
    validation_loss_m = [record["validation_mean_focal_spot_error_m"] for record in history]
    learning_rates = [record.get("learning_rate") for record in history]
    has_lr = all(lr is not None for lr in learning_rates)

    best_index = int(np.argmin(validation_loss_m))
    best_epoch = epochs[best_index]
    best_validation_loss = validation_loss_m[best_index]

    # Epochs where ReduceLROnPlateau fired.
    lr_drop_epochs = []
    if has_lr:
        for i in range(1, len(learning_rates)):
            if learning_rates[i] < learning_rates[i - 1]:
                lr_drop_epochs.append(epochs[i])

    n_rows = 2 if has_lr else 1
    height_ratios = [3, 1] if has_lr else [1]
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(10, 7 if has_lr else 5.5),
        gridspec_kw={"height_ratios": height_ratios},
        sharex=True,
    )
    fig.patch.set_facecolor("white")
    ax = axes[0] if has_lr else axes

    # --- Loss subplot ---
    ax.plot(epochs, train_loss_m, color="#1f77b4", linewidth=2.0, label="Train loss")
    ax.plot(epochs, validation_loss_m, color="#ff7f0e", linewidth=2.0, label="Validation loss")
    ax.axhline(
        test_loss_m,
        color="#d62728",
        linewidth=1.8,
        linestyle=":",
        label=f"Test loss ({test_loss_m:.4f} m)",
    )
    ax.scatter([best_epoch], [best_validation_loss], color="#ff7f0e", s=50, zorder=5)
    ax.annotate(
        f"Best epoch: {best_epoch}\nVal: {best_validation_loss:.4f} m",
        xy=(best_epoch, best_validation_loss),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9},
    )
    for i, ep in enumerate(lr_drop_epochs):
        ax.axvline(ep, color="#9467bd", linewidth=1.2, linestyle="--",
                   label="LR reduced" if i == 0 else None)
    ax.set_ylabel("Loss (m)")
    ax.set_title("Training, Validation, and Test Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(framealpha=0.9)

    # --- LR subplot ---
    if has_lr:
        ax_lr = axes[1]
        ax_lr.plot(epochs, learning_rates, color="#9467bd", linewidth=1.8)
        for ep in lr_drop_epochs:
            ax_lr.axvline(ep, color="#9467bd", linewidth=1.2, linestyle="--")
            new_lr = learning_rates[epochs.index(ep)]
            ax_lr.text(ep + 0.5, new_lr * 1.15, f"{new_lr:.1e}",
                       fontsize=7, color="#9467bd", va="bottom")
        ax_lr.set_yscale("log")
        ax_lr.set_xlabel("Epoch")
        ax_lr.set_ylabel("Learning rate")
        ax_lr.set_title("Learning Rate Schedule")
        ax_lr.grid(True, alpha=0.25)

    if not has_lr:
        ax.set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_vs_corrected_metrics(
    *,
    validation_baseline_metrics: dict[str, object],
    validation_best_metrics: dict[str, object],
    validation_last_metrics: dict[str, object],
    test_baseline_metrics: dict[str, object],
    test_best_metrics: dict[str, object],
    test_last_metrics: dict[str, object],
    output_path: pathlib.Path,
) -> None:
    del validation_last_metrics
    del test_last_metrics

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.patch.set_facecolor("white")

    panels = [
        (
            axes[0],
            "Validation Tracking Error",
            validation_baseline_metrics,
            validation_best_metrics,
            validation_last_metrics,
        ),
        (
            axes[1],
            "Test Tracking Error",
            test_baseline_metrics,
            test_best_metrics,
            test_last_metrics,
        ),
    ]

    for ax, title, baseline_metrics, best_metrics, _ in panels:
        table_rows = [
            [
                "Mean error (mrad)",
                f"{float(baseline_metrics['mean_focal_spot_error_mrad']):.3f}",
                f"{float(best_metrics['mean_focal_spot_error_mrad']):.3f}",
            ],
            [
                "Median error (mrad)",
                f"{float(baseline_metrics['median_focal_spot_error_mrad']):.3f}",
                f"{float(best_metrics['median_focal_spot_error_mrad']):.3f}",
            ],
        ]

        ax.axis("off")
        table = ax.table(
            cellText=table_rows,
            colLabels=["Metric", "Baseline", "Best checkpoint"],
            cellLoc="center",
            colLoc="center",
            loc="center",
            colWidths=[0.48, 0.23, 0.29],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.8)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#d0d7de")
            if row == 0:
                cell.set_facecolor("#e8eef7")
                cell.set_text_props(weight="bold")
            elif col == 0:
                cell.set_facecolor("#f7f7f7")

        ax.set_title(title)

    fig.suptitle("Baseline vs Best Checkpoint Metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_error_histogram(
    *,
    baseline_errors_mrad: list[float],
    corrected_errors_mrad: list[float],
    output_path: pathlib.Path,
) -> None:
    baseline = np.asarray(baseline_errors_mrad, dtype=float)
    corrected = np.asarray(corrected_errors_mrad, dtype=float)
    baseline = baseline[np.isfinite(baseline)]
    corrected = corrected[np.isfinite(corrected)]
    if baseline.size == 0 or corrected.size == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("white")
    bins = np.linspace(min(baseline.min(), corrected.min()), max(baseline.max(), corrected.max()), 30)
    ax.hist(baseline, bins=bins, density=True, alpha=0.35, color="#7f7f7f", label="Baseline")
    ax.hist(corrected, bins=bins, density=True, alpha=0.35, color="#1f77b4", label="Corrected (best)")
    ax.axvline(float(np.median(baseline)), color="#555555", linestyle="--", linewidth=1.6)
    ax.axvline(float(np.median(corrected)), color="#1f77b4", linestyle="--", linewidth=1.6)
    ax.set_xlabel("Tracking error (mrad)")
    ax.set_ylabel("Density")
    ax.set_title("Baseline vs Corrected Error Distribution")
    ax.grid(True, alpha=0.25)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_linear_weights_heatmap(
    *,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    parameter_names: tuple[str, ...],
    output_path: pathlib.Path,
) -> None:
    weight = linear_weight.detach().cpu().numpy()
    bias = linear_bias.detach().cpu().numpy()
    vmax = float(np.max(np.abs(weight)))
    if vmax == 0.0:
        vmax = 1.0

    # X-axis ticks: one per feature group in the aggregated 42-D input.
    # Layout: helpos(3) | kinematic(20) | mean_cen(3) std_cen(3) range_cen(3)
    #         mean_sun(3) slope(3) mean_motor(2) std_motor(2)
    _N_HEL = 3
    _N_KIN = 20
    section_starts = [0, _N_HEL, _N_HEL + _N_KIN,
                      _N_HEL + _N_KIN + 3,  # std_cen
                      _N_HEL + _N_KIN + 6,  # range_cen
                      _N_HEL + _N_KIN + 9,  # mean_sun
                      _N_HEL + _N_KIN + 12, # std_sun
                      _N_HEL + _N_KIN + 15, # mean_motor
                      _N_HEL + _N_KIN + 17, # std_motor
                      ]
    section_labels = ["helpos", "kin", "mean_cen", "std_cen",
                      "range_cen", "mean_sun", "std_sun", "mean_motor", "std_motor"]
    section_ticks = section_starts

    fig, (ax_heatmap, ax_bias) = plt.subplots(
        1,
        2,
        figsize=(14, 8),
        gridspec_kw={"width_ratios": [4.5, 1.2]},
    )
    fig.patch.set_facecolor("white")

    image = ax_heatmap.imshow(weight, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax_heatmap.set_xticks(section_ticks)
    ax_heatmap.set_xticklabels(section_labels, rotation=60, ha="right", fontsize=7)
    ax_heatmap.set_yticks(np.arange(len(parameter_names)))
    ax_heatmap.set_yticklabels(parameter_names)
    ax_heatmap.set_title("Linear Weight Matrix")
    fig.colorbar(image, ax=ax_heatmap, fraction=0.025, pad=0.02)

    ax_bias.barh(np.arange(len(parameter_names)), bias, color="#4c78a8")
    ax_bias.axvline(0.0, color="black", linewidth=1.0)
    ax_bias.set_yticks(np.arange(len(parameter_names)))
    ax_bias.set_yticklabels([])
    ax_bias.set_title("Bias")
    ax_bias.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_predicted_residual_boxplot(
    *,
    predicted_residuals: torch.Tensor,
    parameter_names: tuple[str, ...],
    output_path: pathlib.Path,
) -> None:
    residuals = predicted_residuals.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(14, 6.5))
    fig.patch.set_facecolor("white")
    ax.boxplot(residuals, showfliers=False)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(np.arange(1, len(parameter_names) + 1))
    ax.set_xticklabels(parameter_names, rotation=65, ha="right")
    ax.set_ylabel("Predicted residual value")
    ax.set_title("Predicted Residual Distribution by Parameter")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_response_curves(
    *,
    model: torch.nn.Module,
    calibration_inputs: dict,
    parameter_names: tuple[str, ...],
    output_path: pathlib.Path,
    n_grid: int = 120,
) -> None:
    """
    Partial dependence plot: for each selected input feature, sweep its value
    across the observed training range (all other features held at mean) and
    plot the predicted Wortberg corrections for selected output parameters.

    For a linear model the curves are straight lines by construction.
    Polynomial models (poly2/poly3/poly4) can show curvature — if the curves
    stay flat or straight, the higher-degree terms learned nothing useful.
    """
    model.eval()
    device = next(model.parameters()).device

    # Build 42-D feature matrix for all training heliostats.
    feature_rows = [
        model._select_and_flatten(inp).detach().cpu()
        for inp in calibration_inputs.values()
    ]
    if not feature_rows:
        return
    feature_matrix = torch.stack(feature_rows, dim=0)  # (N_hel, 42)
    baseline = feature_matrix.mean(dim=0)              # (42,)

    selected_out = _RESPONSE_OUTPUT_DIMS
    out_labels = [parameter_names[i] for i in selected_out]
    n_feat = len(_RESPONSE_FEATURE_DIMS)
    n_out = len(selected_out)

    fig, axes = plt.subplots(
        n_feat, n_out,
        figsize=(3.2 * n_out, 2.8 * n_feat),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for row, (feat_dim, feat_label) in enumerate(_RESPONSE_FEATURE_DIMS):
        feat_vals = feature_matrix[:, feat_dim]
        feat_min, feat_max = float(feat_vals.min()), float(feat_vals.max())
        grid = torch.linspace(feat_min, feat_max, n_grid)

        # Build batch: baseline repeated, with one feature swept.
        batch = baseline.unsqueeze(0).expand(n_grid, -1).clone()
        batch[:, feat_dim] = grid

        preds = _model_forward_from_features(model, batch).cpu()  # (n_grid, 20)

        for col, (out_dim, out_label) in enumerate(zip(selected_out, out_labels)):
            ax = axes[row, col]
            y = preds[:, out_dim].numpy()
            bound = float(model.residual_bounds[out_dim].cpu())

            ax.plot(grid.numpy(), y, color="#1f77b4", linewidth=1.8)
            ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
            ax.axhline(+bound, color="#d62728", linewidth=0.7, linestyle=":", alpha=0.5)
            ax.axhline(-bound, color="#d62728", linewidth=0.7, linestyle=":", alpha=0.5)

            # Rug plot: show where actual training heliostats lie on the x-axis.
            ax.scatter(
                feat_vals.numpy(),
                np.full(len(feat_vals), y.min() - 0.05 * (y.max() - y.min() + 1e-9)),
                marker="|", color="gray", s=25, alpha=0.6, zorder=0,
            )

            if row == 0:
                ax.set_title(out_label, fontsize=8)
            if col == 0:
                ax.set_ylabel(feat_label, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2)

    fig.suptitle("Response Curves (partial dependence per input feature)", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    *,
    model: torch.nn.Module,
    calibration_inputs: dict,
    parameter_names: tuple[str, ...],
    output_path: pathlib.Path,
) -> None:
    """
    Two-panel feature importance plot.

    Top panel — weight column norms:
        For each input feature dim j, the L2 norm of its column in the weight
        matrix (summed over output dims).  For poly models, one stacked bar per
        degree block shows how much each degree contributes.

    Bottom panel — gradient × input:
        Model-agnostic sensitivity: run a forward pass with gradients, backprop
        through mean absolute output, then scale grad by feature magnitude.
        Result: (42,) vector averaged over all training heliostats.

    Both panels share the same x-axis (42 feature dims) coloured by group.
    """
    model.eval()

    # ------------------------------------------------------------------
    # Build feature matrix  (N_hel, 42)
    # ------------------------------------------------------------------
    feature_rows = [
        model._select_and_flatten(inp).detach().cpu()
        for inp in calibration_inputs.values()
    ]
    if not feature_rows:
        return
    feature_matrix = torch.stack(feature_rows, dim=0)  # (N_hel, 42)
    n_feat = feature_matrix.shape[1]  # 42
    x_coords = np.arange(n_feat)

    # Group colours for bar coloring
    bar_colors = np.empty(n_feat, dtype=object)
    for _, start, end, color in _FEATURE_GROUPS:
        bar_colors[start:end] = color

    # ------------------------------------------------------------------
    # Panel 1 — weight column norms (linear/poly only; skipped for SNN)
    # ------------------------------------------------------------------
    has_linear_head = hasattr(model, "linear")
    if has_linear_head:
        W = model.linear.weight.detach().cpu()  # (20, expanded_dim)
        if hasattr(model, "degree"):
            degree = model.degree
            blocks = [W[:, k * n_feat:(k + 1) * n_feat] for k in range(degree)]
            block_norms = [b.norm(dim=0).numpy() for b in blocks]
            degree_colors = ["#4c78a8", "#f58518", "#e45756", "#72b7b2"][:degree]
        else:
            block_norms = [W.norm(dim=0).numpy()]
            degree_colors = ["#4c78a8"]
    else:
        block_norms = None
        degree_colors = []

    # ------------------------------------------------------------------
    # Panel 2 — gradient × input
    # ------------------------------------------------------------------
    device = next(model.parameters()).device
    x_grad = feature_matrix.clone().requires_grad_(True)
    preds = _model_forward_from_features(model, x_grad)   # uses model on device
    preds.abs().mean().backward()
    with torch.no_grad():
        grad_x_input = (x_grad.grad.abs() * x_grad.detach().abs()).mean(dim=0).cpu().numpy()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, (ax_w, ax_g) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.patch.set_facecolor("white")

    # Panel 1: stacked bar (one stack per degree) — or a note for SNN
    if has_linear_head:
        bottom = np.zeros(n_feat)
        for k, (norms, dcolor) in enumerate(zip(block_norms, degree_colors)):
            ax_w.bar(x_coords, norms, bottom=bottom, color=dcolor,
                     label=f"degree {k + 1}", alpha=0.85, width=0.8)
            bottom += norms
        ax_w.legend(fontsize=8, framealpha=0.9)
    else:
        ax_w.text(0.5, 0.5, "Weight column norms not available for SNN\n(no single linear input layer)",
                  ha="center", va="center", transform=ax_w.transAxes, fontsize=10, color="gray")

    for label, start, end, color in _FEATURE_GROUPS:
        ax_w.axvspan(start - 0.5, end - 0.5, alpha=0.07, color=color)
    for _, start, _, _ in _FEATURE_GROUPS[1:]:
        ax_w.axvline(start - 0.5, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)

    ax_w.set_ylabel("Weight column L2 norm")
    ax_w.set_title("Feature Importance — Weight Column Norms")
    ax_w.grid(axis="y", alpha=0.2)

    # Panel 2: coloured bars by group
    ax_g.bar(x_coords, grad_x_input, color=list(bar_colors), alpha=0.85, width=0.8)
    for label, start, end, color in _FEATURE_GROUPS:
        ax_g.axvspan(start - 0.5, end - 0.5, alpha=0.07, color=color)
    for _, start, _, _ in _FEATURE_GROUPS[1:]:
        ax_g.axvline(start - 0.5, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)

    ax_g.set_ylabel("|grad| × |input| (mean over heliostats)")
    ax_g.set_title("Feature Importance — Gradient × Input Sensitivity")
    ax_g.grid(axis="y", alpha=0.2)

    # Shared x-axis: group-centre ticks
    group_centers = [(start + end) / 2 - 0.5 for _, start, end, _ in _FEATURE_GROUPS]
    group_labels  = [label for label, _, _, _ in _FEATURE_GROUPS]
    ax_g.set_xticks(group_centers)
    ax_g.set_xticklabels(group_labels, rotation=30, ha="right", fontsize=8)
    ax_g.set_xlim(-0.5, n_feat - 0.5)

    # Legend for group colours (proxy patches)
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, alpha=0.6, label=lbl)
               for lbl, _, _, c in _FEATURE_GROUPS]
    ax_g.legend(handles=patches, fontsize=7, ncol=3, framealpha=0.9,
                loc="upper right", title="Feature group")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_per_heliostat_improvement_scatter(
    *,
    baseline_per_heliostat: dict[str, dict[str, float | None]],
    corrected_per_heliostat: dict[str, dict[str, float | None]],
    output_path: pathlib.Path,
) -> None:
    heliostat_ids = sorted(set(baseline_per_heliostat) & set(corrected_per_heliostat))
    baseline_values = []
    corrected_values = []
    for heliostat_id in heliostat_ids:
        baseline_value = baseline_per_heliostat[heliostat_id].get("focal_spot_error_mrad")
        corrected_value = corrected_per_heliostat[heliostat_id].get("focal_spot_error_mrad")
        if baseline_value is None or corrected_value is None:
            continue
        baseline_values.append(float(baseline_value))
        corrected_values.append(float(corrected_value))

    if not baseline_values:
        return

    baseline_array = np.asarray(baseline_values)
    corrected_array = np.asarray(corrected_values)
    min_value = float(min(baseline_array.min(), corrected_array.min()))
    max_value = float(max(baseline_array.max(), corrected_array.max()))
    improved_count = int(np.sum(corrected_array < baseline_array))
    worsened_count = int(np.sum(corrected_array >= baseline_array))

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    fig.patch.set_facecolor("white")
    ax.scatter(baseline_array, corrected_array, color="#1f77b4", alpha=0.8)
    ax.plot([min_value, max_value], [min_value, max_value], color="#d62728", linestyle="--", linewidth=1.5)
    ax.text(
        0.03,
        0.97,
        f"Improved: {improved_count}\nWorsened or equal: {worsened_count}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9},
    )
    ax.set_xlabel("Baseline heliostat error (mrad)")
    ax.set_ylabel("Corrected heliostat error (mrad)")
    ax.set_title("Per-Heliostat Improvement Scatter")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)