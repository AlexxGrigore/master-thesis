from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch


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
    categories = ["Baseline", "Best", "Last"]
    x = np.arange(len(categories))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
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

    for ax, title, baseline_metrics, best_metrics, last_metrics in panels:
        mean_values = [
            float(baseline_metrics["mean_focal_spot_error_mrad"]),
            float(best_metrics["mean_focal_spot_error_mrad"]),
            float(last_metrics["mean_focal_spot_error_mrad"]),
        ]
        median_values = [
            float(baseline_metrics["median_focal_spot_error_mrad"]),
            float(best_metrics["median_focal_spot_error_mrad"]),
            float(last_metrics["median_focal_spot_error_mrad"]),
        ]

        mean_bars = ax.bar(x - width / 2, mean_values, width=width, color="#1f77b4", alpha=0.9, label="Mean")
        median_bars = ax.bar(x + width / 2, median_values, width=width, color="#ff7f0e", alpha=0.8, label="Median")

        for bars in (mean_bars, median_bars):
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{bar.get_height():.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlabel("Checkpoint")

    axes[0].set_ylabel("Tracking error (mrad)")
    axes[0].legend(framealpha=0.9)
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
                      _N_HEL + _N_KIN + 12, # slope
                      _N_HEL + _N_KIN + 15, # mean_motor
                      _N_HEL + _N_KIN + 17, # std_motor
                      ]
    section_labels = ["helpos", "kin", "mean_cen", "std_cen",
                      "range_cen", "mean_sun", "slope", "mean_motor", "std_motor"]
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