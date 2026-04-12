import pathlib
from typing import Any

import numpy as np
from matplotlib import pyplot as plt


def plot_convergence(history: list[dict[str, Any]], output_path: pathlib.Path, title: str) -> None:
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["loss"] for entry in history]
    eval_epochs = [entry["epoch"] for entry in history if "eval_loss" in entry]
    eval_loss = [entry["eval_loss"] for entry in history if "eval_loss" in entry]

    parameter_series = [
        ("rotation_deviation_mean_abs", "Rotation", "darkorange"),
        ("translation_deviation_mean_abs", "Translation", "steelblue"),
        ("actuator_angle_dev_mean_abs", "Actuator angle", "seagreen"),
        ("actuator_offset_dev_mean_abs", "Actuator offset", "crimson"),
        ("base_pos_dev_e_mean_abs", "Base pos E", "black"),
        ("base_pos_dev_n_mean_abs", "Base pos N", "dimgray"),
        ("base_pos_dev_u_mean_abs", "Base pos U", "silver"),
    ]
    gradient_series = [
        ("grad_rotation", "Rotation grad", "darkorange"),
        ("grad_act_angle", "Actuator angle grad", "seagreen"),
        ("grad_act_offset", "Actuator offset grad", "crimson"),
        ("grad_base_pos", "Base position grad", "black"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=12, fontweight="bold")

    axes[0].plot(epochs, train_loss, color="steelblue", linewidth=1.5, label="Train")
    if eval_epochs:
        axes[0].plot(eval_epochs, eval_loss, color="darkorange", linewidth=1.5, linestyle="--", label="Eval")
        axes[0].legend()
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.3)

    for key, label, color in parameter_series:
        if any(key in entry for entry in history):
            axes[1].plot(epochs, [entry.get(key, 0.0) for entry in history], label=label, color=color, linewidth=1.5)
    axes[1].set_ylabel("Mean |value|")
    axes[1].set_title("Parameter Magnitudes")
    axes[1].grid(alpha=0.3)
    if axes[1].lines:
        axes[1].legend(fontsize=8)

    for key, label, color in gradient_series:
        if any(key in entry for entry in history):
            axes[2].plot(epochs, [entry.get(key, 0.0) for entry in history], label=label, color=color, linewidth=1.5)
    axes[2].set_ylabel("Mean |grad|")
    axes[2].set_xlabel("Epoch")
    axes[2].set_title("Gradient Magnitudes")
    axes[2].grid(alpha=0.3)
    if axes[2].lines:
        axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stage_comparison(
    baseline_metrics: dict[str, Any],
    perturbed_metrics: dict[str, Any],
    recovered_metrics: dict[str, Any],
    output_path: pathlib.Path,
    title: str,
) -> None:
    stages = ["Baseline", "Perturbed", "Recovered"]
    mean_values = [
        baseline_metrics["mean_focal_spot_error_mrad"],
        perturbed_metrics["mean_focal_spot_error_mrad"],
        recovered_metrics["mean_focal_spot_error_mrad"],
    ]
    median_values = [
        baseline_metrics["median_focal_spot_error_mrad"],
        perturbed_metrics["median_focal_spot_error_mrad"],
        recovered_metrics["median_focal_spot_error_mrad"],
    ]

    x = np.arange(len(stages))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.bar(x - width / 2, mean_values, width=width, color="steelblue", label="Mean")
    ax.bar(x + width / 2, median_values, width=width, color="darkorange", label="Median")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Tracking error (mrad)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for xpos, value in zip(x - width / 2, mean_values):
        ax.text(xpos, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    for xpos, value in zip(x + width / 2, median_values):
        ax.text(xpos, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_bucket_pies(
    stage_bucket_summaries: dict[str, dict[str, Any]],
    output_path: pathlib.Path,
    title: str,
) -> None:
    stage_order = ["Baseline", "Perturbed", "Recovered"]
    bucket_order = ["<3 mrad", "3-5 mrad", "5-7 mrad", ">=7 mrad"]
    colors = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for ax, stage_name in zip(axes, stage_order):
        counts = [stage_bucket_summaries[stage_name]["counts"][bucket] for bucket in bucket_order]
        total = sum(counts)
        if total == 0:
            ax.text(0.5, 0.5, "No finite heliostat\nerrors", ha="center", va="center", fontsize=10)
            ax.axis("off")
            continue

        labels = [f"{bucket}\n(n={count})" for bucket, count in zip(bucket_order, counts)]
        ax.pie(
            counts,
            labels=labels,
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
            startangle=90,
            counterclock=False,
            textprops={"fontsize": 8},
        )
        missing = stage_bucket_summaries[stage_name]["num_missing_heliostats"]
        suffix = f"\nmissing={missing}" if missing > 0 else ""
        ax.set_title(f"{stage_name}{suffix}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_bucket_comparison(
    stage_bucket_summaries: dict[str, dict[str, Any]],
    output_path: pathlib.Path,
    title: str,
) -> None:
    stage_order = ["Baseline", "Perturbed", "Recovered"]
    bucket_order = ["<3 mrad", "3-5 mrad", "5-7 mrad", ">=7 mrad"]
    colors = {
        "Baseline": "#2a9d8f",
        "Perturbed": "#e76f51",
        "Recovered": "#264653",
    }

    x = np.arange(len(bucket_order))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    for idx, stage_name in enumerate(stage_order):
        counts = [stage_bucket_summaries[stage_name]["counts"][bucket] for bucket in bucket_order]
        positions = x + (idx - 1) * width
        bars = ax.bar(positions, counts, width=width, label=stage_name, color=colors[stage_name])
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(count),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_order)
    ax.set_ylabel("Number of heliostats")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _sanitize_bucket_name(bucket_label: str) -> str:
    return (
        bucket_label.replace("<", "lt_")
        .replace(">=", "gte_")
        .replace("-", "to")
        .replace(" ", "")
    )



def plot_representative_heliostat_parameter_comparison(
    representative_selection: dict[str, dict[str, Any] | None],
    stage_parameter_values: dict[str, dict[str, dict[str, dict[str, float]]]],
    parameter_group_specs: list[dict[str, Any]],
    output_dir: pathlib.Path,
    title_prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_stage_order = ["Perturbed", "Recovered"]
    delta_stage_display_names = {
        "Perturbed": "After perturbation",
        "Recovered": "After retraining",
    }
    delta_colors = {
        "Perturbed": "#e76f51",
        "Recovered": "#264653",
    }

    for bucket_label, selection in representative_selection.items():
        if selection is None:
            continue

        heliostat_id = selection["heliostat_id"]
        fig, axes = plt.subplots(
            len(parameter_group_specs),
            1,
            figsize=(12, 3.5 * len(parameter_group_specs)),
            squeeze=False,
        )
        fig.patch.set_facecolor("white")
        fig.suptitle(
            f"{title_prefix} — {heliostat_id} ({bucket_label})",
            fontsize=12,
            fontweight="bold",
        )

        for ax, spec in zip(axes[:, 0], parameter_group_specs):
            labels = spec["labels"]
            baseline_values = [
                stage_parameter_values["Baseline"][heliostat_id][spec["key"]][label]
                for label in labels
            ]
            perturbed_deltas = [
                stage_parameter_values["Perturbed"][heliostat_id][spec["key"]][label] - baseline_value
                for label, baseline_value in zip(labels, baseline_values)
            ]
            recovered_deltas = [
                stage_parameter_values["Recovered"][heliostat_id][spec["key"]][label] - baseline_value
                for label, baseline_value in zip(labels, baseline_values)
            ]

            x = np.arange(len(labels))
            width = 0.35
            bars_perturbed = ax.bar(
                x - width / 2,
                perturbed_deltas,
                width=width,
                color=delta_colors["Perturbed"],
                label=delta_stage_display_names["Perturbed"],
            )
            bars_recovered = ax.bar(
                x + width / 2,
                recovered_deltas,
                width=width,
                color=delta_colors["Recovered"],
                label=delta_stage_display_names["Recovered"],
            )

            for bars in (bars_perturbed, bars_recovered):
                for bar in bars:
                    value = bar.get_height()
                    va = "bottom" if value >= 0 else "top"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value,
                        f"{value:.4f}",
                        ha="center",
                        va=va,
                        fontsize=7,
                        rotation=90,
                    )

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel(spec["unit"])
            ax.set_title(
                f"{spec['key'].replace('_', ' ').title()} — delta to baseline"
            )
            ax.grid(alpha=0.3)
            ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--", alpha=0.9)
            ax.legend(fontsize=8)

            baseline_text = " | ".join(
                f"{label}={value:.4f}" for label, value in zip(labels, baseline_values)
            )
            ax.text(
                0.0,
                1.02,
                f"Baseline values: {baseline_text}",
                transform=ax.transAxes,
                fontsize=8,
                ha="left",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            output_dir / f"representative_{_sanitize_bucket_name(bucket_label)}_{heliostat_id}_delta_to_baseline.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
