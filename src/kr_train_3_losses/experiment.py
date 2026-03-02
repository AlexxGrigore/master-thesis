import csv
import json
import logging
import pathlib

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary

from utils.evaluation import evaluate_flux_accuracy
from utils.plotting import (
    _style_ax,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TICK,
    FONT_TITLE,
    GRID_KW,
    plot_tracking_error_histogram,
    plot_training_curves,
    visualize_flux_comparison,
)

log = logging.getLogger(__name__)


def run_experiment(
    loss_name: str,
    loss_fn_factory,
    reconstructor_cls,
    ddp_setup: dict,
    device: torch.device,
    scenario_path: pathlib.Path,
    train_mapping: list,
    test_mapping: list,
    train_data_parser,
    eval_data_parser,
    optimization_configuration: dict,
    output_dir: pathlib.Path,
    save_figures: bool = False,
) -> dict:
    """
    Run one training + evaluation experiment for a given loss function.

    Each call reloads the scenario from disk (fresh kinematic parameters),
    trains with the provided loss, evaluates on the test set, and saves all
    outputs to output_dir / loss_name.

    Returns the test metrics dict from evaluate_flux_accuracy.
    """
    exp_dir = output_dir / loss_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Per-experiment log file — only this experiment's training logs go here.
    exp_log_handler = logging.FileHandler(exp_dir / "training.log")
    exp_log_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    )
    logging.getLogger().addHandler(exp_log_handler)

    try:
        log.info(f"=== Starting experiment: {loss_name} ===")

        # Reload scenario so every experiment starts from the same
        # un-optimised kinematic parameters.
        with h5py.File(scenario_path, "r") as scenario_file:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file,
                device=device,
                number_of_surface_points_per_facet=torch.tensor([25, 25]),
            )

        print(f"  Heliostats: {scenario.heliostat_field.number_of_heliostats_per_group.sum().item()}")

        data = {
            config_dictionary.data_parser: train_data_parser,
            config_dictionary.heliostat_data_mapping: train_mapping,
        }

        reconstructor = reconstructor_cls(
            ddp_setup=ddp_setup,
            scenario=scenario,
            data=data,
            optimization_configuration=optimization_configuration,
            reconstruction_method=config_dictionary.kinematic_reconstruction_raytracing,
        )

        loss_definition = loss_fn_factory(scenario)
        print(f"  Loss: {loss_definition.__class__.__name__}")

        final_loss_per_heliostat = reconstructor.reconstruct_kinematic(
            loss_definition=loss_definition, device=device
        )

        plot_training_curves(log_file=exp_dir / "training.log", output_dir=exp_dir)

        # ---- Final training loss distribution ----
        valid_losses = final_loss_per_heliostat[final_loss_per_heliostat != float("inf")]
        if len(valid_losses) > 0:
            losses_np = valid_losses.detach().cpu().numpy()
            mean_loss = losses_np.mean()
            median_loss = float(np.median(losses_np))
            std_loss = losses_np.std()
            print(f"  Train — mean loss: {mean_loss:.6f}, "
                  f"median: {median_loss:.6f}, "
                  f"min: {losses_np.min():.6f}, max: {losses_np.max():.6f}")

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.patch.set_facecolor("white")
            fig.suptitle(f"Training Loss Summary — {loss_name}",
                         fontsize=FONT_TITLE, fontweight="bold")

            # Left: histogram
            axes[0].hist(losses_np, bins=30, edgecolor="white", linewidth=0.5,
                         alpha=0.85, color="steelblue")
            axes[0].axvline(mean_loss, color="crimson", linestyle="--", linewidth=2.0,
                            label=f"Mean:   {mean_loss:.4f}")
            axes[0].axvline(median_loss, color="darkorange", linestyle="-.", linewidth=2.0,
                            label=f"Median: {median_loss:.4f}")
            axes[0].axvspan(mean_loss - std_loss, mean_loss + std_loss,
                            alpha=0.10, color="crimson", label=f"±1 std: {std_loss:.4f}")
            axes[0].legend(fontsize=FONT_LEGEND, framealpha=0.85)
            axes[0].grid(axis="y", **GRID_KW)
            _style_ax(axes[0], "Final Loss", "Count", "Loss Distribution")

            # Right: sorted losses
            sorted_losses = np.sort(losses_np)
            axes[1].plot(sorted_losses, color="steelblue", linewidth=1.5,
                         marker="o", markersize=3, alpha=0.8)
            axes[1].axhline(mean_loss, color="crimson", linestyle="--", linewidth=1.5,
                            label=f"Mean: {mean_loss:.4f}")
            axes[1].legend(fontsize=FONT_LEGEND, framealpha=0.85)
            axes[1].grid(**GRID_KW)
            _style_ax(axes[1], "Heliostat Index (sorted by loss)", "Final Loss", "Sorted Final Losses")

            plt.tight_layout()
            plt.savefig(exp_dir / "loss_distribution.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        # ---- Test evaluation ----
        test_metrics = evaluate_flux_accuracy(
            scenario=scenario,
            heliostat_data_mapping=test_mapping,
            data_parser=eval_data_parser,
            device=device,
        )

        print(f"  Test  — mean focal spot error:   {test_metrics['mean_focal_spot_error_mrad']:.2f} mrad")
        print(f"  Test  — median focal spot error: {test_metrics['median_focal_spot_error_mrad']:.2f} mrad")

        # ---- Tracking error histogram ----
        plot_tracking_error_histogram(
            errors_mrad=test_metrics["all_errors_mrad"],
            output_path=exp_dir / "tracking_error_histogram.png",
            title=f"Heliostat Tracking Error — {loss_name} (Test Set)",
        )

        # ---- Flux visualizations ----
        visualize_flux_comparison(
            scenario=scenario,
            heliostat_data_mapping=test_mapping,
            data_parser=eval_data_parser,
            device=device,
            output_dir=exp_dir / "visualizations",
            num_samples=5,
            save_figures=save_figures,
        )

        # ---- Save kinematic parameters ----
        all_kinematic_params_json = {}
        for group_index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
            all_kinematic_params_json[f"group_{group_index}"] = {
                "heliostat_names": heliostat_group.names,
                "rotation_deviation_parameters": heliostat_group.kinematic.rotation_deviation_parameters.detach().cpu().tolist(),
                "actuator_parameters": heliostat_group.kinematic.actuators.optimizable_parameters.detach().cpu().tolist(),
            }
        with open(exp_dir / "all_kinematic_parameters.json", "w") as f:
            json.dump(all_kinematic_params_json, f, indent=2)

        # ---- Save test metrics ----
        metrics_to_save = {
            "mean_focal_spot_error_mrad": test_metrics["mean_focal_spot_error_mrad"],
            "median_focal_spot_error_mrad": test_metrics["median_focal_spot_error_mrad"],
            "min_focal_spot_error_mrad": test_metrics["min_focal_spot_error_mrad"],
            "max_focal_spot_error_mrad": test_metrics["max_focal_spot_error_mrad"],
            "num_samples_evaluated": test_metrics["num_samples_evaluated"],
            "per_heliostat": test_metrics["per_heliostat"],
        }
        with open(exp_dir / "test_metrics.json", "w") as f:
            json.dump(metrics_to_save, f, indent=2)

        log.info(f"=== Experiment '{loss_name}' done: {test_metrics['mean_focal_spot_error_mrad']:.2f} mrad ===")
        return test_metrics

    finally:
        logging.getLogger().removeHandler(exp_log_handler)
        exp_log_handler.close()


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

_LOSS_COLORS = {
    "focal_spot_loss": "#4C72B0",
    "pixel_loss": "#DD8452",
    "kl_divergence_loss": "#55A868",
}
_METRIC_LABELS = {
    "mean_focal_spot_error_mrad": "Mean",
    "median_focal_spot_error_mrad": "Median",
    "min_focal_spot_error_mrad": "Min",
    "max_focal_spot_error_mrad": "Max",
}


def save_comparison_report(all_results: dict, output_dir: pathlib.Path) -> None:
    """
    Produce a unified comparison report for all experiments inside
    ``output_dir/comparison/``:

    * ``comparison.json``  – structured summary
    * ``comparison.csv``   – flat table (easy to open in Excel / pandas)
    * ``comparison_chart.png`` – grouped bar chart + box-plot panel
    """
    if not all_results:
        log.warning("No results to compare – skipping comparison report.")
        return

    comp_dir = output_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # ---- Console table ------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("=" * 70)
    header = (
        f"{'Loss Function':<30} "
        f"{'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10} {'N':>6}"
    )
    print(header)
    print("-" * len(header))
    for loss_name, metrics in all_results.items():
        print(
            f"{loss_name:<30} "
            f"{metrics['mean_focal_spot_error_mrad']:>10.2f} "
            f"{metrics['median_focal_spot_error_mrad']:>10.2f} "
            f"{metrics['min_focal_spot_error_mrad']:>10.2f} "
            f"{metrics['max_focal_spot_error_mrad']:>10.2f} "
            f"{metrics['num_samples_evaluated']:>6}"
        )
    print("=" * 70)

    # ---- JSON ---------------------------------------------------------
    summary = {
        name: {
            "mean_mrad": m["mean_focal_spot_error_mrad"],
            "median_mrad": m["median_focal_spot_error_mrad"],
            "min_mrad": m["min_focal_spot_error_mrad"],
            "max_mrad": m["max_focal_spot_error_mrad"],
            "n": m["num_samples_evaluated"],
        }
        for name, m in all_results.items()
    }
    with open(comp_dir / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved comparison JSON  → {comp_dir / 'comparison.json'}")

    # ---- CSV ----------------------------------------------------------
    csv_path = comp_dir / "comparison.csv"
    fieldnames = ["loss_function", "mean_mrad", "median_mrad", "min_mrad", "max_mrad", "n"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for loss_name, m in all_results.items():
            writer.writerow({
                "loss_function": loss_name,
                "mean_mrad": round(m["mean_focal_spot_error_mrad"], 4),
                "median_mrad": round(m["median_focal_spot_error_mrad"], 4),
                "min_mrad": round(m["min_focal_spot_error_mrad"], 4),
                "max_mrad": round(m["max_focal_spot_error_mrad"], 4),
                "n": m["num_samples_evaluated"],
            })
    print(f"Saved comparison CSV   → {csv_path}")

    # ---- Chart --------------------------------------------------------
    loss_names = list(all_results.keys())
    colors = [_LOSS_COLORS.get(n, "#999999") for n in loss_names]
    metric_keys = list(_METRIC_LABELS.keys())
    metric_display = list(_METRIC_LABELS.values())

    has_box = all("all_errors_mrad" in m for m in all_results.values())
    n_cols = 2 if has_box else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")
    fig.suptitle("Loss Function Comparison — mrad Accuracy",
                 fontsize=FONT_TITLE, fontweight="bold")

    # -- Subplot 1: grouped bar chart -----------------------------------
    ax_bar = axes[0]
    n_metrics = len(metric_keys)
    n_losses = len(loss_names)
    bar_width = 0.65 / n_losses
    group_centers = np.arange(n_metrics)

    for i, (loss_name, color) in enumerate(zip(loss_names, colors)):
        m = all_results[loss_name]
        values = [m[k] for k in metric_keys]
        offsets = (i - (n_losses - 1) / 2) * bar_width
        bars = ax_bar.bar(
            group_centers + offsets,
            values,
            width=bar_width * 0.9,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=loss_name.replace("_", " ").title(),
            alpha=0.88,
        )
        # value labels
        for bar in bars:
            h = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.3,
                f"{h:.1f}",
                ha="center", va="bottom",
                fontsize=FONT_TICK - 1,
                color="#333333",
            )

    ax_bar.set_xticks(group_centers)
    ax_bar.set_xticklabels(metric_display, fontsize=FONT_TICK)
    ax_bar.set_ylabel("Error (mrad)", fontsize=FONT_LABEL)
    ax_bar.set_title("Summary Statistics", fontsize=FONT_TITLE - 2)
    ax_bar.legend(fontsize=FONT_LEGEND, framealpha=0.85)
    ax_bar.grid(axis="y", **GRID_KW)
    ax_bar.set_facecolor("white")
    for spine in ["top", "right"]:
        ax_bar.spines[spine].set_visible(False)

    # -- Subplot 2: box plots per loss (per-heliostat errors) -----------
    if has_box:
        ax_box = axes[1]
        box_data = [np.asarray(all_results[n]["all_errors_mrad"]) for n in loss_names]
        bp = ax_box.boxplot(
            box_data,
            patch_artist=True,
            notch=False,
            widths=0.45,
            medianprops=dict(color="black", linewidth=2.0),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=3, alpha=0.4, linestyle="none"),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax_box.set_xticks(range(1, n_losses + 1))
        ax_box.set_xticklabels(
            [n.replace("_", "\n").replace(" loss", "\nloss") for n in loss_names],
            fontsize=FONT_TICK,
        )
        ax_box.set_ylabel("Error per heliostat (mrad)", fontsize=FONT_LABEL)
        ax_box.set_title("Per-Heliostat Error Distribution", fontsize=FONT_TITLE - 2)
        ax_box.grid(axis="y", **GRID_KW)
        ax_box.set_facecolor("white")
        for spine in ["top", "right"]:
            ax_box.spines[spine].set_visible(False)

    plt.tight_layout()
    chart_path = comp_dir / "comparison_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison chart → {chart_path}")
