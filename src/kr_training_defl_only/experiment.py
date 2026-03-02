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
