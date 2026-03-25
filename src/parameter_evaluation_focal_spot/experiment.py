import collections
import gc
import json
import logging
import pathlib
import time

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, index_mapping

from utils.checkpointing import save_kinematic_parameters
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


def _plot_convergence_curves(
    history: list,
    output_dir: pathlib.Path,
    bounds: dict,
) -> None:
    """
    Plot parameter convergence curves from reconstructor._convergence_history.

    One PNG per heliostat group with four rows:
      1. Training loss
      2. Joint translation and rotation deviation magnitudes
      3. Base position deviation magnitudes (δe, δn, δu)
      4. Actuator parameter deviation magnitudes (aᵢ, cᵢ)
    """
    if not history:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    by_group = collections.defaultdict(list)
    for entry in history:
        by_group[entry["group"]].append(entry)

    for group_idx, entries in sorted(by_group.items()):
        epochs = [e["epoch"] for e in entries]

        has_base_pos = entries and "base_pos_dev_e_mean_abs" in entries[0]
        nrows = 4 if has_base_pos else 3
        fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 * nrows), sharex=True)
        fig.patch.set_facecolor("white")
        fig.suptitle(
            f"Parameter Convergence — Group {group_idx}",
            fontsize=FONT_TITLE,
            fontweight="bold",
        )

        # --- Row 0: Loss ---
        eval_epochs = [e["epoch"] for e in entries if "eval_loss" in e]
        eval_losses = [e["eval_loss"] for e in entries if "eval_loss" in e]
        has_eval = bool(eval_epochs)

        axes[0].plot(
            epochs, [e["loss"] for e in entries],
            color="steelblue", linewidth=1.5,
            label="Train" if has_eval else None,
        )
        if has_eval:
            axes[0].plot(
                eval_epochs, eval_losses,
                color="darkorange", linewidth=1.5, linestyle="--", label="Eval (val)",
            )
            axes[0].legend(fontsize=FONT_LEGEND, framealpha=0.85)
        axes[0].grid(**GRID_KW)
        _style_ax(axes[0], "", "Loss", "Training Loss")

        # --- Row 1: Joint translation and rotation deviations ---
        axes[1].plot(
            epochs, [e["translation_deviation_mean_abs"] for e in entries],
            color="steelblue", linewidth=1.5, label="Translation (9 params)",
        )
        axes[1].plot(
            epochs, [e["rotation_deviation_mean_abs"] for e in entries],
            color="darkorange", linewidth=1.5, label="Rotation (4 params)",
        )
        axes[1].axhline(
            bounds["translation"], color="steelblue", linestyle="--",
            linewidth=1.0, alpha=0.5, label=f"Trans. bound ±{bounds['translation']} m",
        )
        axes[1].axhline(
            bounds["rotation"], color="darkorange", linestyle="--",
            linewidth=1.0, alpha=0.5, label=f"Rot. bound ±{bounds['rotation']} rad",
        )
        axes[1].legend(fontsize=FONT_LEGEND, framealpha=0.85)
        axes[1].grid(**GRID_KW)
        _style_ax(axes[1], "", "Mean |deviation|", "Joint Deviations")

        # --- Row 2: Base position deviations (only when trained) ---
        if has_base_pos:
            axes[2].plot(
                epochs, [e["base_pos_dev_e_mean_abs"] for e in entries],
                color="steelblue", linewidth=1.5, label="δe (East)",
            )
            axes[2].plot(
                epochs, [e["base_pos_dev_n_mean_abs"] for e in entries],
                color="darkorange", linewidth=1.5, label="δn (North)",
            )
            axes[2].plot(
                epochs, [e["base_pos_dev_u_mean_abs"] for e in entries],
                color="seagreen", linewidth=1.5, label="δu (Up)",
            )
            axes[2].axhline(
                bounds["base_position"], color="gray", linestyle="--",
                linewidth=1.0, alpha=0.5, label=f"Bound ±{bounds['base_position']} m",
            )
            axes[2].legend(fontsize=FONT_LEGEND, framealpha=0.85)
            axes[2].grid(**GRID_KW)
            _style_ax(axes[2], "", "Mean |deviation| (m)", "Base Position Deviations")

        # --- Row 3 (or 2 when no base position): Actuator deviations ---
        ax_act = axes[3] if has_base_pos else axes[2]
        ax_act.plot(
            epochs, [e["actuator_angle_dev_mean_abs"] for e in entries],
            color="steelblue", linewidth=1.5, label="aᵢ (initial angle)",
        )
        ax_act.plot(
            epochs, [e["actuator_offset_dev_mean_abs"] for e in entries],
            color="darkorange", linewidth=1.5, label="cᵢ (offset)",
        )
        ax_act.axhline(
            bounds["actuator_angle"], color="steelblue", linestyle="--",
            linewidth=1.0, alpha=0.5, label=f"Angle bound ±{bounds['actuator_angle']} rad",
        )
        ax_act.axhline(
            bounds["actuator_offset"], color="darkorange", linestyle="--",
            linewidth=1.0, alpha=0.5, label=f"Offset bound ±{bounds['actuator_offset']} m",
        )
        ax_act.legend(fontsize=FONT_LEGEND, framealpha=0.85)
        ax_act.grid(**GRID_KW)
        _style_ax(ax_act, "Epoch", "Mean |deviation|", "Actuator Deviations")

        plt.tight_layout()
        plt.savefig(
            output_dir / f"convergence_group_{group_idx}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)


def _plot_parameter_histograms(
    scenario,
    output_dir: pathlib.Path,
    bounds: dict,
) -> None:
    """
    Plot final-value histograms and bound saturation for all kinematic parameters.

    One sub-directory ``param_histograms/`` with four PNGs per heliostat group:
      - histograms_translation_group_{i}.png  — 3×3 grid, 9 translation deviations
      - histograms_rotation_group_{i}.png     — 2×2 grid, 4 rotation deviations
      - histograms_base_position_group_{i}.png— 1×3 grid, δe / δn / δu
      - histograms_actuators_group_{i}.png    — 2×2 grid, aᵢ and cᵢ deviations
    """
    hist_dir = output_dir / "param_histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)

    def _saturation_pct(values_np: np.ndarray, bound: float) -> float:
        return 100.0 * float((np.abs(values_np) >= 0.99 * bound).mean())

    def _draw_panel(ax, values_np: np.ndarray, bound: float, title: str, xlabel: str) -> None:
        sat = _saturation_pct(values_np, bound)
        ax.hist(values_np, bins=30, color="steelblue", edgecolor="white",
                linewidth=0.5, alpha=0.85)
        ax.axvline( bound, color="crimson", linestyle="--", linewidth=1.5)
        ax.axvline(-bound, color="crimson", linestyle="--", linewidth=1.5)
        ax.text(
            0.97, 0.95, f"Sat: {sat:.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=FONT_TICK, color="crimson",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )
        ax.grid(axis="y", **GRID_KW)
        _style_ax(ax, xlabel, "Count", title)

    for group_idx, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
        kinematic = heliostat_group.kinematic

        # ---- Translation deviations [N, 9] ----
        trans = kinematic.translation_deviation_parameters.detach().cpu().numpy()
        trans_names = [
            "Joint1 δe", "Joint1 δn", "Joint1 δu",
            "Joint2 δe", "Joint2 δn", "Joint2 δu",
            "Conc. δe",  "Conc. δn",  "Conc. δu",
        ]
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        fig.patch.set_facecolor("white")
        fig.suptitle(f"Translation Deviation Histograms — Group {group_idx}",
                     fontsize=FONT_TITLE, fontweight="bold")
        for i, (name, ax) in enumerate(zip(trans_names, axes.flat)):
            _draw_panel(ax, trans[:, i], bounds["translation"], name, "Deviation (m)")
        plt.tight_layout()
        plt.savefig(hist_dir / f"histograms_translation_group_{group_idx}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---- Rotation deviations [N, 4] ----
        rot = kinematic.rotation_deviation_parameters.detach().cpu().numpy()
        rot_names = [
            "Joint1 tilt N", "Joint1 tilt U",
            "Joint2 tilt E", "Joint2 tilt N",
        ]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.patch.set_facecolor("white")
        fig.suptitle(f"Rotation Deviation Histograms — Group {group_idx}",
                     fontsize=FONT_TITLE, fontweight="bold")
        for i, (name, ax) in enumerate(zip(rot_names, axes.flat)):
            _draw_panel(ax, rot[:, i], bounds["rotation"], name, "Deviation (rad)")
        plt.tight_layout()
        plt.savefig(hist_dir / f"histograms_rotation_group_{group_idx}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---- Base position deviations [N, 3] ----
        if hasattr(kinematic, "_base_position_deviation"):
            base = kinematic._base_position_deviation.detach().cpu().numpy()
            base_names = ["δe (East)", "δn (North)", "δu (Up)"]
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            fig.patch.set_facecolor("white")
            fig.suptitle(f"Base Position Deviation Histograms — Group {group_idx}",
                         fontsize=FONT_TITLE, fontweight="bold")
            for i, (name, ax) in enumerate(zip(base_names, axes.flat)):
                _draw_panel(ax, base[:, i], bounds["base_position"], name, "Deviation (m)")
            plt.tight_layout()
            plt.savefig(hist_dir / f"histograms_base_position_group_{group_idx}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

        # ---- Actuator deviations — aᵢ and cᵢ per actuator ----
        if hasattr(kinematic, "_initial_actuator_initial_angle"):
            a_dev = (
                kinematic.actuators.optimizable_parameters[
                    :, index_mapping.actuator_initial_angle, :
                ].detach().cpu().numpy()
                - kinematic._initial_actuator_initial_angle.cpu().numpy()
            )  # [N, 2]
            c_dev = (
                kinematic.actuators.non_optimizable_parameters[
                    :, index_mapping.actuator_offset, :
                ].detach().cpu().numpy()
                - kinematic._initial_actuator_offset.cpu().numpy()
            )  # [N, 2]

            panel_data = [
                (a_dev[:, 0], bounds["actuator_angle"],  "aᵢ — Actuator 0", "Deviation (rad)"),
                (a_dev[:, 1], bounds["actuator_angle"],  "aᵢ — Actuator 1", "Deviation (rad)"),
                (c_dev[:, 0], bounds["actuator_offset"], "cᵢ — Actuator 0", "Deviation (m)"),
                (c_dev[:, 1], bounds["actuator_offset"], "cᵢ — Actuator 1", "Deviation (m)"),
            ]
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            fig.patch.set_facecolor("white")
            fig.suptitle(f"Actuator Deviation Histograms — Group {group_idx}",
                         fontsize=FONT_TITLE, fontweight="bold")
            for (values, bound, title, xlabel), ax in zip(panel_data, axes.flat):
                _draw_panel(ax, values, bound, title, xlabel)
            plt.tight_layout()
            plt.savefig(hist_dir / f"histograms_actuators_group_{group_idx}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)


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
    train_position_deviation: bool = True,
    validation_mapping: list | None = None,
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

        scenario.set_number_of_rays(10)
        log.info("Number of rays set to 20.")

        print(f"  Heliostats: {scenario.heliostat_field.number_of_heliostats_per_group.sum().item()}")

        data = {
            config_dictionary.data_parser: train_data_parser,
            config_dictionary.heliostat_data_mapping: train_mapping,
        }

        eval_data = None
        if validation_mapping is not None:
            eval_data = {
                "data_parser": eval_data_parser,
                "heliostat_data_mapping": validation_mapping,
            }

        reconstructor = reconstructor_cls(
            ddp_setup=ddp_setup,
            scenario=scenario,
            train_position_deviation=train_position_deviation,
            data=data,
            optimization_configuration=optimization_configuration,
            reconstruction_method=config_dictionary.kinematic_reconstruction_raytracing,
            eval_data=eval_data,
        )

        loss_definition = loss_fn_factory(scenario)
        print(f"  Loss: {loss_definition.__class__.__name__}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        t_train_start = time.time()
        final_loss_per_heliostat = reconstructor.reconstruct_kinematic(
            loss_definition=loss_definition, device=device
        )
        train_time_s = time.time() - t_train_start
        train_peak_gpu_gb = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
        train_end_gpu_gb = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
        log.info(f"Training — time: {train_time_s/60:.1f} min ({train_time_s:.0f}s), peak GPU: {train_peak_gpu_gb:.2f} GB, end GPU: {train_end_gpu_gb:.2f} GB")
        print(f"  Training — time: {train_time_s/60:.1f} min ({train_time_s:.0f}s), peak GPU: {train_peak_gpu_gb:.2f} GB, end GPU: {train_end_gpu_gb:.2f} GB")

        plot_training_curves(log_file=exp_dir / "training.log", output_dir=exp_dir)

        # ---- Parameter convergence curves ----
        with open(exp_dir / "convergence_history.json", "w") as f:
            json.dump(reconstructor._convergence_history, f, indent=2)
        _plot_convergence_curves(
            history=reconstructor._convergence_history,
            output_dir=exp_dir,
            bounds={
                "translation":     reconstructor._BOUND_TRANSLATION_M,
                "rotation":        reconstructor._BOUND_ROTATION_RAD,
                "base_position":   reconstructor._BOUND_BASE_POSITION_M,
                "actuator_angle":  reconstructor._BOUND_ACTUATOR_ANGLE_RAD,
                "actuator_offset": reconstructor._BOUND_ACTUATOR_OFFSET_M,
            },
        )

        # ---- Parameter histograms and bound saturation ----
        _plot_parameter_histograms(
            scenario=scenario,
            output_dir=exp_dir,
            bounds={
                "translation":     reconstructor._BOUND_TRANSLATION_M,
                "rotation":        reconstructor._BOUND_ROTATION_RAD,
                "base_position":   reconstructor._BOUND_BASE_POSITION_M,
                "actuator_angle":  reconstructor._BOUND_ACTUATOR_ANGLE_RAD,
                "actuator_offset": reconstructor._BOUND_ACTUATOR_OFFSET_M,
            },
        )

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
        convergence_history = reconstructor._convergence_history
        del reconstructor
        gc.collect()
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        t_eval_start = time.time()
        test_metrics = evaluate_flux_accuracy(
            scenario=scenario,
            heliostat_data_mapping=test_mapping,
            data_parser=eval_data_parser,
            device=device,
        )
        eval_time_s = time.time() - t_eval_start
        eval_peak_gpu_gb = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
        eval_end_gpu_gb = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
        log.info(f"Evaluation — time: {eval_time_s/60:.1f} min ({eval_time_s:.0f}s), peak GPU: {eval_peak_gpu_gb:.2f} GB, end GPU: {eval_end_gpu_gb:.2f} GB")
        print(f"  Evaluation — time: {eval_time_s/60:.1f} min ({eval_time_s:.0f}s), peak GPU: {eval_peak_gpu_gb:.2f} GB, end GPU: {eval_end_gpu_gb:.2f} GB")

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
        save_kinematic_parameters(scenario, exp_dir / "all_kinematic_parameters.json")

        timing_stats = {
            "training_time_s": round(train_time_s, 1),
            "training_time_min": round(train_time_s / 60, 2),
            "training_peak_gpu_gb": round(train_peak_gpu_gb, 3),
            "training_end_gpu_gb": round(train_end_gpu_gb, 3),
            "evaluation_time_s": round(eval_time_s, 1),
            "evaluation_time_min": round(eval_time_s / 60, 2),
            "evaluation_peak_gpu_gb": round(eval_peak_gpu_gb, 3),
            "evaluation_end_gpu_gb": round(eval_end_gpu_gb, 3),
        }
        with open(exp_dir / "timing_stats.json", "w") as f:
            json.dump(timing_stats, f, indent=2)

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

        # ---- Attach extra info for cross-experiment comparison plots ----
        test_metrics["convergence_history"] = convergence_history
        test_metrics["heliostat_positions"] = {
            name: (
                hg.positions[i, 0].item(),
                hg.positions[i, 1].item(),
            )
            for hg in scenario.heliostat_field.heliostat_groups
            for i, name in enumerate(hg.names)
        }

        log.info(f"=== Experiment '{loss_name}' done: {test_metrics['mean_focal_spot_error_mrad']:.2f} mrad ===")
        return test_metrics

    finally:
        logging.getLogger().removeHandler(exp_log_handler)
        exp_log_handler.close()


# ---------------------------------------------------------------------------
# Cross-experiment comparison plots
# ---------------------------------------------------------------------------

_CONFIG_COLORS = [
    "#2196F3",  # A — blue
    "#FF9800",  # B — orange
    "#4CAF50",  # C — green
    "#9C27B0",  # D — purple
    "#F44336",  # E — red
]


def plot_parameter_comparison(
    all_metrics: dict,
    output_dir: pathlib.Path,
) -> None:
    """
    Generate cross-experiment comparison plots from the results of all 5 configs.

    Produces four figures in output_dir/comparison/:
      1. comparison_bar.png        — mean + median focal spot error per config
      2. comparison_boxplot.png    — per-heliostat error distribution per config
      3. comparison_loss_curves.png— 1×2: overlaid train loss | overlaid val loss
      4. comparison_field_map.png  — 5-panel field scatter, dot colour = error magnitude

    Parameters
    ----------
    all_metrics : dict
        {config_name: metrics_dict} as returned by run_experiment().
        Each metrics_dict must contain keys produced by run_experiment:
        'mean_focal_spot_error_mrad', 'median_focal_spot_error_mrad',
        'all_errors_mrad', 'per_heliostat', 'convergence_history',
        'heliostat_positions'.
    output_dir : pathlib.Path
        Root output directory.  Plots are saved to output_dir/comparison/.
    """
    comp_dir = output_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    config_names = list(all_metrics.keys())
    colors = _CONFIG_COLORS[: len(config_names)]

    # ------------------------------------------------------------------
    # 1. Bar chart — mean + median per config
    # ------------------------------------------------------------------
    means   = [all_metrics[c]["mean_focal_spot_error_mrad"]   for c in config_names]
    medians = [all_metrics[c]["median_focal_spot_error_mrad"] for c in config_names]

    x = np.arange(len(config_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    bars_mean   = ax.bar(x - width / 2, means,   width, label="Mean",   color=colors, alpha=0.85, edgecolor="white")
    bars_median = ax.bar(x + width / 2, medians, width, label="Median", color=colors, alpha=0.50, edgecolor="white")

    for bar in bars_mean:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=FONT_TICK)
    for bar in bars_median:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=FONT_TICK)

    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=15, ha="right", fontsize=FONT_TICK)
    ax.legend(fontsize=FONT_LEGEND)
    ax.grid(axis="y", **GRID_KW)
    _style_ax(ax, "Parameter configuration", "Focal spot error (mrad)",
              "Focal Spot Error by Parameter Configuration")
    plt.tight_layout()
    plt.savefig(comp_dir / "comparison_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Box plot — per-heliostat error distribution
    # ------------------------------------------------------------------
    # Build per-heliostat mean error for each config (one value per heliostat).
    per_heliostat_errors = []
    for c in config_names:
        ph = all_metrics[c]["per_heliostat"]
        vals = [v["focal_spot_error_mrad"] for v in ph.values() if v["focal_spot_error_mrad"] is not None]
        per_heliostat_errors.append(vals)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    bp = ax.boxplot(
        per_heliostat_errors,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for flier, color in zip(bp["fliers"], colors):
        flier.set_markerfacecolor(color)

    ax.set_xticks(range(1, len(config_names) + 1))
    ax.set_xticklabels(config_names, rotation=15, ha="right", fontsize=FONT_TICK)
    ax.grid(axis="y", **GRID_KW)
    _style_ax(ax, "Parameter configuration", "Focal spot error (mrad)",
              "Per-Heliostat Error Distribution by Configuration")
    plt.tight_layout()
    plt.savefig(comp_dir / "comparison_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Loss curves — overlaid train (left) and val (right)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Training and Validation Loss — All Configurations",
                 fontsize=FONT_TITLE, fontweight="bold")

    for c, color in zip(config_names, colors):
        history = all_metrics[c].get("convergence_history", [])
        if not history:
            continue
        epochs      = [e["epoch"] for e in history]
        train_loss  = [e["loss"]  for e in history]
        eval_epochs = [e["epoch"] for e in history if "eval_loss" in e]
        eval_loss   = [e["eval_loss"] for e in history if "eval_loss" in e]

        axes[0].plot(epochs, train_loss, color=color, linewidth=1.5, label=c)
        if eval_loss:
            axes[1].plot(eval_epochs, eval_loss, color=color, linewidth=1.5, label=c)

    for ax, title in zip(axes, ["Train Loss", "Validation Loss"]):
        ax.legend(fontsize=FONT_LEGEND, framealpha=0.85)
        ax.grid(**GRID_KW)
        _style_ax(ax, "Epoch", "Loss", title)

    plt.tight_layout()
    plt.savefig(comp_dir / "comparison_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Field map — 1×N scatter, dot colour = focal spot error
    # ------------------------------------------------------------------
    # Use positions from the first config (same field layout for all).
    heliostat_positions = all_metrics[config_names[0]].get("heliostat_positions", {})
    if not heliostat_positions:
        log.warning("No heliostat positions found; skipping field map.")
        return

    # Gather all per-heliostat errors across configs to build a shared colour scale.
    all_errors_flat = []
    for c in config_names:
        ph = all_metrics[c]["per_heliostat"]
        all_errors_flat += [v["focal_spot_error_mrad"] for v in ph.values()
                            if v["focal_spot_error_mrad"] is not None]
    vmin = float(np.nanpercentile(all_errors_flat, 5))
    vmax = float(np.nanpercentile(all_errors_flat, 95))

    n = len(config_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    fig.patch.set_facecolor("white")
    fig.suptitle("Per-Heliostat Focal Spot Error — Field Map",
                 fontsize=FONT_TITLE, fontweight="bold")

    cmap = plt.cm.RdYlGn_r  # red = high error, green = low error
    sc_last = None
    for ax, c in zip(axes[0], config_names):
        ph = all_metrics[c]["per_heliostat"]
        east, north, errors = [], [], []
        for name, vals in ph.items():
            if name in heliostat_positions and vals["focal_spot_error_mrad"] is not None:
                e, n_coord = heliostat_positions[name]
                east.append(e)
                north.append(n_coord)
                errors.append(vals["focal_spot_error_mrad"])

        # Background: all heliostats without errors in light grey.
        all_e = [v[0] for v in heliostat_positions.values()]
        all_n = [v[1] for v in heliostat_positions.values()]
        ax.scatter(all_e, all_n, s=4, color="#cccccc", zorder=1)

        sc = ax.scatter(east, north, c=errors, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=12, zorder=2, linewidths=0)
        sc_last = sc

        # Tower at origin.
        ax.scatter([0], [0], s=120, marker="*", color="black", zorder=5)

        ax.set_aspect("equal")
        ax.set_xlabel("East (m)", fontsize=FONT_TICK)
        ax.set_ylabel("North (m)", fontsize=FONT_TICK)
        ax.set_title(c, fontsize=FONT_LABEL, fontweight="bold")
        ax.tick_params(labelsize=FONT_TICK)
        ax.grid(True, alpha=0.2)

    # Shared colourbar.
    if sc_last is not None:
        fig.colorbar(sc_last, ax=axes[0].tolist(), label="Focal spot error (mrad)",
                     fraction=0.02, pad=0.04)

    plt.tight_layout()
    plt.savefig(comp_dir / "comparison_field_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
