import gc
import json
import logging
import pathlib
import time

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from artist.core.loss_functions import FocalSpotLoss
from artist_extensions.loss_functions_ext import PixelLossL1
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary

from artist_extensions.kinematic_reconstructors import (
    WortbergKinematicReconstructor,
    WortbergPixelReconstructor,
)
from utils.checkpointing import save_kinematic_parameters
from utils.evaluation import compute_pixel_test_loss, evaluate_flux_accuracy
from utils.plotting import (
    _style_ax,
    FONT_LEGEND,
    FONT_TICK,
    GRID_KW,
    plot_tracking_error_histogram,
    plot_training_curves,
    visualize_flux_comparison,
)

log = logging.getLogger(__name__)

_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#E91E63"]


def run_experiment(
    loss_name: str,
    phase1_opt_config: dict,
    phase2_opt_config: dict,
    ddp_setup: dict,
    device: torch.device,
    scenario_path: pathlib.Path,
    train_mapping: list,
    test_mapping: list,
    train_data_parser,
    eval_data_parser,
    output_dir: pathlib.Path,
    surface_pts: int = 25,
    n_rays: int = 10,
    save_figures: bool = False,
    train_position_deviation: bool = True,
    validation_mapping: list | None = None,
) -> dict:
    """
    Two-phase kinematic reconstruction experiment on the 18 blur-ablation heliostats.

    Phase 1 — Focal spot pretraining (WortbergKinematicReconstructor + FocalSpotLoss).
    Phase 2 — Pixel loss fine-tuning (WortbergPixelReconstructor + PixelLossL1, blur_sigma=2.0).

    The scenario is loaded with `surface_pts × surface_pts` evaluation points per facet
    and `n_rays` per light source, making the resolution a first-class experiment parameter.

    Outputs saved to output_dir / loss_name:
      - phase1/{training.log, training_curves.png, convergence_history.json}
      - phase2/{training.log, training_curves.png, convergence_history.json, training_summary.json}
      - test_metrics.json, test_loss_values.json, timing_stats.json
      - tracking_error_histogram.png, all_kinematic_parameters.json
      - visualizations/flux_comparison_*.png
    """
    exp_dir = output_dir / loss_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    exp_log_handler = logging.FileHandler(exp_dir / "training.log")
    exp_log_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    )
    logging.getLogger().addHandler(exp_log_handler)

    try:
        log.info(f"=== Starting experiment: {loss_name} | surface_pts={surface_pts}, n_rays={n_rays} ===")

        with h5py.File(scenario_path, "r") as scenario_file:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file,
                device=device,
                number_of_surface_points_per_facet=torch.tensor([surface_pts, surface_pts]),
            )

        scenario.set_number_of_rays(n_rays)
        log.info(f"Resolution: surface_pts={surface_pts}, n_rays={n_rays}.")

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

        # ----------------------------------------------------------------
        # Phase 1 — focal spot pretraining
        # ----------------------------------------------------------------
        phase1_dir = exp_dir / "phase1"
        phase1_dir.mkdir(parents=True, exist_ok=True)

        phase1_log_handler = logging.FileHandler(phase1_dir / "training.log")
        phase1_log_handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
        )
        logging.getLogger().addHandler(phase1_log_handler)

        log.info("--- Phase 1: focal spot pretraining ---")
        print(f"\n  Phase 1 — FocalSpotLoss, max_epoch={phase1_opt_config[config_dictionary.max_epoch]}, "
              f"lr={phase1_opt_config[config_dictionary.initial_learning_rate]}")

        phase1_reconstructor = WortbergKinematicReconstructor(
            ddp_setup=ddp_setup,
            scenario=scenario,
            train_position_deviation=train_position_deviation,
            data=data,
            optimization_configuration=phase1_opt_config,
            reconstruction_method=config_dictionary.kinematics_reconstruction_raytracing,
            eval_data=eval_data,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        t_phase1_start = time.time()
        phase1_reconstructor.reconstruct_kinematics(
            loss_definition=FocalSpotLoss(scenario=scenario),
            device=device,
        )
        phase1_time_s = time.time() - t_phase1_start
        phase1_peak_gpu_gb = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
        phase1_end_gpu_gb = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
        log.info(f"Phase 1 — time: {phase1_time_s/60:.1f} min ({phase1_time_s:.0f}s), peak GPU: {phase1_peak_gpu_gb:.2f} GB, end GPU: {phase1_end_gpu_gb:.2f} GB")
        print(f"  Phase 1 — time: {phase1_time_s/60:.1f} min ({phase1_time_s:.0f}s), peak GPU: {phase1_peak_gpu_gb:.2f} GB, end GPU: {phase1_end_gpu_gb:.2f} GB")

        logging.getLogger().removeHandler(phase1_log_handler)
        phase1_log_handler.close()

        with open(phase1_dir / "convergence_history.json", "w") as f:
            json.dump(phase1_reconstructor._convergence_history, f, indent=2)

        del phase1_reconstructor
        gc.collect()
        torch.cuda.empty_cache()

        # ----------------------------------------------------------------
        # Phase 2 — pixel loss fine-tuning
        # ----------------------------------------------------------------
        phase2_dir = exp_dir / "phase2"
        phase2_dir.mkdir(parents=True, exist_ok=True)

        phase2_log_handler = logging.FileHandler(phase2_dir / "training.log")
        phase2_log_handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
        )
        logging.getLogger().addHandler(phase2_log_handler)

        log.info("--- Phase 2: pixel loss fine-tuning ---")
        print(f"\n  Phase 2 — PixelLoss, max_epoch={phase2_opt_config[config_dictionary.max_epoch]}, "
              f"lr={phase2_opt_config[config_dictionary.initial_learning_rate]}")

        phase2_reconstructor = WortbergPixelReconstructor(
            ddp_setup=ddp_setup,
            scenario=scenario,
            train_position_deviation=train_position_deviation,
            data=data,
            optimization_configuration=phase2_opt_config,
            reconstruction_method=config_dictionary.kinematics_reconstruction_raytracing,
            eval_data=eval_data,
            blur_sigma=2.0,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        t_phase2_start = time.time()
        phase2_final_loss = phase2_reconstructor.reconstruct_kinematics(
            loss_definition=PixelLossL1(scenario=scenario),
            device=device,
        )
        phase2_time_s = time.time() - t_phase2_start
        phase2_peak_gpu_gb = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
        phase2_end_gpu_gb = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
        log.info(f"Phase 2 — time: {phase2_time_s/60:.1f} min ({phase2_time_s:.0f}s), peak GPU: {phase2_peak_gpu_gb:.2f} GB, end GPU: {phase2_end_gpu_gb:.2f} GB")
        print(f"  Phase 2 — time: {phase2_time_s/60:.1f} min ({phase2_time_s:.0f}s), peak GPU: {phase2_peak_gpu_gb:.2f} GB, end GPU: {phase2_end_gpu_gb:.2f} GB")

        logging.getLogger().removeHandler(phase2_log_handler)
        phase2_log_handler.close()

        with open(phase2_dir / "convergence_history.json", "w") as f:
            json.dump(phase2_reconstructor._convergence_history, f, indent=2)

        loss_np = phase2_final_loss.detach().cpu().numpy()
        phase2_summary = {
            "num_heliostats_total": int(len(loss_np)),
            "num_nan_loss": int(np.isnan(loss_np).sum()),
            "num_inf_loss": int(np.isinf(loss_np).sum()),
            "num_zero_loss": int((loss_np == 0.0).sum()),
            "num_valid_loss": int(np.isfinite(loss_np).sum()),
            "mean_final_loss": float(np.nanmean(loss_np[np.isfinite(loss_np)])) if np.isfinite(loss_np).any() else None,
            "median_final_loss": float(np.nanmedian(loss_np[np.isfinite(loss_np)])) if np.isfinite(loss_np).any() else None,
        }
        with open(phase2_dir / "training_summary.json", "w") as f:
            json.dump(phase2_summary, f, indent=2)
        print(f"\n  Phase 2 training summary: "
              f"{phase2_summary['num_nan_loss']} NaN, "
              f"{phase2_summary['num_inf_loss']} inf, "
              f"{phase2_summary['num_zero_loss']} zero "
              f"(out of {phase2_summary['num_heliostats_total']} heliostats)")

        del phase2_reconstructor
        gc.collect()
        torch.cuda.empty_cache()

        # ----------------------------------------------------------------
        # Test evaluation
        # ----------------------------------------------------------------
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

        print(f"\n  Test  — mean focal spot error:   {test_metrics['mean_focal_spot_error_mrad']:.2f} mrad")
        print(f"  Test  — median focal spot error: {test_metrics['median_focal_spot_error_mrad']:.2f} mrad")

        # ----------------------------------------------------------------
        # Compute test losses for horizontal reference lines
        # ----------------------------------------------------------------
        phase1_test_loss = test_metrics["mean_focal_spot_error_m"]
        phase2_test_loss = compute_pixel_test_loss(
            scenario=scenario,
            heliostat_data_mapping=test_mapping,
            data_parser=eval_data_parser,
            device=device,
            blur_sigma=2.0,
        )
        with open(exp_dir / "test_loss_values.json", "w") as f:
            json.dump({
                "phase1_test_loss_focal_spot_m": phase1_test_loss,
                "phase2_test_loss_pixel_l1": phase2_test_loss,
            }, f, indent=2)

        # ----------------------------------------------------------------
        # Training curve plots (generated after test_loss is available)
        # ----------------------------------------------------------------
        plot_training_curves(
            log_file=phase1_dir / "training.log",
            output_dir=phase1_dir,
            test_loss=phase1_test_loss,
        )
        plot_training_curves(
            log_file=phase2_dir / "training.log",
            output_dir=phase2_dir,
            test_loss=phase2_test_loss,
        )

        plot_tracking_error_histogram(
            errors_mrad=test_metrics["all_errors_mrad"],
            output_path=exp_dir / "tracking_error_histogram.png",
            title=f"Heliostat Tracking Error — {loss_name} (Test Set)",
        )

        visualize_flux_comparison(
            scenario=scenario,
            heliostat_data_mapping=test_mapping,
            data_parser=eval_data_parser,
            device=device,
            output_dir=exp_dir / "visualizations",
            num_samples=5,
            save_figures=save_figures,
        )

        total_train_time_s = phase1_time_s + phase2_time_s
        timing_stats = {
            "phase1_time_s": round(phase1_time_s, 1),
            "phase1_time_min": round(phase1_time_s / 60, 2),
            "phase1_peak_gpu_gb": round(phase1_peak_gpu_gb, 3),
            "phase1_end_gpu_gb": round(phase1_end_gpu_gb, 3),
            "phase2_time_s": round(phase2_time_s, 1),
            "phase2_time_min": round(phase2_time_s / 60, 2),
            "phase2_peak_gpu_gb": round(phase2_peak_gpu_gb, 3),
            "phase2_end_gpu_gb": round(phase2_end_gpu_gb, 3),
            "total_training_time_s": round(total_train_time_s, 1),
            "total_training_time_min": round(total_train_time_s / 60, 2),
            "evaluation_time_s": round(eval_time_s, 1),
            "evaluation_time_min": round(eval_time_s / 60, 2),
            "evaluation_peak_gpu_gb": round(eval_peak_gpu_gb, 3),
            "evaluation_end_gpu_gb": round(eval_end_gpu_gb, 3),
        }
        with open(exp_dir / "timing_stats.json", "w") as f:
            json.dump(timing_stats, f, indent=2)

        metrics_to_save = {
            "mean_focal_spot_error_mrad": test_metrics["mean_focal_spot_error_mrad"],
            "median_focal_spot_error_mrad": test_metrics["median_focal_spot_error_mrad"],
            "min_focal_spot_error_mrad": test_metrics["min_focal_spot_error_mrad"],
            "max_focal_spot_error_mrad": test_metrics["max_focal_spot_error_mrad"],
            "num_samples_evaluated": test_metrics["num_samples_evaluated"],
            "num_nan_samples": test_metrics["num_nan_samples"],
            "nan_heliostat_ids": test_metrics["nan_heliostat_ids"],
            "per_heliostat": test_metrics["per_heliostat"],
        }
        with open(exp_dir / "test_metrics.json", "w") as f:
            json.dump(metrics_to_save, f, indent=2)

        save_kinematic_parameters(scenario, exp_dir / "all_kinematic_parameters.json")

        log.info(f"=== Experiment '{loss_name}' done: {test_metrics['mean_focal_spot_error_mrad']:.2f} mrad ===")
        return test_metrics

    finally:
        logging.getLogger().removeHandler(exp_log_handler)
        exp_log_handler.close()


def plot_resolution_comparison(
    all_metrics: dict[str, dict],
    output_dir: pathlib.Path,
) -> None:
    """
    Generate comparison plots across all resolution configurations.

    Produces two PNGs in output_dir/comparison/:
      - comparison_bar.png    — mean + median focal spot error per config
      - comparison_boxplot.png— per-heliostat error distribution per config
    """
    comp_dir = output_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    config_names = list(all_metrics.keys())
    colors = _COLORS[: len(config_names)]

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
    _style_ax(ax, "Resolution configuration", "Focal spot error (mrad)",
              "Focal Spot Error by Resolution Configuration")
    plt.tight_layout()
    plt.savefig(comp_dir / "comparison_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Box plot — per-heliostat error distribution
    # ------------------------------------------------------------------
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
        notch=False,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(config_names) + 1))
    ax.set_xticklabels(config_names, rotation=15, ha="right", fontsize=FONT_TICK)
    ax.grid(axis="y", **GRID_KW)
    _style_ax(ax, "Resolution configuration", "Focal spot error (mrad)",
              "Per-Heliostat Error Distribution by Resolution")
    plt.tight_layout()
    plt.savefig(comp_dir / "comparison_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
