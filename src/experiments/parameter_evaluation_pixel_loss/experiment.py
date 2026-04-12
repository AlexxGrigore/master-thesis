import gc
import json
import logging
import pathlib
import time

import h5py
import numpy as np
import torch
from artist.core.loss_functions import FocalSpotLoss
from artist_extensions.loss_functions_ext import PixelLossL1
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary

from utils.checkpointing import save_kinematic_parameters
from utils.evaluation import compute_pixel_test_loss, evaluate_flux_accuracy
from utils.plotting import (
    plot_tracking_error_histogram,
    plot_training_curves,
    visualize_flux_comparison,
    _style_ax,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TICK,
    FONT_TITLE,
    GRID_KW,
)
from matplotlib import pyplot as plt

log = logging.getLogger(__name__)


def run_experiment(
    config_name: str,
    phase1_reconstructor_cls,
    phase2_reconstructor_cls,
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
    save_figures: bool = False,
    train_position_deviation: bool = True,
    validation_mapping: list | None = None,
) -> dict:
    """
    Two-phase kinematic reconstruction experiment for one parameter-subset variant.

    Phase 1 — Focal spot pretraining (phase1_reconstructor_cls + FocalSpotLoss):
        Gets heliostats roughly aligned so reflected light reliably hits the target.

    Phase 2 — Pixel loss fine-tuning (phase2_reconstructor_cls + PixelLossL1):
        Continues from Phase 1 weights with a fresh Adam optimizer.
        The same parameter subset stays active (phase2_reconstructor_cls mirrors
        phase1_reconstructor_cls but inherits from WortbergPixelReconstructor).

    The scenario object is shared across both phases so Phase 2 picks up
    Phase 1's optimised kinematic parameters automatically.

    Outputs saved to output_dir / config_name:
      - phase1/  — training.log, convergence_history.json
      - phase2/  — training.log, convergence_history.json, training_summary.json
      - test_metrics.json, timing_stats.json, tracking_error_histogram.png
      - all_kinematic_parameters.json, visualizations/
    """
    exp_dir = output_dir / config_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    exp_log_handler = logging.FileHandler(exp_dir / "training.log")
    exp_log_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    )
    logging.getLogger().addHandler(exp_log_handler)

    try:
        log.info(f"=== Starting experiment: {config_name} ===")

        # Reload scenario from disk — fresh kinematic parameters for every variant.
        with h5py.File(scenario_path, "r") as scenario_file:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file,
                device=device,
                number_of_surface_points_per_facet=torch.tensor([25, 25]),
            )

        scenario.set_number_of_rays(10)
        log.info("Number of rays set to 10.")
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
        print(f"\n  Phase 1 — FocalSpotLoss [{phase1_reconstructor_cls.__name__}], "
              f"max_epoch={phase1_opt_config[config_dictionary.max_epoch]}, "
              f"lr={phase1_opt_config[config_dictionary.initial_learning_rate]}")

        phase1_reconstructor = phase1_reconstructor_cls(
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
        log.info(f"Phase 1 — time: {phase1_time_s/60:.1f} min, peak GPU: {phase1_peak_gpu_gb:.2f} GB")
        print(f"  Phase 1 — time: {phase1_time_s/60:.1f} min ({phase1_time_s:.0f}s), peak GPU: {phase1_peak_gpu_gb:.2f} GB")

        logging.getLogger().removeHandler(phase1_log_handler)
        phase1_log_handler.close()

        phase1_convergence = phase1_reconstructor._convergence_history
        with open(phase1_dir / "convergence_history.json", "w") as f:
            json.dump(phase1_convergence, f, indent=2)

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
        print(f"\n  Phase 2 — PixelLossL1 [{phase2_reconstructor_cls.__name__}], "
              f"max_epoch={phase2_opt_config[config_dictionary.max_epoch]}, "
              f"lr={phase2_opt_config[config_dictionary.initial_learning_rate]}")

        # Fresh reconstructor = fresh Adam optimizer (no stale momentum from Phase 1).
        # The scenario already holds Phase 1's optimised kinematic parameters.
        phase2_reconstructor = phase2_reconstructor_cls(
            ddp_setup=ddp_setup,
            scenario=scenario,
            train_position_deviation=train_position_deviation,
            data=data,
            optimization_configuration=phase2_opt_config,
            reconstruction_method=config_dictionary.kinematics_reconstruction_raytracing,
            eval_data=eval_data,
            blur_sigma=0.0,
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
        log.info(f"Phase 2 — time: {phase2_time_s/60:.1f} min, peak GPU: {phase2_peak_gpu_gb:.2f} GB")
        print(f"  Phase 2 — time: {phase2_time_s/60:.1f} min ({phase2_time_s:.0f}s), peak GPU: {phase2_peak_gpu_gb:.2f} GB")

        logging.getLogger().removeHandler(phase2_log_handler)
        phase2_log_handler.close()

        phase2_convergence = phase2_reconstructor._convergence_history
        with open(phase2_dir / "convergence_history.json", "w") as f:
            json.dump(phase2_convergence, f, indent=2)

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
        print(f"  Phase 2 summary: {phase2_summary['num_nan_loss']} NaN, "
              f"{phase2_summary['num_inf_loss']} inf out of {phase2_summary['num_heliostats_total']} heliostats")

        # ----------------------------------------------------------------
        # Test evaluation
        # ----------------------------------------------------------------
        del phase2_reconstructor
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
        log.info(f"Evaluation — time: {eval_time_s/60:.1f} min, peak GPU: {eval_peak_gpu_gb:.2f} GB")
        print(f"  Evaluation — time: {eval_time_s/60:.1f} min ({eval_time_s:.0f}s)")
        print(f"  Test — mean: {test_metrics['mean_focal_spot_error_mrad']:.2f} mrad, "
              f"median: {test_metrics['median_focal_spot_error_mrad']:.2f} mrad")

        # Training curve plots (one per phase).
        phase1_test_loss = test_metrics["mean_focal_spot_error_m"]
        phase2_test_loss = compute_pixel_test_loss(
            scenario=scenario,
            heliostat_data_mapping=test_mapping,
            data_parser=eval_data_parser,
            device=device,
            blur_sigma=0.0,
        )
        with open(exp_dir / "test_loss_values.json", "w") as f:
            json.dump({
                "phase1_test_loss_focal_spot_m": phase1_test_loss,
                "phase2_test_loss_pixel_l1": phase2_test_loss,
            }, f, indent=2)

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
            title=f"Heliostat Tracking Error — {config_name} (Test Set)",
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

        save_kinematic_parameters(scenario, exp_dir / "all_kinematic_parameters.json")

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

        # Attach phase2 convergence history for cross-experiment comparison plots.
        test_metrics["convergence_history"] = phase2_convergence
        test_metrics["phase1_convergence_history"] = phase1_convergence
        test_metrics["heliostat_positions"] = {
            name: (
                hg.positions[i, 0].item(),
                hg.positions[i, 1].item(),
            )
            for hg in scenario.heliostat_field.heliostat_groups
            for i, name in enumerate(hg.names)
        }

        log.info(f"=== Experiment '{config_name}' done: {test_metrics['mean_focal_spot_error_mrad']:.2f} mrad ===")
        return test_metrics

    finally:
        logging.getLogger().removeHandler(exp_log_handler)
        exp_log_handler.close()


# ---------------------------------------------------------------------------
# Cross-experiment comparison plots
# ---------------------------------------------------------------------------

_CONFIG_COLORS = [
    "#607D8B",  # 0a — blue-grey
    "#795548",  # 0b — brown
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
    Generate cross-experiment comparison plots for the pixel-loss ablation.

    Produces four figures in output_dir/comparison/:
      1. comparison_bar.png         — mean + median focal spot error per config
      2. comparison_boxplot.png     — per-heliostat error distribution per config
      3. comparison_loss_curves.png — phase1 train loss (left) | phase2 train loss (right)
      4. comparison_field_map.png   — 5-panel field scatter, dot colour = error
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
              "Focal Spot Error by Parameter Configuration (Pixel Loss Training)")
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
    # 3. Loss curves — phase1 train loss (left) | phase2 train loss (right)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Training Loss — All Configurations",
                 fontsize=FONT_TITLE, fontweight="bold")

    for c, color in zip(config_names, colors):
        p1_hist = all_metrics[c].get("phase1_convergence_history", [])
        p2_hist = all_metrics[c].get("convergence_history", [])

        if p1_hist:
            axes[0].plot(
                [e["epoch"] for e in p1_hist],
                [e["loss"] for e in p1_hist],
                color=color, linewidth=1.5, label=c,
            )
        if p2_hist:
            axes[1].plot(
                [e["epoch"] for e in p2_hist],
                [e["loss"] for e in p2_hist],
                color=color, linewidth=1.5, label=c,
            )

    for ax, title in zip(axes, ["Phase 1 — Focal Spot Loss", "Phase 2 — Pixel L1 Loss"]):
        ax.legend(fontsize=FONT_LEGEND, framealpha=0.85)
        ax.grid(**GRID_KW)
        _style_ax(ax, "Epoch", "Loss", title)

    plt.tight_layout()
    plt.savefig(comp_dir / "comparison_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Field map — 1×N scatter, dot colour = focal spot error
    # ------------------------------------------------------------------
    heliostat_positions = all_metrics[config_names[0]].get("heliostat_positions", {})
    if not heliostat_positions:
        log.warning("No heliostat positions found; skipping field map.")
        return

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

    cmap = plt.cm.RdYlGn_r
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

        all_e = [v[0] for v in heliostat_positions.values()]
        all_n = [v[1] for v in heliostat_positions.values()]
        ax.scatter(all_e, all_n, s=4, color="#cccccc", zorder=1)
        sc = ax.scatter(east, north, c=errors, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=12, zorder=2, linewidths=0)
        sc_last = sc
        ax.scatter([0], [0], s=120, marker="*", color="black", zorder=5)
        ax.set_aspect("equal")
        ax.set_xlabel("East (m)", fontsize=FONT_TICK)
        ax.set_ylabel("North (m)", fontsize=FONT_TICK)
        ax.set_title(c, fontsize=FONT_LABEL, fontweight="bold")
        ax.tick_params(labelsize=FONT_TICK)
        ax.grid(True, alpha=0.2)

    if sc_last is not None:
        fig.colorbar(sc_last, ax=axes[0].tolist(), label="Focal spot error (mrad)",
                     fraction=0.02, pad=0.04)

    plt.tight_layout()
    plt.savefig(comp_dir / "comparison_field_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_parameter_deviation_summary(
    config_names: list[str],
    scenario_path: pathlib.Path,
    output_dir: pathlib.Path,
    device: torch.device,
) -> None:
    """
    For each config, compare final kinematic parameters against the fresh
    scenario (initial values) and save a JSON showing which params moved.

    Output: output_dir/comparison/parameter_deviation_summary.json
    """
    from artist.util import index_mapping as idx

    # Load fresh scenario to get initial values.
    with h5py.File(scenario_path, "r") as f:
        scenario_init = Scenario.load_scenario_from_hdf5(
            scenario_file=f, device=device,
            number_of_surface_points_per_facet=torch.tensor([25, 25]),
        )

    # Snapshot initial values (first group — single-group scenario).
    hg0 = scenario_init.heliostat_field.heliostat_groups[0]
    kin0 = hg0.kinematics
    init_translation = kin0.translation_deviation_parameters.detach().cpu()
    init_rotation = kin0.rotation_deviation_parameters.detach().cpu()
    init_act_angle = kin0.actuators.optimizable_parameters[:, idx.actuator_initial_angle, :].detach().cpu()
    init_act_offset = kin0.actuators.non_optimizable_parameters[:, idx.actuator_offset, :].detach().cpu()

    translation_names = [
        "joint1_trans_e", "joint1_trans_n", "joint1_trans_u",
        "joint2_trans_e", "joint2_trans_n", "joint2_trans_u",
        "conc_trans_e",   "conc_trans_n",   "conc_trans_u",
    ]
    rotation_names = [
        "joint1_tilt_n", "joint1_tilt_u",
        "joint2_tilt_e", "joint2_tilt_n",
    ]

    summary = {}
    for config_name in config_names:
        params_file = output_dir / config_name / "all_kinematic_parameters.json"
        if not params_file.exists():
            log.warning(f"Missing {params_file}, skipping.")
            continue

        with open(params_file) as f:
            saved = json.load(f)

        # Reconstruct tensors from the saved JSON (first group).
        group_data = saved["group_0"]
        final_translation = torch.tensor(group_data["translation_deviation_parameters"])
        final_rotation = torch.tensor(group_data["rotation_deviation_parameters"])
        final_act_opt = torch.tensor(group_data["actuator_optimizable_parameters"])
        final_act_nonopt = torch.tensor(group_data["actuator_nonoptimizable_parameters"])
        final_act_angle = final_act_opt[:, idx.actuator_initial_angle, :]
        final_act_offset = final_act_nonopt[:, idx.actuator_offset, :]

        # Compute deviations from initial.
        d_trans = (final_translation - init_translation).abs()
        d_rot = (final_rotation - init_rotation).abs()
        d_angle = (final_act_angle - init_act_angle).abs()
        d_offset = (final_act_offset - init_act_offset).abs()

        has_base_pos = "base_position_deviation_parameters" in group_data
        base_pos_stats = None
        if has_base_pos:
            base_pos = torch.tensor(group_data["base_position_deviation_parameters"])
            # Initial is always 0 (injected fresh each run).
            base_pos_stats = {
                "mean_abs_deviation": [round(base_pos[:, i].abs().mean().item(), 8) for i in range(3)],
                "max_abs_deviation":  [round(base_pos[:, i].abs().max().item(), 8) for i in range(3)],
                "names": ["base_pos_e", "base_pos_n", "base_pos_u"],
            }

        entry = {
            "translation_deviation": {
                "mean_abs_deviation": [round(d_trans[:, i].mean().item(), 8) for i in range(9)],
                "max_abs_deviation":  [round(d_trans[:, i].max().item(), 8) for i in range(9)],
                "names": translation_names,
            },
            "rotation_deviation": {
                "mean_abs_deviation": [round(d_rot[:, i].mean().item(), 8) for i in range(4)],
                "max_abs_deviation":  [round(d_rot[:, i].max().item(), 8) for i in range(4)],
                "names": rotation_names,
            },
            "actuator_initial_angle": {
                "mean_abs_deviation": [round(d_angle[:, i].mean().item(), 8) for i in range(d_angle.shape[1])],
                "max_abs_deviation":  [round(d_angle[:, i].max().item(), 8) for i in range(d_angle.shape[1])],
                "names": [f"actuator_{i}" for i in range(d_angle.shape[1])],
            },
            "actuator_offset": {
                "mean_abs_deviation": [round(d_offset[:, i].mean().item(), 8) for i in range(d_offset.shape[1])],
                "max_abs_deviation":  [round(d_offset[:, i].max().item(), 8) for i in range(d_offset.shape[1])],
                "names": [f"actuator_{i}" for i in range(d_offset.shape[1])],
            },
        }
        if base_pos_stats:
            entry["base_position_deviation"] = base_pos_stats

        # Flag which parameter groups actually moved (mean_abs > 1e-7).
        moved = []
        param_group_keys = [
            "translation_deviation", "rotation_deviation",
            "actuator_initial_angle", "actuator_offset",
        ]
        if base_pos_stats:
            param_group_keys.append("base_position_deviation")
        for group_key in param_group_keys:
            group_val = entry[group_key]
            for name, val in zip(group_val["names"], group_val["mean_abs_deviation"]):
                if val > 1e-7:
                    moved.append(f"{group_key}/{name}")
        entry["parameters_that_moved"] = moved
        entry["num_params_moved"] = len(moved)

        summary[config_name] = entry

    comp_dir = output_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    out_path = comp_dir / "parameter_deviation_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved parameter deviation summary to {out_path}")
    print(f"Saved parameter deviation summary to {out_path}")
