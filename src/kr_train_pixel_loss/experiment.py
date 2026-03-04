import json
import logging
import pathlib

import h5py
import torch
from artist.core.loss_functions import FocalSpotLoss, PixelLoss
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary

from artist_extensions.kinematic_reconstructors import (
    WortbergKinematicReconstructor,
    WortbergPixelReconstructor,
)
from utils.evaluation import evaluate_flux_accuracy
from utils.plotting import (
    plot_tracking_error_histogram,
    plot_training_curves,
    visualize_flux_comparison,
)

log = logging.getLogger(__name__)


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
    save_figures: bool = False,
    train_position_deviation: bool = True,
) -> dict:
    """
    Two-phase kinematic reconstruction experiment.

    Phase 1 — Focal spot pretraining (WortbergKinematicReconstructor):
        Trains the kinematic parameters with FocalSpotLoss so that heliostats
        are roughly aligned and reflected light reliably hits the target.
        This ensures pixel loss receives non-zero gradients in Phase 2.

    Phase 2 — Pixel loss fine-tuning (WortbergPixelReconstructor):
        Continues from Phase 1 weights using PixelLoss. A fresh optimizer is
        created so Adam's momentum is not stale from the different loss scale.
        Deviation bounds and actuator snapshots are preserved from Phase 1 so
        all parameters stay within Wortberg (2025) Table 5.3 limits.

    Outputs saved to output_dir / loss_name:
      - phase1/training.log + training_curves.png
      - phase2/training.log + training_curves.png
      - test_metrics.json
      - tracking_error_histogram.png
      - visualizations/flux_comparison_*.png
    """
    exp_dir = output_dir / loss_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Top-level log captures both phases.
    exp_log_handler = logging.FileHandler(exp_dir / "training.log")
    exp_log_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    )
    logging.getLogger().addHandler(exp_log_handler)

    try:
        log.info(f"=== Starting experiment: {loss_name} ===")

        # Load scenario once — both phases share the same kinematic parameters.
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
            reconstruction_method=config_dictionary.kinematic_reconstruction_raytracing,
        )
        phase1_reconstructor.reconstruct_kinematic(
            loss_definition=FocalSpotLoss(scenario=scenario),
            device=device,
        )

        logging.getLogger().removeHandler(phase1_log_handler)
        phase1_log_handler.close()

        plot_training_curves(log_file=phase1_dir / "training.log", output_dir=phase1_dir)

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

        # Fresh reconstructor = fresh Adam optimizer (no stale momentum from Phase 1).
        # The scenario object already holds Phase 1's optimised kinematic parameters.
        phase2_reconstructor = WortbergPixelReconstructor(
            ddp_setup=ddp_setup,
            scenario=scenario,
            train_position_deviation=train_position_deviation,
            data=data,
            optimization_configuration=phase2_opt_config,
            reconstruction_method=config_dictionary.kinematic_reconstruction_raytracing,
        )
        phase2_final_loss = phase2_reconstructor.reconstruct_kinematic(
            loss_definition=PixelLoss(scenario=scenario),
            device=device,
        )

        logging.getLogger().removeHandler(phase2_log_handler)
        phase2_log_handler.close()

        plot_training_curves(log_file=phase2_dir / "training.log", output_dir=phase2_dir)

        # Save Phase 2 training loss summary — NaN/inf counts indicate heliostats
        # that still produced no flux on the target after Phase 1 pretraining.
        loss_np = phase2_final_loss.detach().cpu().numpy()
        import numpy as np
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

        # ----------------------------------------------------------------
        # Test evaluation
        # ----------------------------------------------------------------
        test_metrics = evaluate_flux_accuracy(
            scenario=scenario,
            heliostat_data_mapping=test_mapping,
            data_parser=eval_data_parser,
            device=device,
        )

        print(f"\n  Test  — mean focal spot error:   {test_metrics['mean_focal_spot_error_mrad']:.2f} mrad")
        print(f"  Test  — median focal spot error: {test_metrics['median_focal_spot_error_mrad']:.2f} mrad")

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

        log.info(f"=== Experiment '{loss_name}' done: {test_metrics['mean_focal_spot_error_mrad']:.2f} mrad ===")
        return test_metrics

    finally:
        logging.getLogger().removeHandler(exp_log_handler)
        exp_log_handler.close()
