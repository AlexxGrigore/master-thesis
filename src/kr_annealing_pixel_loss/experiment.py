import json
import logging
import pathlib

import h5py
import torch
from artist.core.loss_functions import FocalSpotLoss, PixelLoss
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary

from artist_extensions.kinematic_reconstructors import WortbergAnnealingReconstructor
from utils.evaluation import evaluate_flux_accuracy
from utils.plotting import (
    plot_tracking_error_histogram,
    plot_training_curves,
    visualize_flux_comparison,
)

log = logging.getLogger(__name__)


def run_experiment(
    loss_name: str,
    opt_config: dict,
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
    Single-phase kinematic reconstruction with linearly annealed combined loss.

    The loss transitions from pure FocalSpotLoss (epoch 0) to pure PixelLoss
    (epoch max_epoch) via linear annealing.  Both losses are normalised by
    their initial value at epoch 0 so they contribute equally at the midpoint.

    Outputs saved to output_dir / loss_name:
      - training/training.log + training_curves.png
      - training_summary.json
      - test_metrics.json
      - tracking_error_histogram.png
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
        log.info(f"=== Starting experiment: {loss_name} ===")

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

        # ----------------------------------------------------------------
        # Single annealing phase
        # ----------------------------------------------------------------
        train_dir = exp_dir / "training"
        train_dir.mkdir(parents=True, exist_ok=True)

        train_log_handler = logging.FileHandler(train_dir / "training.log")
        train_log_handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
        )
        logging.getLogger().addHandler(train_log_handler)

        print(f"\n  Annealing — FocalSpotLoss→PixelLoss, "
              f"max_epoch={opt_config[config_dictionary.max_epoch]}, "
              f"lr={opt_config[config_dictionary.initial_learning_rate]}")

        reconstructor = WortbergAnnealingReconstructor(
            ddp_setup=ddp_setup,
            scenario=scenario,
            train_position_deviation=train_position_deviation,
            data=data,
            optimization_configuration=opt_config,
            reconstruction_method=config_dictionary.kinematic_reconstruction_raytracing,
            focal_loss=FocalSpotLoss(scenario=scenario),
            pixel_loss=PixelLoss(scenario=scenario),
        )
        final_loss = reconstructor.reconstruct_kinematic(
            loss_definition=None,
            device=device,
        )

        logging.getLogger().removeHandler(train_log_handler)
        train_log_handler.close()

        plot_training_curves(log_file=train_dir / "training.log", output_dir=train_dir)

        import numpy as np
        loss_np = final_loss.detach().cpu().numpy()
        training_summary = {
            "num_heliostats_total": int(len(loss_np)),
            "num_nan_loss": int(np.isnan(loss_np).sum()),
            "num_inf_loss": int(np.isinf(loss_np).sum()),
            "num_zero_loss": int((loss_np == 0.0).sum()),
            "num_valid_loss": int(np.isfinite(loss_np).sum()),
            "mean_final_loss": float(np.nanmean(loss_np[np.isfinite(loss_np)])) if np.isfinite(loss_np).any() else None,
            "median_final_loss": float(np.nanmedian(loss_np[np.isfinite(loss_np)])) if np.isfinite(loss_np).any() else None,
        }
        with open(exp_dir / "training_summary.json", "w") as f:
            json.dump(training_summary, f, indent=2)
        print(f"\n  Training summary: "
              f"{training_summary['num_nan_loss']} NaN, "
              f"{training_summary['num_inf_loss']} inf, "
              f"{training_summary['num_zero_loss']} zero "
              f"(out of {training_summary['num_heliostats_total']} heliostats)")

        # ----------------------------------------------------------------
        # Test evaluation
        # ----------------------------------------------------------------
        del reconstructor
        torch.cuda.empty_cache()

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
