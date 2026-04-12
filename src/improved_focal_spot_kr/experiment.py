import gc
import json
import logging
import pathlib
import time

import h5py
import numpy as np
import torch
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary

from utils.checkpointing import save_kinematic_parameters
from utils.evaluation import evaluate_flux_accuracy
from utils.plotting import (
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
    validation_mapping: list | None = None,
) -> dict:
    """
    Run one training + evaluation experiment.

    Changes vs the original parameter_evaluation_focal_spot experiment:
    - Baseline mrad is logged before any training starts.
    - Eval loss is computed every epoch (scheduler wired to eval loss).
    - Gradient magnitudes are recorded in convergence history.
    - Translations are excluded from all configs.
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

        # ------------------------------------------------------------------
        # Baseline evaluation — untrained scenario parameters
        # ------------------------------------------------------------------
        log.info("Computing baseline (untrained) focal spot error...")
        baseline_metrics = evaluate_flux_accuracy(
            scenario=scenario,
            heliostat_data_mapping=test_mapping,
            data_parser=eval_data_parser,
            device=device,
        )
        baseline_mrad = baseline_metrics["mean_focal_spot_error_mrad"]
        log.info(f"Baseline (untrained) mean focal spot error: {baseline_mrad:.3f} mrad")
        print(f"  Baseline (untrained): {baseline_mrad:.3f} mrad")

        with open(exp_dir / "baseline_metrics.json", "w") as f:
            json.dump(
                {
                    "mean_focal_spot_error_mrad": baseline_metrics["mean_focal_spot_error_mrad"],
                    "median_focal_spot_error_mrad": baseline_metrics["median_focal_spot_error_mrad"],
                    "min_focal_spot_error_mrad": baseline_metrics["min_focal_spot_error_mrad"],
                    "max_focal_spot_error_mrad": baseline_metrics["max_focal_spot_error_mrad"],
                },
                f,
                indent=2,
            )

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
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
            data=data,
            optimization_configuration=optimization_configuration,
            reconstruction_method=config_dictionary.kinematics_reconstruction_raytracing,
            eval_data=eval_data,
        )

        loss_definition = loss_fn_factory(scenario)
        print(f"  Loss: {loss_definition.__class__.__name__}")
        print(f"  Reconstructor: {reconstructor_cls.__name__}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        t_train_start = time.time()
        final_loss_per_heliostat = reconstructor.reconstruct_kinematics(
            loss_definition=loss_definition, device=device
        )
        train_time_s = time.time() - t_train_start
        train_peak_gpu_gb = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
        train_end_gpu_gb = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
        log.info(
            f"Training — time: {train_time_s/60:.1f} min ({train_time_s:.0f}s), "
            f"peak GPU: {train_peak_gpu_gb:.2f} GB, end GPU: {train_end_gpu_gb:.2f} GB"
        )
        print(
            f"  Training — time: {train_time_s/60:.1f} min ({train_time_s:.0f}s), "
            f"peak GPU: {train_peak_gpu_gb:.2f} GB"
        )

        # ------------------------------------------------------------------
        # Convergence history
        # ------------------------------------------------------------------
        with open(exp_dir / "convergence_history.json", "w") as f:
            json.dump(reconstructor._convergence_history, f, indent=2)

        # ------------------------------------------------------------------
        # Test evaluation
        # ------------------------------------------------------------------
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
        log.info(
            f"Evaluation — time: {eval_time_s/60:.1f} min ({eval_time_s:.0f}s), "
            f"peak GPU: {eval_peak_gpu_gb:.2f} GB"
        )

        final_mrad = test_metrics["mean_focal_spot_error_mrad"]
        improvement = baseline_mrad - final_mrad
        print(f"  Baseline: {baseline_mrad:.3f} mrad  →  Final: {final_mrad:.3f} mrad  (Δ {improvement:+.3f} mrad)")
        log.info(
            f"Baseline: {baseline_mrad:.3f} mrad  →  Final: {final_mrad:.3f} mrad  (Δ {improvement:+.3f} mrad)"
        )

        # ------------------------------------------------------------------
        # Save outputs
        # ------------------------------------------------------------------
        test_loss = test_metrics["mean_focal_spot_error_m"]
        with open(exp_dir / "test_loss_values.json", "w") as f:
            json.dump({"test_loss_focal_spot_m": test_loss}, f, indent=2)

        with open(exp_dir / "test_metrics.json", "w") as f:
            json.dump(
                {
                    "baseline_mean_mrad": baseline_mrad,
                    "mean_focal_spot_error_mrad": test_metrics["mean_focal_spot_error_mrad"],
                    "median_focal_spot_error_mrad": test_metrics["median_focal_spot_error_mrad"],
                    "min_focal_spot_error_mrad": test_metrics["min_focal_spot_error_mrad"],
                    "max_focal_spot_error_mrad": test_metrics["max_focal_spot_error_mrad"],
                    "improvement_mrad": improvement,
                    "num_samples_evaluated": test_metrics["num_samples_evaluated"],
                    "num_nan_samples": test_metrics.get("num_nan_samples", 0),
                    "per_heliostat": test_metrics["per_heliostat"],
                },
                f,
                indent=2,
            )

        with open(exp_dir / "timing_stats.json", "w") as f:
            json.dump(
                {
                    "training_time_s": round(train_time_s, 1),
                    "training_time_min": round(train_time_s / 60, 2),
                    "training_peak_gpu_gb": round(train_peak_gpu_gb, 3),
                    "training_end_gpu_gb": round(train_end_gpu_gb, 3),
                    "evaluation_time_s": round(eval_time_s, 1),
                    "evaluation_time_min": round(eval_time_s / 60, 2),
                    "evaluation_peak_gpu_gb": round(eval_peak_gpu_gb, 3),
                },
                f,
                indent=2,
            )

        plot_tracking_error_histogram(
            errors_mrad=test_metrics["all_errors_mrad"],
            output_path=exp_dir / "tracking_error_histogram.png",
            title=f"Heliostat Tracking Error — {loss_name} (Test Set)",
        )

        plot_training_curves(
            log_file=exp_dir / "training.log",
            output_dir=exp_dir,
            test_loss=test_loss,
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

        test_metrics["convergence_history"] = convergence_history
        log.info(f"=== Experiment '{loss_name}' done: {final_mrad:.2f} mrad (baseline {baseline_mrad:.2f} mrad) ===")
        return test_metrics

    finally:
        logging.getLogger().removeHandler(exp_log_handler)
        exp_log_handler.close()
