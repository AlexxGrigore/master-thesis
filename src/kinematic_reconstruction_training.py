"""
Kinematic Reconstruction Training Script

This script uses ARTIST's KinematicReconstructor to optimize heliostat kinematic
parameters using the PAINT benchmark dataset, then evaluates flux image prediction
accuracy on the test set.
"""

import logging
import pathlib
from collections import defaultdict

import h5py
import pandas as pd
import torch
from matplotlib import pyplot as plt

import paint.util.paint_mappings as paint_mappings
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.kinematic_reconstructor import KinematicReconstructor
from artist.core.loss_functions import FocalSpotLoss
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment

# Set random seeds for reproducibility
torch.manual_seed(7)
torch.cuda.manual_seed(7)

# Setup logging
set_logger_config()
log = logging.getLogger(__name__)


def build_heliostat_data_mapping(
    benchmark_csv: pathlib.Path,
    calibration_properties_dir: pathlib.Path,
    flux_image_dir: pathlib.Path,
    split: str = "train",
) -> list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]:
    """
    Build the heliostat_data_mapping from the benchmark CSV file.

    Parameters
    ----------
    benchmark_csv : pathlib.Path
        Path to the benchmark split CSV file.
    calibration_properties_dir : pathlib.Path
        Base directory containing calibration properties JSON files.
    flux_image_dir : pathlib.Path
        Base directory containing flux image PNG files.
    split : str
        Which split to use: "train", "validation", or "test".

    Returns
    -------
    list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
        List of tuples (heliostat_name, calibration_paths, flux_paths).
    """
    # Read the benchmark CSV
    df = pd.read_csv(benchmark_csv)

    # Filter by split
    df_split = df[df["Split"] == split]

    log.info(f"Building heliostat_data_mapping for split '{split}'")
    log.info(f"Total samples in split: {len(df_split)}")

    # Group by heliostat
    heliostat_groups = defaultdict(list)
    for _, row in df_split.iterrows():
        measurement_id = row["Id"]
        heliostat_id = row["HeliostatId"]
        heliostat_groups[heliostat_id].append(measurement_id)

    log.info(f"Number of unique heliostats: {len(heliostat_groups)}")

    # Build the mapping
    heliostat_data_mapping = []
    for heliostat_id, measurement_ids in sorted(heliostat_groups.items()):
        calibration_paths = []
        flux_paths = []

        for mid in measurement_ids:
            cal_path = calibration_properties_dir / split / f"{mid}-calibration-properties.json"
            flux_path = flux_image_dir / split / f"{mid}-flux.png"

            # Only include if both files exist
            if cal_path.exists() and flux_path.exists():
                calibration_paths.append(cal_path)
                flux_paths.append(flux_path)

        if calibration_paths:
            heliostat_data_mapping.append((heliostat_id, calibration_paths, flux_paths))

    log.info(f"Built mapping for {len(heliostat_data_mapping)} heliostats")

    return heliostat_data_mapping


def evaluate_flux_accuracy(
    scenario: Scenario,
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    data_parser: PaintCalibrationDataParser,
    device: torch.device,
    bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
) -> dict:
    """
    Evaluate flux image prediction accuracy after kinematic reconstruction.

    Parameters
    ----------
    scenario : Scenario
        The scenario with optimized kinematic parameters.
    heliostat_data_mapping : list
        Mapping of heliostats to their calibration data.
    data_parser : PaintCalibrationDataParser
        Parser for calibration data.
    device : torch.device
        Device to run computations on.
    bitmap_resolution : torch.Tensor
        Resolution of the flux bitmaps.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """
    all_pixel_losses = []
    all_focal_spot_errors = []
    results_per_heliostat = {}

    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        (
            measured_flux,
            focal_spots,
            incident_ray_directions,
            motor_positions,
            active_heliostats_mask,
            target_area_mask,
        ) = data_parser.parse_data_for_reconstruction(
            heliostat_data_mapping=heliostat_data_mapping,
            heliostat_group=heliostat_group,
            scenario=scenario,
            bitmap_resolution=bitmap_resolution,
            device=device,
        )

        if active_heliostats_mask.sum() == 0:
            continue

        # Activate heliostats
        heliostat_group.activate_heliostats(
            active_heliostats_mask=active_heliostats_mask,
            device=device,
        )

        # Align heliostats with incident ray directions
        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=scenario.target_areas.centers[target_area_mask],
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            device=device,
        )

        # Create ray tracer
        ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=heliostat_group,
            blocking_active=False,
            batch_size=min(heliostat_group.number_of_active_heliostats, 32),
            bitmap_resolution=bitmap_resolution.to(device),
        )

        # Trace rays to get predicted flux
        predicted_flux = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            target_area_mask=target_area_mask,
            device=device,
        )

        # Compute pixel-wise MSE loss
        pixel_loss = ((predicted_flux - measured_flux) ** 2).mean(dim=[1, 2])
        all_pixel_losses.extend(pixel_loss.cpu().tolist())

        # Compute focal spot error (center of mass comparison)
        # Get predicted focal spots from bitmaps
        from artist.util.utils import get_center_of_mass

        predicted_focal_spots = get_center_of_mass(
            bitmaps=predicted_flux,
            scenario=scenario,
            reduction_dimensions=torch.tensor([1, 2]),
            target_area_mask=target_area_mask,
            device=device,
        )

        # Compute Euclidean distance between predicted and measured focal spots
        focal_spot_error = torch.norm(predicted_focal_spots[:, :3] - focal_spots[:, :3], dim=1)
        all_focal_spot_errors.extend(focal_spot_error.cpu().tolist())

        # Store per-heliostat results
        heliostat_names = [
            name for name, _, _ in heliostat_data_mapping
            if name in [h.heliostat_name for h in heliostat_group.heliostats]
        ]
        for i, name in enumerate(heliostat_names):
            if i < len(pixel_loss):
                results_per_heliostat[name] = {
                    "pixel_mse": pixel_loss[i].item(),
                    "focal_spot_error_m": focal_spot_error[i].item() if i < len(focal_spot_error) else None,
                }

    # Compute aggregate metrics
    metrics = {
        "mean_pixel_mse": sum(all_pixel_losses) / len(all_pixel_losses) if all_pixel_losses else float("inf"),
        "mean_focal_spot_error_m": sum(all_focal_spot_errors) / len(all_focal_spot_errors) if all_focal_spot_errors else float("inf"),
        "max_focal_spot_error_m": max(all_focal_spot_errors) if all_focal_spot_errors else float("inf"),
        "min_focal_spot_error_m": min(all_focal_spot_errors) if all_focal_spot_errors else float("inf"),
        "num_samples_evaluated": len(all_pixel_losses),
        "per_heliostat": results_per_heliostat,
    }

    return metrics


def visualize_flux_comparison(
    scenario: Scenario,
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    data_parser: PaintCalibrationDataParser,
    device: torch.device,
    output_dir: pathlib.Path,
    num_samples: int = 5,
):
    """
    Visualize comparison between predicted and measured flux images.

    Parameters
    ----------
    scenario : Scenario
        The scenario with optimized kinematic parameters.
    heliostat_data_mapping : list
        Mapping of heliostats to their calibration data.
    data_parser : PaintCalibrationDataParser
        Parser for calibration data.
    device : torch.device
        Device to run computations on.
    output_dir : pathlib.Path
        Directory to save visualization images.
    num_samples : int
        Number of samples to visualize.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_visualized = 0

    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        if samples_visualized >= num_samples:
            break

        (
            measured_flux,
            _,
            incident_ray_directions,
            _,
            active_heliostats_mask,
            target_area_mask,
        ) = data_parser.parse_data_for_reconstruction(
            heliostat_data_mapping=heliostat_data_mapping,
            heliostat_group=heliostat_group,
            scenario=scenario,
            bitmap_resolution=torch.tensor([256, 256]),
            device=device,
        )

        if active_heliostats_mask.sum() == 0:
            continue

        # Activate and align heliostats
        heliostat_group.activate_heliostats(
            active_heliostats_mask=active_heliostats_mask,
            device=device,
        )
        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=scenario.target_areas.centers[target_area_mask],
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            device=device,
        )

        # Ray trace
        ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=heliostat_group,
            blocking_active=False,
            batch_size=min(heliostat_group.number_of_active_heliostats, 32),
            bitmap_resolution=torch.tensor([256, 256], device=device),
        )

        predicted_flux = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            target_area_mask=target_area_mask,
            device=device,
        )

        # Visualize
        for i in range(min(len(predicted_flux), num_samples - samples_visualized)):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(predicted_flux[i].cpu().detach(), cmap="gray")
            axes[0].set_title("Predicted Flux")
            axes[0].axis("off")

            axes[1].imshow(measured_flux[i].cpu().detach(), cmap="gray")
            axes[1].set_title("Measured Flux")
            axes[1].axis("off")

            plt.tight_layout()
            plt.savefig(output_dir / f"flux_comparison_{samples_visualized}.png", dpi=150)
            plt.close()

            samples_visualized += 1
            if samples_visualized >= num_samples:
                break

    log.info(f"Saved {samples_visualized} flux comparison images to {output_dir}")


def main():
    # ==========================================================================
    # Configuration
    # ==========================================================================

    # Paths
    BASE_DIR = pathlib.Path(__file__).parent
    BENCHMARK_NAME = "benchmark_split-balanced_train-10_validation-30"
    BENCHMARK_DIR = BASE_DIR / "paint_benchmarks"
    SCENARIO_PATH = BASE_DIR.parent / "scenarios" / "all_heliostats_scenario" / "all_heliostats_scenario.h5"
    OUTPUT_DIR = BASE_DIR / "kinematic_reconstruction_results"

    # Benchmark file paths
    BENCHMARK_CSV = BENCHMARK_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
    CALIBRATION_PROPERTIES_DIR = BENCHMARK_DIR / "datasets" / BENCHMARK_NAME / "calibration_properties"
    FLUX_IMAGE_DIR = BENCHMARK_DIR / "datasets" / BENCHMARK_NAME / "flux_image"

    # Training configuration
    SAMPLE_LIMIT_PER_HELIOSTAT = 10  # Limit samples per heliostat for training
    CENTROID_METHOD = paint_mappings.UTIS_KEY  # or paint_mappings.HELIOS_KEY

    # Device
    device = get_device()
    log.info(f"Using device: {device}")

    # ==========================================================================
    # Build heliostat data mappings
    # ==========================================================================

    log.info("Building heliostat data mappings...")

    train_mapping = build_heliostat_data_mapping(
        benchmark_csv=BENCHMARK_CSV,
        calibration_properties_dir=CALIBRATION_PROPERTIES_DIR,
        flux_image_dir=FLUX_IMAGE_DIR,
        split="train",
    )

    test_mapping = build_heliostat_data_mapping(
        benchmark_csv=BENCHMARK_CSV,
        calibration_properties_dir=CALIBRATION_PROPERTIES_DIR,
        flux_image_dir=FLUX_IMAGE_DIR,
        split="test",
    )

    log.info(f"Train mapping: {len(train_mapping)} heliostats")
    log.info(f"Test mapping: {len(test_mapping)} heliostats")

    # ==========================================================================
    # Create data parsers
    # ==========================================================================

    # Parser for training (with sample limit)
    train_data_parser = PaintCalibrationDataParser(
        sample_limit=SAMPLE_LIMIT_PER_HELIOSTAT,
        centroid_extraction_method=CENTROID_METHOD,
    )

    # Parser for evaluation (use all samples)
    eval_data_parser = PaintCalibrationDataParser(
        sample_limit=50,  # Use more samples for evaluation
        centroid_extraction_method=CENTROID_METHOD,
    )

    # ==========================================================================
    # Setup data dictionary for KinematicReconstructor
    # ==========================================================================

    data = {
        config_dictionary.data_parser: train_data_parser,
        config_dictionary.heliostat_data_mapping: train_mapping,
    }

    # ==========================================================================
    # Load scenario and setup distributed environment
    # ==========================================================================

    log.info(f"Loading scenario from: {SCENARIO_PATH}")

    number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
        scenario_path=SCENARIO_PATH
    )
    log.info(f"Number of heliostat groups: {number_of_heliostat_groups}")

    # ==========================================================================
    # Optimization configuration
    # ==========================================================================

    scheduler = config_dictionary.reduce_on_plateau
    scheduler_parameters = {
        config_dictionary.gamma: 0.9,
        config_dictionary.min: 1e-6,
        config_dictionary.max: 1e-3,
        config_dictionary.step_size_up: 500,
        config_dictionary.reduce_factor: 0.1,
        config_dictionary.patience: 30,
        config_dictionary.threshold: 1e-4,
        config_dictionary.cooldown: 10,
    }

    optimization_configuration = {
        config_dictionary.initial_learning_rate: 0.0005,
        config_dictionary.tolerance: 0.0001,
        config_dictionary.max_epoch: 50,
        config_dictionary.batch_size: 50,
        config_dictionary.log_step: 10,
        config_dictionary.early_stopping_delta: 1e-5,
        config_dictionary.early_stopping_patience: 50,
        config_dictionary.scheduler: scheduler,
        config_dictionary.scheduler_parameters: scheduler_parameters,
    }

    # ==========================================================================
    # Run kinematic reconstruction
    # ==========================================================================

    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        # Load the scenario
        with h5py.File(SCENARIO_PATH, "r") as scenario_file:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file, device=device
            )

        log.info("Scenario loaded successfully")
        log.info(f"Number of heliostats in scenario: {scenario.heliostat_field.number_of_heliostat_groups}")

        # Create kinematic reconstructor
        kinematic_reconstructor = KinematicReconstructor(
            ddp_setup=ddp_setup,
            scenario=scenario,
            data=data,
            optimization_configuration=optimization_configuration,
            reconstruction_method=config_dictionary.kinematic_reconstruction_raytracing,
        )

        # Create loss function
        loss_definition = FocalSpotLoss(scenario=scenario)

        log.info("Starting kinematic reconstruction...")

        # Run reconstruction
        final_loss_per_heliostat = kinematic_reconstructor.reconstruct_kinematic(
            loss_definition=loss_definition, device=device
        )

        log.info(f"Reconstruction complete!")
        log.info(f"Final loss per heliostat shape: {final_loss_per_heliostat.shape}")

        # Filter out infinite losses (heliostats without data)
        valid_losses = final_loss_per_heliostat[final_loss_per_heliostat != float("inf")]
        if len(valid_losses) > 0:
            log.info(f"Mean loss (trained heliostats): {valid_losses.mean().item():.6f}")
            log.info(f"Min loss: {valid_losses.min().item():.6f}")
            log.info(f"Max loss: {valid_losses.max().item():.6f}")
            log.info(f"Number of trained heliostats: {len(valid_losses)}")

        # ==========================================================================
        # Evaluate on test set
        # ==========================================================================

        log.info("Evaluating on test set...")

        test_metrics = evaluate_flux_accuracy(
            scenario=scenario,
            heliostat_data_mapping=test_mapping,
            data_parser=eval_data_parser,
            device=device,
        )

        log.info("=" * 60)
        log.info("TEST SET EVALUATION RESULTS")
        log.info("=" * 60)
        log.info(f"Number of samples evaluated: {test_metrics['num_samples_evaluated']}")
        log.info(f"Mean pixel MSE: {test_metrics['mean_pixel_mse']:.6f}")
        log.info(f"Mean focal spot error: {test_metrics['mean_focal_spot_error_m']:.4f} m")
        log.info(f"Min focal spot error: {test_metrics['min_focal_spot_error_m']:.4f} m")
        log.info(f"Max focal spot error: {test_metrics['max_focal_spot_error_m']:.4f} m")
        log.info("=" * 60)

        # ==========================================================================
        # Visualize some results
        # ==========================================================================

        log.info("Generating flux comparison visualizations...")

        visualize_flux_comparison(
            scenario=scenario,
            heliostat_data_mapping=test_mapping,
            data_parser=eval_data_parser,
            device=device,
            output_dir=OUTPUT_DIR / "visualizations",
            num_samples=10,
        )

        # ==========================================================================
        # Save results
        # ==========================================================================

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Save the optimized kinematic parameters for each heliostat group
        log.info("Saving optimized kinematic parameters...")
        kinematic_params_dir = OUTPUT_DIR / "kinematic_parameters"
        kinematic_params_dir.mkdir(parents=True, exist_ok=True)

        all_kinematic_params = {}
        for heliostat_group in scenario.heliostat_field.heliostat_groups:
            group_name = heliostat_group.name
            group_params = {
                "rotation_deviation_parameters": heliostat_group.kinematic.rotation_deviation_parameters.detach().cpu(),
                "actuator_parameters": heliostat_group.kinematic.actuators.optimizable_parameters.detach().cpu(),
            }
            all_kinematic_params[group_name] = group_params

            # Save individual group parameters
            torch.save(group_params, kinematic_params_dir / f"{group_name}_kinematic_params.pt")

        # Save all parameters in one file
        torch.save(all_kinematic_params, OUTPUT_DIR / "all_kinematic_parameters.pt")
        log.info(f"Saved kinematic parameters to {OUTPUT_DIR / 'all_kinematic_parameters.pt'}")

        # Also save the entire optimized scenario to HDF5
        optimized_scenario_path = OUTPUT_DIR / "optimized_scenario.h5"
        with h5py.File(optimized_scenario_path, "w") as f:
            scenario.save_scenario_to_hdf5(scenario_file=f)
        log.info(f"Saved optimized scenario to {optimized_scenario_path}")

        # Save metrics
        import json

        metrics_file = OUTPUT_DIR / "test_metrics.json"
        with open(metrics_file, "w") as f:
            # Convert per_heliostat to serializable format
            serializable_metrics = {
                k: v for k, v in test_metrics.items() if k != "per_heliostat"
            }
            serializable_metrics["per_heliostat"] = test_metrics["per_heliostat"]
            json.dump(serializable_metrics, f, indent=2)

        log.info(f"Saved metrics to {metrics_file}")

        # Save final losses
        torch.save(final_loss_per_heliostat, OUTPUT_DIR / "final_loss_per_heliostat.pt")
        log.info(f"Saved final losses to {OUTPUT_DIR / 'final_loss_per_heliostat.pt'}")

        log.info("Done!")


if __name__ == "__main__":
    main()
