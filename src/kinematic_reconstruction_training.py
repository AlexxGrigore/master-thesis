import datetime
import json
import logging
import pathlib
import traceback
from collections import defaultdict

import h5py
import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

import paint.util.paint_mappings as paint_mappings
from artist.core.loss_functions import FocalSpotLoss
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment
from artist_extensions.kinematic_reconstructors import WortbergKinematicReconstructor

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Setup logging
set_logger_config()
log = logging.getLogger(__name__)

print("Imports completed successfully!")


# ===================================================================
# Helper Functions
# ===================================================================

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
    df = pd.read_csv(benchmark_csv)
    df_split = df[df["Split"] == split]

    log.info(f"Building heliostat_data_mapping for split '{split}'")
    log.info(f"Total samples in split: {len(df_split)}")

    heliostat_groups = defaultdict(list)
    for _, row in df_split.iterrows():
        measurement_id = row["Id"]
        heliostat_id = row["HeliostatId"]
        heliostat_groups[heliostat_id].append(measurement_id)

    log.info(f"Number of unique heliostats: {len(heliostat_groups)}")

    heliostat_data_mapping = []
    for heliostat_id, measurement_ids in sorted(heliostat_groups.items()):
        calibration_paths = []
        flux_paths = []

        for mid in measurement_ids:
            cal_path = calibration_properties_dir / split / f"{mid}-calibration-properties.json"
            flux_path = flux_image_dir / split / f"{mid}-flux.png"

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
    ray_tracing_batch_size: int = 32,
) -> dict:
    """
    Evaluate flux image prediction accuracy after kinematic reconstruction.
    Errors are reported in both meters and milliradians (mrad).
    """
    from artist.util.utils import get_center_of_mass
    from artist.util import index_mapping

    all_pixel_losses = []
    all_focal_spot_errors_m = []
    all_focal_spot_errors_mrad = []
    results_per_heliostat = {}

    # Reference target center (mean over all target areas) used for distance computation.
    # All heliostats aim at roughly the same tower, so this is a good approximation.
    reference_target = scenario.target_areas.centers[:, :3].mean(dim=0).to(device)

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

        ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=heliostat_group,
            blocking_active=False,
            batch_size=min(heliostat_group.number_of_active_heliostats, ray_tracing_batch_size),
            bitmap_resolution=bitmap_resolution.to(device),
        )

        predicted_flux = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            target_area_mask=target_area_mask,
            device=device,
        )

        pixel_loss = ((predicted_flux - measured_flux) ** 2).mean(dim=[1, 2])
        all_pixel_losses.extend(pixel_loss.cpu().tolist())

        target_centers = scenario.target_areas.centers[target_area_mask]
        target_widths = scenario.target_areas.dimensions[target_area_mask][
            :, index_mapping.target_area_width
        ]
        target_heights = scenario.target_areas.dimensions[target_area_mask][
            :, index_mapping.target_area_height
        ]

        predicted_focal_spots = get_center_of_mass(
            bitmaps=predicted_flux,
            target_centers=target_centers,
            target_widths=target_widths,
            target_heights=target_heights,
            device=device,
        )

        focal_spot_error = torch.norm(predicted_focal_spots[:, :3] - focal_spots[:, :3], dim=1)
        all_focal_spot_errors_m.extend(focal_spot_error.cpu().tolist())

        # Compute per-heliostat distances to the reference target for mrad conversion.
        active_indices = torch.where(active_heliostats_mask.bool())[0]
        active_positions = heliostat_group.positions[active_indices, :3].to(device)
        distances = torch.norm(active_positions - reference_target.unsqueeze(0), dim=1)

        # Build name -> distance lookup for per-heliostat results.
        name_to_distance = {
            heliostat_group.names[idx.item()]: dist.item()
            for idx, dist in zip(active_indices, distances)
        }

        # Repeat distances to match number of focal spot error samples
        # (there may be multiple measurements per heliostat).
        num_active = active_indices.shape[0]
        num_focal_spots = focal_spot_error.shape[0]
        samples_per_heliostat = max(num_focal_spots // num_active, 1)
        distances_per_sample = distances.repeat_interleave(samples_per_heliostat)[:num_focal_spots]
        focal_spot_error_mrad = (focal_spot_error / distances_per_sample) * 1000.0
        all_focal_spot_errors_mrad.extend(focal_spot_error_mrad.cpu().tolist())

        heliostat_names = [
            name for name, _, _ in heliostat_data_mapping
            if name in heliostat_group.names
        ]
        for i, name in enumerate(heliostat_names):
            if i < len(pixel_loss):
                fse_m = focal_spot_error[i].item() if i < len(focal_spot_error) else None
                dist_m = name_to_distance.get(name)
                fse_mrad = (fse_m / dist_m * 1000.0) if (fse_m is not None and dist_m) else None
                results_per_heliostat[name] = {
                    "pixel_mse": pixel_loss[i].item(),
                    "focal_spot_error_m": fse_m,
                    "focal_spot_error_mrad": fse_mrad,
                    "distance_to_target_m": dist_m,
                }

    def _safe_mean(lst):
        return sum(lst) / len(lst) if lst else float("inf")

    metrics = {
        "mean_pixel_mse": _safe_mean(all_pixel_losses),
        "mean_focal_spot_error_m": _safe_mean(all_focal_spot_errors_m),
        "max_focal_spot_error_m": max(all_focal_spot_errors_m) if all_focal_spot_errors_m else float("inf"),
        "min_focal_spot_error_m": min(all_focal_spot_errors_m) if all_focal_spot_errors_m else float("inf"),
        "mean_focal_spot_error_mrad": _safe_mean(all_focal_spot_errors_mrad),
        "max_focal_spot_error_mrad": max(all_focal_spot_errors_mrad) if all_focal_spot_errors_mrad else float("inf"),
        "min_focal_spot_error_mrad": min(all_focal_spot_errors_mrad) if all_focal_spot_errors_mrad else float("inf"),
        "num_samples_evaluated": len(all_pixel_losses),
        "all_errors_m": all_focal_spot_errors_m,
        "all_errors_mrad": all_focal_spot_errors_mrad,
        "per_heliostat": results_per_heliostat,
    }
    return metrics


def visualize_flux_comparison(
    scenario: Scenario,
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    data_parser: PaintCalibrationDataParser,
    device: torch.device,
    output_dir: pathlib.Path = None,
    num_samples: int = 5,
    save_figures: bool = False,
):
    """
    Visualize comparison between predicted and measured flux images.
    """
    if save_figures and output_dir is not None:
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

        for i in range(min(len(predicted_flux), num_samples - samples_visualized)):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            im0 = axes[0].imshow(predicted_flux[i].cpu().detach(), cmap="hot")
            axes[0].set_title("Predicted Flux")
            axes[0].axis("off")
            plt.colorbar(im0, ax=axes[0], fraction=0.046)

            im1 = axes[1].imshow(measured_flux[i].cpu().detach(), cmap="hot")
            axes[1].set_title("Measured Flux")
            axes[1].axis("off")
            plt.colorbar(im1, ax=axes[1], fraction=0.046)

            diff = (predicted_flux[i] - measured_flux[i]).cpu().detach()
            im2 = axes[2].imshow(diff, cmap="coolwarm", vmin=-diff.abs().max(), vmax=diff.abs().max())
            axes[2].set_title("Difference (Predicted - Measured)")
            axes[2].axis("off")
            plt.colorbar(im2, ax=axes[2], fraction=0.046)

            plt.tight_layout()

            if save_figures and output_dir is not None:
                plt.savefig(output_dir / f"flux_comparison_{samples_visualized}.png", dpi=150)

            plt.show()

            samples_visualized += 1
            if samples_visualized >= num_samples:
                break

    print(f"Visualized {samples_visualized} flux comparison images")


def plot_tracking_error_histogram(
    errors_mrad: list[float],
    output_path: pathlib.Path,
    title: str = "Tracking Error Distribution",
) -> None:
    """
    Plot a histogram of tracking errors in mrad with a Gaussian fit overlay.
    Mirrors the style used in Tristan's paper.

    Parameters
    ----------
    errors_mrad : list[float]
        All per-sample tracking errors in milliradians.
    output_path : pathlib.Path
        Where to save the PNG.
    title : str
        Plot title.
    """
    errors = np.array(errors_mrad)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel("Tracking Error (mrad)", fontsize=13)
    ax.set_ylabel("Absolute Frequency", fontsize=13)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved tracking error histogram to {output_path}")


# ===================================================================
# Configuration
# ===================================================================

# ===== Environment toggle =====
IS_ON_DAIC = True

if IS_ON_DAIC:
    matplotlib.use("Agg")  # Non-interactive backend for HPC
    BASE_DIR = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
    BENCHMARK_DIR = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/src/paint_benchmarks")
else:
    BASE_DIR = pathlib.Path.cwd().parent
    BENCHMARK_DIR = BASE_DIR / "datasets" / "paint_benchmarks"

BENCHMARK_NAME = "benchmark_split-balanced_train-10_validation-30"
SCENARIO_PATH = BASE_DIR / "scenarios" / "all_heliostats_scenario" / "all_heliostats_scenario.h5"
_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = BASE_DIR / "outputs" / f"kin_recon_{_run_timestamp}"

# Attach a file handler to the ARTIST logger so all training logs are persisted.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_log_file_handler = logging.FileHandler(OUTPUT_DIR / "training.log")
_log_file_handler.setFormatter(
    logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
)
logging.getLogger().addHandler(_log_file_handler)

BENCHMARK_CSV = BENCHMARK_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
CALIBRATION_PROPERTIES_DIR = BENCHMARK_DIR / "datasets" / BENCHMARK_NAME / "calibration_properties"
FLUX_IMAGE_DIR = BENCHMARK_DIR / "datasets" / BENCHMARK_NAME / "flux_image"

# Training configuration
SAMPLE_LIMIT_PER_HELIOSTAT = 8
CENTROID_METHOD = paint_mappings.UTIS_KEY

print(f"\nRunning on DAIC: {IS_ON_DAIC}")
print(f"Base directory: {BASE_DIR}")
print(f"Benchmark CSV: {BENCHMARK_CSV}")
print(f"Scenario path: {SCENARIO_PATH}")
print(f"\nPaths exist:")
print(f"  Benchmark CSV: {BENCHMARK_CSV.exists()}")
print(f"  Scenario: {SCENARIO_PATH.exists()}")
print(f"  Calibration dir: {CALIBRATION_PROPERTIES_DIR.exists()}")
print(f"  Flux image dir: {FLUX_IMAGE_DIR.exists()}")


# ===================================================================
# Device Setup
# ===================================================================

device = get_device()
print(f"\nUsing device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# ===================================================================
# Build Heliostat Data Mappings
# ===================================================================

print("\nBuilding heliostat data mappings...")

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

print(f"\nTrain mapping: {len(train_mapping)} heliostats")
print(f"Test mapping: {len(test_mapping)} heliostats")

# Inspect the mappings
print("\nSample of train mapping:")
for heliostat_id, cal_paths, flux_paths in train_mapping[:3]:
    print(f"  Heliostat: {heliostat_id}, Calibration files: {len(cal_paths)}, Flux files: {len(flux_paths)}")
    print(f"    cal_paths: {cal_paths[0]}")
    print(f"    flux_paths: {flux_paths[0]}")


# ===================================================================
# Create Data Parsers
# ===================================================================

train_data_parser = PaintCalibrationDataParser(
    sample_limit=SAMPLE_LIMIT_PER_HELIOSTAT,
    centroid_extraction_method=CENTROID_METHOD,
)

eval_data_parser = PaintCalibrationDataParser(
    sample_limit=10,
    centroid_extraction_method=CENTROID_METHOD,
)

print(f"\nTrain parser sample limit: {SAMPLE_LIMIT_PER_HELIOSTAT}")
print(f"Eval parser sample limit: 10")
print(f"Centroid method: {CENTROID_METHOD}")


# ===================================================================
# Setup Data Dictionary & Load Scenario
# ===================================================================

data = {
    config_dictionary.data_parser: train_data_parser,
    config_dictionary.heliostat_data_mapping: train_mapping,
}

print("\nData dictionary created")

print(f"Loading scenario from: {SCENARIO_PATH}")
number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
    scenario_path=SCENARIO_PATH
)
print(f"Number of heliostat groups: {number_of_heliostat_groups}")


# ===================================================================
# Optimization Configuration
# ===================================================================

scheduler = config_dictionary.reduce_on_plateau
scheduler_parameters = {
    config_dictionary.gamma: 0.9,
    config_dictionary.min: 1e-6,
    config_dictionary.max: 1e-3,
    config_dictionary.reduce_factor: 0.5,   # less aggressive than 0.1 over 100 epochs
    config_dictionary.patience: 10,          # wait longer before reducing LR
    config_dictionary.threshold: 1e-4,
    config_dictionary.cooldown: 5,           # longer cooldown after each LR reduction
}

optimization_configuration = {
    config_dictionary.initial_learning_rate: 0.0005,
    config_dictionary.tolerance: 0.0001,
    config_dictionary.max_epoch: 100,
    config_dictionary.batch_size: 8,
    config_dictionary.log_step: 5,           # log every 5 epochs
    config_dictionary.early_stopping_delta: 1e-5,
    config_dictionary.early_stopping_patience: 20,  # allow more epochs before stopping
    config_dictionary.scheduler: scheduler,
    config_dictionary.scheduler_parameters: scheduler_parameters,
}

print("\nOptimization configuration:")
for key, value in optimization_configuration.items():
    if key != config_dictionary.scheduler_parameters:
        print(f"  {key}: {value}")


# ===================================================================
# Run Kinematic Reconstruction
# ===================================================================

final_loss_per_heliostat = None

try:
    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        with h5py.File(SCENARIO_PATH, "r") as scenario_file:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file, device=device
            )

        print("Scenario loaded successfully")
        print(f"Number of heliostats in scenario: {scenario.heliostat_field.number_of_heliostat_groups}")

        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

        kinematic_reconstructor = WortbergKinematicReconstructor(
            ddp_setup=ddp_setup,
            scenario=scenario,
            data=data,
            optimization_configuration=optimization_configuration,
            reconstruction_method=config_dictionary.kinematic_reconstruction_raytracing,
        )

        loss_definition = FocalSpotLoss(scenario=scenario)

        print("\nStarting kinematic reconstruction...")

        final_loss_per_heliostat = kinematic_reconstructor.reconstruct_kinematic(
            loss_definition=loss_definition, device=device
        )

    print("\nReconstruction complete!")

except Exception as e:
    print("\n" + "=" * 60)
    print("ERROR: Training crashed!")
    print("=" * 60)
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("=" * 60)

    if torch.cuda.is_available():
        try:
            print(f"\nGPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
            print("Attempting to clear GPU cache...")
            torch.cuda.empty_cache()
            print("GPU cache cleared")
        except Exception:
            print("Could not retrieve GPU memory information")

    print("\nFull traceback:")
    traceback.print_exc()
    raise


# ===================================================================
# Analyze Training Results
# ===================================================================

print(f"\nFinal loss per heliostat shape: {final_loss_per_heliostat.shape}")

valid_losses = final_loss_per_heliostat[final_loss_per_heliostat != float("inf")]

if len(valid_losses) > 0:
    print(f"\nTraining Results:")
    print(f"  Number of trained heliostats: {len(valid_losses)}")
    print(f"  Mean loss: {valid_losses.mean().item():.6f}")
    print(f"  Min loss: {valid_losses.min().item():.6f}")
    print(f"  Max loss: {valid_losses.max().item():.6f}")
    print(f"  Std loss: {valid_losses.std().item():.6f}")

    # Plot loss distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(valid_losses.detach().cpu().numpy(), bins=30, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Final Loss")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Final Losses")
    axes[0].axvline(valid_losses.mean().item(), color="red", linestyle="--", label=f"Mean: {valid_losses.mean().item():.4f}")
    axes[0].legend()

    sorted_losses = valid_losses.detach().cpu().numpy()
    sorted_losses.sort()
    axes[1].plot(sorted_losses, marker="o", markersize=3, linestyle="-", alpha=0.7)
    axes[1].set_xlabel("Heliostat Index (sorted)")
    axes[1].set_ylabel("Final Loss")
    axes[1].set_title("Sorted Final Losses")

    plt.tight_layout()
    if IS_ON_DAIC:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(OUTPUT_DIR / "loss_distribution.png", dpi=150)
        print(f"Saved loss distribution plot to {OUTPUT_DIR / 'loss_distribution.png'}")
    else:
        plt.show()


# ===================================================================
# Evaluate on Test Set
# ===================================================================

print("\nEvaluating on test set...")

test_metrics = evaluate_flux_accuracy(
    scenario=scenario,
    heliostat_data_mapping=test_mapping,
    data_parser=eval_data_parser,
    device=device,
)

print("\n" + "=" * 60)
print("TEST SET EVALUATION RESULTS")
print("=" * 60)
print(f"Number of samples evaluated: {test_metrics['num_samples_evaluated']}")
print(f"Mean pixel MSE:              {test_metrics['mean_pixel_mse']:.6f}")
print(f"Mean focal spot error:       {test_metrics['mean_focal_spot_error_m']:.4f} m  |  {test_metrics['mean_focal_spot_error_mrad']:.2f} mrad")
print(f"Min focal spot error:        {test_metrics['min_focal_spot_error_m']:.4f} m  |  {test_metrics['min_focal_spot_error_mrad']:.2f} mrad")
print(f"Max focal spot error:        {test_metrics['max_focal_spot_error_m']:.4f} m  |  {test_metrics['max_focal_spot_error_mrad']:.2f} mrad")
print("=" * 60)

# Display per-heliostat results
if test_metrics["per_heliostat"]:
    per_heliostat_df = pd.DataFrame.from_dict(test_metrics["per_heliostat"], orient="index")
    per_heliostat_df = per_heliostat_df.sort_values("focal_spot_error_m")
    print("\nPer-heliostat results (sorted by focal spot error):")
    print(per_heliostat_df.to_string())


# ===================================================================
# Visualize Flux Comparisons
# ===================================================================

print("\nGenerating tracking error histogram...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plot_tracking_error_histogram(
    errors_mrad=test_metrics["all_errors_mrad"],
    output_path=OUTPUT_DIR / "tracking_error_histogram.png",
    title="Heliostat Tracking Error Distribution (Test Set)",
)

print("\nGenerating flux comparison visualizations...")

visualize_flux_comparison(
    scenario=scenario,
    heliostat_data_mapping=test_mapping,
    data_parser=eval_data_parser,
    device=device,
    output_dir=OUTPUT_DIR / "visualizations",
    num_samples=5,
    save_figures=IS_ON_DAIC,
)


# ===================================================================
# Save Results
# ===================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Kinematic parameters ----
print("\nSaving optimized kinematic parameters...")
all_kinematic_params_json = {}
for group_index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
    group_name = f"group_{group_index}"
    all_kinematic_params_json[group_name] = {
        "heliostat_names": heliostat_group.names,
        "rotation_deviation_parameters": heliostat_group.kinematic.rotation_deviation_parameters.detach().cpu().tolist(),
        "actuator_parameters": heliostat_group.kinematic.actuators.optimizable_parameters.detach().cpu().tolist(),
    }

kinematic_params_file = OUTPUT_DIR / "all_kinematic_parameters.json"
with open(kinematic_params_file, "w") as f:
    json.dump(all_kinematic_params_json, f, indent=2)
print(f"Saved kinematic parameters to {kinematic_params_file}")

# ---- Final training losses ----
loss_values = final_loss_per_heliostat.tolist()
losses_json = {
    "per_heliostat_index": {
        str(i): (None if v == float("inf") else v)
        for i, v in enumerate(loss_values)
    },
    "summary": {
        "num_trained": int(len(valid_losses)),
        "mean": float(valid_losses.mean().item()),
        "min": float(valid_losses.min().item()),
        "max": float(valid_losses.max().item()),
        "std": float(valid_losses.std().item()),
    },
}
losses_file = OUTPUT_DIR / "final_loss_per_heliostat.json"
with open(losses_file, "w") as f:
    json.dump(losses_json, f, indent=2)
print(f"Saved final losses to {losses_file}")

# ---- Test metrics ----
metrics_file = OUTPUT_DIR / "test_metrics.json"
with open(metrics_file, "w") as f:
    json.dump(test_metrics, f, indent=2)
print(f"Saved metrics to {metrics_file}")

print("\nDone!")
