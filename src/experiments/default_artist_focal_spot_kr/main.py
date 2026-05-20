"""
Baseline experiment: default ARTIST KinematicsReconstructor + FocalSpotLoss.

No custom extensions. No WortbergKinematicReconstructor. No extra parameters.
Trains exactly the parameters ARTIST trains by default:
  - rotation_deviation_parameters (4 tilts)
  - actuators.optimizable_parameters (a_i + b_i, 4 values per heliostat)

Purpose: establish a clean lower bound on what is achievable with the stock
ARTIST optimizer on this dataset. Compare against the Wortberg experiments to
diagnose whether the extensions are helping or hurting.
"""

import pathlib
import sys

_pkg        = pathlib.Path(__file__).parent   # .../default_artist_focal_spot_kr/
_experiments = _pkg.parent                    # .../experiments/
_src        = _experiments.parent             # .../src/
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_experiments))
sys.path.insert(0, str(_pkg))

# ===================================================================
# Configuration
# ===================================================================

IS_ON_DAIC = True
SMOKE_TEST = not IS_ON_DAIC
SMOKE_TEST_HELIOSTAT = "AA31"
SMOKE_TEST_SAMPLE_LIMIT = 8

if IS_ON_DAIC:
    import matplotlib
    matplotlib.use("Agg")

import datetime
import gc
import json
import logging
import time
import traceback

import h5py
import torch
import paint.util.paint_mappings as paint_mappings
from artist.core.kinematics_reconstructor import KinematicsReconstructor
from artist.core.loss_functions import FocalSpotLoss
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment

from utils.evaluation import build_heliostat_data_mapping, evaluate_flux_accuracy
from utils.plotting import plot_training_curves, plot_tracking_error_histogram, visualize_flux_comparison
from utils.checkpointing import save_kinematic_parameters

torch.manual_seed(42)
torch.cuda.manual_seed(42)

set_logger_config()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

print("Imports completed successfully!")

if IS_ON_DAIC:
    BASE_DIR = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
else:
    BASE_DIR = _src.parent

if IS_ON_DAIC:
    PAINT_DIR = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")
else:
    PAINT_DIR = BASE_DIR / "datasets" / "paint"

BENCHMARK_NAME = "benchmark_split-balanced_train-10_validation-30"
SCENARIO_PATH = (
    BASE_DIR / "scenarios" / "one_heliostat_scenarios" / "scenario1.h5"
    if SMOKE_TEST
    else BASE_DIR / "scenarios" / "deflectometry_scenario" / "deflectometry_scenario.h5"
)
_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = (
    BASE_DIR / "outputs" / "local_runs" / f"default_artist_focal_spot_kr_{_run_timestamp}"
    if not IS_ON_DAIC
    else BASE_DIR / "outputs" / f"default_artist_focal_spot_kr_{_run_timestamp}"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_log_file_handler = logging.FileHandler(OUTPUT_DIR / "training.log")
_log_file_handler.setFormatter(
    logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
)
logging.getLogger().addHandler(_log_file_handler)

BENCHMARK_CSV              = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
CALIBRATION_PROPERTIES_DIR = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
FLUX_IMAGE_DIR             = PAINT_DIR / BENCHMARK_NAME / "flux_image"

SAMPLE_LIMIT_PER_HELIOSTAT = SMOKE_TEST_SAMPLE_LIMIT if SMOKE_TEST else 10
CENTROID_METHOD            = paint_mappings.UTIS_KEY

print(f"\nRunning on DAIC: {IS_ON_DAIC}")
print(f"Base directory:  {BASE_DIR}")
print(f"Scenario path:   {SCENARIO_PATH}")

# ===================================================================
# Device
# ===================================================================

device = get_device()
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ===================================================================
# Data mappings
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

if SMOKE_TEST:
    def _filter(mapping):
        return [
            (hid, cal[:SMOKE_TEST_SAMPLE_LIMIT], flux[:SMOKE_TEST_SAMPLE_LIMIT])
            for hid, cal, flux in mapping if hid == SMOKE_TEST_HELIOSTAT
        ]
    train_mapping = _filter(train_mapping)
    test_mapping  = _filter(test_mapping)
    print(f"[SMOKE TEST] Filtered to heliostat {SMOKE_TEST_HELIOSTAT}, {SMOKE_TEST_SAMPLE_LIMIT} samples")

print(f"Train mapping: {len(train_mapping)} heliostats")
print(f"Test mapping:  {len(test_mapping)} heliostats")

# ===================================================================
# Data parsers
# ===================================================================

train_data_parser = PaintCalibrationDataParser(
    sample_limit=SAMPLE_LIMIT_PER_HELIOSTAT,
    centroid_extraction_method=CENTROID_METHOD,
)
eval_data_parser = PaintCalibrationDataParser(
    sample_limit=SMOKE_TEST_SAMPLE_LIMIT if SMOKE_TEST else 10,
    centroid_extraction_method=CENTROID_METHOD,
)

# ===================================================================
# Optimization configuration — nested form required by ARTIST
# ===================================================================

optimization_configuration = {
    config_dictionary.optimization: {
        config_dictionary.initial_learning_rate: 1e-3,
        config_dictionary.tolerance:             1e-6,
        config_dictionary.max_epoch:             11 if SMOKE_TEST else 300,
        config_dictionary.batch_size:            8,
        config_dictionary.log_step:              5,
        config_dictionary.early_stopping_window:   10,
        config_dictionary.early_stopping_delta:    1e-5,
        config_dictionary.early_stopping_patience: 400,  # > max_epoch → always runs fully
    },
    config_dictionary.scheduler: {
        config_dictionary.scheduler_type:   config_dictionary.reduce_on_plateau,
        config_dictionary.min:              1e-6,
        config_dictionary.reduce_factor:    0.5,
        config_dictionary.patience:         10,
        config_dictionary.threshold:        1e-3,
        config_dictionary.cooldown:         5,
    },
}

print("\nOptimization configuration:")
for k, v in optimization_configuration[config_dictionary.optimization].items():
    print(f"  {k}: {v}")

# ===================================================================
# Run
# ===================================================================

try:
    number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
        scenario_path=SCENARIO_PATH
    )
    print(f"\nNumber of heliostat groups: {number_of_heliostat_groups}")

    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        print(f"\n{'=' * 60}")
        print("EXPERIMENT: default ARTIST KinematicsReconstructor + FocalSpotLoss")
        print(f"Parameters optimised: rotation_deviation (4) + actuators.optimizable (a_i + b_i)")
        print("=" * 60)

        # --- Load scenario ---
        with h5py.File(SCENARIO_PATH, "r") as scenario_file:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=scenario_file,
                device=device,
                number_of_surface_points_per_facet=torch.tensor([25, 25]),
            )
        scenario.set_number_of_rays(10)
        print(f"Heliostats: {scenario.heliostat_field.number_of_heliostats_per_group.sum().item()}")

        # --- Build reconstructor ---
        data = {
            config_dictionary.data_parser:          train_data_parser,
            config_dictionary.heliostat_data_mapping: train_mapping,
        }
        reconstructor = KinematicsReconstructor(
            ddp_setup=ddp_setup,
            scenario=scenario,
            data=data,
            optimization_configuration=optimization_configuration,
            reconstruction_method=config_dictionary.kinematics_reconstruction_raytracing,
        )

        loss_definition = FocalSpotLoss(scenario=scenario)

        # --- Train ---
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        t_train = time.time()

        final_loss_per_heliostat = reconstructor.reconstruct_kinematics(
            loss_definition=loss_definition,
            device=device,
        )

        train_time_s      = time.time() - t_train
        train_peak_gpu_gb = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
        print(f"Training — {train_time_s/60:.1f} min, peak GPU: {train_peak_gpu_gb:.2f} GB")
        log.info(f"Training — time: {train_time_s:.0f}s, peak GPU: {train_peak_gpu_gb:.2f} GB")

        # --- Save kinematic parameters ---
        save_kinematic_parameters(scenario, OUTPUT_DIR / "all_kinematic_parameters.json")

        # --- Training loss distribution ---
        valid_losses = final_loss_per_heliostat[final_loss_per_heliostat != float("inf")]
        if len(valid_losses) > 0:
            losses_np = valid_losses.detach().cpu().numpy()
            print(f"Train loss — mean: {losses_np.mean():.6f}, median: {float(losses_np.mean()):.6f}")

        # --- Evaluate ---
        del reconstructor
        gc.collect()
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        t_eval = time.time()

        test_metrics = evaluate_flux_accuracy(
            scenario=scenario,
            heliostat_data_mapping=test_mapping,
            data_parser=eval_data_parser,
            device=device,
        )

        eval_time_s      = time.time() - t_eval
        eval_peak_gpu_gb = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
        print(f"Evaluation — {eval_time_s/60:.1f} min, peak GPU: {eval_peak_gpu_gb:.2f} GB")

        print(f"\nTest — mean focal spot error:   {test_metrics['mean_focal_spot_error_mrad']:.2f} mrad")
        print(f"Test — median focal spot error: {test_metrics['median_focal_spot_error_mrad']:.2f} mrad")
        print(f"Test — min focal spot error:    {test_metrics['min_focal_spot_error_mrad']:.2f} mrad")
        print(f"Test — max focal spot error:    {test_metrics['max_focal_spot_error_mrad']:.2f} mrad")
        print(f"Test — samples evaluated:       {test_metrics['num_samples_evaluated']}")

        # --- Save metrics ---
        test_loss = test_metrics["mean_focal_spot_error_m"]

        metrics_to_save = {
            "mean_focal_spot_error_mrad":   test_metrics["mean_focal_spot_error_mrad"],
            "median_focal_spot_error_mrad": test_metrics["median_focal_spot_error_mrad"],
            "min_focal_spot_error_mrad":    test_metrics["min_focal_spot_error_mrad"],
            "max_focal_spot_error_mrad":    test_metrics["max_focal_spot_error_mrad"],
            "num_samples_evaluated":        test_metrics["num_samples_evaluated"],
            "per_heliostat":                test_metrics["per_heliostat"],
        }
        with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
            json.dump(metrics_to_save, f, indent=2)

        timing_stats = {
            "training_time_s":        round(train_time_s, 1),
            "training_time_min":      round(train_time_s / 60, 2),
            "training_peak_gpu_gb":   round(train_peak_gpu_gb, 3),
            "evaluation_time_s":      round(eval_time_s, 1),
            "evaluation_time_min":    round(eval_time_s / 60, 2),
            "evaluation_peak_gpu_gb": round(eval_peak_gpu_gb, 3),
        }
        with open(OUTPUT_DIR / "timing_stats.json", "w") as f:
            json.dump(timing_stats, f, indent=2)

        # --- Plots ---
        plot_training_curves(
            log_file=OUTPUT_DIR / "training.log",
            output_dir=OUTPUT_DIR,
            test_loss=test_loss,
        )
        plot_tracking_error_histogram(
            errors_mrad=test_metrics["all_errors_mrad"],
            output_path=OUTPUT_DIR / "tracking_error_histogram.png",
            title="Heliostat Tracking Error — default ARTIST (Test Set)",
        )
        visualize_flux_comparison(
            scenario=scenario,
            heliostat_data_mapping=test_mapping,
            data_parser=eval_data_parser,
            device=device,
            output_dir=OUTPUT_DIR / "visualizations",
            num_samples=5,
            save_figures=True,
        )

except Exception:
    print("\n" + "=" * 60)
    print("ERROR: Experiment crashed!")
    print("=" * 60)
    traceback.print_exc()
    raise

print("\nDone!")
