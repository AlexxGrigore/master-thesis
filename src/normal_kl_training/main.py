import pathlib
import sys

_pkg = pathlib.Path(__file__).parent   # .../src/normal_kr_training/
_src = _pkg.parent                     # .../src/
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_pkg))

# ===================================================================
# Configuration
# ===================================================================

IS_ON_DAIC = True

# Must be set before pyplot is imported anywhere (including from plotting).
if IS_ON_DAIC:
    import matplotlib
    matplotlib.use("Agg")

import datetime
import json
import logging
import traceback

import torch
import paint.util.paint_mappings as paint_mappings
from artist.core.loss_functions import FocalSpotLoss
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment
from artist_extensions.kinematic_reconstructors import WortbergKinematicReconstructor
from utils.evaluation import build_heliostat_data_mapping
from experiment import run_experiment

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Setup logging
set_logger_config()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

print("Imports completed successfully!")

if IS_ON_DAIC:
    BASE_DIR = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
    BENCHMARK_DIR = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/src/paint_benchmarks")
else:
    BASE_DIR = pathlib.Path.cwd().parent
    BENCHMARK_DIR = BASE_DIR / "datasets" / "paint_benchmarks"

BENCHMARK_NAME = "benchmark_split-balanced_train-10_validation-30"
SCENARIO_PATH = BASE_DIR / "scenarios" / "all_heliostats_scenario" / "all_heliostats_scenario.h5"
_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = BASE_DIR / "outputs" / f"normal_kr_{_run_timestamp}"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_log_file_handler = logging.FileHandler(OUTPUT_DIR / "training.log")
_log_file_handler.setFormatter(
    logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
)
logging.getLogger().addHandler(_log_file_handler)

BENCHMARK_CSV = BENCHMARK_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
CALIBRATION_PROPERTIES_DIR = BENCHMARK_DIR / "datasets" / BENCHMARK_NAME / "calibration_properties"
FLUX_IMAGE_DIR = BENCHMARK_DIR / "datasets" / BENCHMARK_NAME / "flux_image"

SAMPLE_LIMIT_PER_HELIOSTAT = 10
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
# Load Scenario Metadata
# ===================================================================

print(f"Loading scenario from: {SCENARIO_PATH}")
number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
    scenario_path=SCENARIO_PATH
)
print(f"Number of heliostat groups: {number_of_heliostat_groups}")


# ===================================================================
# Optimization Configuration
# ===================================================================

optimization_configuration = {
    config_dictionary.initial_learning_rate: 0.001,
    config_dictionary.tolerance: 0.0001,
    config_dictionary.max_epoch: 300,
    config_dictionary.batch_size: 8,
    config_dictionary.log_step: 5,
    config_dictionary.early_stopping_window: 10,
    config_dictionary.early_stopping_delta: 1e-5,
    config_dictionary.early_stopping_patience: 20,
    config_dictionary.scheduler: config_dictionary.reduce_on_plateau,
    config_dictionary.scheduler_parameters: {
        config_dictionary.gamma: 0.9,
        config_dictionary.min: 1e-6,
        config_dictionary.max: 1e-2,
        config_dictionary.reduce_factor: 0.5,
        config_dictionary.patience: 10,
        config_dictionary.threshold: 1e-4,
        config_dictionary.cooldown: 5,
    },
}

print("\nOptimization configuration:")
for key, value in optimization_configuration.items():
    if key != config_dictionary.scheduler_parameters:
        print(f"  {key}: {value}")


# ===================================================================
# Run Experiment
# ===================================================================

try:
    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        print(f"\n{'=' * 60}")
        print("EXPERIMENT: focal_spot_loss")
        print("=" * 60)

        metrics = run_experiment(
            loss_name="focal_spot_loss",
            loss_fn_factory=lambda scenario: FocalSpotLoss(scenario=scenario),
            reconstructor_cls=WortbergKinematicReconstructor,
            ddp_setup=ddp_setup,
            device=device,
            scenario_path=SCENARIO_PATH,
            train_mapping=train_mapping,
            test_mapping=test_mapping,
            train_data_parser=train_data_parser,
            eval_data_parser=eval_data_parser,
            optimization_configuration=optimization_configuration,
            output_dir=OUTPUT_DIR,
            save_figures=IS_ON_DAIC,
        )

except Exception as e:
    print("\n" + "=" * 60)
    print("ERROR: Experiment runner crashed!")
    print("=" * 60)
    traceback.print_exc()
    raise


# ===================================================================
# Summary
# ===================================================================

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)
print(f"  Mean focal spot error:   {metrics['mean_focal_spot_error_mrad']:.2f} mrad")
print(f"  Median focal spot error: {metrics['median_focal_spot_error_mrad']:.2f} mrad")
print(f"  Min focal spot error:    {metrics['min_focal_spot_error_mrad']:.2f} mrad")
print(f"  Max focal spot error:    {metrics['max_focal_spot_error_mrad']:.2f} mrad")
print(f"  Samples evaluated:       {metrics['num_samples_evaluated']}")
print("=" * 60)

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(
        {
            "mean_focal_spot_error_mrad": metrics["mean_focal_spot_error_mrad"],
            "median_focal_spot_error_mrad": metrics["median_focal_spot_error_mrad"],
            "min_focal_spot_error_mrad": metrics["min_focal_spot_error_mrad"],
            "max_focal_spot_error_mrad": metrics["max_focal_spot_error_mrad"],
            "num_samples_evaluated": metrics["num_samples_evaluated"],
        },
        f,
        indent=2,
    )
print(f"\nSaved results to {OUTPUT_DIR / 'results.json'}")

print("\nDone!")
