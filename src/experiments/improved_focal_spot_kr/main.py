import pathlib
import sys

_pkg = pathlib.Path(__file__).parent   # .../src/improved_focal_spot_kr/
_experiments = _pkg.parent             # .../src/experiments/
_src = _experiments.parent             # .../src/
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
import logging
import traceback

import torch
import paint.util.paint_mappings as paint_mappings
from artist.core.loss_functions import FocalSpotLoss
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment
from artist_extensions.kinematic_reconstructors import (
    RotationsOnlyReconstructor,
    RotationsActuatorsReconstructor,
    RotationsActuatorsBasePosReconstructor,
)
from utils.evaluation import build_heliostat_data_mapping
from experiment import run_experiment

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
    BASE_DIR / "outputs" / "local_runs" / f"improved_focal_spot_kr_{_run_timestamp}"
    if not IS_ON_DAIC
    else BASE_DIR / "outputs" / f"improved_focal_spot_kr_{_run_timestamp}"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_log_file_handler = logging.FileHandler(OUTPUT_DIR / "training.log")
_log_file_handler.setFormatter(
    logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
)
logging.getLogger().addHandler(_log_file_handler)

BENCHMARK_CSV = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
CALIBRATION_PROPERTIES_DIR = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
FLUX_IMAGE_DIR = PAINT_DIR / BENCHMARK_NAME / "flux_image"

SAMPLE_LIMIT_PER_HELIOSTAT = SMOKE_TEST_SAMPLE_LIMIT if SMOKE_TEST else 10
CENTROID_METHOD = paint_mappings.UTIS_KEY

print(f"\nRunning on DAIC: {IS_ON_DAIC}")
print(f"Base directory: {BASE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Scenario: {SCENARIO_PATH}")

# ===================================================================
# Device setup
# ===================================================================

device = get_device()
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ===================================================================
# Heliostat data mappings
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
validation_mapping = build_heliostat_data_mapping(
    benchmark_csv=BENCHMARK_CSV,
    calibration_properties_dir=CALIBRATION_PROPERTIES_DIR,
    flux_image_dir=FLUX_IMAGE_DIR,
    split="validation",
)

if SMOKE_TEST:
    def _filter_mapping(mapping):
        return [
            (hid, cal[:SMOKE_TEST_SAMPLE_LIMIT], flux[:SMOKE_TEST_SAMPLE_LIMIT])
            for hid, cal, flux in mapping
            if hid == SMOKE_TEST_HELIOSTAT
        ]
    train_mapping = _filter_mapping(train_mapping)
    test_mapping = _filter_mapping(test_mapping)
    validation_mapping = _filter_mapping(validation_mapping)
    print(f"[SMOKE TEST] Filtered to heliostat {SMOKE_TEST_HELIOSTAT}")

print(f"\nTrain:      {len(train_mapping)} heliostats")
print(f"Validation: {len(validation_mapping)} heliostats")
print(f"Test:       {len(test_mapping)} heliostats")

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
# Optimization configuration
# ===================================================================

optimization_configuration = {
    config_dictionary.initial_learning_rate: 1e-4,
    config_dictionary.tolerance: 1e-6,
    config_dictionary.max_epoch: 11 if SMOKE_TEST else 300,
    config_dictionary.batch_size: 8,
    config_dictionary.log_step: 5,
    config_dictionary.early_stopping_window: 10,
    config_dictionary.early_stopping_delta: 1e-5,
    config_dictionary.early_stopping_patience: 400,  # > max_epoch → always runs fully
    config_dictionary.scheduler: config_dictionary.reduce_on_plateau,
    "scheduler_parameters": {
        config_dictionary.min: 1e-6,
        config_dictionary.reduce_factor: 0.5,
        config_dictionary.patience: 10,
        config_dictionary.threshold: 1e-3,
        config_dictionary.cooldown: 5,
    },
}

print("\nOptimization configuration:")
for key, value in optimization_configuration.items():
    if key != "scheduler_parameters":
        print(f"  {key}: {value}")

# ===================================================================
# Experiment configs:
#   A — rotations only           (4 params)
#   B — rotations + actuators    (8 params)
#   C — rotations + actuators + base position  (11 params, no translations)
# ===================================================================

CONFIGS = [
    ("A_rotations_only",              RotationsOnlyReconstructor),
    ("B_rotations_actuators",         RotationsActuatorsReconstructor),
    ("C_rotations_actuators_basepos", RotationsActuatorsBasePosReconstructor),
]

# ===================================================================
# Run
# ===================================================================

number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
    scenario_path=SCENARIO_PATH
)

all_metrics = {}

try:
    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        for config_name, reconstructor_cls in CONFIGS:
            print(f"\n{'=' * 60}")
            print(f"EXPERIMENT: {config_name}")
            print("=" * 60)

            try:
                metrics = run_experiment(
                    loss_name=config_name,
                    loss_fn_factory=lambda scenario: FocalSpotLoss(scenario=scenario),
                    reconstructor_cls=reconstructor_cls,
                    ddp_setup=ddp_setup,
                    device=device,
                    scenario_path=SCENARIO_PATH,
                    train_mapping=train_mapping,
                    test_mapping=test_mapping,
                    train_data_parser=train_data_parser,
                    eval_data_parser=eval_data_parser,
                    optimization_configuration=optimization_configuration,
                    output_dir=OUTPUT_DIR,
                    save_figures=True,
                    validation_mapping=validation_mapping,
                )
                all_metrics[config_name] = metrics
            except Exception:
                log.exception(f"Experiment {config_name} failed — continuing with next.")
                traceback.print_exc()

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
print("RESULTS SUMMARY")
print("=" * 60)
for config_name, metrics in all_metrics.items():
    print(f"\n  [{config_name}]")
    print(f"    Baseline (untrained): {metrics.get('baseline_mean_mrad', float('nan')):.2f} mrad")
    print(f"    Mean focal spot error:   {metrics['mean_focal_spot_error_mrad']:.2f} mrad")
    print(f"    Median focal spot error: {metrics['median_focal_spot_error_mrad']:.2f} mrad")
    print(f"    Improvement: {metrics.get('improvement_mrad', float('nan')):+.2f} mrad")
print("=" * 60)

print("\nDone!")
