import pathlib
import sys

_pkg = pathlib.Path(__file__).parent   # .../src/kr_train_pixel_loss/
_src = _pkg.parent                     # .../src/
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_pkg))

# ===================================================================
# Configuration
# ===================================================================

IS_ON_DAIC = True
SMOKE_TEST = not IS_ON_DAIC  # runs locally with 1 heliostat, 3 epochs — smoke-tests the full code path
SMOKE_TEST_HELIOSTAT = "AA31"
SMOKE_TEST_SAMPLE_LIMIT = 8

# Must be set before pyplot is imported anywhere (including from plotting).
if IS_ON_DAIC:
    import matplotlib
    matplotlib.use("Agg")

import datetime
import logging
import traceback

import torch
import paint.util.paint_mappings as paint_mappings
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment
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
    BASE_DIR = pathlib.Path(__file__).parent.parent.parent
    BENCHMARK_DIR = BASE_DIR / "datasets" / "paint_benchmarks"

BENCHMARK_NAME = "benchmark_split-balanced_train-10_validation-30"
SCENARIO_PATH = (
    BASE_DIR / "scenarios" / "one_heliostat_scenarios" / "scenario1.h5"
    if SMOKE_TEST
    else BASE_DIR / "scenarios" / "deflectometry_scenario" / "deflectometry_scenario.h5"
)
_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = (
    BASE_DIR / "outputs" / "local_runs" / f"pixel_loss_kr_blurr_{_run_timestamp}"
    if not IS_ON_DAIC
    else BASE_DIR / "outputs" / f"pixel_loss_kr_{_run_timestamp}"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_log_file_handler = logging.FileHandler(OUTPUT_DIR / "training.log")
_log_file_handler.setFormatter(
    logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
)
logging.getLogger().addHandler(_log_file_handler)

BENCHMARK_CSV = BENCHMARK_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
CALIBRATION_PROPERTIES_DIR = BENCHMARK_DIR / "datasets" / BENCHMARK_NAME / "calibration_properties"
FLUX_IMAGE_DIR = BENCHMARK_DIR / "datasets" / BENCHMARK_NAME / "flux_image"

SAMPLE_LIMIT_PER_HELIOSTAT = SMOKE_TEST_SAMPLE_LIMIT if SMOKE_TEST else 10
TRAIN_BASE_POSITION_DEVIATION = True
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
    print(f"\n[SMOKE TEST] Filtered to heliostat {SMOKE_TEST_HELIOSTAT}, {SMOKE_TEST_SAMPLE_LIMIT} samples/split")

print(f"\nTrain mapping: {len(train_mapping)} heliostats")
print(f"Validation mapping: {len(validation_mapping)} heliostats")
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
    sample_limit=SMOKE_TEST_SAMPLE_LIMIT if SMOKE_TEST else 10,
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
#
# Two-phase training:
#
# Phase 1 — FocalSpotLoss pretraining (WortbergKinematicReconstructor).
#   Gets heliostats roughly aligned so reflected light reliably hits the
#   target before pixel loss takes over. ReduceLROnPlateau keeps the LR
#   high while progress is happening and only reduces on genuine plateaus.
#   Early stopping patience > max_epoch so Phase 1 always runs fully.
#   100 epochs at lr=1e-4: with patience=10+cooldown=5=15 epochs/LR cycle,
#   we get ~6 LR reductions — more effective than 300 epochs at 1e-3 with
#   a late LR drop.
#
# Phase 2 — PixelLoss fine-tuning (WortbergPixelReconstructor).
#   Continues from Phase 1 weights. Fresh Adam optimizer (no stale
#   momentum). LR starts at 1e-5 for conservative pixel-level fine-tuning.
#   Early stopping patience=150: with ReduceLROnPlateau patience=30+
#   cooldown=5=35 epochs/cycle, this allows ~4 LR reductions before
#   stopping — enough to exhaust the LR schedule if needed.

LOSS_NAME = "pixel_loss"

PHASE1_OPT_CONFIG = {
    config_dictionary.initial_learning_rate: 1e-4,
    config_dictionary.tolerance: 1e-6,
    config_dictionary.max_epoch: 11 if SMOKE_TEST else 100,
    config_dictionary.batch_size: 8,
    config_dictionary.log_step: 5,
    config_dictionary.early_stopping_window: 10,
    config_dictionary.early_stopping_delta: 1e-5,
    config_dictionary.early_stopping_patience: 200,  # > max_epoch → always runs fully
    config_dictionary.scheduler: config_dictionary.reduce_on_plateau,
    config_dictionary.scheduler_parameters: {
        config_dictionary.min: 1e-8,
        config_dictionary.reduce_factor: 0.5,
        config_dictionary.patience: 10,
        config_dictionary.threshold: 1e-3,
        config_dictionary.cooldown: 5,
    },
}

PHASE2_OPT_CONFIG = {
    config_dictionary.initial_learning_rate: 1e-5,
    config_dictionary.tolerance: 1e-8,
    config_dictionary.max_epoch: 11 if SMOKE_TEST else 300,
    config_dictionary.batch_size: 8,
    config_dictionary.log_step: 5,
    config_dictionary.early_stopping_window: 10,
    config_dictionary.early_stopping_delta: 1e-5,
    config_dictionary.early_stopping_patience: 150,  # ~4 LR cycles (35 epochs each)
    config_dictionary.scheduler: config_dictionary.reduce_on_plateau,
    config_dictionary.scheduler_parameters: {
        config_dictionary.min: 1e-8,
        config_dictionary.reduce_factor: 0.5,
        config_dictionary.patience: 30,
        config_dictionary.threshold: 1e-3,
        config_dictionary.cooldown: 5,
    },
}


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
        print(f"EXPERIMENT: {LOSS_NAME} (two-phase)")
        print("=" * 60)
        print(f"  Phase 1 — FocalSpotLoss: lr={PHASE1_OPT_CONFIG[config_dictionary.initial_learning_rate]}, "
              f"max_epoch={PHASE1_OPT_CONFIG[config_dictionary.max_epoch]}")
        print(f"  Phase 2 — PixelLoss:     lr={PHASE2_OPT_CONFIG[config_dictionary.initial_learning_rate]}, "
              f"max_epoch={PHASE2_OPT_CONFIG[config_dictionary.max_epoch]}")

        metrics = run_experiment(
            loss_name=LOSS_NAME,
            phase1_opt_config=PHASE1_OPT_CONFIG,
            phase2_opt_config=PHASE2_OPT_CONFIG,
            ddp_setup=ddp_setup,
            device=device,
            scenario_path=SCENARIO_PATH,
            train_mapping=train_mapping,
            test_mapping=test_mapping,
            validation_mapping=validation_mapping,
            train_data_parser=train_data_parser,
            eval_data_parser=eval_data_parser,
            output_dir=OUTPUT_DIR,
            save_figures=True,
            train_position_deviation=TRAIN_BASE_POSITION_DEVIATION,
        )

        print(f"\n{'=' * 60}")
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"  Mean focal spot error:   {metrics['mean_focal_spot_error_mrad']:.2f} mrad")
        print(f"  Median focal spot error: {metrics['median_focal_spot_error_mrad']:.2f} mrad")
        print(f"  Min focal spot error:    {metrics['min_focal_spot_error_mrad']:.2f} mrad")
        print(f"  Max focal spot error:    {metrics['max_focal_spot_error_mrad']:.2f} mrad")
        print(f"  Samples evaluated:       {metrics['num_samples_evaluated']}")
        print("=" * 60)

except Exception as e:
    print("\n" + "=" * 60)
    print("ERROR: Experiment runner crashed!")
    print("=" * 60)
    traceback.print_exc()
    raise

print("\nDone!")
