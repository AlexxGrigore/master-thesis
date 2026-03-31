import pathlib
import sys

_pkg = pathlib.Path(__file__).parent   # .../src/resolution_ablation/
_src = _pkg.parent                     # .../src/
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_pkg))

# ===================================================================
# Configuration
# ===================================================================

IS_ON_DAIC = True
SMOKE_TEST = not IS_ON_DAIC
SMOKE_TEST_HELIOSTAT = "AH29"   # guaranteed in blur_ablation_scenario.h5
SMOKE_TEST_SAMPLE_LIMIT = 8

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
from experiment import run_experiment, plot_resolution_comparison

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
else:
    BASE_DIR = pathlib.Path(__file__).parent.parent.parent

if IS_ON_DAIC:
    PAINT_DIR = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")
else:
    PAINT_DIR = BASE_DIR / "datasets" / "paint"

BENCHMARK_NAME = "benchmark_split-balanced_train-10_validation-30"
SCENARIO_PATH = (
    BASE_DIR / "scenarios" / "one_heliostat_scenarios" / "scenario1.h5"
    if SMOKE_TEST
    else BASE_DIR / "scenarios" / "blur_ablation_scenario" / "blur_ablation_scenario.h5"
)
_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = (
    BASE_DIR / "outputs" / "local_runs" / f"resolution_ablation_{_run_timestamp}"
    if not IS_ON_DAIC
    else BASE_DIR / "outputs" / f"resolution_ablation_{_run_timestamp}"
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
TRAIN_BASE_POSITION_DEVIATION = True
CENTROID_METHOD = paint_mappings.UTIS_KEY

# 18 heliostats in blur_ablation_scenario.h5 — stratified sample across distance × lateral position.
BLUR_ABLATION_HELIOSTAT_IDS = frozenset({
    "AH29", "AA30", "AA33", "AL35", "AC37", "AB47",
    "AQ28", "AQ25", "BA37", "AW40", "AZ43", "AZ45",
    "BC32", "BF29", "AZ55", "AZ52", "AY72", "BA71",
})

# Resolution configurations: (name, surface_pts_per_facet, n_rays)
RESOLUTION_CONFIGS = [
    ("low_10x5",    10,  5),
    ("med_25x10",   25, 10),
    ("med_25x20",   25, 20),
    ("high_50x20",  50, 20),
    ("best_75x20",  75, 20),
]

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


def _filter_to_blur_ablation(mapping: list) -> list:
    """Keep only the 18 blur-ablation heliostats."""
    return [(hid, cal, flux) for hid, cal, flux in mapping if hid in BLUR_ABLATION_HELIOSTAT_IDS]


if SMOKE_TEST:
    def _filter_mapping(mapping):
        return [
            (hid, cal[:SMOKE_TEST_SAMPLE_LIMIT], flux[:SMOKE_TEST_SAMPLE_LIMIT])
            for hid, cal, flux in mapping
            if hid == SMOKE_TEST_HELIOSTAT
        ]
    train_mapping      = _filter_mapping(train_mapping)
    test_mapping       = _filter_mapping(test_mapping)
    validation_mapping = _filter_mapping(validation_mapping)
    RESOLUTION_CONFIGS = [("smoke_10x5", 10, 5)]
    print(f"\n[SMOKE TEST] Filtered to heliostat {SMOKE_TEST_HELIOSTAT}, {SMOKE_TEST_SAMPLE_LIMIT} samples/split")
else:
    train_mapping      = _filter_to_blur_ablation(train_mapping)
    test_mapping       = _filter_to_blur_ablation(test_mapping)
    validation_mapping = _filter_to_blur_ablation(validation_mapping)

print(f"\nTrain mapping:      {len(train_mapping)} heliostats")
print(f"Validation mapping: {len(validation_mapping)} heliostats")
print(f"Test mapping:       {len(test_mapping)} heliostats")

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
    "scheduler_parameters": {
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
    config_dictionary.early_stopping_patience: 150,
    config_dictionary.scheduler: config_dictionary.reduce_on_plateau,
    "scheduler_parameters": {
        config_dictionary.min: 1e-8,
        config_dictionary.reduce_factor: 0.5,
        config_dictionary.patience: 30,
        config_dictionary.threshold: 1e-3,
        config_dictionary.cooldown: 5,
    },
}

print("\nOptimization configuration:")
print(f"  Phase 1 — lr={PHASE1_OPT_CONFIG[config_dictionary.initial_learning_rate]}, "
      f"max_epoch={PHASE1_OPT_CONFIG[config_dictionary.max_epoch]}")
print(f"  Phase 2 — lr={PHASE2_OPT_CONFIG[config_dictionary.initial_learning_rate]}, "
      f"max_epoch={PHASE2_OPT_CONFIG[config_dictionary.max_epoch]}")
print(f"  Configs to run: {[c[0] for c in RESOLUTION_CONFIGS]}")


# ===================================================================
# Run all configurations
# ===================================================================

all_metrics = {}

try:
    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        for config_name, surface_pts, n_rays in RESOLUTION_CONFIGS:
            print(f"\n{'=' * 60}")
            print(f"EXPERIMENT: {config_name}  (surface_pts={surface_pts}, n_rays={n_rays})")
            print("=" * 60)

            metrics = run_experiment(
                loss_name=config_name,
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
                surface_pts=surface_pts,
                n_rays=n_rays,
                save_figures=True,
                train_position_deviation=TRAIN_BASE_POSITION_DEVIATION,
            )
            all_metrics[config_name] = metrics

        # ---- Cross-experiment comparison plots ----
        print(f"\n{'=' * 60}")
        print("Generating comparison plots...")
        print("=" * 60)
        plot_resolution_comparison(all_metrics=all_metrics, output_dir=OUTPUT_DIR)

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
print("RESULTS SUMMARY — Resolution Ablation")
print("=" * 60)
for config_name, metrics in all_metrics.items():
    print(f"\n  [{config_name}]")
    print(f"    Mean focal spot error:   {metrics['mean_focal_spot_error_mrad']:.2f} mrad")
    print(f"    Median focal spot error: {metrics['median_focal_spot_error_mrad']:.2f} mrad")
    print(f"    Min:  {metrics['min_focal_spot_error_mrad']:.2f} mrad  "
          f"Max: {metrics['max_focal_spot_error_mrad']:.2f} mrad")
print("=" * 60)

print("\nDone!")
