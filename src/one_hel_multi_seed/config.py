"""Configuration for the one-heliostat multi-seed sensitivity experiment."""
import pathlib

import paint.util.paint_mappings as paint_mappings
from artist.util import constants as config_dictionary

IS_ON_DAIC = False

BENCHMARK_NAME = "benchmark_split-balanced_train-100_validation-50_deflectometry"

SEEDS      = [42, 123, 456, 789, 1337, 2024, 31415, 271828, 99999, 12345]
HELIOSTATS = ["AC36", "AG33", "AO34", "AW36", "BE35"]

if IS_ON_DAIC:
    BASE_DIR  = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
    PAINT_DIR = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")
else:
    BASE_DIR  = pathlib.Path(__file__).resolve().parents[2]
    PAINT_DIR = BASE_DIR / "datasets" / "paint"

ONE_HELIOSTAT_SCENARIOS_DIR = BASE_DIR / "scenarios" / "one_heliostat_scenarios"
BENCHMARK_CSV               = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
CALIBRATION_PROPERTIES_DIR  = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
FLUX_IMAGE_DIR              = PAINT_DIR / BENCHMARK_NAME / "flux_image"

CENTROID_METHOD = paint_mappings.UTIS_KEY

TRAIN_SIZES = [1, 5, 10, 20, 25, 50, 75, 100]
VAL_SAMPLES  = 50
TEST_SAMPLES = 50

SYNTH_GEN_RAYS           = 100
TRAIN_RAYS               = 10
SURFACE_POINTS_PER_FACET = 25

LOSS_TYPE     = "focal_spot"
STAGE1_EPOCHS = 20
STAGE2_EPOCHS = 200

PERTURBATION_RANGES = {
    "rotation_rad":       0.003,
    "actuator_angle_rad": 0.003,
    "actuator_stroke_m":  0.003,
    "actuator_offset_m":  0.003,
    "translation_m":      0.015,
    "base_position_m":    0.05,
}

OPTIMIZATION_CONFIG = {
    config_dictionary.initial_learning_rate: 1e-4,
    config_dictionary.tolerance:             1e-6,
    config_dictionary.max_epoch:             100,
    config_dictionary.batch_size:            8,
    config_dictionary.log_step:              5,
    config_dictionary.early_stopping_window:   10,
    config_dictionary.early_stopping_delta:    1e-5,
    config_dictionary.early_stopping_patience: 400,
    config_dictionary.scheduler: config_dictionary.reduce_on_plateau,
    "scheduler_parameters": {
        config_dictionary.lr_min:       1e-6,
        config_dictionary.reduce_factor: 0.5,
        config_dictionary.patience:      10,
        config_dictionary.threshold:     1e-3,
        config_dictionary.cooldown:      5,
    },
}
