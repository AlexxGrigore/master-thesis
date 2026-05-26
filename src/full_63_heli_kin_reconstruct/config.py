"""
Configuration for the full-63-heliostat kinematic reconstruction experiment.

Corrected data pipeline: the synthetic dataset is generated from the PERTURBED
scenario, so the KR must learn the perturbation values from a clean starting
point (the real inverse problem), rather than learn to undo known perturbations
against a clean reference dataset.
"""
import pathlib

import paint.util.paint_mappings as paint_mappings
from artist.util import constants as config_dictionary

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

IS_ON_DAIC = False

BENCHMARK_NAME = "benchmark_split-balanced_train-100_validation-50_deflectometry"

if IS_ON_DAIC:
    BASE_DIR  = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
    PAINT_DIR = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")
else:
    BASE_DIR  = pathlib.Path(__file__).resolve().parents[2]
    PAINT_DIR = BASE_DIR / "datasets" / "paint"

SCENARIO_PATH              = BASE_DIR / "scenarios" / "full_63_heli_kin_reconstruct" / "scenario.h5"
BENCHMARK_CSV              = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
CALIBRATION_PROPERTIES_DIR = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
FLUX_IMAGE_DIR             = PAINT_DIR / BENCHMARK_NAME / "flux_image"

# Synthetic data lives in a separate directory from the old experiment so both
# datasets can coexist without conflict.
SYNTHETIC_DATA_DIR = BASE_DIR / "scenarios" / "full_63_heli_kin_reconstruct" / "synthetic_data"

# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------

CENTROID_METHOD = paint_mappings.UTIS_KEY

TRAIN_SAMPLES = 100
VAL_SAMPLES   = 50
TEST_SAMPLES  = 50

SYNTH_GEN_RAYS           = 100   # high ray count → clean centroids
TRAIN_RAYS               = 10
SURFACE_POINTS_PER_FACET = 25    # 25×25 = 625 pts/facet

# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------

PERTURBATION_SEED = 42

PERTURBATION_RANGES = {
    "rotation_rad":        0.003,
    "actuator_angle_rad":  0.003,
    "actuator_stroke_m":   0.003,
    "actuator_offset_m":   0.003,
    "translation_m":       0.015,
    "base_position_m":     0.05,
}

# ---------------------------------------------------------------------------
# Loss / training
# ---------------------------------------------------------------------------

# Available loss types: "focal_spot", "pixel", "contour", "alignment"
LOSS_TYPE     = "focal_spot"
STAGE1_EPOCHS = 20
STAGE2_EPOCHS = 100

# Hyperparameters for ContourLoss (only used when LOSS_TYPE = "contour").
# Defaults follow Wortberg (2025); τ and η were Bayesian-optimised by Tristan.
CONTOUR_PARAMS = {
    "smoothing_rounds":     2,
    "gaussian_kernel_size": 5,
    "gaussian_sigma":       1.0,
    "threshold_tau":        0.58,   # sigmoid centre
    "threshold_eta":        70.0,   # sigmoid sharpness
    "weight_coarse":        0.3,    # β  (distance-field term)
    "weight_gravity":       0.2,    # γ  (COM-distance term); fine = 1 − β − γ = 0.5
}

# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

OPTIMIZATION_CONFIG = {
    config_dictionary.initial_learning_rate: 1e-4,
    config_dictionary.tolerance:             1e-6,
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
