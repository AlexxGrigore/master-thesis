"""
Configuration for the full-field 200-samples synthetic perturbation experiment.
Edit this file to change hyperparameters, paths, and perturbation ranges.
"""
import pathlib

import paint.util.paint_mappings as paint_mappings
from artist.util import config_dictionary

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

SCENARIO_PATH              = BASE_DIR / "scenarios" / "full_field_200_samples_scenario" / "scenario.h5"
BENCHMARK_CSV              = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
CALIBRATION_PROPERTIES_DIR = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
FLUX_IMAGE_DIR             = PAINT_DIR / BENCHMARK_NAME / "flux_image"

# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------

CENTROID_METHOD = paint_mappings.UTIS_KEY

TRAIN_SAMPLES = 100
VAL_SAMPLES   = 50
TEST_SAMPLES  = 50

# Rays for synthetic data generation (high → clean, near-noiseless centroids).
# Must match what was used in generate_dataset.py (100).
SYNTH_GEN_RAYS = 100
# Rays during training (lower for speed; 10 works fine on CPU)
TRAIN_RAYS = 10
# Surface points per facet (N×N); 4 facets/heliostat → 4×N² pts total.
# 25×25 = 625 pts/facet, matches the scenario generation setting.
SURFACE_POINTS_PER_FACET = 25

# ---------------------------------------------------------------------------
# Perturbation
#
# Random per heliostat, seeded for reproducibility.
# Ranges match WortbergKinematicReconstructor deviation bounds (Wortberg 2025 Table 5.3).
# ---------------------------------------------------------------------------

PERTURBATION_SEED = 42

PERTURBATION_RANGES = {
    "rotation_rad":        0.005,  # ±5 mrad  — 4 joint tilts
    "actuator_angle_rad":  0.005,  # ±5 mrad  — a_i: 2 actuator initial angles (optimized)
    "actuator_stroke_m":   0.005,  # ±5 mm    — b_i: 2 actuator stroke lengths  (frozen)
    "actuator_offset_m":   0.005,  # ±5 mm    — c_i: 2 actuator offsets         (optimized)
    "translation_m":       0.05,   # ±50 mm   — 9 joint + concentrator translations (optimized)
    "base_position_m":     0.05,   # ±50 mm   — (east, north, up) base position  (optimized)
}

# ---------------------------------------------------------------------------
# Dataset
#
# "synthetic" — pre-generated ray-traced data (run generate_dataset.py first)
# "real"      — PAINT calibration images; kinematic perturbations are skipped
# ---------------------------------------------------------------------------

DATASET_TYPE = "real"

# ---------------------------------------------------------------------------
# Loss
#
# "focal_spot" — Euclidean distance between predicted and measured centroids (mrad-aligned)
# "pixel"      — MSE on Gaussian-blurred, peak-normalized flux bitmaps
# "alignment"  — MSE on motor positions converted to joint angles (no ray tracing)
# ---------------------------------------------------------------------------

LOSS_TYPE = "focal_spot"

# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

OPTIMIZATION_CONFIG = {
    config_dictionary.initial_learning_rate: 1e-4,
    config_dictionary.tolerance:             1e-6,
    config_dictionary.max_epoch:             100,
    config_dictionary.batch_size:            8,
    config_dictionary.log_step:              5,
    config_dictionary.early_stopping_window:   10,
    config_dictionary.early_stopping_delta:    1e-5,
    config_dictionary.early_stopping_patience: 400,  # > max_epoch → always runs fully
    config_dictionary.scheduler: config_dictionary.reduce_on_plateau,
    "scheduler_parameters": {
        config_dictionary.lr_min:       1e-6,
        config_dictionary.reduce_factor: 0.5,
        config_dictionary.patience:      10,
        config_dictionary.threshold:     1e-3,
        config_dictionary.cooldown:      5,
    },
}
