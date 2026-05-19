"""
Configuration for the 5-heliostat synthetic perturbation experiment.
"""
import pathlib

import paint.util.paint_mappings as paint_mappings
from artist.util import config_dictionary

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

IS_ON_DAIC = False

HELIOSTAT_IDS = ["AA31", "AQ28", "BA37", "BC33", "AZ55"]

BENCHMARK_NAME = "benchmark_split-balanced_train-50_validation-10"

if IS_ON_DAIC:
    BASE_DIR = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
    PAINT_DIR = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")
else:
    BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
    PAINT_DIR = BASE_DIR / "datasets" / "paint"

SCENARIO_PATH = BASE_DIR / "scenarios" / "five_heliostats_scenario" / "scenario.h5"
BENCHMARK_CSV = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
CALIBRATION_PROPERTIES_DIR = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
FLUX_IMAGE_DIR = PAINT_DIR / BENCHMARK_NAME / "flux_image"

# ---------------------------------------------------------------------------
# Parameter subset ablation
# Each tuple: (dir_name, human-readable label)
# The corresponding reconstructor class is resolved in main.py to avoid
# importing artist_extensions before sys.path is configured.
# ---------------------------------------------------------------------------

PARAM_SUBSETS = [
    ("rotations_only", "Rotations only (4 params)"),
    ("rot_act",        "Rot + Actuators (8 params)"),
    ("rot_act_base",   "Rot + Act + BasePos (11 params)"),
    ("full",           "Full Wortberg (20 params)"),
]

# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------

CENTROID_METHOD = paint_mappings.UTIS_KEY
# Training data ablation: experiment is run once per count below.
# Validation and test sets are fixed at VAL_SAMPLES / TEST_SAMPLES.
TRAIN_SAMPLE_COUNTS = [1, 5, 10, 15, 25, 50]
TRAIN_SAMPLES = 50   # kept for backward compatibility (max of ablation counts)
VAL_SAMPLES = 10
TEST_SAMPLES = 10

# Rays for synthetic data generation (high → clean, near-noiseless centroids)
SYNTH_GEN_RAYS = 100
# Rays during training (lower for speed)
TRAIN_RAYS = 10
# Surface points per facet (N×N grid); 4 facets/heliostat → 4×N² pts total
SURFACE_POINTS_PER_FACET = 50

# ---------------------------------------------------------------------------
# Perturbation
#
# Random per heliostat, seeded for reproducibility.
# Ranges match the WortbergKinematicReconstructor deviation bounds (Table 5.3).
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
# Optimisation
# ---------------------------------------------------------------------------

OPTIMIZATION_CONFIG = {
    config_dictionary.initial_learning_rate: 1e-4,
    config_dictionary.tolerance: 1e-6,
    config_dictionary.max_epoch: 300,
    config_dictionary.batch_size: 8,
    config_dictionary.log_step: 5,
    config_dictionary.early_stopping_window: 10,
    config_dictionary.early_stopping_delta: 1e-5,
    config_dictionary.early_stopping_patience: 400,  # > max_epoch → always runs fully
    config_dictionary.scheduler: config_dictionary.reduce_on_plateau,
    "scheduler_parameters": {
        config_dictionary.lr_min: 1e-6,
        config_dictionary.reduce_factor: 0.5,
        config_dictionary.patience: 10,
        config_dictionary.threshold: 1e-3,
        config_dictionary.cooldown: 5,
    },
}
