"""
Configuration for the one-heliostat train-size sensitivity experiment.

Uses the corrected synthetic dataset from full_63_heli_kin_reconstruct:
data was generated from a PERTURBED scenario, so the KR starts clean and
must discover the perturbation values — the real inverse problem.

Edit this file to change the heliostat, training sample sweep, hyperparameters,
and dataset type.
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

# Per-heliostat scenario dir; main.py appends /{heliostat_id}/scenario.h5 at runtime.
ONE_HELIOSTAT_SCENARIOS_DIR = BASE_DIR / "scenarios" / "one_heliostat_scenarios"
BENCHMARK_CSV              = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
CALIBRATION_PROPERTIES_DIR = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
FLUX_IMAGE_DIR             = PAINT_DIR / BENCHMARK_NAME / "flux_image"

# Synthetic data from the corrected full-63-heliostat experiment.
# Generated from a perturbed scenario — no separate generate_dataset.py step needed.
SYNTH_DATA_DIR = BASE_DIR / "scenarios" / "full_63_heli_kin_reconstruct" / "synthetic_data"

# ---------------------------------------------------------------------------
# Heliostat selection
#
# Set to a specific heliostat ID (e.g. "AA23") to fix the experiment to that
# heliostat.  None means auto-select: the first heliostat found in the
# benchmark CSV that also has a local scenario will be used.  Must match the
# heliostat used when running create_scenario.py.
# ---------------------------------------------------------------------------

HELIOSTAT_ID: str | None = None

# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------

CENTROID_METHOD = paint_mappings.UTIS_KEY

# Training sample counts to sweep — the experiment runs one full train+eval
# cycle per entry.  Val and test sizes are fixed across all runs.
TRAIN_SIZES = [1, 5, 10, 20, 25, 50, 75, 100]

VAL_SAMPLES  = 50
TEST_SAMPLES = 50

# Rays for synthetic data generation (must match what was used to generate the
# existing full_field_200_samples synthetic data — do not change).
SYNTH_GEN_RAYS = 100
# Rays during training (lower for speed).
TRAIN_RAYS = 10
# Surface points per facet (N×N); 4 facets/heliostat → 4×N² pts total.
SURFACE_POINTS_PER_FACET = 25

# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------
# Perturbations are fixed at dataset-generation time (see
# full_63_heli_kin_reconstruct/generate_dataset.py and its config.py).
# The ground-truth values are loaded at runtime from:
#   SYNTH_DATA_DIR / "perturbations.json"
# The ranges below are for reference only — do not use them to re-perturb.
#
# rotation_rad:        ±3 mrad   (4 joint tilts)
# actuator_angle_rad:  ±3 mrad   (a_i)
# actuator_stroke_m:   ±3 mm     (b_i, frozen during training)
# actuator_offset_m:   ±3 mm     (c_i)
# translation_m:       ±15 mm    (9 joint + concentrator translations)
# base_position_m:     ±15 mm    (e, n, u)

# ---------------------------------------------------------------------------
# Dataset
#
# "synthetic" — reuses pre-generated data from full_field_200_samples_scenario/synthetic_data/
# "real"      — PAINT calibration images; kinematic perturbations are skipped
# ---------------------------------------------------------------------------

DATASET_TYPE = "synthetic"

# ---------------------------------------------------------------------------
# Loss
#
# "focal_spot" — Euclidean distance between predicted and measured centroids (mrad)
# "pixel"      — MSE on Gaussian-blurred, peak-normalised flux bitmaps
# "alignment"  — MSE on motor positions converted to joint angles (no ray tracing)
# ---------------------------------------------------------------------------

LOSS_TYPE = "focal_spot"

STAGE1_EPOCHS = 20    # AlignmentLoss pre-training (no ray tracing)
STAGE2_EPOCHS = 200   # Configured loss fine-tuning

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
