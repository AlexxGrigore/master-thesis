"""
Configuration for the one_heliostat_demo experiment.

Edit HELIOSTAT_ID and DATA_MODE before running main.py.
All other parameters are documented inline.
"""
import pathlib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = pathlib.Path(__file__).resolve().parents[3]  # master-thesis/

# Single-heliostat scenario (one per heliostat, created by create_scenarios.py).
# The literal "{heliostat_id}" placeholder is filled in at runtime.
SCENARIO_PATH_TEMPLATE = str(
    BASE_DIR / "scenarios" / "one_heliostat_scenarios" / "{heliostat_id}" / "scenario.h5"
)

# Pre-generated synthetic dataset (all heliostats combined).
# Copy the contents of outputs/one_hel_demo_dataset_all_<timestamp>/dataset/ here.
SYNTHETIC_DATASET_DIR = BASE_DIR / "datasets" / "synthetic" / "dataset"

# PAINT benchmark dataset — used by generate_dataset.py to supply sun ray directions.
# Only incident_ray_direction, active_mask, and target_mask are read from here;
# flux and centroids are re-generated from the perturbed kinematics.
BENCHMARK_NAME  = "benchmark_split-balanced_train-100_validation-50_deflectometry"
PAINT_DIR       = BASE_DIR / "datasets" / "paint"
BENCHMARK_CSV   = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
CALIBRATION_DIR = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
REAL_FLUX_DIR   = PAINT_DIR / BENCHMARK_NAME / "flux_image"

# ---------------------------------------------------------------------------
# Heliostat & data mode
# ---------------------------------------------------------------------------

HELIOSTAT_ID = "AC33"

# "random_synthetic" : sample random perturbations (seed RANDOM_SEED, ±RANDOM_PERT_BOUNDS)
# "synthetic"        : use CUSTOM_PERTURBATIONS_SPEC below (fully reproducible)
DATA_MODE = "random_synthetic"

# ---------------------------------------------------------------------------
# Surface resolution & ray counts
# ---------------------------------------------------------------------------

SURFACE_POINTS_PER_FACET = 25    # 25×25 = 625 pts/facet
TRAIN_RAYS   = 10                # rays per surface point during Stage 2 training
DISPLAY_RAYS = 50                # rays for pre/post-training evaluation plots
GENERATE_RAYS = 100              # rays for synthetic GT data generation

# ---------------------------------------------------------------------------
# Training schedule
# ---------------------------------------------------------------------------

STAGE1_EPOCHS   = 20
STAGE2_EPOCHS   = 100
MINI_BATCH_SIZE = 25             # Stage 2 samples per mini-batch
BASE_LR         = 1e-4
PLOT_EVERY      = 1              # capture trail snapshot every N epochs (1 = max detail)

# ---------------------------------------------------------------------------
# Data filtering
# ---------------------------------------------------------------------------

MIN_ACTIVE_PIXEL_PERCENT = 2.0   # discard samples with < 2% active pixels

# Minimum filtered sample counts per split — generate_dataset.py resamples
# the perturbations (up to MAX_RESAMPLE_ATTEMPTS) until these are satisfied.
MIN_TRAIN_SAMPLES     = 50
MIN_VAL_SAMPLES       = 20
MIN_TEST_SAMPLES      = 20
MAX_RESAMPLE_ATTEMPTS = 10

# ---------------------------------------------------------------------------
# Perturbation bounds (Wortberg 2025)
# ---------------------------------------------------------------------------

_BOUND_TRANSLATION_M      = 0.05
_BOUND_ROTATION_RAD       = 0.005
_BOUND_ACTUATOR_ANGLE_RAD = 0.005
_BOUND_ACTUATOR_STROKE_M  = 0.005
_BOUND_ACTUATOR_OFFSET_M  = 0.005
_BOUND_BASE_POSITION_M    = 0.05

RANDOM_PERT_BOUNDS = {
    "rotation_rad":       _BOUND_ROTATION_RAD,
    "actuator_angle_rad": _BOUND_ACTUATOR_ANGLE_RAD,
    "actuator_stroke_m":  _BOUND_ACTUATOR_STROKE_M,
    "actuator_offset_m":  _BOUND_ACTUATOR_OFFSET_M,
    "translation_m":      _BOUND_TRANSLATION_M,
    "base_position_m":    _BOUND_BASE_POSITION_M,
}

RANDOM_SEED = 42

# Used when DATA_MODE == "synthetic" (or as fallback reference values).
CUSTOM_PERTURBATIONS_SPEC = {
    "rotation":        [ 0.0028902976773679256, -0.002185862511396408,
                         0.002886323258280754,   0.0008946311427280307],
    "actuator_angle":  [ 0.004728531930595636,  -0.0011793976882472634],
    "actuator_stroke": [-0.001398988300934434,   0.003376838518306613],
    "actuator_offset": [ 0.0019239198882132769,  0.0009438949637115002],
    "translation":     [-0.025944454595446587,   0.03422738239169121,
                        -0.04707298427820206,   -0.0435216911137104,
                         0.02801002934575081,    0.0269764494150877,
                         0.04111963510513306,   -0.037746936082839966,
                        -0.03659498691558838],
    "base_position":   [ 0.024333560839295387,   0.008520788513123989,
                         0.013595938682556152],
}
