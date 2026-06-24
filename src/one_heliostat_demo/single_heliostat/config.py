"""
Configuration for the one_heliostat_demo experiment.

Quick-start
-----------
The most commonly changed settings are at the top of each section.
CLI flags on main.py / run_all.py override any value set here.
"""
import pathlib

# ============================================================================
# Paths
# ============================================================================

BASE_DIR = pathlib.Path(__file__).resolve().parents[3]  # master-thesis/

# Single-heliostat scenario (one per heliostat, created by create_scenarios.py).
# The "{heliostat_id}" placeholder is filled at runtime.
SCENARIO_PATH_TEMPLATE = str(
    BASE_DIR / "scenarios" / "one_heliostat_scenarios" / "{heliostat_id}" / "scenario.h5"
)

# Pre-generated synthetic dataset used when --skip-dataset-gen is set.
# Each sub-folder (train / val / test / {heliostat_id} / {idx:04d}/) contains
# calibration_properties.json and flux_image.png.
# Switch between "balanced_dataset" and "azimuth_dataset" to change the pool.
SYNTHETIC_DATASET_DIR = BASE_DIR / "datasets" / "synthetic" / "balanced_dataset" / "dataset"

# PAINT benchmark — used only by generate_dataset.py to supply sun ray directions.
# Flux and centroids are re-synthesised from the perturbed kinematics, not read here.
BENCHMARK_NAME  = "benchmark_split-balanced_train-100_validation-50_deflectometry"
PAINT_DIR       = BASE_DIR / "datasets" / "paint"
BENCHMARK_CSV   = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
CALIBRATION_DIR = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
REAL_FLUX_DIR   = PAINT_DIR / BENCHMARK_NAME / "flux_image"

# ============================================================================
# Dataset — splitter & split sizes
# ============================================================================

# Split strategy applied to the pooled data.
#   "balanced" — KMeans on (azimuth, elevation); val and test are fixed
#                across all training sizes (candidates selected first).
#   "azimuth"  — sort by azimuth; train=morning, val=late-afternoon (fixed),
#                test=noon (shrinks slightly as training size grows).
SPLITTER_TYPE = "balanced"

# Training samples drawn from the pool.
# Overridden by --train-size on the CLI.
SPLITTER_TRAIN_SIZE = 100

# Samples reserved for val and test (each gets this many).
# Pool must contain at least SPLITTER_TRAIN_SIZE + 2 × SPLITTER_VAL_SIZE samples
# after the active-pixel filter.
SPLITTER_VAL_SIZE = 50

# PAINT's DatasetSplitter assigns VALIDATION_INDEX as the intended final-evaluation set.
#   True  (recommended): test_flux = VALIDATION_INDEX (final eval)
#                        val_flux  = TEST_INDEX        (scheduler / early-stopping)
#   False: keep the splitter's original assignment.
SWAP_VAL_TEST = True

# ============================================================================
# Data generation mode  (only relevant when NOT using --skip-dataset-gen)
# ============================================================================

# Heliostat to process in single-heliostat runs (overridden by --heliostat-id).
HELIOSTAT_ID = "AC33"

# Data source used during training:
#   "synthetic"        — load from SYNTHETIC_DATASET_DIR (pre-generated, GT perturbations known)
#   "random_synthetic" — same as "synthetic" when --skip-dataset-gen is set; otherwise
#                        generates a new dataset with random perturbations before training
#   "real"             — load actual PAINT calibration images from CALIBRATION_DIR / REAL_FLUX_DIR;
#                        GT perturbations are unknown, parameter trajectory plots are skipped;
#                        use with --skip-dataset-gen (no generation step needed)
DATA_MODE = "synthetic"

# Centroid extraction method used by PaintCalibrationDataParser (real data only).
CENTROID_METHOD = "UTIS"

# ============================================================================
# Ray counts & surface resolution
# ============================================================================

SURFACE_POINTS_PER_FACET = 25    # 25×25 = 625 pts/facet
TRAIN_RAYS   = 10                # rays per surface point during Stage 2 training
DISPLAY_RAYS = 50                # rays for pre/post-training evaluation plots
GENERATE_RAYS = 100              # rays for synthetic GT data generation

# ============================================================================
# Training schedule
# ============================================================================

STAGE1_EPOCHS   = 100
STAGE2_EPOCHS   = 100
STAGE1_LOSS     = "forward_aim"  # "forward_aim" (DEFAULT — forward normal vs geometric
                                 #   desired normal; only consistent objective, see
                                 #   STAGE1_ALIGNMENT_LOSS_FINDINGS.md)
                                 # The following use the inverse map and are BROKEN
                                 #   (loss minimum not at theta*), kept for comparison:
                                 #   "motor_steps" (MotorStepLoss, increment-normalized) |
                                 #   "motor_mse" (AlignmentLoss) | "normal_mrad" (NormalAlignmentLoss)
MINI_BATCH_SIZE = 25             # Stage 2: samples per mini-batch

# ----------------------------------------------------------------------------
# Aim-point / motor-position formulation  (see CALIBRATION_FORMULATION.md)
# ----------------------------------------------------------------------------
# The formulation is hardcoded centre-free: the motor position m_c and the
# observed centroid c_gt are the only real observables; the intended aim point is
# never used. Stage 1 aims the inverse map at c_gt; Stage 2 and evaluation orient
# from the recorded motors m_c. The legacy centre-aim modes have been removed.

# OPTIMIZE_ACTUATOR_OFFSET — whether the actuator offset c_i is calibrated.
# c_i lives in the actuators' `non_optimizable_parameters` tensor. ARTIST's
# default treats that whole tensor as fixed (manufacturing geometry); Wortberg
# 2025 deliberately re-enables c_i. Set False to honour ARTIST's "non-optimizable"
# designation and freeze the entire tensor; True reproduces Wortberg (only c_i
# in that tensor receives gradients, everything else stays frozen).
OPTIMIZE_ACTUATOR_OFFSET = True

BASE_LR         = 1e-4
PLOT_EVERY      = 1              # capture trail snapshot every N epochs (1 = all epochs)

# ============================================================================
# Data quality filter
# ============================================================================

# Discard samples whose GT flux image has fewer than this percentage of active pixels.
MIN_ACTIVE_PIXEL_PERCENT = 2.0

# ============================================================================
# Dataset generation — resampling limits  (generate_dataset.py)
# ============================================================================

MIN_TRAIN_SAMPLES     = 50
MIN_VAL_SAMPLES       = 50
MIN_TEST_SAMPLES      = 50
MAX_RESAMPLE_ATTEMPTS = 25

# Rays used for the fast pre-scan (pool-level active-pixel check before the
# full GENERATE_RAYS pass). Low count keeps the scan cheap; flux presence
# correlates well enough with the full-quality result.
SCAN_RAYS = 10

# ============================================================================
# Perturbation bounds  (Wortberg 2025, Table 5.3)
# ============================================================================

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

# Used when DATA_MODE == "synthetic".
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
