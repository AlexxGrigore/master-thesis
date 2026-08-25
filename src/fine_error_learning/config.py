"""
Configuration for the fine_error_learning experiment.

One shared transformer model predicts a 24-D kinematic correction Δθ per
heliostat from a set of calibration measurements (flux images + scalars),
on top of a frozen stage-1 warm start θ_KR. Training is end-to-end through
the ARTIST ray tracer with the stage-2 focal-spot centroid loss.

Quick-start
-----------
    python fine_error_learning/main.py                 # full run (config defaults)
    python fine_error_learning/main.py --smoke-test    # tiny local sanity check
"""
import pathlib

# ============================================================================
# Paths
# ============================================================================

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]  # master-thesis/

# Single-heliostat scenarios (one per heliostat, created by create_scenarios.py).
SCENARIO_PATH_TEMPLATE = str(
    BASE_DIR / "scenarios" / "one_heliostat_scenarios" / "{heliostat_id}" / "scenario.h5"
)

# Synthetic calibration data: {train,val,test}/<HID>/<NNNN>/{flux_image.png,
# calibration_properties.json}. Ground truth in perturbations.json next to them.
# This is the balanced_dataset the stage-1 checkpoints below were trained on
# (perturbations.json matches the run's all_perturbations.json exactly).
SYNTHETIC_DATA_DIR = BASE_DIR / "datasets" / "synthetic" / "balanced_dataset" / "dataset"

# Completed stage-1 run ON THE SYNTHETIC DATASET above:
# <HID>/stage1_checkpoint.pt per heliostat (absolute kinematic tensors:
# translation [1,9], rotation [1,4], act_angle [1,2,2], act_offset [1,7,2],
# base_pos [1,3]). Generated with:
#   cd src && python one_heliostat_demo/run_all.py --data-mode synthetic \
#       --skip-dataset-gen --skip-stage2 --no-plots \
#       --output-dir ../outputs/fine_error_learning/all63_stage1_synth
# NOTE: outputs/new_mapping_function/all63_stage1 was a REAL-data run
# (pre-training direction error ~153 mrad) — its checkpoints leave many
# heliostats off-target on synthetic data and are NOT a valid warm start here.
STAGE1_CHECKPOINT_DIR = BASE_DIR / "outputs" / "fine_error_learning" / "all63_stage1_synth"

OUTPUT_ROOT = BASE_DIR / "outputs" / "fine_error_learning"

# ============================================================================
# Held-out generalization test set (built by prepare_heldout.py)
# ============================================================================
# 12 heliostats with deflectometry that are NOT in the 63-heliostat benchmark,
# spread 25-273 m from the tower, with fresh perturbations (seed 4242 vs 42).
# Evaluate a trained run on them with:
#   python fine_error_learning/main.py --evaluate <run_dir> --eval-split test \
#       --heliostats $HELDOUT_IDS \
#       --checkpoint-dir outputs/fine_error_learning/heldout_stage1_synth \
#       --data-dir datasets/synthetic/heldout_dataset/dataset \
#       --scenario-template 'scenarios/heldout_heliostat_scenarios/{heliostat_id}/scenario.h5'
# Keep in sync with prepare_heldout.py.
HELDOUT_IDS = [
    "AA36", "AA29", "AF32", "AG29", "AJ46", "AP32",
    "AW39", "AZ33", "BB28", "BE40", "BA70", "BH65",
]
HELDOUT_SCENARIO_TEMPLATE = str(
    BASE_DIR / "scenarios" / "heldout_heliostat_scenarios" / "{heliostat_id}" / "scenario.h5"
)
HELDOUT_DATA_DIR = BASE_DIR / "datasets" / "synthetic" / "heldout_dataset" / "dataset"
HELDOUT_CHECKPOINT_DIR = OUTPUT_ROOT / "heldout_stage1_synth"

# ============================================================================
# Data
# ============================================================================

# Heliostats to train on. None → discover all from STAGE1_CHECKPOINT_DIR.
HELIOSTAT_IDS = None

# Tokens per heliostat sample: the measurement set is subsampled (train, random
# per epoch) or padded (zeros + boolean mask) to this length.
K_TOKENS = 50

# Cap on measurements per heliostat per split (None = all; used by --smoke-test).
MAX_MEASUREMENTS = None

# ============================================================================
# Model
# ============================================================================

D_MODEL = 128        # token / encoder hidden dimension
N_HEADS = 4
N_LAYERS = 2
D_FF = 512
DROPOUT = 0.1
D_IMG = 64           # CNN image-feature dimension (before token projection)
USE_FLUX = True      # False → skip the CNN, zero image features (ablation)
BOUNDED_HEAD = False  # False → unbounded zero-init head (default);
                      # True → tanh × RESIDUAL_BOUNDS fallback (old scheme)
OUTPUT_GAIN = 0.01    # unbounded head: Δθ = raw × OUTPUT_GAIN × PARAMETER_SCALE
                      # (tames Adam's first-step jump to ~1e-4 × scale)

# ============================================================================
# Warm start
# ============================================================================

# "stage1" — load STAGE1_CHECKPOINT_DIR/<HID>/stage1_checkpoint.pt (default).
# "nominal" — keep the scenario's nominal kinematics (θ_KR = nominal; useful
#             sanity baseline that is guaranteed on-target on synthetic data).
WARM_START = "stage1"

SURFACE_POINTS_PER_FACET = 25   # 25×25 = 625 pts/facet

# On-target-fraction diagnostic at warm-start time: ray-trace this many
# measurements per heliostat under θ_KR and log the fraction whose predicted
# centroid is finite and lands within the target bitmap extent.
ON_TARGET_MAX_MEASUREMENTS = 20
ON_TARGET_RAYS = 10

# ============================================================================
# Ray tracing
# ============================================================================

TRAIN_RAYS = 10   # rays per surface point during training
VAL_RAYS = 10     # rays per surface point during validation

# ============================================================================
# Training schedule
# ============================================================================

EPOCHS = 100
HELI_BATCH_SIZE = 8     # heliostats per optimizer step
MINI_BATCH_SIZE = 25    # measurements per ray-traced mini-batch within a heliostat

BASE_LR = 1e-3
WEIGHT_DECAY = 1e-5
PLATEAU_FACTOR = 0.5    # ReduceLROnPlateau on the val centroid error [mrad]
PLATEAU_PATIENCE = 10
GRAD_CLIP = 1.0

RESIDUAL_L2_WEIGHT = 1e-4   # penalty on Δθ.pow(2).mean()
LOSS_REDUCTION = "l2"       # robust_reduce_squared mode: "l2" | "huber" | "soft_l1" | "trimmed"
HUBER_DELTA_MRAD = 3.0      # only for "huber"/"soft_l1" (converted with hel distance)

# Experiment A — auxiliary pixelwise flux-distribution loss.
# 0 disables. When > 0, the pixel-loss weight lambda is auto-calibrated on the
# first training mini-batch so that lambda * L_pixel ≈ PIXEL_LOSS_RATIO * L_centroid
# (the two terms have unrelated units/scales; this keeps the auxiliary term a
# fixed fraction of the centroid objective at the warm start).
PIXEL_LOSS_RATIO = 0.0
PIXEL_LOSS_DOWNSIZE = 32    # avg-pool both images to 32×32 before the MSE

# Experiment B — query-conditioned decoder.
# False: one static Δθ per heliostat (mean pooling + head). True: a query token
# built from a measurement's sun direction cross-attends to the encoder memory
# → per-measurement Δθ(query). Training ray-traces QUERIES_PER_STEP randomly
# sampled measurements per heliostat per epoch (one predict_flux call each —
# per-query Δθ prevents batching them into one trace).
QUERY_DECODER = False
QUERIES_PER_STEP = 8
VAL_QUERY_MEASUREMENTS = 4  # fixed val subset for the per-epoch val metric

RANDOM_SEED = 7
