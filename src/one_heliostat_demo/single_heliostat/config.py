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

# Restrict real (PAINT) calibration samples to a single aim target.
# None = use all targets (default). Set to a target_name, e.g.
# "solar_tower_juelich_upper", to keep only samples aimed at that target.
# Overridden by --target on the CLI (run_all.py). Affects DATA_MODE == "real" only.
TARGET_FILTER = None

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

# Per-axis motor-encoder-offset correction (real data only).
# MOTOR_OFFSET_STEPS: [axis1, axis2] in raw motor steps, subtracted from every
#   recorded motor position (manual override). None = disabled.
# AUTO_MOTOR_OFFSET: estimate the offset from the training split as the per-axis
#   median of (m_gt − inverse(c_gt)) under nominal kinematics, then subtract it.
#   Removes constant encoder-zero biases (e.g. AY39: −4986 steps ≈ −32 mm on
#   axis 2) that lie far outside the deviation-parameter bounds.
MOTOR_OFFSET_STEPS = None
AUTO_MOTOR_OFFSET  = False
# Shape of the auto-estimated correction:
#   "constant" — per-axis constant in STEPS (encoder-zero / b_i-type fault)
#   "angle"    — per-axis constant in JOINT ANGLE (home-angle / a_i-type fault);
#                applied per sample as Δα·(ds/dα)(m)·increment. Fresh-look analysis
#                (outputs/really_important/fresh_look/) found the angle model fits
#                48/63 heliostats better than the constant-step model.
AUTO_MOTOR_OFFSET_MODE = "constant"

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

# ReduceLROnPlateau (Stage 1). The scheduler only lowers the LR once the val loss
# stops improving by more than THRESHOLD (relative) for PATIENCE epochs. With the
# geometric-init orientation LR (1e-3) the loss descends steadily and may not
# plateau within a short run — raise STAGE1_EPOCHS to let it converge and the
# scheduler will anneal the tail. FACTOR >= 1.0 disables the decay entirely.
STAGE1_PLATEAU_FACTOR    = 0.5
STAGE1_PLATEAU_PATIENCE  = 8
# Relative threshold: an epoch counts as "improving" only if the val loss drops by
# more than this fraction. 1e-4 (0.01%) is so lenient that any slow steady descent
# resets the patience counter and the LR never anneals. 1e-3 (0.1%) lets the
# scheduler fire once the improvement genuinely slows, annealing the LR to squeeze
# the tail — the way to push a long run to its floor.
STAGE1_PLATEAU_THRESHOLD = 1e-3
# Stage 2 scheduler.
STAGE2_PLATEAU_FACTOR    = 0.5
STAGE2_PLATEAU_PATIENCE  = 10
STAGE2_PLATEAU_THRESHOLD = 1e-4

STAGE1_LOSS     = "forward_aim"  # "forward_aim" (DEFAULT — forward normal vs geometric
                                 #   desired normal; only consistent objective, see
                                 #   STAGE1_ALIGNMENT_LOSS_FINDINGS.md)
                                 # The following use the inverse map and are BROKEN
                                 #   (loss minimum not at theta*), kept for comparison:
                                 #   "motor_steps" (MotorStepLoss, increment-normalized) |
                                 #   "motor_mse" (AlignmentLoss) | "normal_mrad" (NormalAlignmentLoss)
MINI_BATCH_SIZE = 25             # Stage 2: samples per mini-batch

# ----------------------------------------------------------------------------
# Stage-1 robust reduction  (forward_aim only)
# ----------------------------------------------------------------------------
# How the per-sample residuals are AGGREGATED — not what is measured. "l2" is
# the plain mean of the squared chord (least squares), so a handful of bad
# calibration samples dominate the fit. Robust modes cap or discard that
# influence: they improve the MEDIAN pointing error but cost mean and centroid
# accuracy, because they buy the bulk of the distribution by sacrificing the
# tail (and the centroid metric is tail-sensitive). Always report both.
#   "l2"      — mean of squared residuals (DEFAULT, historical behaviour)
#   "huber"   — quadratic within δ, linear beyond (δ = STAGE1_HUBER_DELTA, mrad)
#   "soft_l1" — pseudo-Huber, smooth everywhere
#   "trimmed" — drop the worst STAGE1_TRIM_FRACTION of samples, mean the rest
# When != "l2" the best-epoch selection follows the objective instead of the
# mean alignment mrad (otherwise it would reject the epochs robustness buys).
# DEFAULT since the 63-heliostat sweep (outputs/new_mapping_function/robust_loss_sweep/):
# soft_l1 improves ALL FOUR metrics vs l2 field-wide — dir median 3.34 -> 2.84,
# centroid median 4.21 -> 3.55 — and beats Mathias on 31/41 shared heliostats
# (3.84 vs 4.56). It is NOT the median-for-mean trade-off the single-heliostat
# AA23 study suggested; that heliostat is well behaved, so l2 had little to be
# robust against. Set to "l2" to reproduce pre-2026-07-21 runs.
STAGE1_REDUCTION      = "soft_l1"
STAGE1_HUBER_DELTA    = 1.5      # mrad, for "huber" / "soft_l1"
STAGE1_TRIM_FRACTION  = 0.25     # for "trimmed"
# Which quantity picks the best Stage-1 epoch to restore:
#   "mrad"      — mean alignment mrad (DEFAULT; safe, and identical across configs
#                 so a reduction sweep stays a controlled comparison)
#   "objective" — the training objective itself. UNSAFE for "trimmed": the
#                 discarded fraction is unpenalized, so the optimizer can minimize
#                 the objective by abandoning it and this rule then locks that in
#                 (AA23: trimmed-40% diverges to 21.5 mrad vs 3.4 under "mrad").
STAGE1_SELECT_ON      = "mrad"

# ----------------------------------------------------------------------------
# Stage-2 loss selection
# ----------------------------------------------------------------------------
#   "focal_spot" — DEFAULT. Squared metre distance between the predicted-flux
#                  COM and the parsed centroid c_gt (UTIS). All existing
#                  baselines were produced with this.
#   "contour"    — Wortberg (2025) upper-contour loss (artist_extensions/
#                  contour_loss.py): matches the occlusion-robust upper edge of
#                  the measured flux image instead of its centroid. Stage 1 is
#                  the alignment warm-up that puts the beam on target, so the
#                  contour loss runs from Stage-2 epoch 1 (no in-stage ramp);
#                  a guardrail falls back to ForwardAimLoss on divergence.
STAGE2_LOSS = "focal_spot"

# ----------------------------------------------------------------------------
# Stage-2 structure (focal_spot loss)
# ----------------------------------------------------------------------------
# Which parameters Stage 2 may move:
#   "all"              — every group (DEFAULT, historical). Because the
#                        geometric-init Stage 1 pins translation / base position /
#                        offset / pivot at lr=0, Stage 2 is the ONLY place those
#                        are ever trained.
#   "orientation_only" — the same free set Stage 1 uses (orientation + a/b), so
#                        Stage 2 refines pointing on the ray-traced objective
#                        without the landing DOFs that can drift the direction
#                        metric.
# DEFAULT "all", confirmed by the field matrix. The two stages deliberately use
# DIFFERENT parameter sets, and the metrics show why: freezing the landing DOFs
# protects the pointing fit (direction median -0.02 vs +0.03) while opening them
# is what lets Stage 2 move the beam on the target plane (centroid median -0.27
# vs -0.18). Stage 1 therefore stays restricted (GEOMETRIC_INIT_ORIENTATION_ONLY)
# and Stage 2 opens everything — so Stage 2 is the ONLY place translation, base
# position, offset and pivot are ever trained. Choosing "orientation_only" here
# would leave those four at nominal for the entire pipeline, which is close to
# Mathias's free set (a/b + mount orientation) if a more parsimonious model is
# wanted; it costs ~0.05-0.09 mrad of centroid accuracy.
STAGE2_PARAM_SET = "all"

# Robust aggregation of the focal-spot residual, same idea as STAGE1_REDUCTION.
# The residual is a miss distance in metres; the delta below is given in mrad and
# converted per heliostat using its own distance to the target.
#   "l2" (DEFAULT, historical) | "huber" | "soft_l1" | "trimmed"
# DEFAULT since the 3x2 field matrix (outputs/new_mapping_function/stage2_matrix/):
# the reduction is what makes Stage 2 work at all. Under l2, Stage 2 is a near
# no-op (centroid mean -0.02 mrad) that degrades the direction median (+0.22) and
# helps only 37/63 heliostats. Under soft_l1 it gives a real centroid gain
# (median -0.23) on 54/63, with direction left neutral.
STAGE2_REDUCTION         = "soft_l1"
STAGE2_HUBER_DELTA_MRAD  = 3.0
STAGE2_TRIM_FRACTION     = 0.25

# Contour extraction (thesis §4.2.3). τ/η are the Bayesian-optimized values for
# simulated STJ flux — the thesis says to RETUNE on other data (our real PAINT
# flux is noisier; revisit τ, η, q, σ after the first results).
CONTOUR_TAU              = 0.58   # soft-threshold centre on [0,1] flux
CONTOUR_ETA              = 70.0   # sigmoid sharpness
CONTOUR_SMOOTHING_ROUNDS = 2      # q bilinear up/down denoising passes
# Denoising σ/kernel: the thesis default (σ=1, k=5) is too weak for OUR
# predicted flux — at TRAIN_RAYS=10 the Monte-Carlo speckle survives, the soft
# mask is full of holes, and the Sobel fires all over the blob interior instead
# of only the upper edge. A fragmented predicted contour vs the clean GT arc
# biases coarse+gravity toward "move the beam up" (verified: one such epoch
# moved AA23 2.0 → 8.4 mrad). σ=3 closes the speckle holes; both sides use the
# same extractor, so the GT contour thickens/shifts symmetrically.
CONTOUR_GAUSS_SIGMA      = 3.0
CONTOUR_GAUSS_KSIZE      = 13

# Term weights (eq. 4.41): Fine (DICE) gets 1 − β − γ. The raw coarse term is
# an unnormalized pixel sum (~1e3–1e5 on 256² images), hence the small β.
# The thesis tuned β/γ by Bayesian optimization but does not print them —
# these are starting points, sweep on the simplex later.
CONTOUR_BETA  = 1e-4              # β — coarse (soft distance field)
CONTOUR_GAMMA = 0.3               # γ — gravity (COM distance, metres)

# Divide raw Coarse/Gravity by these before weighting (default 1.0 = no-op,
# exact prior behavior). Set to a representative empirical magnitude (e.g.
# ~1200 for coarse, ~0.1 for gravity, from this project's own runs) to turn
# beta/gamma into genuine 0-1 mixing weights before a beta/gamma sweep --
# see contour_beta_gamma_sweep.py.
CONTOUR_COARSE_SCALE  = 1.0
CONTOUR_GRAVITY_SCALE = 1.0

# Optional post-hoc Gaussian blur (std dev, px) widening the extracted
# upper-contour band beyond its native ~1-2 px Sobel width (default 0.0 =
# no-op). See HybridFocalContourLoss / STAGE2_LOSS="hybrid" below for the
# other structural fix in the same direction.
CONTOUR_BAND_SIGMA = 0.0

# STAGE2_LOSS="hybrid": blend FocalSpotLoss (whole image) with
# WortbergContourLoss, total = HYBRID_FOCAL_WEIGHT*focal + (1-...)*contour.
# HYBRID_FOCAL_SCALE normalizes the raw (squared-metres) focal term the same
# way CONTOUR_COARSE_SCALE/GRAVITY_SCALE do for the other two terms.
HYBRID_FOCAL_WEIGHT = 0.5
HYBRID_FOCAL_SCALE  = 1.0

# Guardrail (eq. 4.45): if the Stage-2 val centroid error exceeds
# max(MIN_MRAD, FACTOR × post-Stage-1 val error), train on ForwardAimLoss until
# it recovers (release at 0.8× the threshold). Per-sample empty-flux rescue:
# samples whose predicted flux misses the target entirely also fall back.
CONTOUR_GUARDRAIL_FACTOR   = 3.0
CONTOUR_GUARDRAIL_MIN_MRAD = 2.0
CONTOUR_EMPTY_FLUX_EPS     = 1e-6

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

# OPTIMIZE_ACTUATOR_STROKE — whether the actuator initial stroke length b_i is
# calibrated. Full-parameter regime (supervisor 2026-07-14, see
# OPTIMIZATION_PARAMETERS.md): TRAINABLE by default with a ±50 mm bound
# (_BOUND_ACTUATOR_STROKE_TRAIN_M, encoder re-referencing scale). b_i is the
# encoder-zero anchor of the motor→angle map; training it replaces the
# AUTO_MOTOR_OFFSET data-side correction (same correction space). Do not enable
# AUTO_MOTOR_OFFSET simultaneously — that double-corrects the same fault.
OPTIMIZE_ACTUATOR_STROKE = True

# OPTIMIZE_PIVOT_RADIUS — whether the linkage pivot radius r_i is calibrated
# (±_BOUND_PIVOT_RADIUS_TRAIN_M around the loaded value). Part of the
# full-parameter regime. The rest of the non_optimizable tensor (type,
# clockwise flag, min/max motor limits) is NEVER optimized: IDs and
# inverse-kinematics branch-selection limits.
OPTIMIZE_PIVOT_RADIUS = True

# LR multiplier for the actuator param group (a_i, b_i) applied ONLY when
# OPTIMIZE_ACTUATOR_STROKE is True. b_i must travel a re-referencing-scale
# distance; an Adam step ≈ lr, so at BASE_LR (~0.1 mm/step) it can't reach ~32 mm
# in 100 epochs. 20× → ~2 mm/step, reachable in ~16 epochs.
ACTUATOR_STROKE_LR_MULT = 20.0

BASE_LR         = 1e-4
PLOT_EVERY      = 1              # capture trail snapshot every N epochs (1 = all epochs)

# Ray-traced trail snapshots during STAGE 1 (diagnostics only).
# Stage 1 optimizes a purely kinematic objective (ForwardAimLoss: forward normal
# vs the sun<->c_gt bisector) — no surface, no ray tracing, no flux image — so
# these snapshots never affect the trained parameters. They only add points to
# the trail / flux-GIF plots, at a cost that scales with the SQUARE of
# SURFACE_POINTS_PER_FACET: ~0.3 s/epoch at 25x25 but ~5 s/epoch at 100x100,
# which is what made a 500-epoch Stage 1 take 45 min instead of 3.
# OFF by default; enable with --stage1-trail-plots. A single snapshot of the
# final (restored) Stage-1 state is always taken so the plots stay connected.
STAGE1_TRAIL_PLOTS = False

# ----------------------------------------------------------------------------
# Geometric initialization (Stage 1)  — see AA23_METHOD_STUDY.md
# ----------------------------------------------------------------------------
# Before the Stage-1 forward-aim refine, seed the orientation parameters from a
# closed-form estimate of the mount misorientation: build the nominal-vs-desired
# normal clouds on the training split, solve Kabsch/Wahba for the single rotation
# that best maps one onto the other, and fit the 4 tilts + 2 phi_0 to it. This
# lands Stage 1 in the correct basin deterministically (no random restarts), which
# a single gradient descent from nominal could not reach for large mount faults.
# Only applied when STAGE1_LOSS == "forward_aim".
GEOMETRIC_INIT           = True
GEOMETRIC_INIT_EPOCHS    = 200      # inner fit of the orientation params to the Kabsch rotation
GEOMETRIC_INIT_LR        = 3e-3
# With a correct orientation seed, the extra translation / base-position DOFs of
# the full-parameter regime only shift the ray-traced landing (hurting the centroid
# metric) without improving pointing. Restrict the Stage-1 refine to Mathias's free
# set — orientation (tilts) + phi_0 + stroke — freezing translation, base, offset,
# pivot. Applied only when GEOMETRIC_INIT is on. See AA23_METHOD_STUDY.md.
GEOMETRIC_INIT_ORIENTATION_ONLY = True
# Stage-1 rotation-deviation learning rate used AFTER the geometric seed (the
# refine only needs to travel a little, but 1e-4 is too slow for that). Applied
# only when GEOMETRIC_INIT is on; otherwise the rotation group keeps BASE_LR.
S1_ORIENTATION_LR        = 1e-3

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
_BOUND_ACTUATOR_STROKE_M  = 0.005   # random-perturbation scale (dataset generation only)
_BOUND_ACTUATOR_OFFSET_M  = 0.005
_BOUND_BASE_POSITION_M    = 0.05

# ----------------------------------------------------------------------------
# TRAINING bounds — full-parameter regime (supervisor 2026-07-14).
# Separate from the _BOUND_* generation constants above (those describe synthetic
# perturbation magnitudes and stay untouched). All clamps are centred on the
# LOADED initial values (translation[7] holds the physical 0.175 m concentrator
# offset; a_1 is stored with a −π/2 shift), never on zero.
# ----------------------------------------------------------------------------
# ±50 mm: physical scale of an encoder re-referencing (AY39 ≈ 32 mm); on axis 2
# this spans ≈ ±165 mrad of joint-angle correction — Stage 1's reachable set now
# contains the reference errors that AUTO_MOTOR_OFFSET used to remove.
_BOUND_ACTUATOR_STROKE_TRAIN_M = 0.05
# Joint tilts: widened 0.020 → 0.5 rad. AA23-class heliostats have a real mount
# reorientation of ~0.16 rad that the old ±20 mrad clamp could not represent; the
# geometric init (below) plus this bound let Stage 1 reach it. See
# outputs/new_mapping_function/aa23_method_comparison/AA23_METHOD_STUDY.md.
_BOUND_ROTATION_TRAIN_RAD       = 0.5     # joint tilts (was ±20 mrad)
# a_i home angle: effectively UNBOUNDED (±500 mrad). Decisive AY39 experiment
# (fullparam_regime/AY39_opt_widea): the ±20 mrad bound was the binding
# constraint — freed, a₂ walks to the true fault (−99 mrad) and AY39 reaches
# 19.4 mrad (beats even the old offset's 22.9). Sanity-checked on AY37/AY36:
# no regression, a stops at the data's value (−4.8 / −27.8 mrad).
_BOUND_ACTUATOR_ANGLE_TRAIN_RAD = 0.5
_BOUND_ACTUATOR_OFFSET_TRAIN_M  = 0.020   # c_i linkage side (was ±5 mm)
_BOUND_PIVOT_RADIUS_TRAIN_M     = 0.020   # r_i linkage side (was frozen)
# Base-position bound: the WATCHED one (supervisor) — keep tight, monitor clamps.
_BOUND_BASE_POSITION_TRAIN_M    = 0.05
_BOUND_TRANSLATION_TRAIN_M      = 0.05

RANDOM_PERT_BOUNDS = {
    "rotation_rad":       _BOUND_ROTATION_RAD,
    "actuator_angle_rad": _BOUND_ACTUATOR_ANGLE_RAD,
    "actuator_stroke_m":  _BOUND_ACTUATOR_STROKE_M,
    "actuator_offset_m":  _BOUND_ACTUATOR_OFFSET_M,
    "translation_m":      _BOUND_TRANSLATION_M,
    "base_position_m":    _BOUND_BASE_POSITION_M,
}

RANDOM_SEED = 42

# Replay mode: path to a perturbations.json (as saved next to a generated
# dataset). When set, generate() uses exactly these per-heliostat perturbations
# instead of sampling — used to regenerate the identical synthetic dataset on
# another machine (e.g. DAIC) from the committed JSON.
FIXED_PERTURBATIONS_JSON = None

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
