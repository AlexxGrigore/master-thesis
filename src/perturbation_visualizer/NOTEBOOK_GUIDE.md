# One-Heliostat Training Demo — Notebook Guide

**File:** `src/perturbation_visualizer/one_heliostat_training_demo.ipynb`

This notebook is a fully self-contained kinematic reconstruction experiment on a **single heliostat**. It applies (or loads) a known perturbation to the heliostat's kinematic parameters, then recovers those parameters using a two-stage optimizer. Every step from raw data to final accuracy is visible in a single run.

---

## Table of Contents

1. [High-level workflow](#1-high-level-workflow)
2. [Configuration reference](#2-configuration-reference)
3. [Data modes](#3-data-modes)
4. [Scenario and kinematic parameters](#4-scenario-and-kinematic-parameters)
5. [Data loading and in-memory generation](#5-data-loading-and-in-memory-generation)
6. [GT flux filtering](#6-gt-flux-filtering)
7. [Dataset overview plots](#7-dataset-overview-plots)
8. [Pre-training baseline](#8-pre-training-baseline)
9. [Optimizer setup](#9-optimizer-setup)
10. [Trail capture infrastructure](#10-trail-capture-infrastructure)
11. [Stage 1 — AlignmentLoss](#11-stage-1--alignmentloss)
12. [Stage 2 — FocalSpotLoss](#12-stage-2--focalsplotloss)
13. [Centroid trail visualization](#13-centroid-trail-visualization)
14. [Convergence plot (mrad)](#14-convergence-plot-mrad)
15. [Gradient norms and parameter trajectories](#15-gradient-norms-and-parameter-trajectories)
16. [Final evaluation](#16-final-evaluation)
17. [Key tensor shapes and conventions](#17-key-tensor-shapes-and-conventions)
18. [Dependencies and file paths](#18-dependencies-and-file-paths)

---

## 1. High-level workflow

```
Clean scenario (HDF5)
        │
        ▼
Load / generate GT dataset  ─── three modes: real / synthetic / random_synthetic
        │
        ▼
Filter empty GT flux images  ─── flux sum > FLUX_SUM_MIN  AND  active pixels ≥ MIN_ACTIVE_PIXEL_PERCENT
        │
        ▼
Dataset overview plots  ─── train / val / test grids sorted by solar elevation
        │
        ▼
Pre-training baseline  ─── clean params vs perturbed GT → large mrad error
        │
        ▼
Optimizer setup  ─── gradient hooks, parameter snapshots, Adam + ReduceLROnPlateau
        │
        ▼
Stage 1 — AlignmentLoss  ─── motor-position MSE, no ray tracing, fast
        │
        ▼
Stage 2 — FocalSpotLoss  ─── ray-traced centroid MSE, mini-batched
        │
        ▼
Test evaluation  ─── predicted centroid vs GT centroid on held-out test set
```

The test set is **completely isolated** from training. It is only touched by
`_capture_test_eval()`, which is called exactly three times: pre-training, after
Stage 1, and after Stage 2. During training, the model is monitored with the
**validation set** only.

---

## 2. Configuration reference

All knobs live in **Cell 4**. Grouped by purpose:

### Heliostat and data source

| Variable | Default | Meaning |
|----------|---------|---------|
| `HELIOSTAT_ID` | `"AW36"` | Which heliostat to use. Options: `AC36 AG33 AO34 AW36 BE35` |
| `DATA_MODE` | `"random_synthetic"` | Data source — see §3 |
| `SCENARIO_PATH` | derived | Path to single-heliostat HDF5 scenario |
| `SYNTH_DATASET_DIR` | `scenarios/full_63_heli_kin_reconstruct/synthetic_data` | Root of the pre-generated synthetic dataset |
| `PERTURBATIONS_JSON` | `SYNTH_DATASET_DIR/perturbations.json` | GT perturbations for the synthetic dataset |
| `BENCHMARK_NAME` | `benchmark_split-balanced_train-100_...` | PAINT benchmark CSV name (real mode) |
| `CENTROID_METHOD` | `"UTIS"` | Centroid extraction method for real images |

### Resolution and ray counts

| Variable | Default | Meaning |
|----------|---------|---------|
| `SURFACE_POINTS_PER_FACET` | `25` | Each facet gets 25×25 = 625 surface points. Must match the dataset generation resolution |
| `TRAIN_RAYS` | `10` | Rays per surface point during Stage 2 training (speed vs. accuracy trade-off) |
| `DISPLAY_RAYS` | `50` | Rays used for visualization and test evaluation (cleaner images) |
| `GENERATE_RAYS` | `100` | Rays used when generating GT data in memory (`random_synthetic` / custom mode) |

### Training schedule

| Variable | Default | Meaning |
|----------|---------|---------|
| `STAGE1_EPOCHS` | `20` | Epochs for AlignmentLoss (fast; no ray tracing) |
| `STAGE2_EPOCHS` | `100` | Epochs for FocalSpotLoss (slow; ray tracing per mini-batch) |
| `MINI_BATCH_SIZE` | `25` | Samples per mini-batch in Stage 2. Stage 1 always uses all samples |
| `BASE_LR` | `1e-4` | Base Adam learning rate. Translation and base-position params get `5 × BASE_LR` |
| `PLOT_EVERY` | `1` | Capture centroid trail every N epochs (1 = every epoch) |
| `N_DISPLAY` | `10` | How many training sun positions to show in the pre-training grid |

### GT flux filtering

| Variable | Default | Meaning |
|----------|---------|---------|
| `FLUX_SUM_MIN` | `1e-4` | Minimum total flux sum for a sample to be kept |
| `MIN_ACTIVE_PIXEL_PERCENT` | `0.1` | Minimum fraction of strictly-positive pixels (in %) |

Both criteria must be satisfied to keep a sample.

### Wortberg deviation bounds (used for both clamping and random sampling)

| Variable | Value | Parameter group |
|----------|-------|----------------|
| `_BOUND_TRANSLATION_M` | `0.05 m` | 9 joint/concentrator translations |
| `_BOUND_ROTATION_RAD` | `0.005 rad` | 4 joint tilts |
| `_BOUND_ACTUATOR_ANGLE_RAD` | `0.005 rad` | aᵢ (actuator initial angle) |
| `_BOUND_ACTUATOR_STROKE_M` | `0.005 m` | bᵢ — **frozen during training**, not recoverable |
| `_BOUND_ACTUATOR_OFFSET_M` | `0.005 m` | cᵢ (actuator offset) |
| `_BOUND_BASE_POSITION_M` | `0.05 m` | Heliostat base position (E, N, U) |

After every optimizer step, all parameters are **hard-clamped** to these bounds
(relative to the initial value for translation/angle/offset; absolute for rotation
and base position).

### Random-synthetic perturbation bounds

```python
RANDOM_SEED = 42
RANDOM_PERT_BOUNDS = {
    "rotation_rad":       _BOUND_ROTATION_RAD,
    "actuator_angle_rad": _BOUND_ACTUATOR_ANGLE_RAD,
    "actuator_stroke_m":  _BOUND_ACTUATOR_STROKE_M,
    "actuator_offset_m":  _BOUND_ACTUATOR_OFFSET_M,
    "translation_m":      _BOUND_TRANSLATION_M,
    "base_position_m":    _BOUND_BASE_POSITION_M,
}
```

Each component is drawn independently from `Uniform(-bound, +bound)`. The same
dictionary is used for clamping during training, so the optimizer can always
recover the full perturbation.

### Custom perturbation mode (only for `DATA_MODE = "synthetic"`)

```python
USE_CUSTOM_PERTURBATIONS = False   # set True to override the on-disk GT
CUSTOM_PERTURBATIONS_SPEC = {
    "rotation":        [0.002, 0.000, 0.000, 0.000],  # rad (4 joint tilts)
    "actuator_angle":  [0.001, -0.001],                # rad
    "actuator_stroke": [0.000,  0.000],                # m (frozen, not recovered)
    "actuator_offset": [0.002,  0.000],                # m
    "translation":     [0.005, 0., 0., 0., 0., 0., 0., 0., 0.],  # m (9)
    "base_position":   [0.010, -0.020, 0.005],         # m [E, N, U]
}
```

---

## 3. Data modes

### `"synthetic"` — pre-generated dataset from disk

The GT dataset was already ray-traced offline with the perturbations listed in
`perturbations.json`. The notebook loads flux images, centroids, motor positions,
and ray directions from:

```
SYNTH_DATASET_DIR/
  perturbations.json           ← per-heliostat GT perturbation values
  train/<HELIOSTAT_ID>/0000/
    calibration_properties.json
    flux_image.png
  val/<HELIOSTAT_ID>/0000/...  ← optional; if absent, no val monitoring
  test/<HELIOSTAT_ID>/0000/...
```

GT perturbations are **known** (`pert_tensors` is loaded from JSON), so
parameter trajectory plots are available.

With `USE_CUSTOM_PERTURBATIONS = True`: the on-disk flux/centroids are replaced
by in-memory generation using `CUSTOM_PERTURBATIONS_SPEC`. Rays, active masks,
and target masks still come from disk; only the physics outputs are regenerated.

### `"random_synthetic"` — in-memory generation from random perturbations

1. The notebook draws one perturbation set per `RANDOM_SEED` from `RANDOM_PERT_BOUNDS`.
2. Rays, masks, and target indices are loaded from disk (same as `"synthetic"`).
3. The perturbation is applied to the clean kinematic parameters, `GENERATE_RAYS`
   rays per surface point are shot, and centroids + flux images are computed in memory.
4. The kinematics are then reset to clean. Training starts from the clean state.

GT perturbations are **known** (`pert_tensors` holds the drawn values), so all
plots are available. Useful for testing recovery ability with different perturbation
magnitudes without generating a separate disk dataset.

### `"real"` — actual PAINT calibration images

Loads from the PAINT benchmark CSV split (`BENCHMARK_NAME`). Only samples
belonging to `HELIOSTAT_ID` are selected. The parser is `PaintCalibrationDataParser`
which reads physical flux images and extracts centroids via `CENTROID_METHOD` (UTIS).

GT perturbations are **unknown** (`pert_tensors = None`), so:
- Parameter trajectory plots are skipped (Cell 29 prints a skip message)
- The mrad metric still works (predicted centroid vs. PAINT-measured centroid)
- Stage 1 / Stage 2 convergence and test accuracy are fully available

The validation split for real data comes from the `"validation"` split in the
benchmark CSV. If `HELIOSTAT_ID` is absent from that split, `val_flux = None` and
no val monitoring happens.

---

## 4. Scenario and kinematic parameters

**Cell 6** loads the single-heliostat HDF5 scenario with `SURFACE_POINTS_PER_FACET²`
surface points per facet (default: 625 per facet, 2500 per heliostat).

```python
scenario   # ARTIST Scenario object (one heliostat group, one light source)
heliostat_group = scenario.heliostat_field.heliostat_groups[0]
kinematic  = heliostat_group.kinematics  # WortbergKinematicModel
```

The heliostat-to-tower distance `hel_dist_m` is computed once here as:

```
hel_dist_m = ||heliostat_position - mean(target_area_centers)||
```

This scalar is reused throughout the notebook to convert centroid displacement
on the receiver from metres to milliradians:

```
error_mrad = error_m / hel_dist_m * 1000
```

### Optimized parameters (Wortberg 2025, Table 5.3)

| Tensor | Shape | Description |
|--------|-------|-------------|
| `kinematic.translation_deviation_parameters` | `[1, 9]` | Joint + concentrator position deviations (m) |
| `kinematic.rotation_deviation_parameters` | `[1, 4]` | Joint tilt deviations (rad) |
| `kinematic.actuators.optimizable_parameters[:, actuator_initial_angle, :]` | `[1, 2]` | aᵢ — initial angles (rad) |
| `kinematic.actuators.non_optimizable_parameters[:, actuator_offset, :]` | `[1, 2]` | cᵢ — actuator offsets (m) |
| `kinematic._base_position_deviation` | `[1, 3]` | ENU position offset (m), injected separately |

**Frozen:** `actuator_initial_stroke_length` (bᵢ) — zeroed by gradient hook.

---

## 5. Data loading and in-memory generation

**Cell 8** performs all data loading. The six arrays returned for each split are:

| Variable | Shape | Description |
|----------|-------|-------------|
| `*_flux` | `[N, H, W]` | GT flux images on `device` |
| `*_centroids` | `[N, 4]` | GT centroid ENU coordinates + homogeneous |
| `*_rays` | `[N, 3]` | Incident ray directions (pointing toward heliostat) |
| `*_motor_pos` | `[N, 2]` | GT motor positions (one pair per sample) |
| `*_active_mask` | `[1]` | Tensor holding N (count of samples for this heliostat) |
| `*_target_mask` | `[N]` | Target area index per sample |

`active_mask` is a 1-element tensor whose value is the sample count, **not** a
binary mask. This is how ARTIST's `activate_heliostats` works for a single-heliostat
scenario.

### In-memory regeneration (random_synthetic / custom)

When `DATA_MODE == "random_synthetic"` or `USE_CUSTOM_PERTURBATIONS == True`:

1. `apply_perturbations(kinematic, pert_tensors, device)` modifies the kinematic
   parameters in-place and returns a `_snap` dict for later reset.
2. `_pert_bpd = kinematic._base_position_deviation.detach().clone()` captures the
   applied base-position shift before it can be reset.
3. `_forward_pass(...)` activates the heliostat, injects `_pert_bpd`, aligns
   surfaces, ray-traces, and returns `(centroids, flux)` in natural order. The
   sampler permutation is internally inverted so outputs align with input ray order.
4. Motor positions are captured from `kinematic.active_motor_positions` after the
   forward pass.
5. `reset_perturbations(kinematic, _snap)` restores all parameters to their
   clean values. Training starts from clean.

---

## 6. GT flux filtering

**Cell 9** — `_apply_flux_filter()` — drops samples where the GT flux image is
effectively empty. Two independent criteria:

- `flux.sum() > FLUX_SUM_MIN` (default: `1e-4`) — rejects completely black images
- `(flux > 0).sum() / flux.numel() * 100 >= MIN_ACTIVE_PIXEL_PERCENT` (default: `0.1%`)
  — rejects very sparse images with almost no lit pixels

Both must be satisfied. The filter reports how many samples were dropped and for
which reason, then reassigns all six split arrays to contain only the kept samples.
`active_mask` is updated to `[kept]`.

After filtering, `N_TRAIN`, `N_VAL`, `N_TEST` are set and a subset of
`N_DISPLAY` training indices evenly spread by solar elevation is selected
for the pre-training visualization.

---

## 7. Dataset overview plots

**Cell 12** prints a dataset-type description before plotting, then shows one grid
per split (train / val / test).

The description varies by mode:
- **real**: benchmark name, heliostat ID, centroid method, scenario path
- **synthetic (disk)**: source directory, perturbations JSON path
- **synthetic (custom)**: lists all values in `CUSTOM_PERTURBATIONS_SPEC`
- **random_synthetic**: seed, rays-per-point, all perturbation bounds

Each grid (`_plot_split_grid`) shows all samples sorted by **solar elevation**
(low elevation = early morning / late afternoon, high elevation = solar noon).
Each cell shows the peak-normalised flux image with the GT centroid marked as a
green `+`. The subtitle of each cell is the elevation angle in degrees.

Val grid appears only when `val_flux is not None`. Test grid always appears.

---

## 8. Pre-training baseline

**Cell 14** runs `_forward_pass` with the **clean** kinematic parameters (no
perturbation) against the **perturbed** GT data. This gives the starting-point
error — how far off the clean model is before training.

The cell plots all training samples as predicted/GT pairs, sorted by elevation.
Each GT cell is annotated with the per-sample mrad error. The mean and max over
all training samples are shown in the figure title.

This is a diagnostic: in synthetic modes the error should approximate the
perturbation magnitude. In real mode it shows how much the clean scenario
disagrees with the actual heliostat.

---

## 9. Optimizer setup

**Cell 17** configures everything for Stage 1 (and its infrastructure is reused by
Stage 2). Key steps:

### Gradient enabling
```python
kinematic.translation_deviation_parameters.requires_grad_(True)
kinematic.rotation_deviation_parameters.requires_grad_(True)
kinematic.actuators.optimizable_parameters.requires_grad_(True)
kinematic.actuators.non_optimizable_parameters.requires_grad_(True)
```

### Gradient hooks
Two hooks are registered to enforce the Wortberg parameter selection:

- `_freeze_stroke`: zeros the gradient row corresponding to `actuator_initial_stroke_length`
  (bᵢ index) inside `optimizable_parameters`, keeping it frozen during training.
- `_only_c_i`: zeros all gradient rows in `non_optimizable_parameters` **except** the
  `actuator_offset` row (cᵢ index), so only the offset is optimized from that tensor.

### Base-position deviation
`kinematic._base_position_deviation` is a dynamically-attached attribute that does
not exist in the original ARTIST kinematic model. It is created as `zeros(1, 3)`
here and added to the active heliostat positions inside each training step. This
lets the optimizer move the heliostat's assumed base position without modifying
the scenario geometry itself.

### Initial-value snapshots
```python
_init_translation  # [1, 9] — for relative clamping
_init_angle        # [1, 2] — for relative clamping of actuator angles
_init_offset       # [1, 2] — for relative clamping of actuator offsets
```

Rotation and base-position are clamped to absolute bounds (initial value assumed 0).

### `_apply_bounds()`
Called after every optimizer step. Clamps each parameter group:
- Translation: `[init ± 0.05 m]`
- Rotation: `[-0.005, +0.005] rad` (absolute)
- Actuator angle (aᵢ): `[init ± 0.005 rad]`
- Actuator offset (cᵢ): `[init ± 0.005 m]`
- Base position: `[-0.05, +0.05 m]` (absolute, per component)

### Adam optimizer — differential learning rates
Large-scale parameters (translation, base-position) get `5 × BASE_LR`; all others
get `BASE_LR`. ReduceLROnPlateau scheduler: `factor=0.5`, `patience=5`,
`cooldown=3`, `min_lr=1e-8`.

---

## 10. Trail capture infrastructure

**Cell 19** defines three monitoring utilities:

### `_capture_trails(label, epoch)`
Ray-traces the **training set** with the current kinematics (no grad) and stores:
- Per-sample pixel-space centroid `(col, row)` of the predicted flux bitmap
- Mean and median mrad over training samples
- Mean mrad over the **validation set** (if `val_flux is not None`)

The test set is **never** evaluated here. Test evaluation is only done at the three
milestone snapshots described below.

Stored as a list of dicts in `_trail_checkpoints`. One entry per captured epoch.

### `_capture_grad_and_params(abs_epoch)`
Called after `optimizer.step()` + `_apply_bounds()` to record:
- Gradient L2-norm per parameter group (after clipping but before step affects grads)
- Current parameter values relative to their initial snapshot (for deviation plots)

Results in `_grad_history` and `_param_history`.

### `_capture_test_eval(label)`
Evaluates the full test set using `DISPLAY_RAYS` (50) for cleaner images and stores
predicted flux bitmaps, per-sample mrad errors, and metre errors. Used only at:
- `'Pre-training'` — before any optimization
- `'After Stage 1'` — after Stage 1 best-checkpoint restore
- `'After Stage 2'` — after Stage 2 best-checkpoint restore

---

## 11. Stage 1 — AlignmentLoss

**Cell 21** — motor-position MSE, no ray tracing.

### Loss definition
```
AlignmentLoss(pred_motor_pos, meas_motor_pos, actuators, device)
  → per_sample_loss [N_TRAIN]
```

Both predicted and measured motor positions are first converted to **joint angles**
via the actuator model (`motor_positions_to_angles`), then the squared difference
is summed over the two actuators. This makes the loss invariant to the absolute
motor-position scale of the specific actuator family.

Unit: **rad²** (squared actuator joint-angle error, summed over 2 actuators).

### Per-epoch training step
1. `activate_heliostats(train_active_mask)` — sets `active_heliostat_positions` to
   the heliostat's static position replicated N_TRAIN times.
2. Base-position injection:
   `active_heliostat_positions += repeat(base_pos_deviation, N_TRAIN) padded to [N, 4]`
3. `align_surfaces_with_incident_ray_directions(...)` — runs the kinematic forward
   model, yielding `active_motor_positions`.
4. `AlignmentLoss` is applied, averaged over samples, and backpropagated.
5. Gradient clip (max norm 1.0) → optimizer step → bounds clamp → parameter capture.

### Validation monitoring
Each epoch runs the same alignment-loss computation on the val set (no grad). The
val loss is used for the scheduler and for **checkpoint selection**.

### Best-checkpoint restore
After the loop, the parameter set that produced the lowest val alignment loss
(or train loss if no val data) is restored. This prevents Stage 1 from
overfitting to noise in the last few epochs.

---

## 12. Stage 2 — FocalSpotLoss

**Cell 23** — ray-traced centroid MSE, mini-batched.

### Loss definition
```
FocalSpotLoss(scenario)(prediction_flux, gt_centroids, target_indices, ...)
  → per_sample_loss [mb_size]
```

The ray tracer produces a flux bitmap per sample. The ARTIST `get_center_of_mass`
function computes the weighted centroid of that bitmap and `bitmap_coordinates_to_target_coordinates`
converts it to ENU target coordinates. The loss is the squared Euclidean distance
between the predicted centroid and the GT centroid.

Unit: **m²** (squared centroid displacement on the receiver plane).

### Sampler permutation
`HeliostatRayTracer` internally uses a `RestrictedDistributedSampler` that may
reorder the samples for GPU efficiency. The output flux tensor is in the
**sampler's order**, not the input ray order. After tracing:

```python
sample_idx = ray_tracer.get_sampler_indices()
lps = focal_spot_loss_fn(
    prediction=flux,
    ground_truth=mb_gt_cents[sample_idx],  # reindex GT to match flux order
    target_area_indices=mb_target[sample_idx],
    ...
)
```

### Mini-batching
Training samples are processed in mini-batches of `MINI_BATCH_SIZE` (default: 25).
Gradients accumulate across mini-batches before `optimizer.step()`:

```python
(lps.mean() * (mb_size / N_TRAIN)).backward()  # weight by sample count, not batch count
```

Dividing by `N_TRAIN` ensures each sample contributes equally to the gradient,
even when the last mini-batch is smaller.

### Per-epoch random seed
```python
random_seed = epoch * 1000 + mb  # varies per epoch and per mini-batch
```

A different seed per epoch prevents the optimizer from fitting to a fixed ray
sampling pattern (which would happen with `random_seed=42` every epoch).

### Validation monitoring
Same mini-batch structure as training, but under `torch.no_grad()`. Val loss drives
the scheduler and checkpoint selection.

### Best-checkpoint restore
After the Stage 2 loop, parameters from the epoch with the lowest val loss
(or train loss if no val) are restored. This is the final model used for test
evaluation and all visualization.

---

## 13. Centroid trail visualization

**Cell 25** — shows how the predicted centroid moved across training.

### Layout
For each time-of-day bin (morning / solar noon / afternoon / other), the cell
produces one figure:
- **Left panel**: polar sky-chart showing all sun positions. Current bin = blue,
  other bins = dimmed gray. Orientation is **European solar convention** — South at
  top, West at right, East at left, North at bottom. Radius = zenith angle (0° =
  overhead, 90° = horizon).
- **Right panels**: one flux image per sample in the bin. The **GT flux** is shown
  (grayscale, normalized). Overlaid: the centroid trail across all captured epochs.

### Colormap convention
- **Blues** (light → dark): Stage 1 (AlignmentLoss). Light blue = pre-training /
  epoch 1, dark blue = end of Stage 1.
- **RdYlGn** (red → yellow → green): Stage 2 (FocalSpotLoss). Red = epoch 1 of
  Stage 2, green = final epoch. Each dot is one captured epoch; segments connect
  consecutive valid (non-None) centroids.
- **Green × marker**: GT centroid (from the flux bitmap center-of-mass).
- **Red × marker**: final predicted centroid (last trail checkpoint).

Corner annotation on each image: final mrad error + metres for that sample.

A two-panel horizontal colorbar below the per-bin figures summarizes the stage
color coding.

---

## 14. Convergence plot (mrad)

**Cell 27** — both stages on a single axis with the same unit.

The x-axis spans all captured epochs (pre-training at epoch 0, then Stage 1
epochs 1…S1, then Stage 2 epochs S1+1…S1+S2).

- **Train curve (solid blue)**: `mrad_mean` from `_trail_checkpoints` — mean focal-
  spot error over training samples computed by ray-tracing with the current kinematics.
- **Val curve (dashed orange)**: `mrad_val_mean` from `_trail_checkpoints` — same
  computation on validation samples. Shown only when `val_flux is not None`.

The vertical dashed line marks the Stage 1 → Stage 2 transition. A gap between
train and val curves indicates overfitting. Both curves use mrad regardless of which
loss was being optimized — the unit is consistent because it is always derived from
centroid displacement on the receiver.

---

## 15. Gradient norms and parameter trajectories

**Cell 29** — only shown when GT perturbations are known (synthetic or random_synthetic
modes). Skipped in real mode.

**Top panel**: log-scale gradient L2-norm per parameter group across all epochs.
Five groups: translation, rotation, actuator_angle, actuator_offset, base_position.
Each group uses a separate color. Vertical dashed line = Stage 1 → Stage 2.

**Bottom panels** (one per parameter group):
- **Solid lines**: current parameter value (relative to initial for angles/offsets;
  absolute for translation, rotation, base-position).
- **Dashed lines of the same color**: GT perturbation target value.
- Convergence means solid lines reach the dashed lines.

Parameter values are recorded **after** each optimizer step + bounds clamp,
so they represent the actual state used for the next forward pass.

---

## 16. Final evaluation

**Cell 31** outputs three things:

### Accuracy table
```
Stage                    Mean [mrad]  Median [mrad]  Mean [m]  Median [m]
----------------------------------------------------------------------
Pre-training             ...
After Stage 1            ...
After Stage 2            ...
```

Computed from `_test_evals`, which holds the three milestone snapshots. Each row
is the mean/median over all test samples for this single heliostat.

### Raw loss curves
Two side-by-side subplots:
- **Stage 1** — AlignmentLoss [rad²] vs. epoch; train (solid) and val (dashed, if available)
- **Stage 2** — FocalSpotLoss [m²] vs. epoch; train (solid) and val (dashed, if available)

These show the raw optimizer signals. For both stages in a common mrad unit, see the
convergence plot in Cell 27.

### Test image grid
All test samples sorted by solar elevation. For each sample, two rows:
- **Top**: final predicted flux (normalized grayscale). Red `+` = predicted centroid,
  green `×` = GT centroid. Corner annotation = mrad and metre error.
- **Bottom**: GT flux with green `×` centroid.

---

## 17. Key tensor shapes and conventions

### Coordinate system
ARTIST uses **ENU (East-North-Up)** with homogeneous coordinates `[E, N, U, 1]`.
The `[:, :3]` slice gives the ENU 3-vector.

### Incident ray directions
`train_rays` shape: `[N, 3]`. These point **from the sun toward the heliostat**
(i.e., the direction light travels). The sun direction is `−train_rays[:, :3]`.
Solar elevation: `arcsin(−ray[:, 2])` (negative z = sun above horizon for a
downward-pointing ray).

### `active_heliostats_mask`
Shape: `[1]`. For a single-heliostat scenario, contains the integer N (number of
active sample instances), not a boolean 0/1. The ARTIST activation logic replicates
the single heliostat's geometry N times in memory.

### Flux images
- GT synthetic flux: physical ray-tracer output in intensity units (~1e-4 per pixel
  at ~100 m distance). Stored as `float32` on `device`.
- GT real flux (PAINT): `uint8` PNG normalized to `[0, 1]` on load.
- Both are displayed using `_to_norm()` which peak-normalises to `[0, 1]` for display.
- The centroid computation (`get_center_of_mass`) divides by the total sum, so it is
  scale-invariant and works correctly for both unit conventions.

### mrad conversion
```
error_mrad = torch.norm(pred_centroid[:3] - gt_centroid[:3]) / hel_dist_m * 1000
```

`hel_dist_m` is computed once from the scenario geometry.

---

## 18. Dependencies and file paths

### Relative to `master-thesis/`
```
scenarios/one_heliostat_scenarios/<HELIOSTAT_ID>/scenario.h5  ← clean single-heliostat scenario
scenarios/full_63_heli_kin_reconstruct/synthetic_data/        ← pre-generated synthetic dataset
datasets/paint/splits/<BENCHMARK_NAME>.csv                    ← PAINT benchmark CSV
datasets/paint/<BENCHMARK_NAME>/calibration_properties/       ← PAINT calibration JSONs
datasets/paint/<BENCHMARK_NAME>/flux_image/                   ← PAINT flux PNGs
```

### Python imports
```python
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario
from artist.util import get_device, set_logger_config, indices, constants
from artist.optim.loss import FocalSpotLoss

from utils.synth_data import (
    _forward_pass,
    apply_perturbations, reset_perturbations, sample_perturbations,
    SyntheticDatasetParser,
)
from artist_extensions.loss_functions_ext import AlignmentLoss
```

### ARTIST path setup
The notebook adds `_src = notebook_dir.parent` to `sys.path`, where `_src` is the
`src/` directory. Both `utils/` and `artist_extensions/` live there.

### `_forward_pass` function (`src/utils/synth_data.py`)
Signature: `(scenario, heliostat_group, rays, active_mask, target_mask, base_pos_delta, device) → (centroids, flux)`

Internally: activates heliostat → injects `base_pos_delta` → aligns surfaces →
ray-traces → computes center-of-mass centroids → inverts sampler permutation →
returns everything in original ray order.

### `AlignmentLoss` (`src/artist_extensions/loss_functions_ext.py`)
Converts motor positions to joint angles via `actuators.motor_positions_to_angles`,
then returns per-sample `sum((pred_angles − meas_angles)²)` over 2 actuators.

### `WortbergKinematicReconstructor` (`src/artist_extensions/kinematic_reconstructors.py`)
The full production version of this notebook's training loop. The notebook replicates
its core logic inline for visibility. If productionising, use the reconstructor class
instead of the notebook cells.
