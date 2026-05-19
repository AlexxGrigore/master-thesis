# Five-Heliostat Synthetic Perturbation Experiment

## Goal

Evaluate how well a `WortbergKinematicReconstructor` can recover known kinematic
perturbations in a fully controlled synthetic setting — where the ground-truth
perturbation is known exactly and no real PAINT flux images are involved.

A secondary goal is to measure how the **amount of training data** (number of
calibration samples per heliostat) affects recovery quality.

---

## Heliostats

Five heliostats from the PAINT dataset:

```
AA31, AQ28, BA37, BC33, AZ55
```

Scenario file: `scenarios/five_heliostats_scenario/scenario.h5`

---

## Data Generation

All data (train, validation, test) is generated **synthetically from the clean
(unperturbed) scenario**. No real PAINT flux images are used for evaluation.

Real PAINT sun positions (from the benchmark split) drive the ray tracer, but the
focal-spot centroids and flux bitmaps are computed from the clean scenario geometry.

**Run once before training:**
```
python generate_dataset.py
python generate_dataset.py --force   # overwrite existing files
```

This saves files to `scenarios/five_heliostats_scenario/synthetic_data/`:

```
synthetic_data/
  {train|val|test}/
    {heliostat_id}/
      {idx:04d}/
        calibration_properties.json   ← incident_ray_direction, focal_spot_enu (ENU, 4D),
                                         motor_position, target_area_index
        flux_image.png                ← uint8 grayscale flux bitmap, normalised to [0,255]
```

No WGS84 conversion is needed — centroids are stored directly in local ENU coordinates
as computed by `bitmap_coordinates_to_target_coordinates`.

The train folder always contains 50 files per heliostat. The mapping passed to
`SyntheticDatasetParser` at training time controls how many are loaded (so all
6 train-size runs share the same on-disk folder).

| Split      | Samples / heliostat | Purpose                          |
|------------|---------------------|----------------------------------|
| train      | 50 (ablation loads 1–50) | Drives the optimizer        |
| validation | 10                  | Scheduler / early stopping       |
| test       | 10                  | Final evaluation (fixed)         |

Rays used during data generation: **100** (high, for clean noiseless centroids).
Rays used during training: **10** (lower, for speed).

---

## Evaluation Protocol

The test dataset is generated **once from the clean scenario** and never changed.
Only the scenario's kinematic parameters are modified between stages.

```
Stage 1 — Pre-perturbation
  Scenario params : CLEAN (original)
  Test data       : clean synthetic
  Expected result : ~0 mrad  (sanity check — perfect params match clean data)

Stage 2 — Post-perturbation
  Scenario params : PERTURBED  (random offsets applied in-place to all 6 groups)
  Test data       : same clean synthetic
  Expected result : high mrad  (wrong params degrade prediction)

Stage 3 — Post-training
  Scenario params : TRAINED  (optimizer corrects from perturbed state)
  Train/val data  : clean synthetic (same source as test)
  Test data       : same clean synthetic
  Expected result : low mrad  (optimizer recovered toward clean params)
```

A kinematic snapshot (`kinematic_stages.json`) is captured at each stage for
post-hoc analysis and presentation plots.

> **Note:** `evaluate_flux_accuracy` does not inject `_base_position_deviation`
> into the ray tracer, so the base-position component of recovery is not reflected
> in the mrad metrics. Rotation and actuator-angle components ARE reflected (they
> are baked directly into the scenario's kinematic state).

---

## Perturbations

Random per heliostat, seeded for reproducibility (`PERTURBATION_SEED = 42`).
All six parameter groups of `WortbergKinematicReconstructor` are perturbed,
including the frozen `b_i` (actuator stroke length):

| Parameter group                    | Range | Unit | Optimized? |
|------------------------------------|-------|------|------------|
| Rotation (4 joint tilts)           | ±5    | mrad | Yes        |
| Actuator initial angle aᵢ (×2)     | ±5    | mrad | Yes        |
| Actuator stroke length bᵢ (×2)     | ±5    | mm   | **No (frozen)** |
| Actuator offset cᵢ (×2)            | ±5    | mm   | Yes        |
| Translation deviation (9 joints)   | ±50   | mm   | Yes        |
| Base position (east, north, up)    | ±50   | mm   | Yes        |

Perturbations are saved to `perturbations.json` at the start of every run.

The frozen `b_i` perturbation is permanent — the optimizer cannot recover it.
All `abs_residual` values in `param_recovery` measure deviation from the clean state
and therefore approach **0 on perfect recovery** (except `actuator_stroke`, which
remains at `|perturbation|` regardless of training).

---

## Training Data Ablation

The experiment is run **six times** with different training sample counts:

| Run | Train samples / heliostat |
|-----|---------------------------|
| A   | 1                         |
| B   | 5                         |
| C   | 10                        |
| D   | 15                        |
| E   | 25                        |
| F   | 50                        |

Validation and test sets are kept constant across all runs (10 samples each).
All other hyperparameters are identical.

---

## Evaluation Metrics

Two metrics are reported per stage and per heliostat:

**1. Focal spot error (mrad)**
Centroid-based. Computed as `|predicted_centroid - ground_truth_centroid|` in metres,
then converted to mrad using each heliostat's individual distance to the reference
target center. Sampler reordering is undone with `torch.argsort(sample_indices)`
before per-heliostat slicing.

**2. Pixel-wise L1 loss**
Image-based. Both predicted (physical intensity units) and measured ([0,1]) flux
bitmaps are:
1. Blurred with a Gaussian kernel (σ=1) to smooth ray-tracing noise.
2. Peak-normalised per image (divide by max pixel value).
3. Compared with per-pixel absolute difference, summed over the image.

This is scale-invariant and handles the unit mismatch between ray-tracer output
and stored PNG values.

---

## Optimisation Configuration

| Parameter               | Value                                                  |
|-------------------------|--------------------------------------------------------|
| Reconstructor           | `WortbergKinematicReconstructor` (full Wortberg set)   |
| Learning rate           | 1e-4                                                   |
| Scheduler               | ReduceLROnPlateau (factor=0.5, patience=10, cooldown=5)|
| Max epochs              | 300                                                    |
| Early stopping patience | 400 (disabled — always runs full 300 epochs)           |
| Batch size              | 8                                                      |
| Log step                | 5                                                      |
| Surface pts / facet     | 50×50 (→ 10,000 pts/heliostat, 4 facets)               |

---

## Output Structure

Each run produces a timestamped directory under `outputs/local_runs/` (local) or
`outputs/` (DAIC):

```
five_hel_synth_{timestamp}/
  perturbations.json                  — true perturbation values per heliostat (all 6 groups)
  run.log                             — full execution log

  train_1/                            — results for 1 training sample/heliostat
    results.json                      — summary of all 3 mrad + pixel-loss values + param_recovery
    kinematic_parameters.json         — final trained kinematic parameters (loadable by pipeline)
    kinematic_stages.json             — kinematic snapshot at 3 stages: clean / perturbed / trained
    convergence_history.json          — train/val loss per epoch
    kinematic_history.json            — per-heliostat param values at each log step (epoch-by-epoch)
    convergence.png                   — loss plot + 3 horizontal reference lines
    recovery_rotation.png             — perturbation vs residual: rotation
    recovery_actuator_angle.png       — perturbation vs residual: actuator angle aᵢ
    recovery_actuator_stroke.png      — perturbation vs residual: actuator stroke bᵢ (frozen)
    recovery_actuator_offset.png      — perturbation vs residual: actuator offset cᵢ
    recovery_translation.png          — perturbation vs residual: translation (9 params)
    recovery_base_position.png        — perturbation vs residual: base position
    kinematic_stages_AA31.png         — bar chart: clean / perturbed / trained params per heliostat
    kinematic_stages_AQ28.png
    kinematic_stages_BA37.png
    kinematic_stages_BC32.png
    kinematic_stages_AZ55.png
    kinematics_AA31.png               — param evolution over training epochs
    kinematics_AQ28.png
    kinematics_BA37.png
    kinematics_BC32.png
    kinematics_AZ55.png
    flux_comparisons/
      flux_comparison_AA31.png        — measured vs predicted flux + difference (post-training)
      flux_comparison_AQ28.png
      flux_comparison_BA37.png
      flux_comparison_BC32.png
      flux_comparison_AZ55.png

  train_5/    — same structure
  train_10/   — same structure
  train_15/   — same structure
  train_25/   — same structure
  train_50/   — same structure

  ablation_summary.json               — comparison across all 6 runs
  ablation_comparison.png             — table: mrad per stage per run
  combined_convergence.png            — overlay of training curves for all sizes
```

---

## Presentation Plots

Three types of plots are generated specifically for presentations:

### 1. Kinematic stages (`kinematic_stages_{hid}.png`)
One figure per heliostat. Shows all kinematic parameter groups (rotation, actuator
angle, actuator stroke, actuator offset, translation, base position) as grouped bar
charts across three stages:
- **Clean** — original scenario values (deviation params all ≈ 0)
- **Perturbed** — after random perturbations are applied
- **Trained** — after the optimizer runs

A bar that returns to ~0 in the trained stage indicates good recovery.

### 2. Flux comparison (`flux_comparisons/flux_comparison_{hid}.png`)
One figure per heliostat. Side-by-side view of:
- **Measured** — the ground-truth flux bitmap from the synthetic dataset (blurred σ=1, peak-normalised)
- **Predicted** — the ray-traced flux bitmap from the trained scenario (blurred σ=1, peak-normalised)
- **Difference** — absolute difference between the two preprocessed images

Pixel-wise L1 loss is computed on the preprocessed (blurred, peak-normalised) images before plotting.
Annotated with focal spot error (mrad) and pixel-wise L1 loss for the first test sample.

Images are saved under a `flux_comparisons/` subfolder within each `train_N/` run directory.

### 3. Parameter evolution (`kinematics_{hid}.png`)
One figure per heliostat. Plots kinematic parameter values over training epochs,
with dashed horizontal lines marking the true perturbation target. Shows whether
the optimizer converges toward the correct values.
