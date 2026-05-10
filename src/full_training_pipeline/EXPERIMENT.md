# Full-Field Fine-Error-Learning Pipeline

## Goal

Learn a **field-level residual correction** on top of coarse kinematic parameters that were
already optimised by `WortbergKinematicReconstructor` on the full 63-heliostat field.

The hypothesis is that a shared model can capture systematic prediction errors that remain
after coarse kinematic calibration — e.g., systematic biases or non-linearities that are
shared across all heliostats.

---

## Architecture

```
heliostat feature vector (42D)
         │
  SharedLinearResidualModel   ← or SharedPolyResidualModel(degree=2/3/4)
    Linear(42×d → 20)  →  tanh  ×  bounds
         │
  residual correction (20D)
         │
coarse kinematic params (20D, from checkpoint)
         │  +
  corrected params (20D)
         │
  ray tracer / kinematic solver
         │
  loss (focal_spot | pixel | alignment)
```

### Model Variants

Selected via `--model-type` CLI flag (default: `linear`).

| Model type | Architecture             | Input dim | Parameters | Ratio (63 hel) |
|------------|--------------------------|-----------|------------|----------------|
| `linear`   | Linear(42 → 20)          | 42        | 860        | 14:1           |
| `poly2`    | Linear(84 → 20), [x, x²] | 84        | 1,700      | 27:1           |
| `poly3`    | Linear(126 → 20), [x, x², x³] | 126  | 2,540      | 40:1           |
| `poly4`    | Linear(168 → 20), [x..x⁴] | 168     | 3,380      | 54:1           |

Polynomial models expand the 42D feature vector with power terms (no cross-terms) before
the linear output layer. All models use zero-weight initialisation and `tanh × bounds`
output activation to keep corrections within physical limits.

### Output bounds (tanh scaling)

| Parameter group             | Count | Bound  | Unit |
|-----------------------------|-------|--------|------|
| Translation deviation       | 9     | ±0.05  | m    |
| Rotation deviation          | 4     | ±0.005 | rad  |
| Actuator initial angle (aᵢ) | 2     | ±0.005 | rad  |
| Actuator offset (cᵢ)        | 2     | ±0.005 | m    |
| Base position (E, N, U)     | 3     | ±0.05  | m    |

These match the Wortberg (2025) Table 5.3 optimisable parameter set.

---

## Feature Engineering

Each heliostat is represented by a **42-dimensional summary vector**:

| Dims  | Feature group        | Description                                           |
|-------|----------------------|-------------------------------------------------------|
| 0–2   | `heliostat_position` | Absolute ENU position from scenario (3D)              |
| 3–22  | `kinematic_params`   | Flattened 20-D Wortberg coarse checkpoint params      |
| 23–25 | `mean_centroid`      | Mean ENU flux centroid across all measurements (3D)   |
| 26–28 | `std_centroid`       | Std of centroid positions (3D)                        |
| 29–31 | `range_centroid`     | Max − min centroid per axis (3D)                      |
| 32–34 | `mean_sun`           | Mean sun direction unit vector (3D)                   |
| 35–37 | `cen_sun_slope`      | OLS slope of centroid on sun elevation (3D)           |
| 38–39 | `mean_motor`         | Mean motor encoder readings (2D)                      |
| 40–41 | `std_motor`          | Std of motor readings (2D)                            |

The OLS slope (`cen_sun_slope`) captures how the measured focal-spot position changes with
sun elevation — the key kinematic signature distinguishing heliostats with different errors.

All features are z-score normalised using training-set statistics (mean/std computed from
the training split; applied identically to validation and test).

---

## Coarse Checkpoint

Two separate checkpoints, selected automatically by `--dataset-type`:

| Dataset type | File                                                      |
|--------------|-----------------------------------------------------------|
| `synthetic`  | `coarse_learning_parameters/kinematic_parameters_synthetic.json` |
| `real`       | `coarse_learning_parameters/kinematic_parameters_real.json`      |

Each file was produced by running `WortbergKinematicReconstructor` on the full 63-heliostat
field scenario. The pipeline loads these into the scenario at startup; the residual model
predicts a correction on top of this frozen base.

---

## Dataset

### Real (PAINT calibration images)

Benchmark: `benchmark_split-balanced_train-100_validation-50_deflectometry`

| Split      | Samples / heliostat |
|------------|---------------------|
| train      | up to 100           |
| validation | up to 50            |
| test       | up to 50            |

Parser: `CachedPaintCalibrationDataParser` — wraps `PaintCalibrationDataParser` with an
in-memory cache (CPU RAM). On the first epoch all flux PNGs and calibration JSONs are read
from disk (NFS on DAIC); every subsequent epoch retrieves the cached tensors, eliminating
repeated network filesystem reads and reducing per-epoch time to match synthetic.

Three separate parser instances (train / val / test) to avoid cross-split cache collisions.

### Synthetic (pre-generated)

Generated by `full_field_200_samples/generate_dataset.py`. Same scenario as coarse checkpoint.

| Split      | Samples / heliostat |
|------------|---------------------|
| train      | 100                 |
| validation | 50                  |
| test       | 50                  |

Directory: `scenarios/full_field_200_samples_scenario/synthetic_data/{train|val|test}/`

Parser: `SyntheticDatasetParser` (three separate instances, one per split).

---

## Loss Functions

Configured via `LOSS_TYPE` in `config.py` (default: `focal_spot`).

| Key           | Class             | Description                                                  | Ray tracing |
|---------------|-------------------|--------------------------------------------------------------|-------------|
| `focal_spot`  | `FocalSpotLoss`   | Euclidean distance (m) between predicted and measured centroid| Yes         |
| `pixel`       | `PixelLoss`       | MSE on Gaussian-blurred (σ=1), peak-normalised flux bitmaps  | Yes         |
| `alignment`   | `AlignmentLoss`   | MSE on motor positions converted to joint-angle space        | No          |

---

## Training Loop

1. Load scenario from HDF5; load coarse checkpoint.
2. Capture `GroupParameterState` for each heliostat group (frozen base vectors).
3. Build train/val/test feature bundles; z-score normalise using training statistics.
4. Instantiate model via `build_residual_model(model_type)` (zero-weight init).
5. Optimise with AdamW + `ReduceLROnPlateau` on `val_mean_mrad`.
6. For each epoch:
   a. For each heliostat group: predict residual via model → add to base → apply to scenario.
   b. Parse data, run ray tracer (or kinematic solver for alignment loss).
   c. Accumulate task loss + L2 regularisation on residual magnitudes.
   d. Backprop; gradient clip (`max_norm=1.0`); AdamW step.
   e. Evaluate on validation split; step scheduler.
7. Early stopping (patience = max(5, max_epochs // 10)).
8. Evaluate best and last-epoch checkpoints on test set.

**Regularisation**: `loss = task_loss + residual_l2_weight × ||residual||²` (default 1e-4)

---

## Optimisation Configuration

| Parameter                   | Default value |
|-----------------------------|---------------|
| Optimiser                   | AdamW         |
| Learning rate               | 1e-3          |
| Weight decay                | 1e-5          |
| Max epochs                  | 200           |
| Gradient clip max norm      | 1.0           |
| LR scheduler patience       | 10            |
| LR scheduler factor         | 0.5           |
| LR scheduler min LR         | 1e-6          |
| Sample limit / heliostat    | 100           |
| Number of rays              | 10            |
| Ray-tracing batch size      | 32            |
| Surface pts / facet         | 25×25         |
| Bitmap resolution           | 256×256       |
| Residual L2 weight          | 1e-4          |

---

## How to Run

```bash
# Synthetic data, linear model (default)
python main.py --dataset-type synthetic

# Real PAINT data, linear model
python main.py --dataset-type real --daic

# Polynomial models
python main.py --dataset-type synthetic --model-type poly2
python main.py --dataset-type synthetic --model-type poly3
python main.py --dataset-type synthetic --model-type poly4

# Smoke test (3 epochs)
python main.py --smoke-test

# Sanity check before submitting to DAIC
python check_env.py --dataset-type synthetic --daic
```

### SLURM jobs (sbatch_files/)

| Script                                 | Dataset   | Model  | Time   |
|----------------------------------------|-----------|--------|--------|
| `run_full_pipeline_synthetic.sh`       | synthetic | linear | 1:30   |
| `run_full_pipeline_real.sh`            | real      | linear | 1:30   |
| `run_full_pipeline_synthetic_poly2.sh` | synthetic | poly2  | 1:30   |
| `run_full_pipeline_synthetic_poly3.sh` | synthetic | poly3  | 1:30   |
| `run_full_pipeline_synthetic_poly4.sh` | synthetic | poly4  | 1:30   |
| `run_full_pipeline_real_poly2.sh`      | real      | poly2  | 1:30   |
| `run_full_pipeline_real_poly3.sh`      | real      | poly3  | 1:30   |
| `run_full_pipeline_real_poly4.sh`      | real      | poly4  | 1:30   |

---

## Output Structure

```
full_training_pipeline_{model_type}_{timestamp}/
  training.log
  pipeline_details.md          — human-readable run summary

  json/
    config.json                — full PipelineConfig serialised
    norm_stats.json            — z-score normalisation statistics
    history.json               — per-epoch train/val metrics
    timing.json                — wall-clock times + peak GPU memory
    predicted_residuals.json   — per-heliostat predicted correction vectors
    validation_baseline_metrics.json
    validation_corrected_metrics.json
    test_baseline_metrics.json
    test_corrected_metrics.json

  models/
    linear_residual_model.pt            — best-validation checkpoint (PyTorch)
    linear_residual_model.json          — same, human-readable JSON
    linear_residual_model_last_epoch.pt
    linear_residual_model_last_epoch.json
    corrected_kinematic_parameters_best.json

  plots/
    loss_curve.png                      — train/val loss + LR schedule
    baseline_vs_corrected_metrics.png   — mean/median mrad before and after
    error_histogram.png                 — per-heliostat error distribution
    per_heliostat_improvement_scatter.png
    predicted_residual_boxplot.png      — distribution of predicted corrections
    response_curves.png                 — partial dependence: correction vs input feature
    feature_importance.png              — weight column norms + gradient×input sensitivity
    linear_weights_heatmap.png          — linear model only
```

---

## Relationship to Other Experiments

| Experiment                  | Role                                                         |
|-----------------------------|--------------------------------------------------------------|
| `full_field_200_samples/`   | Produces the coarse kinematic checkpoint and synthetic data  |
| `full_training_pipeline/`   | Learns a shared residual correction on top of that checkpoint|
| `five_heliostats_synth/`    | Smaller closed-loop ablation (5 heliostats, 6 train sizes)   |
