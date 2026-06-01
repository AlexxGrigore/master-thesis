# Full Training Pipeline

## Overview

The full training pipeline is the first reusable fine-error-learning baseline for thesis experiments.
It starts from a frozen coarse kinematic checkpoint and learns a shared linear residual model that predicts a bounded correction vector for each heliostat.

The key idea is:

1. Use the existing coarse kinematic parameters as the starting point.
2. Build one compact feature vector per heliostat from calibration metadata.
3. Predict a residual correction in the same Wortberg-style parameter space already used by kinematic reconstruction.
4. Apply the corrected parameters in ARTIST.
5. Optimize the residual model against the final focal-spot tracking error.

This gives a simple, controllable baseline before moving to larger models.

## Inputs

The pipeline uses four main input groups.

### Scenario

- An ARTIST scenario `.h5` file.
- This provides the heliostat field, target areas, optical setup, and the differentiable forward model used during training and evaluation.

### Coarse Checkpoint

- `coarse_learning_parameters/kinematic_parameters.json`
- This is the frozen starting point for all heliostats.
- The fine model does not replace these parameters; it predicts residuals on top of them.

### PAINT Benchmark Split

- A benchmark CSV describing which samples belong to `train`, `validation`, and `test`.
- Calibration-property JSON files.
- Flux-image files.

### Runtime Configuration

- Number of epochs.
- Learning rate and weight decay.
- Residual penalty weight.
- Number of rays.
- Bitmap resolution.
- Sample limit per heliostat.

These settings are defined through `PipelineConfig` in [config.py](/Users/alexandru/Master Thesis/master-thesis/src/full_training_pipeline/config.py).

## Feature Construction

The model does not consume raw images directly.
Instead, it builds one aggregated feature vector per heliostat from the calibration metadata in the selected split.

### Per-sample features

Each sample contributes:

- Sun direction in 3D.
- Axis 1 motor position.
- Axis 2 motor position.

The sun direction is computed from elevation and azimuth.

### Aggregated heliostat features

For each heliostat, the pipeline aggregates those per-sample values into:

- Mean of all sample features.
- Standard deviation of all sample features.
- Sample count.

That produces the current 11-dimensional feature vector:

- `mean_sun_x`
- `mean_sun_y`
- `mean_sun_z`
- `mean_axis_1_motor_position`
- `mean_axis_2_motor_position`
- `std_sun_x`
- `std_sun_y`
- `std_sun_z`
- `std_axis_1_motor_position`
- `std_axis_2_motor_position`
- `sample_count`

Feature construction lives in [features.py](/Users/alexandru/Master Thesis/master-thesis/src/full_training_pipeline/features.py) and [data.py](/Users/alexandru/Master Thesis/master-thesis/src/full_training_pipeline/data.py).

## Model

The first pipeline version uses one shared linear model for all heliostats.

### What “shared” means

- There is not one separate linear model per heliostat.
- All heliostats use the same learned weight matrix and bias.
- Different heliostats still get different outputs because they have different input features.

### Input and output dimensions

- Input dimension: 11 aggregated features.
- Output dimension: 20 residual parameters.

### Output parameter space

The model predicts residuals for the shared Wortberg parameter vector:

- 9 translation parameters.
- 4 rotation parameters.
- 2 actuator initial-angle parameters.
- 2 actuator offset parameters.
- 3 base-position parameters.

These names are defined in [model.py](/Users/alexandru/Master Thesis/master-thesis/src/full_training_pipeline/model.py).

### Output bounding

The linear output is passed through `tanh` and then scaled by fixed residual bounds.

This keeps predictions in a physically reasonable range and matches the structure of the underlying kinematic parameters.

## Training Logic

The training step is built around ARTIST as the differentiable simulator.

For each epoch:

1. Load the frozen coarse parameters into the scenario.
2. Build feature tensors per heliostat group.
3. Predict residual corrections with the shared linear model.
4. Add the residuals to the frozen coarse parameter vector.
5. Apply the corrected parameters to the scenario.
6. Ray trace the active heliostats in ARTIST.
7. Compute focal-spot loss on the training split.
8. Add a small residual L2 penalty.
9. Backpropagate through the residual model.
10. Evaluate the current model on the validation split.

The best checkpoint is selected by the lowest validation mean focal-spot error in mrad.

The training/evaluation orchestration lives mainly in [train.py](/Users/alexandru/Master Thesis/master-thesis/src/full_training_pipeline/train.py), [pipeline.py](/Users/alexandru/Master Thesis/master-thesis/src/full_training_pipeline/pipeline.py), and [evaluate.py](/Users/alexandru/Master Thesis/master-thesis/src/full_training_pipeline/evaluate.py).

## Data Splits

The pipeline uses three splits:

- `train`: optimize the shared residual model.
- `validation`: select the best checkpoint.
- `test`: report final held-out performance.

The best model and the last epoch are both evaluated and saved.

## Outputs

Each training run writes a structured output directory.

### Top-level files

- `training_summary.json`: compact run summary.
- `training.log`: raw training log.
- `pipeline_details.md`: run-specific copy of the pipeline description.

### `json/`

- `config.json`
- `feature_normalization.json`
- `history.json`
- `validation_baseline_metrics.json`
- `validation_corrected_metrics.json`
- `validation_corrected_metrics_last_epoch.json`
- `test_baseline_metrics.json`
- `test_corrected_metrics.json`
- `test_corrected_metrics_last_epoch.json`

The evaluation JSONs intentionally retain `all_errors_mrad` for downstream analysis.

### `models/`

- `linear_residual_model.pt`
- `linear_residual_model.json`
- `linear_residual_model_last_epoch.pt`
- `linear_residual_model_last_epoch.json`
- `corrected_kinematic_parameters_best.json`

The `.json` exports are human-readable and contain the learned weights, bias, normalization statistics, parameter names, and residual bounds.

### `plots/`

- `loss_curve.png`
- `baseline_vs_corrected_metrics.png`
- `error_histogram.png`
- `linear_weights_heatmap.png`
- `predicted_residual_boxplot.png`
- `per_heliostat_improvement.png`

## Plot Meaning

### Loss Curve

- Training loss over epochs.
- Validation loss over epochs.
- Test loss shown as a horizontal dotted line.

### Baseline vs Corrected Metrics

- Compares baseline, best-checkpoint, and last-epoch validation/test tracking errors.

### Error Histogram

- Compares the distribution of baseline and corrected test errors.

### Linear Weights Heatmap

- Shows how each input feature contributes to each predicted residual parameter.

### Predicted Residual Boxplot

- Shows the distribution of predicted residual values across heliostats for each parameter.

### Per-Heliostat Improvement Scatter

- Each point is one heliostat.
- The diagonal indicates no change.
- Points below the diagonal improved after correction.

## Package Structure

- [config.py](/Users/alexandru/Master Thesis/master-thesis/src/full_training_pipeline/config.py): runtime configuration and defaults.
- [features.py](/Users/alexandru/Master Thesis/master-thesis/src/full_training_pipeline/features.py): per-sample and aggregated feature construction.
- [data.py](/Users/alexandru/Master Thesis/master-thesis/src/full_training_pipeline/data.py): split loading, normalization, and grouped feature tensors.
- [model.py](/Users/alexandru/Master Thesis/master-thesis/src/full_training_pipeline/model.py): shared linear residual model and parameter definitions.
- [pipeline.py](/Users/alexandru/Master Thesis/master-thesis/src/full_training_pipeline/pipeline.py): parameter-vector composition and differentiable ARTIST loss path.
- [evaluate.py](/Users/alexandru/Master Thesis/master-thesis/src/full_training_pipeline/evaluate.py): applying a trained model and measuring tracking accuracy.
- [plotting.py](/Users/alexandru/Master Thesis/master-thesis/src/full_training_pipeline/plotting.py): all run-level plots.
- [train.py](/Users/alexandru/Master Thesis/master-thesis/src/full_training_pipeline/train.py): main entrypoint.
- [concise.md](/Users/alexandru/Master Thesis/master-thesis/src/full_training_pipeline/concise.md): short planning note.

## Current Limitations

- The model is intentionally small and linear.
- Inputs are summary features only; no raw-image encoder is used.
- The coarse checkpoint is frozen rather than jointly optimized.
- The current design prioritizes interpretability and control over maximum capacity.

These are acceptable constraints for the first baseline because the goal is to verify that a reusable fine-error-learning pipeline works end to end.

## Intended Next Steps

Potential next iterations include:

1. Try a larger residual model such as an MLP.
2. Expand feature engineering beyond the current summary statistics.
3. Compare frozen-coarse versus partially trainable coarse parameter strategies.
4. Add richer evaluation plots or ablation studies once the baseline is stable.