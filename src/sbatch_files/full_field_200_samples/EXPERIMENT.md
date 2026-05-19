# Full-Field 200-Samples Experiment

## Goal

Scale the synthetic perturbation experiment from 5 heliostats to the full field of
**63 heliostats** — those in the PAINT dataset that have both deflectometry data and
at least 200 calibration measurements. The primary motivation is to test whether the
equifinality finding from the 5-heliostat experiment persists at scale, and whether
more training data (100 samples vs 5–50) reduces compensating-parameter solutions.

A secondary goal is to use a **pixel-wise loss** (blurred MSE) instead of the focal-spot
loss used in the deflectometry-only experiments, since the pixel signal carries more
information than a single centroid coordinate.

---

## Status

**Ready to run.** All infrastructure is complete:

| Component              | File                          | Status  |
|------------------------|-------------------------------|---------|
| Benchmark download     | `src/download_paint_benchmark_200.py` | ✓ done |
| Scenario creation      | `src/full_field_200_samples/create_scenario.py` | ✓ done |
| Synthetic data gen     | `src/full_field_200_samples/generate_dataset.py` | ✓ done |
| Loss function          | `src/artist_extensions/loss_functions_ext.py` | ✓ done |
| Experiment entry point | `src/full_field_200_samples/main.py` | ✓ done |
| Training / evaluation  | `src/full_field_200_samples/train.py` | ✓ done |
| Hyperparameter config  | `src/full_field_200_samples/config.py` | ✓ done |
| Reporting              | reuses `five_heliostats_synth/reporting.py` | ✓ done |

Run with:
```
cd src/full_field_200_samples
python main.py
```

---

## Heliostats

63 heliostats selected by the intersection of:
- `≥ 200` calibration measurements in the PAINT metadata
- Filled deflectometry h5 file present locally under `datasets/paint/heliostats/`

Benchmark CSV:
```
datasets/paint/splits/benchmark_split-balanced_train-100_validation-50_deflectometry.csv
```

Scenario file (built from these 63 heliostats):
```
scenarios/full_field_200_samples_scenario/scenario.h5
```

---

## Data

### Real PAINT Benchmark

Downloaded with `src/download_paint_benchmark_200.py`.

| Split      | Samples / heliostat | Total items |
|------------|---------------------|-------------|
| train      | 100                 | 6 300       |
| validation | 50                  | 3 150       |
| test       | 50                  | 3 150       |

Note: 82 flux images (~0.5 %) were missing after the initial download (network
gaps). Re-running the download script fills them in. After re-download all splits
are complete.

### Synthetic Dataset

Generated with `src/full_field_200_samples/generate_dataset.py`. Saved to:

```
scenarios/full_field_200_samples_scenario/synthetic_data/
  {train|val|test}/
    {heliostat_id}/
      {idx:04d}/
        calibration_properties.json   ← incident_ray_direction, focal_spot_enu (ENU 4D),
                                         motor_position, target_area_index
        flux_image.png                ← uint8 grayscale, peak-normalised to [0, 255]
```

Real PAINT sun positions drive the ray tracer; focal-spot centroids and flux bitmaps
are computed from the **clean (unperturbed) scenario**. No real flux images are used
as training targets.

| Split | Samples / heliostat | Rays | Notes |
|-------|---------------------|------|-------|
| train | 100                 | 100  | OOM-safe: chunked (10 hel/chunk) + batch_size=8 |
| val   | 50                  | 100  | same |
| test  | 50                  | 100  | same |

Surface points per facet: **25 × 25** (625 pts/facet, 2 500 pts/heliostat).

**Status:** generated and verified. All 63 heliostats present in all three splits.
Actual sample counts are 97–100 / 47–50 / 48–50 due to the `_equalize_mapping` chunk
minimum — negligible effect on training (< 3 % shortfall). Re-generating with
`--force` after completing the flux-image re-download would give exact 100/50/50.

---

## Loss Function

Training loss: **BlurredPixelLoss** (`src/artist_extensions/loss_functions_ext.py`)

Pipeline per sample:
1. Gaussian blur σ=1 applied to both predicted and measured flux bitmaps.
2. Each image peak-normalised to [0, 1] independently.
3. Pixel-wise MSE between the two normalised blurred images.

This matches the evaluation metric in `utils/evaluation.py` (which uses L1; training
uses MSE for smoother gradients). Scale-invariant: handles the unit mismatch between
ray-tracer physical-intensity output and PNG-loaded [0, 1] measured images.

---

## Evaluation Protocol

Same three-stage protocol as the 5-heliostat experiment:

| Stage            | Scenario params | Expected result                  |
|------------------|-----------------|----------------------------------|
| Pre-perturbation | Clean           | ~0 mrad (sanity check)           |
| Post-perturbation| Perturbed       | High mrad                        |
| Post-training    | Trained         | Low mrad (recovery)              |

Two metrics reported: **focal spot error (mrad)** and **pixel-wise L1** (blurred,
peak-normalised — to match loss and evaluation consistency).

---

## Perturbations

Same parameter groups and ranges as the 5-heliostat experiment:

| Parameter group                 | Range | Unit | Optimized? |
|---------------------------------|-------|------|------------|
| Rotation (4 joint tilts)        | ±5    | mrad | Yes        |
| Actuator initial angle aᵢ (×2)  | ±5    | mrad | Yes        |
| Actuator stroke length bᵢ (×2)  | ±5    | mm   | **No (frozen)** |
| Actuator offset cᵢ (×2)         | ±5    | mm   | Yes        |
| Translation deviation (9 joints)| ±50   | mm   | Yes        |
| Base position (E, N, U)         | ±50   | mm   | Yes        |

Perturbation seed: 42. Applied per-heliostat independently.

---

## Hyperparameters

| Parameter               | Value  |
|-------------------------|--------|
| Learning rate (initial) | 1e-4   |
| Scheduler               | ReduceLROnPlateau (factor=0.5, patience=10) |
| Min LR                  | 1e-6   |
| Max epochs              | 300    |
| Batch size              | 8      |
| Early stopping patience | 400 (disabled — always runs full 300 epochs) |
| Train rays              | 10     |
| Surface pts/facet       | 25 × 25 |

---

## Output Structure

Each run produces a timestamped directory under `outputs/local_runs/` (local) or `outputs/` (DAIC):

```
full_field_200_{timestamp}/
  config.json                    — full configuration snapshot (all cfg.* values + CLI flags)
  perturbations.json             — true kinematic perturbations per heliostat (synthetic only)
  run.log                        — full execution log

  convergence_history.json       — train/val loss per epoch
  kinematic_history.json         — per-heliostat kinematic param values at each log step
  kinematic_parameters.json      — final trained kinematic parameters (loadable by pipeline)
  kinematic_stages.json          — kinematic snapshots at 3 stages: clean / perturbed / trained
  results.json                   — all 3 mrad + pixel-loss values per stage

  convergence.png                — loss curve + 3 horizontal mrad reference lines
  recovery_rotation.png          — perturbation vs residual: rotation (synthetic only)
  recovery_actuator_angle.png
  recovery_actuator_stroke.png   — frozen b_i: always at |perturbation| (synthetic only)
  recovery_actuator_offset.png
  recovery_translation.png
  recovery_base_position.png
  summary.txt                    — human-readable run summary

  flux_comparisons/
    flux_comparison_{hid}.png    — measured vs predicted flux + absolute difference per heliostat
```

---

## Connection to 5-Heliostat Findings

The 5-heliostat experiment (`src/five_heliostats_synth`) showed that with only 5 samples
the optimizer finds a **degenerate solution**: near-zero focal-spot error but large
residuals in rotation joints 3/4 (~5 mrad) and base position clamped at the ±50 mm
bound. This is **equifinality** — the focal-spot observable is insufficient to uniquely
identify all 20 kinematic parameters per heliostat.

Hypothesis: more training samples (100 here vs 5–50 there) constrain more viewing
angles, reducing the degeneracy. This experiment tests that hypothesis at full scale.

---

## Dependencies

ARTIST must be installed from the local source (not the pip package) because several
symbols used here — `bitmap_coordinates_to_target_coordinates`, `config_dictionary.batch_size`,
`config_dictionary.early_stopping_window`, `config_dictionary.lr_min` — are not present
in the published pip release:

```
pip install -e "/path/to/ARTIST/"
```
