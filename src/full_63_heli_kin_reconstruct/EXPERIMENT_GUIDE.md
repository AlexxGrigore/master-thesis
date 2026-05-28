# Experiment Guide: `full_63_heli_kin_reconstruct`

## Purpose

This experiment performs **kinematic reconstruction** (KR) on a field of 63 heliostats.
Given calibration images (flux bitmaps on a receiver target), the goal is to recover the
kinematic error parameters of each heliostat — the small deviations in joint angles,
actuator offsets, and concentrator position that cause a real heliostat to aim differently
from an ideal one.

This is the core inverse problem of heliostat calibration: you observe where the light lands,
and you infer what is geometrically wrong with the mirror.

---

## The Inverse Problem (Corrected Pipeline)

The experiment uses a **corrected closed-loop pipeline**:

1. A **clean** scenario (`scenario.h5`, deflectometry surfaces) is loaded — one file per heliostat
   at `scenarios/one_heliostat_scenarios/{hid}/scenario.h5`.
2. **Random perturbations** are applied to every heliostat's kinematic parameters to create a
   "perturbed" (simulated real-world) field. The perturbations are saved to `perturbations.json`.
3. The perturbed field is **ray-traced** for a set of sun positions to produce synthetic
   calibration images (flux bitmaps) and GT centroids. This is the synthetic dataset.
4. At training time, the KR loads the **clean scenario** again and **starts from zero deviations**.
   It optimises the kinematic parameters until the predicted flux matches the perturbed
   dataset — effectively learning the perturbation values.

**Why this direction matters**: an earlier version applied perturbations and then trained a KR to
undo them against a clean reference. That is the wrong direction — it is not the real inverse
problem. Here the KR truly starts blind and must discover the perturbations from scratch, exactly
as it would for a real heliostat field.

---

## Architecture: Per-Heliostat Training

The key architectural decision is that **each heliostat is trained completely independently**,
using its own scenario file. This differs from an earlier joint-training approach and has several
advantages:

- **No uniform-count constraint**: ARTIST's distributed sampler requires equal sample counts per
  heliostat when training jointly. Training one at a time removes this constraint entirely —
  each heliostat uses exactly the samples it has after flux filtering, with no padding needed.
- **Clean quality gating**: heliostats with too few valid samples can be skipped or have Stage 2
  omitted without affecting other heliostats.
- **Simpler debugging**: a crash or bad run for one heliostat does not abort the rest.

The 63 individual scenario files live at `scenarios/one_heliostat_scenarios/{hid}/scenario.h5`.

---

## Files

| File | Role |
|------|------|
| `config.py` | All configurable parameters (paths, splits, perturbation bounds, loss, training) |
| `generate_dataset.py` | Pre-generates the synthetic dataset on disk |
| `main.py` | Entry point: builds mappings, filters data, runs per-heliostat training loop, aggregates |
| `train.py` | Two-stage training loop, trail recorder, GT grid collection |
| `reporting.py` | Per-heliostat output plot functions |
| `aggregate.py` | Aggregates per-heliostat results into combined summary plots and JSON |

---

## Step 1 — Dataset Generation (`generate_dataset.py`)

Run once before training:

```bash
python generate_dataset.py          # first time
python generate_dataset.py --force  # regenerate (e.g. after changing perturbation bounds)
python generate_dataset.py --daic --force  # on the cluster
```

**What it does:**

1. Loads the clean scenario.
2. Samples one random perturbation vector per heliostat (seed = `PERTURBATION_SEED = 42`).
   Perturbation bounds follow Wortberg (2025) Table 5.3:

   | Parameter | Bound |
   |-----------|-------|
   | Joint rotation tilts (4 params) | ±0.005 rad |
   | Actuator initial angle `a_i` (2 params) | ±0.005 rad |
   | Actuator offset `c_i` (2 params) | ±0.005 m |
   | Translation deviations (9 params: joints + concentrator) | ±0.05 m |
   | Base position deviation (3 params: E/N/U) | ±0.05 m |

3. Applies the perturbations to the scenario.
4. For each heliostat, ray-traces `SYNTH_GEN_RAYS = 100` rays per sun position from the
   PAINT benchmark split (train: 100, val: 50, test: 50 sun positions per heliostat).
5. Applies a Gaussian blur (σ = `BLUR_SIGMA = 1.0`) to each flux image before saving, to
   match the blur applied to predicted images during training (see Flux Preprocessing below).
6. Saves per-sample into:
   ```
   synthetic_data/{split}/{hid}/{idx:04d}/
       flux_image.png              — 256×256 normalised, blurred flux bitmap
       calibration_properties.json — sun direction, motor positions, GT centroid (ENU)
   ```
7. Saves `synthetic_data/perturbations.json` — the ground-truth perturbation for each heliostat.

**Note:** If perturbation bounds or `BLUR_SIGMA` change, the dataset must be regenerated with `--force`.

---

## Step 2 — Flux Filtering

Before training, each split's mapping is filtered to remove samples where the flux image is
effectively empty. This is critical for synthetic data: a fixed perturbation can cause a
heliostat to miss the target entirely for certain sun angles, producing black flux images that
would yield meaningless centroid estimates and corrupt the loss.

**Criterion:** a sample is removed if fewer than `MIN_ACTIVE_PIXEL_PCT = 1.0%` of pixels
have a normalised value > 0.01.

- Applied to **all three splits** (train, val, test).
- The filter reads the actual flux PNG files from disk.

**Quality gates applied after filtering:**

| Gate | Threshold | Effect |
|------|-----------|--------|
| Val hard skip | `MIN_VAL_SAMPLES = 2` | Heliostat skipped entirely — no Stage 1, no Stage 2 |
| Test hard skip | `MIN_TEST_SAMPLES = 2` | Same |
| Stage-2 soft skip | `MIN_FOCAL_SPOT_TRAIN_SAMPLES = 10` | Stage 1 still runs; Stage 2 skipped |

**Note on `MIN_ACTIVE_PIXEL_PCT`:** the threshold is intentionally set to 1.0% to exclude borderline cases where the perturbed heliostat barely clips the receiver edge for a given sun angle. Such images carry an unreliable GT centroid estimate and would produce noisy training signal.

Heliostats that are hard-skipped are logged as warnings and excluded from `results_combined.json`.
Heliostats that hit the soft-skip still produce Stage 1 results and are included in aggregated
Stage-1 statistics; they are excluded from post-Stage-2 aggregated statistics.

---

## Step 3 — Two-Stage Training

Each heliostat is trained independently. The full loop is:

```
for hid in sorted(heliostat_ids):
    load clean scenario for hid
    apply flux filtering
    [quality gate checks]
    Stage 1 — AlignmentLoss (all heliostats)
    [Stage-2 soft-skip check]
    Stage 2 — FocalSpotLoss (or alternative)
    evaluate on val and test sets
    save per-heliostat outputs to run_dir/{hid}/

aggregate_results(all hel_results, run_dir)
```

### Flux Preprocessing

Both the generated GT images and the predicted images during training/eval are processed
identically before centroid computation and loss evaluation:

1. **Gaussian blur** with σ = `BLUR_SIGMA = 1.0` (applied to GT at generation time, to predicted at
   inference time).
2. **Peak normalisation**: divide by the maximum pixel value so the image is in [0, 1]. This makes
   the loss scale-invariant with respect to sun intensity and distance attenuation.

### Stage 1 — AlignmentLoss (motor-position MSE)

**Epochs:** `STAGE1_EPOCHS = 20`

Optimises the kinematic parameters to minimise the MSE between *predicted* motor positions
(computed from the scenario's kinematics and the sun direction) and the *measured* motor positions
stored in the calibration properties. **No ray tracing** — fast.

**Purpose:** pre-conditions the kinematic parameters into a sensible region before the expensive
ray-tracing stage. Without Stage 1, the flux spot may be so far off-target that the FocalSpotLoss
gradient is zero or meaningless.

### Stage 2 — FocalSpotLoss (or alternative)

**Epochs:** `STAGE2_EPOCHS = 100`

Ray-traces the current scenario (`TRAIN_RAYS = 10` rays per surface point), applies blur +
peak normalisation to the predicted flux, computes its centroid, and minimises the Euclidean
distance between the predicted centroid and the GT centroid from the dataset.

**Loss options** (set via `LOSS_TYPE` in `config.py` or `--loss-type` on the command line):

| Loss type | Description |
|-----------|-------------|
| `focal_spot` | Centroid distance (ENU, metres) after blur + normalisation. **Default.** |
| `pixel` | Pixel-wise MSE between predicted and measured flux bitmaps. |
| `contour` | Edge-based loss comparing contours of predicted and measured flux. |
| `alignment` | Motor-position MSE only (no ray tracing — useful for ablations). |

### Optimiser

Both stages share the same base config:
- Optimiser: Adam, `lr = 1e-4`
- Scheduler: ReduceLROnPlateau on validation loss (`factor=0.5`, `patience=10`, `cooldown=5`, `min_lr=1e-6`)
- Early stopping: patience = 400 epochs, delta = 1e-5
- Batch size: 8 (ray-tracing batch, not heliostat batch)

### What is optimised

Following Wortberg (2025), the KR optimises per heliostat:
- `rotation_deviation_parameters` — 4 joint tilt angles
- `actuators.optimizable_parameters[:, actuator_initial_angle]` — actuator initial angles `a_i`
- `actuators.non_optimizable_parameters[:, actuator_offset]` — actuator offsets `c_i`
- `translation_deviation_parameters` — 9 translation deviations (joints + concentrator)
- `_base_position_deviation` — 3D base position offset (E/N/U)

Frozen: `actuator_initial_stroke_length` (`b_i`).

---

## Evaluation Checkpoints

Three exact evaluations are run on the **test split** for each heliostat:

| Checkpoint | Scenario state | Expected mrad |
|------------|----------------|---------------|
| Pre-training | Clean (zero deviations) | High — clean scenario mispredicts perturbed data |
| Post-Stage-1 | Alignment-trained | Intermediate — motor positions corrected |
| Post-training | Stage-2 best-val checkpoint | Low — the result |

A fourth evaluation on the **val split** is run post-training for the summary table.

**mrad metric:**
```
FSE_mrad = ||pred_centroid[:3] - gt_centroid[:3]||₂  /  dist(heliostat, tower)  ×  1000
```

**Evaluation skip — off-receiver samples:**
During evaluation, if the predicted flux for a test sample has a peak value below `1e-6` (after
Gaussian blur), the heliostat is pointing entirely off the receiver for that sun angle. Such
samples produce a degenerate centroid at pixel (0, 0) — the top-left corner of the bitmap — which
maps to a far-away ENU point and would inflate the reported mrad by 50–100×. These samples are
marked as `NaN` in the focal-spot-error tensor and excluded from mean/median mrad statistics by
`_safe_mean`. The number of excluded samples is reported in `results.json` under
`"num_nan_samples"`.

This is distinct from the GT flux filter (Step 2): that filter removes GT images with too few
active pixels *before* training; this filter removes *predicted* outputs that miss the receiver
*during evaluation*. Both work together to keep the mrad metric meaningful.

---

## Centroid Trail Capture

During Stage 2, a `_CentroidTrailRecorder` is attached as an epoch callback. Every
`CENTROID_TRAIL_STRIDE = 1` epoch it runs a lightweight forward pass on up to
`CENTROID_TRAIL_N_DISP = 25` training samples and records the predicted centroid (ENU). After
training, these trail positions are overlaid on GT flux images to visualise convergence.

A Stage-1 trail recorder is also attached, so the unified mrad plot covers both stages.

---

## Output Structure

Outputs are split between **per-heliostat** subdirectories and the **run root**.

### Run root (`run_dir/`)

| File | Description |
|------|-------------|
| `config.json` | Full config snapshot for reproducibility |
| `perturbations.json` | Copy of the GT perturbation vectors (synthetic only) |
| `run.log` | Full training log |
| `results_combined.json` | Aggregated pre/post-S1/post-S2 mrad (mean + median across heliostats) |
| `results_histogram.png` | Distribution of mrad across heliostats — pre / post-S1 / post-S2 |
| `accuracy_table_all.png` | Per-heliostat accuracy table — all heliostats (including S2-skipped) |
| `accuracy_table_stage2.png` | Per-heliostat accuracy table — Stage-2 heliostats only, with improvement % |
| `aggregated_unified_mrad.png` | Averaged mrad trajectory across heliostats (both stages, train + val) |
| `aggregated_stage1_loss.png` | Averaged AlignmentLoss (original units) across all heliostats (train + val) |
| `aggregated_stage2_loss.png` | Averaged FocalSpotLoss (original units) across Stage-2 heliostats (train + val) |

### Per-heliostat (`run_dir/{hid}/`)

| File | Description |
|------|-------------|
| `results.json` | All numerical results for this heliostat (pre/post mrad, pixel loss, param recovery) |
| `convergence_history_stage1.json` | Full AlignmentLoss history per epoch |
| `convergence_history_stage2.json` | Full Stage-2 loss history per epoch |
| `mrad_trajectory.json` | Per-epoch mrad from trail recorders (consumed by unified mrad plot) |
| `convergence_stage1.png` | Stage-1 AlignmentLoss curve (train + val) |
| `convergence_stage2.png` | Stage-2 loss curve in native units (train + val) |
| `convergence_unified_mrad.png` | Both stages on a single mrad y-axis (train + val) |
| `per_heliostat_accuracy.png` | Table: pre/post-S1/post-S2 mrad for this heliostat |
| `per_heliostat_accuracy_histogram.png` | Histogram of per-sample mrad (test set) |
| `field_accuracy_map.png` | ENU scatter of heliostats coloured by post-training FSE |
| `summary_table.png` | 2-row table: val + test mean/median mrad |
| `summary.json` | Summary metrics as JSON |
| `kinematic_parameters.json` | Final trained kinematic parameter values |
| `field_positions.json` | Heliostat ENU positions + tower position |
| `timing.json` | Wall-clock times |
| `gt_grids/train.png` | GT measured flux grid — training split |
| `gt_grids/val.png` | Same for validation |
| `gt_grids/test.png` | Same for test |
| `centroid_trails/{hid}_trail.png` | Stage-2 centroid trail overlaid on training samples |
| `contour_components_{train,val}.png` | (contour loss only) Per-component loss curves |
| `contour_overlay_{best,worst}_{hid}.png` | (contour loss only) GT vs predicted contour |

---

## Running the Experiment

**Smoke test (3 heliostats, minimal epochs — end-to-end check):**
```bash
cd src/full_63_heli_kin_reconstruct
python generate_dataset.py --force   # only if data is missing or bounds changed
python main.py --smoke-test
```

**Full run (local):**
```bash
python main.py
python main.py --loss-type contour
python main.py --no-deflectometry    # use ideal flat-surface scenario
```

**Full run (DAIC cluster):**
```bash
python generate_dataset.py --daic --force
python main.py --daic --dataset-type synthetic --loss-type focal_spot \
               --stage1-epochs 20 --stage2-epochs 100
```

**Real PAINT data:**
```bash
python main.py --daic --dataset-type real --loss-type focal_spot
```

**Regenerate aggregated plots from an existing run directory (without retraining):**
```bash
python aggregate.py outputs/local_runs/full_63_synthetic_focal_spot_deflectometry_<timestamp>
```

---

## Key Configuration Parameters (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_SAMPLES` | 100 | Sun positions per heliostat in the training split |
| `VAL_SAMPLES` | 50 | Sun positions per heliostat in the validation split |
| `TEST_SAMPLES` | 50 | Sun positions per heliostat in the test split |
| `SYNTH_GEN_RAYS` | 100 | Rays used during dataset generation (higher = cleaner GT) |
| `TRAIN_RAYS` | 10 | Rays used during Stage-2 training forward pass |
| `TRAIN_SURFACE_POINTS` | 25 | Surface resolution: 25×25 = 625 pts/facet (controls GPU memory) |
| `PERTURBATION_SEED` | 42 | RNG seed for reproducible perturbations |
| `BLUR_SIGMA` | 1.0 | Gaussian blur σ applied to both GT (at generation) and predicted flux (at inference) |
| `MIN_ACTIVE_PIXEL_PCT` | 1.0 | Flux filter: min % of pixels > 0.01 to keep a sample |
| `MIN_VAL_SAMPLES` | 2 | Min valid val samples after filtering; fewer → hard skip |
| `MIN_TEST_SAMPLES` | 2 | Min valid test samples after filtering; fewer → hard skip |
| `MIN_FOCAL_SPOT_TRAIN_SAMPLES` | 10 | Min valid train samples; fewer → Stage-2 soft skip |
| `LOSS_TYPE` | `"focal_spot"` | Stage-2 loss function |
| `STAGE1_EPOCHS` | 20 | AlignmentLoss pre-training epochs |
| `STAGE2_EPOCHS` | 100 | Main training epochs |
| `CENTROID_TRAIL_STRIDE` | 1 | Capture centroid trail every N Stage-2 epochs |
| `CENTROID_TRAIL_N_DISP` | 25 | Max training samples shown in trail grid per heliostat |

---

## Scenario Details

- **Per-heliostat scenarios:** `scenarios/one_heliostat_scenarios/{hid}/scenario.h5` (63 files)
- **Full-field scenario:** `scenarios/full_63_heli_kin_reconstruct/scenario.h5` (deflectometry)
- **Alternative full-field:** `scenario_ideal.h5` (flat surfaces, use with `--no-deflectometry`)
- **Heliostats:** 63, each scenario contains exactly 1 heliostat group
- **Facets per heliostat:** 4, each with B-spline deflectometry surface
- **Surface points:** 25×25 = 625 per facet = 2 500 per heliostat (at default resolution)
- **Target:** single planar receiver area on the solar tower

---

## Dataset vs. Scenario Relationship

```
generate_dataset.py
  └─ loads full clean scenario.h5
  └─ applies random perturbations → perturbed scenario (in memory only)
  └─ ray-traces perturbed scenario × PAINT sun positions
  └─ applies Gaussian blur (σ=BLUR_SIGMA) to each flux image
  └─ writes synthetic_data/{split}/{hid}/{idx}/...
  └─ writes synthetic_data/perturbations.json

main.py / train.py  (per heliostat)
  └─ loads clean one_heliostat_scenarios/{hid}/scenario.h5  ← KR starts from zero deviations
  └─ reads synthetic_data/    ← GT data from the PERTURBED + blurred field
  └─ optimises scenario parameters to match GT
  └─ result: scenario.parameters ≈ perturbations.json values
```

The dataset never changes between runs (unless regenerated). Only the scenario's
kinematic parameters change during training.
