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

1. A **clean** scenario (`scenario.h5`, deflectometry surfaces, 63 heliostats) is loaded.
2. **Random perturbations** are applied to every heliostat's kinematic parameters to create a
   "perturbed" (simulated real-world) field. The perturbations are saved to `perturbations.json`.
3. The perturbed field is **ray-traced** for a set of sun positions to produce synthetic
   calibration images (flux bitmaps) and GT centroids. This is the synthetic dataset.
4. At training time, the KR loads the **clean scenario** again and **starts from zero deviations**.
   It optimises the kinematic parameters until the predicted flux matches the perturbed
   dataset — effectively learning the perturbation values.

**Why this direction matters**: an earlier version of the experiment applied perturbations and
then trained a KR to undo them against a clean reference. That is the wrong direction — it is
not the real inverse problem. Here the KR truly starts blind and must discover the perturbations
from scratch, exactly as it would for a real heliostat field.

---

## Files

| File | Role |
|------|------|
| `config.py` | All configurable parameters (paths, splits, perturbation bounds, loss, training) |
| `generate_dataset.py` | Pre-generates the synthetic dataset on disk |
| `main.py` | Entry point: builds mappings, filters data, runs training, calls reporting |
| `train.py` | Two-stage training loop, trail recorder, GT grid collection |
| `reporting.py` | All output plot functions |

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
5. Saves per-sample into:
   ```
   synthetic_data/{split}/{hid}/{idx:04d}/
       flux_image.png              — 256×256 normalised flux bitmap
       calibration_properties.json — sun direction, motor positions, GT centroid (ENU)
   ```
6. Saves `synthetic_data/perturbations.json` — the ground-truth perturbation for each heliostat.

**Note:** If perturbation bounds in `config.py` change, the dataset must be regenerated.

---

## Step 2 — Flux Filtering

Before training, each split's mapping is filtered to remove samples where the flux image is
effectively empty. This is critical for synthetic data: a fixed perturbation can cause a
heliostat to miss the target entirely for certain sun angles (e.g. low elevation, extreme
azimuth), producing black or near-black flux images that would produce meaningless centroid
estimates and corrupt the loss.

**Criterion:** a sample is removed if fewer than `MIN_ACTIVE_PIXEL_PCT = 0.25%` of pixels
have a normalised value > 0.01.

- Applied to **all three splits** (train, val, test) for both synthetic and real data.
- The filter reads the actual synthetic PNG files from disk (not the PAINT mapping paths).
- Output: `filter_stats.png` — a table showing how many samples per heliostat remain after
  filtering for each split (green = full, yellow = partial, red = zero).

**Architecture note:** `SyntheticDatasetParser` reads samples sequentially by index
(0000, 0001, …). Filtering reduces the per-heliostat count; the parser then reads that many
samples starting from index 0000. Specific bad samples in the middle of the index range are
not individually skipped — only the total count is reduced.

---

## Step 3 — Two-Stage Training

The training is split into two stages with different loss functions.

### Stage 1 — AlignmentLoss (motor-position MSE)

**Epochs:** `STAGE1_EPOCHS = 20`

**What it does:** optimises the kinematic parameters to minimise the MSE between the
*predicted* motor positions (computed from the scenario's kinematics and the sun direction)
and the *measured* motor positions stored in the calibration properties. This requires no
ray tracing and is therefore fast.

**Purpose:** pre-conditions the kinematic parameters into a sensible region before the
expensive ray-tracing stage begins. Without Stage 1, the parameters start completely wrong
and the FocalSpotLoss gradient signal can be very noisy or zero (if the flux spot is far
off-target).

**Loss:** `AlignmentLoss` — mean squared error over motor encoder readings.

### Stage 2 — FocalSpotLoss (or alternative)

**Epochs:** `STAGE2_EPOCHS = 100`

**What it does:** ray-traces the current scenario (using `TRAIN_RAYS = 10` rays per surface
point) to produce a predicted flux bitmap, computes the centroid of that bitmap, and minimises
the Euclidean distance between the predicted centroid and the GT centroid stored in the dataset.

**Purpose:** the main optimisation — directly minimises the focal-spot error, which is the
quantity we care about.

**Loss options** (set via `LOSS_TYPE` in `config.py` or `--loss-type` on the command line):

| Loss type | Description |
|-----------|-------------|
| `focal_spot` | Distance between predicted and GT centroid (ENU, metres). Default. |
| `pixel` | Pixel-wise MSE between predicted and measured flux bitmaps. |
| `contour` | Edge-based loss comparing contours of predicted and measured flux. |
| `alignment` | Motor-position MSE only (skips ray tracing — useful for ablations). |

### Optimiser

Both stages share the same base config:
- Optimiser: Adam, `lr = 1e-4`
- Scheduler: ReduceLROnPlateau (`factor=0.5`, `patience=10`, `cooldown=5`, `min_lr=1e-6`)
- Early stopping: patience = 400 epochs, delta = 1e-5
- Batch size: 8 (ray-tracing batch size, not heliostat batch size)

### What is optimised

Following Wortberg (2025), the KR optimises:
- `rotation_deviation_parameters` — 4 joint tilt angles per heliostat
- `actuators.optimizable_parameters[:, actuator_initial_angle]` — actuator initial angles `a_i`
- `actuators.non_optimizable_parameters[:, actuator_offset]` — actuator offsets `c_i`
- `translation_deviation_parameters` — 9 translation deviations (joints + concentrator)
- `_base_position_deviation` — 3D base position offset (E/N/U)

Frozen (not optimised): `actuator_initial_stroke_length` (`b_i`).

---

## Evaluation Checkpoints

Three exact evaluations are run on the **test split** (50 samples per heliostat, unfiltered):

| Checkpoint | Scenario state | Expected mrad |
|------------|---------------|---------------|
| Pre-training | Clean (zero deviations) | High — clean scenario mispredicts perturbed data |
| Post-Stage-1 | Alignment-trained | Intermediate — motor positions corrected |
| Post-training | Fully trained | Low — the result |

A fourth evaluation on the **val split** is run post-training for the summary table.

**mrad metric:**
```
FSE_mrad = ||pred_centroid[:3] - gt_centroid[:3]||₂  /  dist(heliostat, tower)  ×  1000
```

---

## Centroid Trail Capture

During Stage 2, a `_CentroidTrailRecorder` is attached to the reconstructor as an
`epoch_callback`. Every `CENTROID_TRAIL_STRIDE = 5` epochs it runs a lightweight forward pass
on up to `CENTROID_TRAIL_N_DISP = 25` training samples per heliostat and records the predicted
centroid position (ENU). After training, these trail positions are plotted on top of the GT
flux images to visualise convergence.

A Stage-1 trail recorder is also attached, so the unified mrad plot covers both stages.

---

## Output Files

All outputs are written to the run directory (e.g. `outputs/local_runs/full_63_synthetic_focal_spot_deflectometry_<timestamp>/`).

| File | Description |
|------|-------------|
| `config.json` | Full config snapshot for reproducibility |
| `filter_stats.png` | Per-heliostat sample counts after flux filtering (train/val/test) |
| `convergence_unified_mrad.png` | Stage 1 + Stage 2 on a single mrad y-axis |
| `convergence_stage1.png` | Stage 1 AlignmentLoss in rad² over epochs |
| `convergence_stage2.png` | Stage 2 loss in native units over epochs |
| `gt_grids/train.png` | GT measured flux grid — training split (one row per heliostat) |
| `gt_grids/val.png` | Same for validation |
| `gt_grids/test.png` | Same for test |
| `centroid_trails/{hid}_trail.png` | Stage-2 centroid trail overlaid on training samples |
| `per_heliostat_accuracy.png` | Table: pre/post mrad for every heliostat |
| `per_heliostat_accuracy_histogram.png` | Histogram of post-training mrad across heliostats |
| `field_accuracy_map.png` | ENU scatter plot of heliostats coloured by post-training FSE |
| `summary_table.png` | 2-row table: val + test mean/median mrad |
| `results.json` | All numerical results (pre/post mrad, per-heliostat, pixel loss, …) |
| `mrad_trajectory.json` | Per-epoch mean mrad from trail recorders (consumed by unified plot) |
| `convergence_history_stage1.json` | Full loss history for Stage 1 |
| `convergence_history_stage2.json` | Full loss history for Stage 2 |
| `kinematic_parameters.json` | Final trained kinematic parameters for all heliostats |
| `field_positions.json` | Heliostat ENU positions + tower position |
| `timing.json` | Wall-clock times and GPU/RAM peak usage |
| `run.log` | Full training log |
| `contour_components_{train,val}.png` | (contour loss only) Per-component loss curves |
| `contour_overlay_{best,worst}_{hid}.png` | (contour loss only) GT vs predicted contour |
| `pipeline_steps_{best,worst}_{hid}.png` | (contour loss only) Step-by-step contour pipeline |

---

## Running the Experiment

**Local (smoke test, 5 epochs total):**
```bash
cd src/full_63_heli_kin_reconstruct
python generate_dataset.py --force   # only needed if data is missing or bounds changed
python main.py --smoke-test
```

**Local (full run):**
```bash
python main.py
python main.py --loss-type contour
python main.py --no-deflectometry    # use ideal flat-surface scenario
```

**On DAIC (cluster):**
```bash
python generate_dataset.py --daic --force
python main.py --daic --dataset-type synthetic --loss-type focal_spot \
               --stage1-epochs 20 --stage2-epochs 100
```

**Real PAINT data:**
```bash
python main.py --daic --dataset-type real --loss-type focal_spot
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
| `SURFACE_POINTS_PER_FACET` | 25 | Surface resolution: 25×25 = 625 pts/facet (controls GPU memory) |
| `PERTURBATION_SEED` | 42 | RNG seed for reproducible perturbations |
| `MIN_ACTIVE_PIXEL_PCT` | 0.25 | Flux filter: min % of pixels > 0.01 to keep a sample |
| `LOSS_TYPE` | `"focal_spot"` | Stage-2 loss function |
| `STAGE1_EPOCHS` | 20 | AlignmentLoss pre-training epochs |
| `STAGE2_EPOCHS` | 100 | Main training epochs |
| `CENTROID_TRAIL_STRIDE` | 5 | Capture centroid trail every N Stage-2 epochs |
| `CENTROID_TRAIL_N_DISP` | 25 | Max training samples shown in trail grid per heliostat |

---

## Scenario Details

- **File:** `scenarios/full_63_heli_kin_reconstruct/scenario.h5` (deflectometry)
- **Alternative:** `scenario_ideal.h5` (flat surfaces, use with `--no-deflectometry`)
- **Heliostats:** 63, organised in one heliostat group
- **Facets per heliostat:** 4, each with B-spline deflectometry surface
- **Surface points:** 25×25 = 625 per facet = 2 500 per heliostat (at default resolution)
- **Target:** single planar receiver area on the solar tower

---

## Dataset vs. Scenario Relationship

```
generate_dataset.py
  └─ loads clean scenario.h5
  └─ applies random perturbations → perturbed scenario (in memory only)
  └─ ray-traces perturbed scenario × PAINT sun positions
  └─ writes synthetic_data/{split}/{hid}/{idx}/...
  └─ writes synthetic_data/perturbations.json

main.py / train.py
  └─ loads clean scenario.h5  ← KR starts from here (zero deviations)
  └─ reads synthetic_data/    ← GT data from the PERTURBED field
  └─ optimises scenario parameters to match GT
  └─ result: scenario.parameters ≈ perturbations.json values
```

The dataset never changes between runs (unless regenerated). Only the scenario's
kinematic parameters change during training.
