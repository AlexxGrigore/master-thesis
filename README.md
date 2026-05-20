# master-thesis

Kinematic reconstruction of heliostat fields using [ARTIST](https://github.com/ARTIST-Association/ARTIST)
and real PAINT calibration data.

**Thesis topic:** Learning the kinematic parameters of 63 heliostats from PAINT flux images, using
a corrected closed-loop pipeline where perturbations are injected before data generation (the real
inverse problem).

---

## Quick navigation

| You want to… | Go to |
|---|---|
| Run the primary KR experiment | [`src/full_63_heli_kin_reconstruct/`](#experiment-3--full_63_heli_kin_reconstruct--primary) |
| Sweep training data sizes per heliostat | [`src/one_heliostat_train_sizes/`](#experiment-1--one_heliostat_train_sizes) |
| Run the coarse-to-fine residual pipeline | [`src/full_training_pipeline/`](#experiment-2--full_training_pipeline) |
| Understand ARTIST extensions | [`src/artist_extensions/`](#artist-extensions) |
| Submit to DAIC cluster | [`src/sbatch_files/`](#daic--sbatch-files) |

---

## Repository layout

```
master-thesis/
├── datasets/paint/              # PAINT benchmark data (not committed)
├── scenarios/                   # Pre-built scenario HDF5 files (not committed)
├── outputs/                     # Training run outputs (not committed)
└── src/
    ├── artist_extensions/       # Custom reconstructors, loss functions, parsers
    ├── utils/                   # Shared evaluation, checkpointing, reporting helpers
    ├── sbatch_files/            # SLURM job scripts for DAIC
    ├── one_heliostat_train_sizes/       # Experiment 1
    ├── full_training_pipeline/          # Experiment 2
    └── full_63_heli_kin_reconstruct/    # Experiment 3  ← primary
```

---

## Setup

**Conda environment:** `thesisenv` (Python 3.10, PyTorch CPU on macOS; CUDA on DAIC).

```bash
# Install ARTIST as editable (already done; re-run if ARTIST is re-cloned)
pip install -e /path/to/ARTIST

# Also requires: paint, h5py, tqdm, matplotlib, Pillow, colorlog, psutil
```

**PAINT benchmark data** (not committed):
- Local: `datasets/paint/`
- DAIC: `/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint/`
- Benchmark used: `benchmark_split-balanced_train-100_validation-50_deflectometry`

**Shared 63-heliostat scenario** (required by Experiments 1 and 3):
```
scenarios/full_field_200_samples_scenario/scenario.h5
```

---

## Corrected pipeline (key methodological point)

All three experiments use a **corrected closed-loop pipeline**:

1. Sample kinematic perturbations from known ranges (seed 42).
2. Apply perturbations to the scenario → generate synthetic flux images from that perturbed scenario.
3. The KR starts from a **clean (ideal) scenario** and must discover the perturbations from the data.

This is the real inverse problem. Previous work generated data from the clean scenario, applied
perturbations, then trained the KR to reverse them — a trivial task that doesn't reflect reality.

---

## Experiment 1 — `one_heliostat_train_sizes`

Sweeps training data sizes `[1, 5, 10, 20, 25, 50, 75, 100]` for 5 heliostats spread by
distance from tower: AC36 (34 m), AG33 (54 m), AO34 (90 m), AW36 (139 m), BE35 (210 m).
Each size runs a full two-stage train + eval cycle. Goal: how many images are needed to
recover kinematics to a target accuracy, and does distance matter?

**Data:** reuses synthetic data from `full_63_heli_kin_reconstruct/generate_dataset.py` — no
separate generation step.

**One-time setup** (per-heliostat scenario files, run once):
```bash
cd src && python one_heliostat_train_sizes/create_scenarios.py
```
Writes `scenarios/one_heliostat_scenarios/<ID>/scenario.h5` for each heliostat.

**Key config** (`src/one_heliostat_train_sizes/config.py`):
```python
HELIOSTAT_ID  = None              # None → auto-select first available; or e.g. "AC36"
TRAIN_SIZES   = [1, 5, 10, 20, 25, 50, 75, 100]
VAL_SAMPLES   = 50
TEST_SAMPLES  = 50
DATASET_TYPE  = "synthetic"       # "synthetic" | "real"
LOSS_TYPE     = "focal_spot"      # "focal_spot" | "pixel" | "alignment"
STAGE1_EPOCHS = 20                # AlignmentLoss pre-training (no ray tracing)
STAGE2_EPOCHS = 200               # FocalSpotLoss / PixelLoss fine-tuning
TRAIN_RAYS    = 10

# Optimizer (ARTIST format)
OPTIMIZATION_CONFIG = {
    initial_learning_rate: 1e-4,
    batch_size:            8,
    scheduler:             reduce_on_plateau  (factor=0.5, patience=10, min_lr=1e-6),
    early_stopping_patience: 400,             # effectively off (> max_epoch)
}
```

**Run:**
```bash
cd src
python one_heliostat_train_sizes/main.py --smoke-test         # quick end-to-end check
python one_heliostat_train_sizes/main.py --heliostat-id AC36  # single heliostat
python one_heliostat_train_sizes/run_all_heliostats.py        # all 5 + comparison plots
sbatch sbatch_files/run_one_hel_all.sh                        # DAIC (10 h, A40 GPU)
```

**Outputs** (per heliostat in `outputs/.../train_size_N/`):
results table (`summary_table.txt`), convergence plots, per-size mrad scores.
Cross-heliostat: `comparison_mrad_vs_train_size.png`, `comparison_table.txt`.

---

## Experiment 2 — `full_training_pipeline`

Coarse-to-fine pipeline for all 63 heliostats. Stage 1 runs standard kinematic reconstruction
(coarse alignment). Stage 2 trains a lightweight residual model (linear / polynomial /
transformer) to predict remaining per-sample error from heliostat features and motor positions.
The residual output is a 20D Wortberg-style correction vector (9 translations, 4 rotations,
2 actuator angles, 2 actuator offsets, 3 base positions).

**Key config** (`src/full_training_pipeline/config.py`):
```python
DATASET_TYPE             = "synthetic"   # "synthetic" | "real"
LOSS_TYPE                = "focal_spot"  # "focal_spot" | "pixel" | "alignment"
MAX_EPOCHS               = 200
LEARNING_RATE            = 1e-3
WEIGHT_DECAY             = 1e-5
RESIDUAL_L2_WEIGHT       = 1e-4
GRAD_CLIP_MAX_NORM       = 1.0
LR_SCHEDULER_PATIENCE    = 10
LR_SCHEDULER_FACTOR      = 0.5
NUMBER_OF_RAYS           = 10
RAY_TRACING_BATCH_SIZE   = 32
SURFACE_POINTS_PER_FACET = (25, 25)
BITMAP_RESOLUTION        = (256, 256)
SAMPLE_LIMIT_PER_HELIOSTAT = 100
```

**Run:**
```bash
cd src
python full_training_pipeline/main.py --smoke-test
python full_training_pipeline/main.py --dataset-type real --model-type linear
# --model-type: linear | poly2 | poly3 | poly4 | transformer
```

**Outputs** (`outputs/.../`): `training_summary.json`, `linear_residual_model.pt`,
`corrected_kinematic_parameters_best.json`, loss curves, per-heliostat improvement scatter.

---

## Experiment 3 — `full_63_heli_kin_reconstruct` *(primary)*

Closed-loop kinematic reconstruction for all 63 heliostats using the corrected pipeline.

**Perturbations** (applied at dataset generation, seed 42, uniform ±range):

| Parameter | ±Range | Count | Frozen? |
|---|---|---|---|
| Rotation deviations | 5 mrad | 4 | No |
| Actuator initial angle `a_i` | 5 mrad | 2 | No |
| Actuator stroke `b_i` | 5 mm | 2 | **Yes** |
| Actuator offset `c_i` | 5 mm | 2 | No |
| Translation deviations | 50 mm | 9 | No |
| Base position `(e, n, u)` | 50 mm | 3 | No |

> Ranges chosen so <2 % of generated flux images miss the target entirely.

**Key config** (`src/full_63_heli_kin_reconstruct/config.py`):
```python
PERTURBATION_SEED  = 42
PERTURBATION_RANGES = {
    "rotation_rad":       0.005,  # ±5 mrad
    "actuator_angle_rad": 0.005,  # ±5 mrad
    "actuator_stroke_m":  0.005,  # ±5 mm  (frozen)
    "actuator_offset_m":  0.005,  # ±5 mm
    "translation_m":      0.050,  # ±50 mm
    "base_position_m":    0.050,  # ±50 mm
}
LOSS_TYPE     = "focal_spot"   # "focal_spot" | "pixel"
STAGE1_EPOCHS = 50             # AlignmentLoss
STAGE2_EPOCHS = 250            # FocalSpotLoss / PixelLoss
TRAIN_SAMPLES = 100
VAL_SAMPLES   = 50
TEST_SAMPLES  = 50
TRAIN_RAYS    = 10
SYNTH_GEN_RAYS = 100           # used during dataset generation only

# Optimizer
OPTIMIZATION_CONFIG = {
    initial_learning_rate: 1e-4,
    batch_size:            8,
    scheduler:             reduce_on_plateau  (factor=0.5, patience=10, min_lr=1e-6),
    early_stopping_patience: 400,
}
```

**Data generation** (run once, or after changing perturbation ranges):
```bash
cd src
python full_63_heli_kin_reconstruct/generate_dataset.py           # ~10 min locally
python full_63_heli_kin_reconstruct/generate_dataset.py --force   # overwrite
```
Writes `scenarios/full_63_heli_kin_reconstruct/synthetic_data/{train,val,test}/`.

**Pre-flight check (DAIC):**
```bash
cd src && python full_63_heli_kin_reconstruct/check_daic.py
```

**Run:**
```bash
cd src
python full_63_heli_kin_reconstruct/main.py --smoke-test
python full_63_heli_kin_reconstruct/main.py \
    --dataset-type synthetic --loss-type focal_spot \
    --stage1-epochs 50 --stage2-epochs 250
sbatch sbatch_files/run_full63_synth_focal.sh    # DAIC (2 h, A40 GPU)
```

**Key outputs** (`outputs/.../`):
`summary_table.txt` (mean/median val+test mrad), `per_heliostat_accuracy.json`,
`field_accuracy_map.png`, `flux_grid_best_*.png` / `flux_grid_worst_*.png`,
`convergence_stage1.png` / `convergence_stage2.png`, `recovery_*.png` (synthetic only).

---

## ARTIST extensions (`src/artist_extensions/`)

Custom subclasses that keep ARTIST a clean dependency.

### `kinematic_reconstructors.py`

Three reconstructors implementing Wortberg (2025) Table 5.3 parameter set:

| Class | Loss | Notes |
|---|---|---|
| `WortbergKinematicReconstructor` | `FocalSpotLoss` | Stage 2, focal-spot centroid |
| `WortbergPixelReconstructor` | `PixelLoss` (blurred) | Stage 2, flux bitmap |
| `WortbergAlignmentReconstructor` | `AlignmentLoss` | Stage 1, no ray tracing |

All optimise: 9 translation deviations, 4 rotation deviations, 2 actuator angles,
2 actuator offsets, 3 base-position deviations (±50 mm / ±5 mrad bounds).
Actuator stroke `b_i` is **frozen**. Mini-batching over heliostats, gradient
clipping (`max_norm=1.0`), best-val checkpointing.

### `loss_functions_ext.py`

`AlignmentLoss` — motor-position MSE in joint-angle space; no ray tracing required.

### `cached_paint_parser.py`

`CachedPaintCalibrationDataParser` — caches PAINT tensors in CPU RAM after the first
epoch. Eliminates repeated NFS reads on DAIC (significant speedup from epoch 2 onward).

---

## DAIC / sbatch files

All scripts use one A40 GPU, 16 GB RAM, 4 CPU cores, Apptainer image at
`/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif`.

| Script | Experiment | Dataset | Loss | Epochs (S1+S2) | Limit |
|---|---|---|---|---|---|
| `run_full63_synth_focal.sh` | Exp 3 | synthetic | focal_spot | 50+250 | 2 h |
| `run_full63_real_focal.sh`  | Exp 3 | real       | focal_spot | 50+250 | 2 h |
| `run_full63_synth_pixel.sh` | Exp 3 | synthetic  | pixel      | 50+500 | 2 h |
| `run_full63_real_pixel.sh`  | Exp 3 | real       | pixel      | 50+500 | 2 h |
| `run_one_hel_all.sh`        | Exp 1 | synthetic  | focal_spot | 20+200 | 10 h |

```bash
# Submit from DAIC login node
sbatch src/sbatch_files/<script>.sh
```
Logs → `~/projects/githubProjects/master-thesis/logs/`.

