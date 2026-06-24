# ARTIST upscaling cost experiment

Measures the two resources that matter for scaling ARTIST to commercial fields —
**wall-clock time** and **GPU VRAM** — as a function of field size `N ∈ {1, 10, 20, 50}`
heliostats, across the two expensive offline phases:

| Phase | Script | What it measures |
|-------|--------|------------------|
| **A. Scenario creation** | `create_scenarios.py` | NURBS-fit + scenario write, per N |
| **B. Joint training** | `run_training.py` | joint kinematics reconstruction, per N |

The scenarios built in A are the exact ones trained in B (same heliostats, same order —
both read `selection.py`). Operational *aiming* is deliberately **not** profiled: it is
pure kinematics (no ray tracing), microseconds, and scales linearly — that's the cheap
contrast you make on the slides, not something this experiment needs numbers for.

## What gets recorded

Per N, written to `outputs/new_mapping_function/profiling_experiment/`:

- **creation_*.json** — `seconds`, `peak_vram_alloc_gb`, `peak_vram_reserved_gb`
- **training_*.json** — same, plus `raytrace_seconds` / `raytrace_share_pct` (the share of
  training time spent in `HeliostatRayTracer.trace_rays`, i.e. forward ray tracing)
- **summary.csv** — both phases merged into one table for the deck

Timing is CUDA-synced; VRAM peak is reset per phase; CPU peak RSS and the GPU model are
recorded once per file.

## Run it

One job does everything (Phase A → Phase B → summary):

```bash
sbatch src/sbatch_files/run_profiling_experiment.sh
```

Or step by step on DAIC:

```bash
cd .../master-thesis/src
apptainer exec --nv --bind /tudelft.net:/tudelft.net <sif> \
    python profiling_experiment/create_scenarios.py --daic --sizes 1 10 20 50
apptainer exec --nv --bind /tudelft.net:/tudelft.net <sif> \
    python profiling_experiment/run_training.py     --daic --sizes 1 10 20 50
apptainer exec --nv --bind /tudelft.net:/tudelft.net <sif> \
    python profiling_experiment/summarize.py        --daic
```

Drop `--daic` to run locally (no GPU → VRAM fields are omitted, timings still recorded).

## Pinned configuration (held constant across all N)

NURBS: 20×20 control points, 400 fit epochs, fit-from-normals (matches the thesis's real
deflectometry scenario). Training: 25×25 surface points/facet, 10 rays/point (scenario
light source), 10 uniform calibration samples/heliostat, 256² bitmap, blocking **off**,
fixed `--max-epoch 150` (early stopping off so work-per-heliostat is constant across N),
single A40, one heliostat group. Training uses ARTIST's own `KinematicsReconstructor` +
`FocalSpotLoss` (the canonical reference path), processing all N heliostats jointly in one
batched ray-tracing pass — not the one-heliostat-at-a-time loop used elsewhere.

## Reading the result for the deck

`summary.csv` gives two scaling curves (time and VRAM vs N) for each phase. Fit a line to
each and extrapolate toward commercial scale (10k–100k heliostats); mark where the VRAM
line crosses the A40's 48 GB. Rule-of-thumb anchor from earlier runs (25×25, 10 samples):
~27 MB of training-graph VRAM per heliostat → the single-A40 wall sits near ~1,700
heliostats at 25×25.

## Files

- `selection.py` — single source of truth for the nested heliostat list (eligible =
  deflectometry data **and** ≥10 train samples). `select(10) ⊂ select(20) ⊂ …`.
- `profiling.py` — CUDA-synced timer + peak-VRAM harness; `accumulate_raytrace_time`
  monkeypatches `trace_rays` at runtime (ARTIST source untouched).
- `create_scenarios.py` / `run_training.py` / `summarize.py` — the three steps.
