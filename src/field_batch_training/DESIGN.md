# field_batch_training — full-field (1277) kinematic calibration on DAIC

Design only (no code yet). Goal: run the settled full-parameter regime (wide a-bound,
no AUTO_MOTOR_OFFSET, Stage 1 forward-aim + Stage 2 focal-spot centroid) on **all 1277
heliostats** of the 50-20-20 benchmark, as **13 SLURM jobs of ≤100 heliostats each**,
each training its ~100 heliostats **in parallel inside one ARTIST heliostat group**.

## Feasibility — verified up front

| requirement | status |
|---|---|
| same sample count per heliostat (parallel-group training) | ✔ all 1277 have **exactly 50 train / 20 val / 20 test** (verified from the split CSV — one signature across the field) |
| enough data per heliostat | ✔ 50 train samples is the same budget the 63-heliostat runs use successfully |
| flux images | ⚠ 844 of 114,930 PNGs missing → **train and evaluate centroid-only** (Stage 1 needs no flux; Stage 2 loss = predicted-flux centroid vs measured centroid from the JSON; measured PNGs are only for plots) |
| scenario source | ✔ `scenarios/full_benchmark_ideal/ideal_1277.h5` exists (1 group, real per-heliostat kinematics from PAINT properties, ideal surfaces — only the 63 deflectometry heliostats could have measured surfaces; the field trains with ideal + canting) |
| 40 GB VRAM | ✔ with Stage-2 **sample mini-batching**: 100 heliostats × 10–25 samples/mini-batch = 1000–2500 active instances ≈ 3–7 GB at 25×25 surface pts × 10 rays (reference point: 3760 instances ≈ 10 GB). Stage 1 is ray-trace-free (negligible). Large headroom — could even go to 200/batch, but 100 keeps jobs short and restarts cheap |

Batch layout: 1277 = **12 × 100 + 1 × 77**, split alphabetically (composition is
irrelevant to the loss — heliostats are independent; alphabetical = reproducible).

## Project structure (inspired by one_heliostat_demo)

```
src/field_batch_training/
├── DESIGN.md                    ← this file
├── README.md                    – how to build, submit, aggregate
├── config.py                    – LOCAL + DAIC path blocks (--daic switch, like run_all);
│                                  regime constants imported/copied from the settled setup:
│                                  full-param, wide-a (±0.5 rad), S1 200 + S2 200 ep,
│                                  TRAIN_RAYS 10, 25×25 surface pts, MINI_BATCH samples 10–25,
│                                  AUTO_MOTOR_OFFSET permanently off
├── build_batch_scenarios.py     – ✔ BUILT — creates field_batch_{00..12}.h5 directly from
│                                  the downloaded PAINT tower + Properties (100 heliostats
│                                  each, last 77; 1 group per file; ideal surfaces; smoke-
│                                  tested locally). No dependency on ideal_1277.h5 (whose
│                                  creator script was lost)
├── build_batch_manifests.py     – per batch: heliostat list + calibration-JSON paths for
│                                  the 3 splits (uses the benchmark CSV's own split — NO
│                                  pooled re-split, NO active-pixel filter: both would
│                                  break the uniform 50/20/20)
├── train_field_batch.py         – the grouped two-stage trainer:
│                                  · tensors shaped [100·N_samples, …], group parameter
│                                    tensors [100, …]
│                                  · Stage 1 forward-aim on all instances, per-heliostat
│                                    loss reduction
│                                  · Stage 2 centroid loss, mini-batched over samples
│                                  · bounds clamped per heliostat around ITS initial values
│                                  · best-val checkpoint kept PER HELIOSTAT (not global —
│                                    one bad heliostat must not decide everyone's restore)
│                                  · per-heliostat results.json equivalents in one file
├── run_batch.py                 – entry point: --batch-id N [--daic]
├── slurm/                       (all jobs are .sh files, submitted manually in order)
│   ├── download_dataset.sh      – ✔ BUILT — CPU job, downloads the 50-20-20 benchmark
│   │                              (+ tower + Properties, no deflectometry) to umbrella
│   │                              storage; idempotent; self-verifying
│   ├── create_scenarios.sh      – ✔ BUILT — CPU job, runs build_batch_scenarios.py --daic;
│   │                              submit after the download job has finished
│   └── train_batch.sh           – 1 GPU (a40), array job (`sbatch --array=0-12`); task ID
│                                  = batch ID; logs to $REPO/logs/
└── aggregate_field.py           – merge 13 batch summaries → field table, ECDF,
                                   rescued-vs-stuck classification, comparison to the
                                   63-heliostat run and the initial-aim floors
```

Outputs: `outputs/field_batch_training/batch_{00..12}/` + `aggregated/`.

## What is deliberately different from one_heliostat_demo

1. **No pooled re-splitting.** one_heliostat_demo pools train+val+test and re-splits with
   the PAINT DatasetSplitter (+ an active-pixel filter). Here the benchmark's own
   train/validation/test assignment is used verbatim — it is what guarantees the uniform
   50/20/20 that group training needs. (Filtering, if ever wanted, must drop whole
   heliostats, never single samples.)
2. **No flux PNG loading.** Centroid-only losses and evaluation (844 PNGs missing anyway).
   Per-heliostat plots are dropped; the aggregate step produces field-level figures.
3. **Group tensors everywhere.** Parameters, bounds, LR groups and checkpoints are
   [N_hel, …]-shaped; per-heliostat best-val selection replaces the single-heliostat
   "best epoch" logic.
4. **DAIC-first.** Paths resolve via a --daic flag (remote root
   `/home/nfs/agrigore/projects/githubProjects/master-thesis/`); everything CPU-tested
   locally on a 5-heliostat mini-batch before submission.

## Runtime estimate

Stage 2 dominates: 200 epochs × (50 samples / mini-batch 25 → 2 ray-traced forwards of
2500 instances) per epoch, plus 200 cheap Stage-1 epochs. On an A40-class GPU expect
roughly **2–5 h per batch**; 13 array tasks run concurrently as the queue allows → the
whole field in one evening. Local single-heliostat CPU reference: ~2 min per heliostat
(≈ 3.5 h if run serially on CPU — the GPU batch is ~100× more parallel per step).

## Expected outcome (pre-registered from the diagnostics)

- The **rescuable majority** (constant-reference faults within ±165 mrad equivalent)
  should land in single digits to ~15 mrad — the 63-heliostat run with this regime is
  tracking a ~8 mrad median on a *harder-than-average* subset.
- The **broken tail** (AW36-class axis-1 faults > b-range, mid-campaign jumpers,
  heavy-scatter heliostats — several hundred field-wide per the 1277 initial-aim
  analysis) will plateau at their data floors; the aggregate step should classify them
  (post error vs the per-heliostat corrected-miss floor) rather than hide them in a mean.
- Field median target: **≤ ~15–20 mrad** (the initial-aim analysis puts the constants-only
  floor at 18.8 mrad median; the trainable regime beat that floor on the 63).

## Decisions (settled 2026-07-14)

1. **Batch composition: alphabetical.**
2. **Epoch budget: Stage 1 = 200, Stage 2 = 200.**
3. **Per-heliostat kinematic_parameters.json: YES** — emitted for every heliostat in every
   batch (it is the calibration product).
4. **Eval: centroid-only everywhere** (uniformity; 844 PNGs missing).

## Missing flux PNGs — inventory (for a future image-based loss)

844 of 114,930 samples lack a flux PNG, spread thinly over **588 of the 1277 heliostats**
(median 1 missing per affected heliostat, max 8 of 90 — worst: AW55 8, AH51 7, AH59 6).
33 of the 63 deflectometry heliostats are affected (1–3 samples each, incl. AW36, AF40,
AC27). Full list: `outputs/field_batch_training/missing_flux_pngs.csv`.

Consequence: no heliostat is unusable, but a future image-based loss cannot simply drop
the affected samples — that would break the uniform 50/20/20 the group training needs.
The right mechanism is a **per-sample flux-availability mask** carried in the batch
manifests: image-loss terms are zero-weighted where the PNG is missing (optionally
falling back to the centroid loss for those samples). `build_batch_manifests.py` should
emit this mask from day one so the switch to an image loss later is config-only.
