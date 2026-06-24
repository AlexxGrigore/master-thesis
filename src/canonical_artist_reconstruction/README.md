# Canonical ARTIST Kinematics Reconstruction (reference baseline)

A deliberately minimal subproject that runs **ARTIST's own `KinematicsReconstructor`**
exactly as the official tutorial prescribes, with **no project-specific modifications**.
It exists as a trustworthy reference to compare against the custom two-stage, centre-free
pipeline in `src/one_heliostat_demo`.

## What the canonical method does (and what it does NOT)

Mirrors `ARTIST/tutorials/04_kinematics_reconstruction_tutorial.py`:

- **Single stage.** No Stage-1 alignment loss.
- **Loss = `FocalSpotLoss`** (ray-traced focal-spot centroid vs measured centroid).
- **Aligns by aiming at the TARGET CENTRE** (`get_centers_of_target_areas`), not from the
  recorded motors. (This is the "aim-at-centre" formulation — what `CALIBRATION_FORMULATION.md`
  replaced with the centre-free version.)
- **Optimizes** `rotation_deviation`, `initial_angles` (a_i), `initial_stroke_length` (b_i).
  It does NOT touch translation, actuator offset c_i, or base position.

Contrast with `one_heliostat_demo`: two stages, custom forward-aim Stage-1 loss, orient from
recorded motors `m_c` (centre-free), freezes b_i, adds translation / c_i / base position.

## Run

```bash
python run_canonical.py                                   # AC33 AB33 BE35
python run_canonical.py --heliostats AC33 BE35 --max-epoch 300 --train-samples 20
```

Reports focal-spot error (mrad) on a held-out TEST split, before vs after reconstruction.
Outputs to `outputs/new_mapping_function/canonical_artist_reconstruction_<timestamp>/`.

## Result (3 real PAINT heliostats, 2026-06-22)

20 train / 20 test samples, 600 epochs, tutorial LRs.

| heliostat | dist | BEFORE mean / median | AFTER mean / median |
|-----------|------|----------------------|---------------------|
| AC33 | 56 m  | 12.33 / 5.23 | 12.20 / 6.04 |
| AB33 | 54 m  | 6.29 / 4.87  | 5.80 / 4.39  |
| BE35 | 217 m | 2.17 / 1.77  | 2.67 / 2.00  |

### Key takeaways

1. **The aim-at-centre "before" error is only a few mrad (median 2–5).** This is the number we
   remembered. The 43 mrad field-wide figure came entirely from the centre-free metric
   (orient-from-`m_c`), NOT from real miscalibration.
2. **The canonical reconstruction makes only small adjustments** (loss drops ~5–10% over 600
   epochs; test error roughly unchanged). Consistent with the tutorial's own "the adjustments
   are small" caveat — there is little error to remove because the heliostats already start near
   the surface/measurement floor in this formulation.
3. The mean is heavy-tailed (AC33 mean 12 vs median 5): a few outlier samples that the
   rotation+a_i+b_i parameter set cannot fix.
