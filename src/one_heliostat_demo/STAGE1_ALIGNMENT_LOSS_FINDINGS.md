# Stage-1 Alignment Loss — Findings

**Date:** 2026-06-20
**Heliostat analysed:** AC33 (synthetic, balanced dataset)
**Status:** problem identified and root-caused; fix proposed but **not yet implemented**

---

## TL;DR

The Stage-1 alignment loss minimises the **distance between two motor positions**:
the recorded motors `m_c`, and the motors `m_pred` that the model's **inverse kinematics**
says are needed to aim at the observed centroid `c_gt`.

**The problem:** this loss does **not** reach its minimum at the true parameters θ\*.
Minimising it drives *away* from correct calibration. The root cause is that ARTIST's
inverse kinematics is **not a consistent inverse of its forward model** — it ignores the
rotation-deviation parameters — so `m_pred` is systematically wrong even when θ is correct.

**The fix:** use a **forward-only** objective (compare the forward normal at the real motors
`m_c` against the geometric desired normal). This provably has its minimum at θ\*.

---

## 1. What we are trying to do

A heliostat is a mirror aimed by **two motors**. The **kinematic model** maps motor positions
to where the mirror points (its surface normal) and hence where the reflected beam lands. The
model has small unknown **geometric error parameters θ** (tilts, translations, actuator offsets).
The real mirror has true values **θ\***. **Calibration = recover θ.**

The synthetic dataset gives, per calibration sample:
- the **sun direction**,
- the **recorded motor positions `m_c`**,
- the **beam centroid `c_gt`** (where the beam actually landed).

This data was generated with the true θ\*, so by construction the motors `m_c` send the beam
to `c_gt`.

## 2. The model runs in two directions

- **Forward:** motors + sun → mirror normal → beam landing point.
  *"I set the motors here; where does the light go?"*
- **Inverse:** aim point + sun → motors.
  *"I want to hit there; what motors do I need?"*

These should be **exact opposites**. Running **inverse → forward** should return you to the
point you started from. This round-trip property is the heart of the bug.

## 3. What the alignment loss actually computes

Stage 1 uses the **inverse**:

> Given the sun and the observed centroid `c_gt`, compute the motors `m_pred` that the current
> model thinks are needed.

Then:

```
loss = distance( m_pred , m_c )
```

i.e. exactly the **distance between two motor positions**. The intent: if θ is correct, the
motors predicted to aim at `c_gt` should match the motors that actually produced `c_gt`.

## 4. What we expected

At θ\*, `m_pred` should equal `m_c`, so the loss should be **0**. Driving the loss down should
recover θ\*.

## 5. What actually happened

| run | loss behaviour | calibration result |
|---|---|---|
| train-size 1  | dropped almost to 0 | wrong — test focal error stuck at ~30 mrad |
| train-size 25 | **plateaued**, never reached 0 | wrong — no improvement |

The mirror-aim plot (`plots/normal_aim_convergence.png`) showed the predicted normal **stuck at
its starting point**, never moving toward the true normal. Loss going down, calibration not
improving — that contradiction is what had to be explained.

## 6. The decisive test — evaluate the loss at the known θ\*

Because the data is synthetic, we know θ\*. If the objective were sound, θ\* would be the
**best-scoring** point (loss ≈ 0). Measured for AC33 (50 samples):

| θ (parameters) | alignment loss | forward pointing error |
|---|---|---|
| starting point (nominal) | 0.0045 | 14.4 mrad |
| **true answer θ\*** | 0.0025 | **0.67 mrad** ✓ |
| what training settled on | **0.0011** ✓ | ~13 mrad ✗ |

Two things to read carefully:

1. At θ\*, the **forward pointing error is 0.67 mrad** → θ\* is genuinely correct; the forward
   model and the data are fine.
2. Training reached a **lower loss (0.0011)** than the true answer scores (0.0025).

**⟹ The true answer is not the best-scoring answer.** The loss prefers a *wrong* θ, so
minimising it pulls *away* from the truth. The optimiser was not failing — it was faithfully
chasing a bad target.

## 7. Why 8.1 mrad in one place and 0.67 mrad in another (same θ\*)

Both are measured at θ\*. The difference is **which motors are fed into the forward model.**

**Check A → 0.67 mrad — uses the real recorded motors `m_c`:**
1. Take the genuine recorded motors `m_c`.
2. Run them forward at θ\* → mirror normal.
3. Compare to the geometric correct normal (reflects sun onto `c_gt`).
4. → **0.67 mrad, they match.** ✓ (The real motors point correctly — as they must, since
   `c_gt` was produced by `m_c`.)

**Check B → 8.1 mrad — uses the motors the inverse invented:**
1. Take the aim point `c_gt`.
2. Ask the inverse at θ\*: "what motors hit `c_gt`?" → answer `m_pred`.
3. Run `m_pred` forward at θ\* → normal.
4. Compare to the same correct normal.
5. → **8.1 mrad, they don't match.** ✗

The one sentence:

> Even at the true θ\*, the inverse's answer **`m_pred` ≠ the real `m_c`**.

- Feed the **real** motors forward → correct (0.67 mrad).
- Feed the **inverse's** motors forward → wrong (8.1 mrad).

The alignment loss is `distance(m_pred, m_c)`, so it measures the *broken* quantity (Check B).
That residual is exactly why the loss is non-zero at θ\* (it appeared as ~7.8 mrad of
"self-consistency" error in the diagnostic).

## 8. Root cause — the inverse is not a true inverse

Round-trip test at θ\* (aim → inverse → forward → compare to aim):

| inverse iterations | round-trip normal error |
|---|---|
| 2  | 8.12 mrad |
| 5  | 8.12 mrad |
| 20 | 8.12 mrad |
| 80 | 8.12 mrad |

The error is **independent of iteration count** → it is **structural, not under-convergence.**

ARTIST's analytical inverse `incident_ray_directions_to_orientations`
(`ARTIST/artist/field/kinematics_rigid_body.py:336`) solves for the joint angles using only the
**translation-deviation** parameters. It does **not** invert the **rotation-deviation
parameters** (±5 mrad × 4) that the forward model applies. So the forward applies tilts the
inverse never accounts for, the round trip never closes, and `m_pred` is systematically off.

## 9. Why that one fact breaks the loss

The loss compares `m_pred` (from the broken inverse) against `m_c` (real). Because the inverse
has a blind spot, `m_pred` is distorted. The optimiser can shrink the loss by choosing a θ that
**compensates for the inverse's blind spot** — but that θ is not the true one. So "small loss"
and "correct calibration" come apart, and the loss minimum sits at the wrong θ.

This explains every earlier observation: train-size 1 overfitting to 0.4 mrad; train-size 25
flooring at ~3.9 mrad; the forward normal stuck at nominal while the true normal sits ~13 mrad
away.

## 10. The fix — a forward-only objective

Stop using the inverse. Use only the forward direction, which is self-consistent:

```
loss = angle( normal(m_c, θ) ,  desired_normal )
desired_normal = normalize( -sun_direction + normalize(c_gt - mirror_origin) )
```

i.e. push the real motors `m_c` **forward** to get the mirror normal, and compare it to the
geometric desired normal (the sun↔`c_gt` bisector). This is **Check A** from §7, turned into
the training objective.

Evidence it is the right objective (AC33): it scores **0.67 mrad at θ\*** and **14.4 mrad at the
start** → its minimum **is** at the true parameters. It uses no inverse kinematics, so it avoids
the inconsistency entirely, and it is differentiable in θ.

> Small caveat: the 0.67 mrad floor at θ\* is the residual between the rigid-body normal and the
> full B-spline-surface-traced centroid. It is small here; Stage 2 (focal-spot loss, full ray
> tracing) is what accounts for the surface.

## 11. How to reproduce

Diagnostic scripts used (kept in `/tmp`, regenerate as needed):
- `diag_stage1.py` — loss + forward aim error at nominal vs true θ\* (the §6 / §7 table).
- `diag_fwdinv.py` — forward∘inverse round-trip error vs inverse iteration count (the §8 table).

True perturbation θ\* for each heliostat:
`datasets/synthetic/balanced_dataset/dataset/perturbations.json`.

Runs referenced:
- `outputs/new_mapping_function/AC33_trainsize1_stage1_100ep_synthetic/`
- `outputs/new_mapping_function/AC33_trainsize25_stage1_100ep_synthetic_balanced/`
