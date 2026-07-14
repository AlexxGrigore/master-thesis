# Kinematic optimization — full-parameter regime (supervisor feedback, 2026-07-14)

Supervisor: the Wortberg Table 5.3 bounds are outdated (they addressed a deadlock that no
longer exists). Relax bounds to ~20 mrad, optimize **all available parameters** including the
currently frozen ones, keep a close watch only on the heliostat-position offset bounds.

This document is the parameter inventory, the proposed regime, and the pitfalls.

## 1. Complete parameter inventory (rigid_body + linear actuators)

### Kinematics tensors

| parameter | shape / index | current | proposed | bound (around initial value) |
|---|---|---|---|---|
| `rotation_deviation_parameters` (first-joint tilt N,U; second-joint tilt E,N) | [1,4] | optimized, ±5 mrad | optimized | **±20 mrad** |
| `translation_deviation_parameters` (first joint E,N,U; second joint E,N,U; concentrator E,N,U) | [1,9] | optimized, ±50 mm | optimized | ±50 mm (keep) |
| `_base_position_deviation` (heliostat position E,N,U — our extension) | [1,3] | optimized, ±50 mm | optimized | **±50 mm — the WATCHED bound** (supervisor) |
| `initial_orientation` | [4] | frozen | frozen | — (a frame convention, not a parameter) |

⚠ `translation_deviation_parameters[7]` holds the *physical* concentrator offset (0.175 m) —
bounds must stay centred on initial values, never on zero (already the case in `_apply_bounds`).

### Actuator `optimizable_parameters` [1,2,2] (per axis)

| idx | parameter | current | proposed | bound |
|---|---|---|---|---|
| 0 | `initial_angle` aᵢ | optimized, ±5 mrad | optimized | **±20 mrad** |
| 1 | `initial_stroke_length` bᵢ | **frozen** | **OPTIMIZE** | **±50 mm** (encoder re-referencing scale) |

⚠ a₁ is stored with a −π/2 shift vs the PAINT file (−1.5555 = 0.0152 − π/2) — another reason
all clamps are relative to loaded initial values.

### Actuator `non_optimizable_parameters` [1,7,2] (per axis)

| idx | parameter | nature | proposed | bound |
|---|---|---|---|---|
| 0 | `type` | **ID** (categorical) | never optimize | — |
| 1 | `clockwise_movement` | **ID** (boolean) | never optimize | — |
| 2 | `min_motor_position` | operational limit | never optimize | — used by the inverse for **branch selection** (picks the reachable arccos solution); corrupting it breaks the solver |
| 3 | `max_motor_position` | operational limit | never optimize | — same |
| 4 | `increment` (steps/m) | continuous physical | **optional, phase 2** (off by default) | ±0.5 % relative |
| 5 | `offset` cᵢ (linkage side) | optimized, ±5 mm | optimized | **±20 mm** |
| 6 | `pivot_radius` rᵢ (linkage side) | **frozen** | **OPTIMIZE** | **±20 mm** |

Why increment is phase-2: (a) evidence is weak — the motor-position-dependent residual in the
field data is explained by the a-vs-b curve shape, not by a gear-ratio error; (b) scale trap:
an Adam step moves a raw parameter by ≈ lr, and increment ≈ 154 166 steps/m, so at lr 1e-4 it
is immobile; it needs its own LR group (lr ≈ 1–2 for ~0.1 % travel per 100 epochs) or a
relative reparameterisation. Enable only behind a flag with that plumbing.

### Out of scope
Surface (B-spline control points, canting, facet positions): calibrated by deflectometry, not
by the kinematic reconstructor. `sun` / target geometry: fixed plant data.

## 2. Summary of the new regime

**Optimize (16 continuous parameters per heliostat + position):**
4 rotation tilts, 9 translations, 3 base-position, a₁ a₂, b₁ b₂, c₁ c₂, r₁ r₂ — everything
continuous and physical. Frozen forever: type, clockwise, min/max motor (IDs and limits).
Optional later: increment.

**Bounds:** angles ±20 mrad (rotations, aᵢ); linkage lengths ±20 mm (cᵢ, rᵢ);
stroke reference ±50 mm (bᵢ); translations ±50 mm; base position ±50 mm (watched).
All bounds centred on loaded initial values.

**Learning-rate groups** (Adam step ≈ lr per parameter, so group by physical scale):
- rotations + aᵢ (rad): base LR
- cᵢ, rᵢ, translations, base position (m, mm-scale travel): base LR × 5 (existing pattern)
- bᵢ (m, up-to-50 mm travel): base LR × 20 (exists as `ACTUATOR_STROKE_LR_MULT`)
- increment (if enabled): dedicated group, lr ≈ 1–2 in raw units

## 3. Cautions (measured, not hypothetical)

1. **Identifiability / overfitting.** a, b, c, r all shape the same per-axis arccos curve and
   the data covers a limited stroke arc. Unfreezing b alone already produced measurable
   overfit (AY36: train 18.8→12.1 mrad while test only 20.2→18.0; gap +1→+5 mrad,
   b2_freeze_experiment). With four curve parameters free per axis this can only grow.
   Keep best-val checkpointing (exists) and report the train-vs-test gap per run.
2. **Wide bounds do not solve beam-off-target.** With the beam metres off the receiver the
   focal loss has no gradient and Stage-1's proxy drives b the wrong way (measured: AY39 b ran
   to the wrong bound). The heavy tail of reference errors (implied Δα up to 380 mrad —
   far beyond ±20 mrad anyway) still needs `AUTO_MOTOR_OFFSET` first. Recommended default for
   real data: offset ON, then full-parameter refinement inside the new bounds.
3. **20 mrad covers the moderate cases only.** From the field diagnosis the implied angle
   errors of the stuck population are 30–380 mrad; ±20 mrad rescues the AY36-class boundary
   cases at best. That is fine — the constants handle the bulk — but do not expect the bound
   relaxation alone to un-stick the bad half.
4. **Physics-informed transforms.** increment, c, r, b pass through softplus(β=100)+ε in
   `_physics_informed_parameters`; at their magnitudes this is identity to <1e-6, so raw-space
   clamps are physical. aᵢ is untransformed.

## 4. Implementation checklist (train.py / config.py)

- [ ] `OPTIMIZE_ACTUATOR_STROKE` default True (b trainable; LR mult already wired)
- [ ] extend `_non_opt_grad` mask: {offset, pivot_radius} (+increment behind new flag);
      keep {type, clockwise, min, max} at zero grad
- [ ] `_apply_bounds`: add pivot_radius ±20 mm; keep b ±50 mm; new constants:
      `_BOUND_ROTATION_RAD = 0.020`, `_BOUND_ACTUATOR_ANGLE_RAD = 0.020`,
      `_BOUND_ACTUATOR_OFFSET_M = 0.020`, `_BOUND_PIVOT_RADIUS_M = 0.020`
- [ ] snapshot/restore + kinematic_parameters.json: add pivot_radius delta (b delta exists)
- [ ] keep base-position bound at ±50 mm and log when it clamps (the watched bound)
- [ ] leave `RANDOM_PERT_BOUNDS` (dataset generation) untouched — those describe synthetic
      perturbation magnitudes, not training freedom
