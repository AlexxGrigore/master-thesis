# Parameter Ablation Experiment — Focal Spot Loss

This experiment compares how kinematic reconstruction accuracy depends on **which subset of
heliostat parameters is optimised**.  All five configurations use the same focal spot loss,
the same 100-epoch budget, and the same train/test split.

---

## Background: the kinematic parameter space

Each heliostat in the ARTIST model has the following optimisable parameters (from Wortberg 2025,
Table 5.3):

| Group | Symbol | Physical meaning | Bound |
|---|---|---|---|
| **Translations** | — | Position offsets of Joint 1, Joint 2, and the concentrator along (e, n, u) | ±0.05 m |
| **Rotations** | — | Tilt deviations of the two rotation axes (4 values total) | ±0.005 rad |
| **Actuator initial angle** | aᵢ | Angular offset of each actuator drive at its reference position | ±0.005 rad |
| **Actuator offset** | cᵢ | Distance offset along each actuator joint | ±0.005 m |
| **Base position deviation** | δe, δn, δu | Shift of the heliostat's ground anchor point | ±0.05 m |

Parameters **not** optimised in any configuration (fixed by design):
- `bᵢ` — actuator initial stroke length (frozen in all configs)
- `dᵢ` — actuator pivot radius (non-optimisable in ARTIST by design)

---

## The five configurations

Each configuration activates a different subset of the parameters above.
The configurations are ordered by **increasing complexity**.

### Config A — Rotations only (`RotationsOnlyReconstructor`)

**Active:** rotations (4 params per heliostat)

The minimal structural model.  Corrects only the tilt deviations of the two rotation axes.
This is the natural baseline: axis misalignment is the dominant pointing error in most
real heliostats.

---

### Config B — Rotations + Actuators (`RotationsActuatorsReconstructor`)

**Active:** rotations (4) + actuator aᵢ (2) + actuator cᵢ (2) = 8 params per heliostat

Adds drive-mechanism calibration on top of Config A.  The actuator parameters capture
systematic errors in how the drive converts a motor command into a mirror angle.

---

### Config C — Rotations + Translations (`RotationsTranslationsReconstructor`)

**Active:** rotations (4) + joint/concentrator translations (9) = 13 params per heliostat

Adds the geometric offsets of the joints and concentrator in space, but leaves the
actuator drive model uncorrected.  Translations use a 5× higher learning rate than
rotations because their deviation bound is ten times larger.

---

### Config D — Full structural, no base position (`FullStructuralReconstructor`)

**Active:** rotations (4) + translations (9) + actuator aᵢ (2) + actuator cᵢ (2) = 17 params

The full Wortberg parameter set **without** correcting where the heliostat is planted in
the ground.  This is equivalent to `WortbergKinematicReconstructor(train_position_deviation=False)`.

---

### Config E — Full Wortberg (`WortbergKinematicReconstructor`)

**Active:** rotations (4) + translations (9) + actuator aᵢ (2) + actuator cᵢ (2) + base position δ(e,n,u) (3) = 20 params

The complete parameter set from Wortberg (2025).  Adds a correction to the heliostat's
ground anchor position on top of Config D.

---

## Outputs

Each configuration produces its own sub-directory under the experiment output folder
(`outputs/<timestamp>/<config_name>/`) containing per-run diagnostics.  After all five
runs complete, a `comparison/` directory is created with cross-experiment summary plots.

### Per-configuration outputs

| File | What it shows |
|---|---|
| `training.log` | Epoch-by-epoch loss, LR, and GPU memory |
| `convergence_group_0.png` | Training + validation loss curves, plus the mean absolute deviation of each parameter group over epochs — shows whether parameters are actually moving and whether they saturate their bounds |
| `param_histograms/` | Final-value histograms for every individual parameter with bound saturation percentage — reveals if the optimiser hits the ±bound wall for many heliostats (a sign the bound may be too tight, or the true error is large) |
| `loss_distribution.png` | Histogram and sorted plot of the final per-heliostat training loss — identifies heliostats that did not converge |
| `tracking_error_histogram.png` | Distribution of per-sample focal spot errors on the **test set** in mrad |
| `visualizations/` | Side-by-side flux image comparisons (predicted vs measured) for a few test samples |
| `test_metrics.json` | Scalar summary: mean / median / min / max focal spot error and per-heliostat breakdown |
| `all_kinematic_parameters.json` | Final optimised parameter values for every heliostat |

### Comparison outputs (`comparison/`)

| File | What it shows |
|---|---|
| `comparison_bar.png` | **Bar chart** — mean (solid) and median (transparent) focal spot error in mrad for each of the five configurations side by side.  The primary headline result: which parameter set achieves the lowest error? |
| `comparison_boxplot.png` | **Box plot** — distribution of per-heliostat errors for each configuration.  Shows not just the average but the spread and outliers — a config can look good on the mean while leaving many heliostats poorly calibrated. |
| `comparison_loss_curves.png` | **Overlaid loss curves** — left panel: training loss for all five configs on the same axes; right panel: validation loss.  Useful for comparing convergence speed and checking whether any config overfits (validation diverges from training). |
| `comparison_field_map.png` | **Field map** — birds-eye scatter plot of the heliostat field (east vs north, tower at origin).  Each dot is coloured by the focal spot error of that heliostat using a red–yellow–green spectrum (red = high error, green = low error).  All five panels share the same colour scale so spatial patterns are directly comparable across configurations — e.g. "Config A struggles with far-field heliostats; Config E fixes this". |
