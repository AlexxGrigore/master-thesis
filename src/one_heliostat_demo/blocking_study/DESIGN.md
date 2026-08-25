# blocking_study - does modelling blocking improve calibration accuracy?

Target dataset: `benchmark_split-balanced_train-50_validation-20` (50-20-20, already on DAIC).

## The three premises under test

| # | premise | status |
|---|---|---|
| P1 | Heliostats block each other's reflected flux, usually in the bottom half, which raises the measured centroid | **CONFIRMED** on real data, 59/59 samples |
| P2 | Training WITH blocking gives better centroid-metric accuracy, because the predicted centroid is then aligned with the measured one | **NOT YET TESTED** - needs training |
| P3 | The assumed pose of the surrounding heliostats (horizontal / one of 3 targets / receiver) changes training accuracy, so it has to be accounted for | **PARTIALLY** - the pose matters as stow vs aimed, but the 4 aimed targets are indistinguishable |

Everything below is organised around getting P2 and P3 answered.

---

## Findings so far (all measured, no training)

### P1 is confirmed

Tracing each real calibration sample twice (blocking off, then on, identical ray seed) on
three heliostats with confirmed exposure:

| heliostat | rays blocked (median) | deficit sits below spot centre | centroid pushed upward |
|---|---|---|---|
| BE25 | 10.3 % | 20/20 samples, -0.417 m | 20/20, +0.178 mrad |
| AY43 | 9.9 % | 19/19 samples, -0.408 m | 19/19, +0.211 mrad |
| AY44 | 13.2 % | 20/20 samples, -0.420 m | 20/20, +0.283 mrad |

So the mechanism is real and its sign is exactly as expected: the lost flux sits about
0.4 m below the spot centre, and the centroid moves up by 0.18 to 0.28 mrad.

**That 0.18-0.28 mrad is the honest pre-registered effect size for P2.** It is the
mismatch a blocking-unaware model is forced to absorb into its kinematics, so it is roughly
the most that modelling blocking can recover. Against the real-data baseline of 3.5 mrad it
would be invisible; against the synthetic noise floor of about 0.06 mrad it is a factor of
3 to 5, hence measurable. **This is the reason to run the experiment on simulated data
first, and it is a quantitative reason, not a stylistic one.**

### Only the 2 nearest neighbours matter (the scenario-sizing answer)

Built deliberately oversized scenarios (22 to 36 candidate neighbours out to 72 m) and
measured each candidate's individual contribution by neutralising all the others:

| heliostat | contributing neighbours | distance | everything else |
|---|---|---|---|
| BE25 | BD25 (4.74 %), BD26 (3.97 %) | 10.2 m, 11.3 m | 34 candidates, 0.00 % each |
| AY43 | AX43 (7.33 %), AX42 (0.79 %) | 7.8 m, 8.9 m | 22 candidates, 0.00 % each |
| AY44 | AX44 (8.77 %), AX43 (0.90 %) | 7.8 m, 8.9 m | 20 candidates, 0.00 % each |

**Rule: only the immediately adjacent row, within about 12 m, blocks anything.** Confirmed
independently by the fact that the total blocked fraction is identical in a 6-primitive and
a 37-primitive scenario (BE25 11.27 % both, AY43 10.48 % both, AY44 11.20 % both). Geometry
explains it: at 9 deg beam elevation the beam clears a 2.8 m mirror top after roughly 20 m.

Practical consequence: a blocking scenario needs the trained heliostat plus its 2 to 4
nearest in-beam neighbours. That is cheap, and it means this study never needs a field-scale
scenario. Keep using `blockers.py` with its conservative margin to pick candidates, then
confirm real exposure with the raytracer.

### P3, the part already answerable without training

From `GATE_RESULTS.md`: stowed neighbours block 0.06 % of rays, aimed neighbours block
about 10 %. But the four aimed hypotheses agree to within 0.5 percentage points (upper
10.42, lower 10.59, mft 10.06, receiver 10.15), because from 200 m the four targets are
within about 1 deg of each other, so a neighbour aimed at any of them presents nearly the
same silhouette.

**So P3 collapses to a binary: stowed versus aimed.** Which specific target does not
matter, and no downstream training metric can resolve a 0.5 pp difference. This shrinks the
pose sweep from 5 arms to 2, which is what makes the training matrix affordable.

### A blocking bug in ARTIST that had to be fixed first

`build_linear_bounding_volume_hierarchies` produces a **disconnected forest** rather than a
single tree for most primitive counts. `lbvh_filter_blocking_planes` starts its traversal at
node 0, so it only ever reaches node 0's component and silently drops every other blocker.

| n primitives | 2 | 5 | 6 | 7 | 9 | 19-21 | 24 | 36 | 37 |
|---|---|---|---|---|---|---|---|---|---|
| disjoint roots | 1 | 1 | **2** | 1 | **2** | 1 | **2** | **10** | 1 |
| primitives reachable | 2 | 5 | **3** | 7 | **6** | all | **12** | **8** | 37 |

Real consequences measured on the field: for BE25 with 37 primitives the filter returned
**zero** hits when the truth is 11.27 % of rays blocked; AY43 was under-reported as 7.16 %
against a true 10.48 %.

Fix: `brute_blocking.py` provides an `exact_blocking()` context manager that replaces the
filter with "every primitive except the ray's own heliostat". The LBVH is only an
acceleration structure, so this changes cost, not physics, and with a few tens of
primitives the cost is negligible. Self-exclusion must be retained (leaving a heliostat's
own plane in produced 29 % spurious self-blocking, because its surface points sit on tilted
facets a few centimetres off the single fitted corner plane).

`gate.py`, `wortberg_premise.py` and `neighbour_census.py` all route through
`exact_blocking()`. **Any future blocking work in this repo must do the same, or report
silently wrong numbers.** `lbvh_is_connected()` is provided to assert the stock path would
have been trustworthy for a given scenario.

---

## Heliostat selection

Chosen by requiring, in order: confirmed real blocking exposure (raytracer, not the coarse
screen), a tight baseline (mean close to median in `standard_results.csv`, so blocking is
not competing with unrelated kinematic error), and available deflectometry.

| heliostat | distance | blocked | baseline dir med/mean | role |
|---|---|---|---|---|
| **BE25** | 218 m | 10.3 % | 1.56 / 2.45 | primary, most exposed of the standard 63 |
| **AY43** | 172 m | 9.9 % | 1.60 / 1.86 | primary, tightest baseline |
| **AY44** | 173 m | 13.2 % | 2.39 / 2.73 | primary, highest blocked fraction |
| AA27 | 62 m | 0.06 % | - | negative control, must show no effect |

Rejected: AZ27, AY42, BA35, AQ24, BA42 (mean 2 to 4x the median, outlier-heavy).
**AP43 rejected after verification** despite having the best baseline in the field: it
passes the coarse geometric screen but the raytracer finds exactly zero intercepted rays,
identical flux to the decimal. A conservative silhouette screen is a candidate list, never
evidence of exposure.

---

## Experiment S: simulated, generate with blocking, train with and without

The core experiment, and the one that can actually resolve a 0.2 mrad effect.

**Scenarios.** Reuse the existing small neighbourhood scenarios
(`scenarios/neighbourhoods/{BE25,AY43,AY44}/`), rebuilt with deflectometry surfaces for the
trained heliostat. Blockers keep ideal surfaces, since only their silhouette is used.

**Generation.** Extend the existing synthetic path (`generate_dataset.py`, which calls
`_forward_pass` in `src/utils/synth_data.py`) so that generation runs with blocking on and
the neighbours in a chosen pose. Reuse the perturbation already assigned in the existing
synthetic datasets (`dataset/perturbations.json`) so results stay comparable with every
earlier synthetic run. Perturb only the trained heliostat; keep neighbours nominal, since a
40 mrad neighbour error moves its silhouette edge by centimetres at 10 m, a second-order
effect on top of a first-order pose question.

This yields, per heliostat, one dataset generated with `neighbours = aimed` whose recorded
centroids carry the real upward bias.

**Training arms**, all starting from the same Stage-1 checkpoint so differences are Stage 2
alone:

| arm | model during training | tests |
|---|---|---|
| A0 | blocking off | P2 baseline: today's pipeline, carries the mismatch |
| A1 | blocking on, neighbours aimed (**correct**) | P2 upper bound: how much is recoverable |
| A2 | blocking on, neighbours stowed (**wrong**) | P3: cost of assuming the wrong pose |

Three arms, three heliostats, plus AA27 as control on A0/A1. `MINI_BATCH_SIZE = 1` is
required whenever blocking is on (ARTIST's blocking assumes one active instance per
heliostat row, while this pipeline encodes N samples as N repeats of one row).

**Scoring.** Synthetic data has known ground-truth kinematics, so report, per arm:
1. **centroid error in mrad** (the headline for P2) and **direction error in mrad** - both,
   always, never one alone;
2. **parameter recovery** against the true perturbation, which no real-data run can give.

**Reading the result.** A1 better than A0 by roughly 0.2 mrad confirms P2. A1 equal to A0
means blocking is not worth modelling even when the model is otherwise perfect, which is a
clean negative and a genuinely useful thesis result. A2 much worse than A1 confirms P3 and
means the neighbour pose has to be inferred; A2 close to A1 means it can be ignored.

---

## Experiment R: the same three arms on real data

Run only after S, and only with S's effect size in hand, because on real data a 0.2 mrad
shift sits under a 3.5 mrad baseline. Value is the sim-versus-real contrast: S has known
ground truth and a clean forward model, R has the encoder bias, deflectometry error and
genuinely unknown neighbour state. If S shows a gain and R does not, the gain is being
swamped by real-data error, which is itself the answer to "should we take blocking into
account".

---

## Work remaining

1. Wire blocking into the training pipeline. Group-index generalisation in `train.py`
   (currently hard-codes heliostat 0 of group 0 and length-1 active masks): resolve
   `hel_idx` via `hg.names`, one-hot masks at `train.py:297-306`, `:2519`, `:2615`,
   `_base_position_deviation` to `[N_hel, 3]` at `:655`, `_steps_per_rad` at `:1782-1792`.
   Blocking injection at the two Stage-2 tracers (`:2531`, `:2626`) and in
   `synth_data.py:231`. Config: `BLOCKING_ENABLED`, `BLOCKING_NEIGHBOUR_POSE`.
2. Blocking-aware synthetic generation (Experiment S datasets).
3. Rebuild the three scenarios with deflectometry surfaces.
4. Runner plus aggregation, `--daic` paths, one SLURM file mirroring
   `src/field_batch_training/slurm/train_batch.sh`.

## Limitations to state

- **Shading is not modelled at all.** ARTIST occludes reflected rays only. The same
  neighbours that block about 10 % of the outgoing beam also shade the incoming beam at low
  sun elevation, and nothing represents that. At the geometry where blocking bites, shading
  is probably the larger effect.
- One rectangular plane per heliostat: no facet gaps, no torque tube.
- Blockers carry nominal kinematics, so their pose error is roughly the field median even
  under the correct hypothesis.
- Some neighbours cannot physically be aimed at some targets (ARTIST logs "no valid motor
  position combination"); those fall back to an unreachable-pose result and should be
  excluded from the aimed hypothesis rather than silently accepted.
