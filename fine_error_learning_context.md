# Fine Error Learning Context

## Project Summary

This thesis is about improving heliostat pointing accuracy with a two-stage, physics-aware calibration pipeline built on top of ARTIST and the PAINT dataset.

The high-level idea is:

1. Use a differentiable kinematic optimizer to estimate coarse geometric deviations for each heliostat.
2. Add a second learning module on top of that coarse solution to model the residual error that rigid-body kinematics alone cannot explain.

In thesis terms:

- **Coarse Error Learning (CEL)** = the existing kinematic reconstructor.
- **Fine Error Learning (FEL)** = the next module to be implemented.

The final target is a modular coarse-to-fine pipeline in which the coarse module produces a baseline parameter estimate $\theta_{KR}$ and the fine module predicts a residual correction $\Delta \theta$, such that:

$$
\theta_{final} = \theta_{KR} + \Delta \theta
$$

The corrected parameters are then passed through the differentiable ARTIST ray tracer, which produces a predicted flux image. The loss is computed against the measured calibration image, and gradients are backpropagated through the ray tracer into the fine module.

## Intended Final Pipeline

The intended final pipeline corresponds to the architecture shown in the thesis figure:

1. **Inputs**
   - Sun position
   - Heliostat data, including location and motor state
   - Calibration image

2. **Coarse Error Learning**
   - Runs first and provides the baseline kinematic estimate.
   - This is the current kinematic reconstructor stage.
   - In the fine-learning stage, this module is treated as fixed or precomputed.

3. **Fine Error Learning**
   - Learns a residual correction $\Delta \theta$ on top of the coarse estimate.
   - Its output is added to the coarse parameters before ray tracing.

4. **Differentiable Ray Tracing**
   - Uses $\theta_{KR} + \Delta \theta$ to render a predicted flux image.
   - Provides the physical link between learned parameters and image-space supervision.

5. **Loss and Backpropagation**
   - Compares predicted and measured flux.
   - Backpropagates through the differentiable renderer into the fine module.

This means the fine module is not learning arbitrary image regression. It is learning parameter corrections that remain physically grounded because the supervision passes through the simulator.

## Current State of the Project

### What already exists

The current codebase is still centered on **Coarse Error Learning**.

Implemented pieces include:

- ARTIST-based scenario generation and PAINT data handling.
- Multiple kinematic reconstructor variants in [src/artist_extensions/kinematic_reconstructors.py](/Users/alexandru/Master%20Thesis/master-thesis/src/artist_extensions/kinematic_reconstructors.py).
- Several coarse-learning experiment folders under [src](/Users/alexandru/Master%20Thesis/master-thesis/src), such as focal-spot, pixel-loss, alignment-loss, blur-ablation, and parameter-evaluation experiments.
- A real-data recovery benchmark in [src/synthetic_error_recovery/main.py](/Users/alexandru/Master%20Thesis/master-thesis/src/synthetic_error_recovery/main.py), which tests whether the coarse reconstructor can recover after known perturbations are injected into the parameters it optimizes.

### What is not implemented yet

There is currently **no dedicated Fine Error Learning module** in the codebase.

In particular, the following pieces still need to be built:

- A modular interface for FEL models.
- A training loop that freezes or reuses KR outputs and only trains the residual learner.
- A residual-parameter representation that plugs cleanly into ARTIST before ray tracing.
- Single-heliostat masking and training logic for the first FEL stage.
- Later, multi-heliostat fine-tuning logic for field-level objectives.

### Current empirical situation

The coarse reconstructor is useful, but it is not sufficient on its own.

Based on the current thesis notes and experiments:

- The present KR pipeline appears to plateau around roughly **4.5 mrad mean error** on the analysed benchmark runs.
- Some optimizer and training issues have already been identified, such as convergence quality, scheduler usage, checkpointing, overfitting on very limited training samples, and weak identifiability for some parameter groups.
- Even if KR is improved further, the broader thesis direction assumes that **rigid-body geometry alone will not close the final accuracy gap**.

That is the main justification for the Fine Error Learning stage.

## Conceptual Role of Fine Error Learning

The Fine Error Learning module is meant to learn the part of the pointing error that the coarse geometric model does not explain well.

Examples of effects that may end up in this residual include:

- Model mismatch in the kinematic parameterization.
- Residual actuator effects not captured by the current rigid formulation.
- Unmodeled structural or optical effects that manifest consistently in the measured flux.
- Input-dependent residual behavior that changes with sun position, heliostat state, or operating region.

The key idea is not to replace the kinematic reconstructor. The key idea is to use KR as a physically meaningful baseline and then learn only the remaining correction.

## Why the Pipeline Must Be Modular

Your supervisor's recommendation is structurally correct: before introducing a large model such as a transformer, the full pipeline should be made modular and testable.

That modularity should separate at least these concerns:

1. **Baseline provider**
   - Supplies $\theta_{KR}$.
   - Can come from a stored KR result or from an on-the-fly KR stage.

2. **Residual model**
   - Predicts $\Delta \theta$.
   - Should be swappable, so a polynomial model and a transformer can use the same training and evaluation interfaces.

3. **Parameter combiner**
   - Forms $\theta_{KR} + \Delta \theta$.
   - Applies corrections in a controlled and explicit way.

4. **Renderer / simulator**
   - Runs differentiable ray tracing.

5. **Loss layer**
   - Compares predicted and measured flux.

6. **Training orchestration**
   - Handles masking, batching, logging, checkpoints, and evaluation.

If these pieces are modular from the start, the first polynomial baseline and the later transformer become model choices rather than architectural rewrites.

## Immediate Next Step: Polynomial Fine Error Learning

Before implementing the transformer, the first FEL version should be a **simple polynomial residual model**.

This is not the intended final method. It is a controlled intermediate step to validate that:

- the FEL data flow is correct,
- the residual parameters are injected into ARTIST correctly,
- gradients propagate through the full pipeline,
- the loss decreases for the right reasons,
- the masking and training logic are sound,
- and the optimization behaves as expected.

### What the polynomial model should do

The polynomial model should take the same kind of inputs that the later FEL model will use, but in the simplest differentiable form possible.

Depending on the design choice, these inputs may include:

- sun position,
- heliostat identity or geometry descriptors,
- motor state,
- coarse KR parameters,
- and possibly compact image-derived features if needed later.

The output should remain a residual correction vector $\Delta \theta$ with the same semantic meaning that the future transformer will predict.

### Why this polynomial stage matters

This stage is valuable because it answers infrastructure questions before model-capacity questions.

If a simple polynomial learner cannot train correctly, then introducing a transformer would only make debugging harder. The polynomial baseline should therefore be treated as a **pipeline validation model**.

### What success looks like for the polynomial stage

The polynomial stage does not need to be state of the art. It only needs to demonstrate that:

- residual corrections can be learned end-to-end,
- the predicted $\Delta \theta$ produces meaningful changes in the rendered flux,
- gradients remain stable,
- training is reproducible,
- and evaluation outputs are interpretable.

If those conditions hold, then the pipeline is ready for a higher-capacity model.

## Later Stage: Transformer-Based Fine Error Learning

Once the modular residual-learning pipeline is verified with the polynomial model, the next model should be a transformer.

The transformer is the more ambitious FEL model because it can potentially learn richer dependencies such as:

- nonlinear coupling between sun position and residual correction,
- shared patterns across heliostats,
- structured relationships between motor state, geometry, and flux behavior,
- and more complex residual patterns than a low-order polynomial can represent.

In that sense, the transformer is the **capacity upgrade**, not the **pipeline validation step**.

The sequence should therefore be:

1. Build the modular FEL pipeline.
2. Validate it with a polynomial residual model.
3. Replace only the residual model with a transformer.
4. Reuse the same renderer, losses, logging, and evaluation flow.

## Recommended Engineering Direction

For implementation, the cleanest structure is:

1. Define a common residual-model interface.
   - Input: FEL features.
   - Output: residual correction vector $\Delta \theta$.

2. Implement a first residual model using polynomial basis functions.
   - Keep it small, deterministic, and easy to debug.

3. Build the parameter-injection layer.
   - Convert model outputs into the exact parameter tensors expected by ARTIST.

4. Build a dedicated FEL training script.
   - Freeze KR.
   - Train only the residual learner.
   - Start with one-heliostat masking.

5. Reuse existing evaluation and plotting patterns where possible.
   - JSON summaries
   - stage-wise plots
   - reconstruction metrics in mrad

6. Only after the polynomial version is stable, add the transformer residual model.

## Practical Thesis Positioning

The thesis now has a clear story:

- **Stage A:** build and analyse the coarse kinematic reconstructor.
- **Stage B:** verify coarse-stage behavior with recovery-style experiments.
- **Stage C:** build a modular residual-learning pipeline.
- **Stage D:** validate the FEL pipeline with a polynomial model.
- **Stage E:** replace the polynomial residual learner with a transformer.

This makes the development path defensible:

- first establish physical calibration,
- then verify recovery and gradient behavior,
- then validate the residual-learning infrastructure,
- and only then introduce a larger neural model.

## Current Goal

The immediate goal is **not** to jump directly to the transformer.

The immediate goal is to build the first correct, modular, end-to-end Fine Error Learning pipeline, using a polynomial residual model as the initial learner.

Once that works, the transformer becomes a contained model replacement rather than a high-risk first implementation.

## One-Sentence Version for Future Chats

This thesis uses ARTIST and PAINT to build a coarse-to-fine heliostat calibration pipeline where the existing kinematic reconstructor provides coarse parameter corrections, and the next task is to implement a modular Fine Error Learning stage that first uses a polynomial residual model to validate the end-to-end differentiable pipeline before upgrading that residual model to a transformer.