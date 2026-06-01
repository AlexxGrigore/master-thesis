# Full Training Pipeline Plan

## Goal

Build one reusable pipeline that starts from coarse kinematic learning and then adds fine residual correction on top of it. The immediate target is a simple, controllable first version, not the final transformer-based system.

## Current State

- `coarse_learning_parameters/kinematic_parameters.json` contains old focal-spot-trained kinematic parameters and can be used as the initial coarse stage artifact.
- The recovery benchmark in `src/experiments/synthetic_error_recovery/` is the current sanity check for whether coarse parameter errors can be re-learned.
- The next step is not new ARTIST code. ARTIST should remain the differentiable simulator used inside the training loop.

## First Pipeline Version

1. Load frozen coarse kinematic parameters for each heliostat.
2. Run ARTIST with those parameters to obtain the coarse prediction.
3. Build fine-learning inputs from the observation context.
4. Predict a residual correction in the same parameter space that KR already optimizes.
5. Add the residual correction to the frozen coarse parameters.
6. Re-run ARTIST with the corrected parameters.
7. Train the fine model end-to-end through the final image-space loss.

## Recommended FEL V1 Design

- Model: shared linear residual model.
- Inputs: sun direction in 3D plus the features needed to condition the residual prediction.
- Output: the same parameter set currently optimized by the chosen KR variant.
- Training rule: coarse parameters stay frozen during FEL training.
- Objective: reduce final focal-spot error, not parameter error directly.

## Suggested Implementation Order

1. Define the exact KR parameter vector that FEL is allowed to correct.
2. Add a loader that reads `coarse_learning_parameters/kinematic_parameters.json` into a structured parameter representation.
3. Create a small dataset builder that returns per-sample FEL inputs, targets, and metadata.
4. Implement a minimal shared linear residual model.
5. Implement one training loop that does coarse forward pass, residual correction, corrected forward pass, and loss backpropagation.
6. Add evaluation outputs comparable to the current recovery benchmark.
7. Only after the linear model is stable, replace it with a higher-capacity model such as an MLP or transformer.

## Proposed Folder Direction

- `data.py`: load FEL samples and metadata.
- `features.py`: build sun and context features.
- `model.py`: linear residual model first, larger models later.
- `pipeline.py`: compose coarse parameters, residual correction, and ARTIST forward pass.
- `train.py`: training entrypoint.
- `evaluate.py`: validation, plots, and summaries.
- `config.py`: shared constants and experiment settings.

## What To Keep Separate

- `src/experiments/`: one-off studies, ablations, smoke tests, recovery benchmark.
- `src/full_training_pipeline/`: reusable coarse-to-fine training code intended to become the main thesis pipeline.

## Immediate Next Tasks

1. Decide the exact FEL-corrected parameter vector.
2. Convert the stored coarse parameters into a clean loader API.
3. Implement the linear residual baseline end-to-end.
4. Reuse the recovery benchmark metrics and plots for FEL evaluation.