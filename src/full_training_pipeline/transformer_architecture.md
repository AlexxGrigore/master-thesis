# FEL Transformer Architecture

## Overview

The transformer model replaces the linear/polynomial residual models with a sequence-aware architecture that processes each heliostat's calibration measurements as a **set of tokens**, letting every measurement attend to every other before predicting the kinematic correction.

```
[N_meas calibrations per heliostat]
           │
    Token Construction          → [N_meas, d_model]
           │
    Transformer Encoder         → [N_meas, d_model]
           │
    Mean Pooling                → [d_model]
           │
    Concatenate global features → [d_model + 23]
           │
    Linear → tanh × bounds      → Δθ  (20D)
           │
    coarse θ + Δθ → ARTIST → loss
```

The model is **shared across all heliostats**: one set of weights is trained on every heliostat in the field simultaneously, just as with the linear baseline.

---

## Stage 1 — Token Construction

Each calibration measurement is encoded into a single `d_model`-dimensional token by fusing two modalities.

### Scalar token

Seven scalar features are extracted per measurement:

| Feature | Dim | Description |
|---|---|---|
| `sun_directions` | 3 | Unit vector pointing toward the sun (ENU) |
| `motor_positions` | 2 | Encoder readings for axis 1 and axis 2 |
| `centroids` | 3 | ENU position of the flux centroid on the receiver |
| **Total** | **8** | |

These are projected linearly into the token space:

```
scalar_token = Linear(8 → d_model)    shape: (N_meas, d_model)
```

### Image token (optional)

When flux images are available (`flux_images` is not `None`), each grayscale 256×256 image is encoded by a shared CNN:

```
Conv2d(1→16,  k=3, stride=2, pad=1) → ReLU    # 256×256 → 128×128
Conv2d(16→32, k=3, stride=2, pad=1) → ReLU    # 128×128 → 64×64
Conv2d(32→64, k=3, stride=2, pad=1) → ReLU    # 64×64   → 32×32
AdaptiveAvgPool2d(1)                           # 32×32   → [64]
Linear(64 → d_model)                           # shape: (N_meas, d_model)
```

`AdaptiveAvgPool2d` makes the CNN resolution-agnostic. The **CNN is shared** across all measurements and all heliostats — one set of weights learns a general flux-pattern encoder, rather than overfitting to individual images.

### Token fusion

The two tokens are **added** (not concatenated):

```
token_i = scalar_token_i + image_token_i    shape: (N_meas, d_model)
```

If no flux images are available, the image token is simply omitted:

```
token_i = scalar_token_i
```

**No positional encoding** is applied. Calibration order is arbitrary (measurements are taken on different days in no fixed order), and sun position — the most informative "position" signal — is already explicitly present in the scalar features.

---

## Stage 2 — Transformer Encoder

A standard `torch.nn.TransformerEncoder` with `n_layers` stacked layers. Each layer contains:

### Multi-head self-attention

```
Q = token × W_Q    (N_meas, d_model)
K = token × W_K    (N_meas, d_model)
V = token × W_V    (N_meas, d_model)

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) × V
```

The `(N_meas × N_meas)` attention matrix lets every calibration attend to every other. The model can learn, for example, that measurements taken at similar sun elevations produce correlated residuals, or that disagreement between two measurements signals a specific kinematic fault.

### Position-wise feed-forward network

```
FFN(x) = Linear(d_model → d_ff) → ReLU → Linear(d_ff → d_model)
```

### Sub-layer wrappers

Each sub-layer uses residual connections, layer normalisation, and dropout:

```
x = LayerNorm(x + Dropout(MultiHeadAttention(x)))
x = LayerNorm(x + Dropout(FFN(x)))
```

**Output:** `(N_meas, d_model)` — each token is now context-aware, having attended to all other calibrations for this heliostat.

---

## Stage 3 — Pooling

Mean pooling collapses the variable-length sequence into a single fixed-size heliostat representation:

```
h = mean(encoder_output, dim=0)    shape: (d_model,)
```

This vector summarises the entire calibration history of the heliostat. Mean pooling was chosen over CLS-token pooling because it treats every measurement equally and is robust to different numbers of available calibrations.

---

## Stage 4 — Output Head

The global heliostat-level features are concatenated to the pooled representation:

```
global = cat(heliostat_position (3,), kinematic_params (20,))    shape: (23,)
combined = cat(h, global)                                         shape: (d_model + 23,)
```

The combined vector is then projected to the 20D kinematic correction:

```
Δθ = tanh(Linear(d_model + 23 → 20)) × residual_bounds
```

The **output linear layer is zero-initialised** (both weight and bias). This guarantees that at epoch 0, every heliostat receives `Δθ = 0`, so training starts exactly at the coarse checkpoint — identical behaviour to the linear and polynomial baselines.

`residual_bounds` is a fixed buffer (not a learnable parameter) that clamps each correction dimension to a physically meaningful range.

---

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `d_model` | 64 | Token and encoder hidden dimension |
| `n_heads` | 4 | Attention heads |
| `n_layers` | 2 | Stacked transformer encoder layers |
| `d_ff` | 128 | FFN hidden dimension |
| `dropout` | 0.1 | Dropout rate inside attention and FFN |

With these defaults the model has **~96,700 learnable parameters**, compared to 860 for the linear baseline. The suggested exploration ranges are `d_model ∈ {64, 128}`, `n_layers ∈ {2, 4}`, `d_ff ∈ {128, 256}`.

---

## Implementation Notes

### File layout

| File | Contents |
|---|---|
| `model_transformer.py` | `_FluxImageCNN`, `SharedTransformerResidualModel` |
| `model.py` | `build_residual_model("transformer")` factory |
| `data.py` | `_load_flux_images`, `load_flux_images` flag in all builders |
| `config.py` | `load_flux_images: bool`, auto-set `True` when `model_type == "transformer"` |

### Forward pass signature

`SharedTransformerResidualModel.forward()` is **interface-compatible** with `SharedLinearResidualModel`:

```python
def forward(self, inputs: list[HeliostatCalibrationInput | None]) -> torch.Tensor:
    # returns (N_heliostats, 20)
```

Each heliostat is processed independently (no cross-heliostat attention). `None` entries (heliostats with no calibration data) produce a zero correction vector.

### Flux image loading

Flux images are loaded on demand by `_load_flux_images` in `data.py` using PIL, converted to grayscale `L` mode, and normalised to `[0, 1]` float32. Loading is gated by `config.load_flux_images`, which is automatically `True` when `model_type == "transformer"` and `False` for all other model types — so non-transformer runs pay zero I/O cost.

### Training

Run with:

```bash
python main.py --model-type transformer --dataset-type synthetic
```

The JSON checkpoint export (used by the linear model for human-readable weight inspection) is skipped for the transformer; only the `.pt` binary checkpoint is saved. The response-curve and feature-importance plots (which rely on `_select_and_flatten` and a linear head) are also skipped.
