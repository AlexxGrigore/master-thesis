from __future__ import annotations

import torch
import torch.nn as nn

from full_training_pipeline.features import HeliostatCalibrationInput
from full_training_pipeline.model import (
    SHARED_WORTBERG_PARAMETER_NAMES,
    SHARED_WORTBERG_RESIDUAL_BOUNDS,
)

# Scalar features per measurement: sun_xyz (3) + motor (2) + centroid_ENU (3)
_N_SCALARS = 8
# Global heliostat features concatenated after pooling: position (3) + kinematic params (20)
_N_GLOBAL = 23
# CNN output channels before linear projection
_CNN_OUT_CHANNELS = 64


class _FluxImageCNN(nn.Module):
    """
    Shared CNN encoder for grayscale flux images.

    Architecture (for a 256×256 input):
        Conv2d(1→16,  k=3, stride=2, pad=1) → ReLU   → 128×128
        Conv2d(16→32, k=3, stride=2, pad=1) → ReLU   → 64×64
        Conv2d(32→64, k=3, stride=2, pad=1) → ReLU   → 32×32
        AdaptiveAvgPool2d(1)                           → [64]
        Linear(64 → d_model)

    AdaptiveAvgPool2d handles arbitrary input sizes gracefully.
    One set of weights shared across all measurements and all heliostats.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, _CNN_OUT_CHANNELS, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(_CNN_OUT_CHANNELS, d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (N, H, W) float32, values normalised to [0, 1]
        Returns:
            (N, d_model)
        """
        x = images.unsqueeze(1)  # (N, 1, H, W)
        x = self.conv(x)         # (N, 64, 1, 1)
        x = x.flatten(1)         # (N, 64)
        return self.proj(x)      # (N, d_model)


class SharedTransformerResidualModel(nn.Module):
    """
    Transformer-based residual model for heliostat kinematic correction.

    Per heliostat, every calibration measurement is encoded into a d_model-dimensional
    token by fusing two modalities:
      - Scalar features (sun_xyz, motor, centroid_ENU) → Linear(8 → d_model)
      - Flux image (H×W) → shared CNN → Linear(64 → d_model)
    The two tokens are summed: token_i = scalar_token_i + image_token_i.
    When flux images are unavailable (flux_images is None), only the scalar token is used.

    A standard multi-head self-attention transformer encoder then lets every measurement
    attend to every other.  Mean pooling collapses the sequence to a single d_model-vector.
    Global heliostat features (position + coarse kinematic params, 23D total) are
    concatenated before the output linear projects to the 20D correction.

    Zero-weight initialisation on the output head ensures Δθ = 0 at epoch 0, so training
    starts exactly at the coarse checkpoint.

    forward() signature is identical to SharedLinearResidualModel:
        inputs  : list[HeliostatCalibrationInput | None]  (one per heliostat)
        returns : (N_heliostats, 20) correction tensor

    Hyperparameters:
        d_model   : token and encoder hidden dimension (suggested: 64–128)
        n_heads   : attention heads (suggested: 4–8)
        n_layers  : encoder layers (suggested: 2–4)
        d_ff      : FFN hidden dimension (suggested: 128–256)
        dropout   : dropout rate (suggested: 0.1)
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Token construction
        self.scalar_proj = nn.Linear(_N_SCALARS, d_model)
        self.cnn = _FluxImageCNN(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,  # (batch, seq, d_model)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head: (d_model + 23) → 20
        self.output_head = nn.Linear(
            d_model + _N_GLOBAL, len(SHARED_WORTBERG_PARAMETER_NAMES)
        )
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

        self.register_buffer("residual_bounds", SHARED_WORTBERG_RESIDUAL_BOUNDS.clone())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_tokens(self, inp: HeliostatCalibrationInput) -> torch.Tensor:
        """Return (N_meas, d_model) token sequence for one heliostat."""
        device = self.residual_bounds.device
        sun = inp.sun_directions.to(device)     # (N, 3)
        motor = inp.motor_positions.to(device)  # (N, 2)
        centroid = inp.centroids.to(device)     # (N, 3)

        scalars = torch.cat([sun, motor, centroid], dim=-1)  # (N, 8)
        token = self.scalar_proj(scalars)                    # (N, d_model)

        if inp.flux_images is not None:
            images = inp.flux_images.to(device)    # (N, H, W)
            token = token + self.cnn(images)       # (N, d_model)

        return token

    def _global_features(self, inp: HeliostatCalibrationInput) -> torch.Tensor:
        """Return (23,) = heliostat_position (3) + kinematic_params (20)."""
        device = self.residual_bounds.device
        return torch.cat([
            inp.heliostat_position.to(device),
            inp.kinematic_params.to(device),
        ])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs: list[HeliostatCalibrationInput | None]) -> torch.Tensor:
        device = self.residual_bounds.device
        n_params = len(SHARED_WORTBERG_PARAMETER_NAMES)
        rows: list[torch.Tensor] = []

        for inp in inputs:
            if inp is None:
                rows.append(torch.zeros(n_params, device=device))
                continue

            tokens = self._build_tokens(inp)              # (N, d_model)
            encoded = self.transformer(tokens.unsqueeze(0))   # (1, N, d_model)
            pooled = encoded.squeeze(0).mean(dim=0)       # (d_model,)

            combined = torch.cat([pooled, self._global_features(inp)])  # (d_model+23,)
            correction = torch.tanh(self.output_head(combined)) * self.residual_bounds
            rows.append(correction)

        return torch.stack(rows, dim=0)  # (N_heliostats, 20)
