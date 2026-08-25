"""
Shared transformer residual model for fine_error_learning.

One shared model across all heliostats. Per heliostat, each calibration
measurement becomes one token:

    token_i = Linear( concat( CNN(flux_i) [d_img], scalars_i [8] ) )  →  d_model

(with ``use_flux=False`` the CNN is skipped and the image features are zero —
the images-on/off ablation is a config flag, not a code edit).

A pre-norm ``nn.TransformerEncoder`` with NO positional encoding (permutation
invariance by design) processes the token set; ``src_key_padding_mask`` and
masked mean pooling handle heliostats with fewer than K real measurements.

The pooled vector is concatenated with the normalized warm-start vector θ_KR
(24) and the heliostat position (3), and an MLP predicts the 24-D correction
Δθ. The final layer is zero-initialized (Δθ = 0 at epoch 0, so training starts
exactly at the stage-1 warm start). Unbounded mode scales the raw output by
``output_gain × PARAMETER_SCALE`` — Adam's ~lr-sized first steps would
otherwise move Δθ by ~0.05 rad/m at once and throw the beam off target; a
tanh × bounds fallback (old scheme) is kept behind the ``bounded_head`` flag.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fine_error_learning import pipeline as fel_pipeline
from fine_error_learning.data import N_SCALARS

_CNN_OUT_CHANNELS = 64


class FluxCNN(nn.Module):
    """
    Lightweight shared CNN encoder for grayscale flux images.

    Architecture (for a 256×256 input):
        Conv2d(1→16,  k=3, stride=2, pad=1) → GELU   → 128×128
        Conv2d(16→32, k=3, stride=2, pad=1) → GELU   → 64×64
        Conv2d(32→64, k=3, stride=2, pad=1) → GELU   → 32×32
        AdaptiveAvgPool2d(1)                           → [64]
        Linear(64 → d_img)
    """

    def __init__(self, d_img: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, _CNN_OUT_CHANNELS, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(_CNN_OUT_CHANNELS, d_img)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (N, H, W) float32, values in [0, 1]
        Returns:
            (N, d_img)
        """
        x = images.unsqueeze(1)  # (N, 1, H, W)
        x = self.conv(x)         # (N, 64, 1, 1)
        x = x.flatten(1)         # (N, 64)
        return self.proj(x)      # (N, d_img)


class FelTransformerModel(nn.Module):
    """
    Set-transformer residual model: measurement set → 24-D Δθ.

    forward():
        flux       : [B, K, H, W] or None (when use_flux=False)
        scalars    : [B, K, 8]  — standardized incident-ray xyz + motor + target center
        mask       : [B, K] bool — True for real measurements, False for padding
        theta_kr   : [B, 24] — frozen warm-start vector (normalized internally)
        positions  : [B, 3]  — absolute heliostat ENU position (normalized internally)
        returns    : [B, 24] Δθ in the pipeline's parameter ordering
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        dropout: float = 0.1,
        d_img: int = 64,
        use_flux: bool = True,
        bounded_head: bool = False,
        output_gain: float = 0.01,
        query_decoder: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_img = d_img
        self.use_flux = use_flux
        self.bounded_head = bounded_head
        self.query_decoder = query_decoder

        # Token construction
        self.cnn = FluxCNN(d_img) if use_flux else None
        self.token_proj = nn.Linear(d_img + N_SCALARS, d_model)

        # Transformer encoder (pre-norm, batch-first, no positional encoding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Experiment B: query-conditioned decoder. A query token built from a
        # (standardized) sun direction cross-attends to the encoder memory →
        # Δθ conditioned on sun position instead of one static Δθ per heliostat.
        if query_decoder:
            self.query_proj = nn.Linear(3, d_model)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        # Output head: (d_model + 24 + 3) → 24, final layer zero-initialized
        self.head = nn.Sequential(
            nn.Linear(d_model + fel_pipeline.N_PARAMS + 3, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, fel_pipeline.N_PARAMS),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        self.register_buffer("param_scale", fel_pipeline.PARAMETER_SCALE.clone())
        self.register_buffer("residual_bounds", fel_pipeline.RESIDUAL_BOUNDS.clone())
        # Unbounded mode: Δθ = raw × output_scale. Adam takes ~lr-sized steps on
        # the head weights regardless of gradient magnitude, so without a small
        # gain the very first step already moves Δθ by ~0.05 rad/m — physically
        # enormous (the beam leaves the target) and training diverges. The gain
        # keeps per-step Δθ at ~1e-4 × scale while staying unbounded in range.
        self.register_buffer(
            "output_scale", fel_pipeline.PARAMETER_SCALE.clone() * output_gain
        )

    def _encode(
        self,
        flux: torch.Tensor | None,
        scalars: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Token construction + encoder. Returns memory [B, K, d_model]."""
        device = self.param_scale.device
        batch_size, k = scalars.shape[:2]

        if self.use_flux and flux is not None:
            image_features = self.cnn(flux.reshape(batch_size * k, *flux.shape[2:]))
            image_features = image_features.reshape(batch_size, k, self.d_img)
        else:
            image_features = scalars.new_zeros(batch_size, k, self.d_img)

        tokens = self.token_proj(torch.cat([image_features, scalars], dim=-1))
        # src_key_padding_mask: True marks positions to IGNORE (padding).
        return self.encoder(tokens, src_key_padding_mask=~mask.to(device))

    def _apply_head(self, combined: torch.Tensor) -> torch.Tensor:
        raw = self.head(combined)
        if self.bounded_head:
            return torch.tanh(raw) * self.residual_bounds
        return raw * self.output_scale

    def _globals(self, theta_kr: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        device = self.param_scale.device
        return torch.cat(
            [
                theta_kr.to(device) / self.param_scale,
                positions.to(device) / fel_pipeline.POSITION_SCALE_M,
            ],
            dim=-1,
        )

    def forward(
        self,
        flux: torch.Tensor | None,
        scalars: torch.Tensor,
        mask: torch.Tensor,
        theta_kr: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        device = self.param_scale.device
        encoded = self._encode(flux, scalars, mask)

        # Masked mean pooling over real tokens only.
        weights = mask.to(device).unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

        combined = torch.cat([pooled, self._globals(theta_kr, positions)], dim=-1)
        return self._apply_head(combined)

    def forward_queries(
        self,
        flux: torch.Tensor | None,
        scalars: torch.Tensor,
        mask: torch.Tensor,
        theta_kr: torch.Tensor,
        positions: torch.Tensor,
        query_sun: torch.Tensor,
    ) -> torch.Tensor:
        """Query-conditioned Δθ. ``query_sun``: [B, Q, 3] standardized sun
        directions. Returns [B, Q, 24]. Requires ``query_decoder=True``."""
        device = self.param_scale.device
        memory = self._encode(flux, scalars, mask)
        queries = self.query_proj(query_sun.to(device))
        decoded = self.decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=~mask.to(device),
        )  # [B, Q, d_model]
        b, q = decoded.shape[:2]
        globals_q = self._globals(theta_kr, positions).unsqueeze(1).expand(b, q, -1)
        combined = torch.cat([decoded, globals_q], dim=-1)
        return self._apply_head(combined)
