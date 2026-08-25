"""
Data loading for fine_error_learning.

One training sample = one heliostat = its set of calibration measurements.
The raw per-measurement tensors come from ``SyntheticDatasetParser``
(``src/utils/synth_data.py``); this module adds:

  - discovery of heliostat IDs from a completed stage-1 run directory,
  - per-heliostat measurement containers (raw values kept for ray tracing),
  - scalar-feature standardization with train-split statistics
    (mean/std stored in ``scaler_stats.json`` next to the outputs),
  - token construction: subsample or zero-pad the measurement set to
    ``K`` tokens with a boolean mask (True = real measurement).
"""
from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass

import torch

from artist.util import indices as index_mapping

from utils.synth_data import SyntheticDatasetParser

log = logging.getLogger(__name__)

# Scalar features per measurement: incident-ray xyz (3) + motor position (2)
# + aim-target center ENU (3). The target matters because measurements of one
# heliostat are spread over several target areas (indices 0/1/2 in the
# balanced dataset) — without it the encoder cannot tell which target a
# measurement was aimed at.
N_SCALARS = 8


@dataclass
class HeliostatMeasurements:
    """All measurements of one heliostat of one split (raw, unstandardized)."""

    heliostat_id: str
    flux: torch.Tensor            # [N, 256, 256] float32 in [0, 1]
    focal_spots: torch.Tensor     # [N, 4] absolute ENU centroid of the measured flux
    incident_rays: torch.Tensor   # [N, 4] sun direction (homogeneous)
    motor_positions: torch.Tensor # [N, 2] recorded encoder ticks
    target_indices: torch.Tensor  # [N] long
    target_centers: torch.Tensor  # [N, 3] absolute ENU center of the aim target

    @property
    def n(self) -> int:
        return self.flux.shape[0]

    def capped(self, max_measurements: int | None) -> "HeliostatMeasurements":
        """Return a copy truncated to the first ``max_measurements`` (None = all)."""
        if max_measurements is None or self.n <= max_measurements:
            return self
        return HeliostatMeasurements(
            heliostat_id=self.heliostat_id,
            flux=self.flux[:max_measurements],
            focal_spots=self.focal_spots[:max_measurements],
            incident_rays=self.incident_rays[:max_measurements],
            motor_positions=self.motor_positions[:max_measurements],
            target_indices=self.target_indices[:max_measurements],
            target_centers=self.target_centers[:max_measurements],
        )

    def raw_scalars(self) -> torch.Tensor:
        """[N, 8] = incident-ray xyz + motor positions + target center ENU."""
        return torch.cat(
            [self.incident_rays[:, :3], self.motor_positions, self.target_centers],
            dim=-1,
        )


def discover_heliostat_ids(stage1_checkpoint_dir: pathlib.Path | str) -> list[str]:
    """Heliostat IDs = sub-directories of the stage-1 run holding a checkpoint."""
    root = pathlib.Path(stage1_checkpoint_dir)
    hids = sorted(p.name for p in root.iterdir() if (p / "stage1_checkpoint.pt").exists())
    if not hids:
        raise FileNotFoundError(f"No stage1_checkpoint.pt found under {root}")
    return hids


def load_measurements(
    data_dir: pathlib.Path | str,
    heliostat_id: str,
    heliostat_group,
    scenario,
    device: torch.device,
) -> HeliostatMeasurements | None:
    """Load one split of one heliostat via ``SyntheticDatasetParser``.

    ``data_dir`` is the split directory (``.../synthetic_data/train`` etc.).
    Returns None when the heliostat has no data in this split.
    """
    data_dir = pathlib.Path(data_dir)
    hel_dir = data_dir / heliostat_id
    if not hel_dir.exists():
        return None
    n_samples = sum(1 for d in hel_dir.iterdir() if d.is_dir() and d.name.isdigit())
    if n_samples == 0:
        return None

    parser = SyntheticDatasetParser(data_dir)
    mapping = [(heliostat_id, list(range(n_samples)), list(range(n_samples)))]
    flux, focal_spots, incident_rays, motor_positions, _, target_mask = (
        parser.parse_data_for_reconstruction(
            heliostat_data_mapping=mapping,
            heliostat_group=heliostat_group,
            scenario=scenario,
            device=device,
        )
    )
    # Aim-target center per measurement (same indexing as the ray tracer /
    # warm-start diagnostic: target_area_index → planar target areas).
    planar = scenario.solar_tower.target_areas[index_mapping.planar_target_areas]
    target_centers = planar.centers[target_mask, :3].float()
    return HeliostatMeasurements(
        heliostat_id=heliostat_id,
        flux=flux,
        focal_spots=focal_spots,
        incident_rays=incident_rays,
        motor_positions=motor_positions,
        target_indices=target_mask,
        target_centers=target_centers,
    )


# ---------------------------------------------------------------------------
# Scalar standardization (train-split statistics)
# ---------------------------------------------------------------------------

def compute_scaler(measurements: list[HeliostatMeasurements]) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean/std [8] of the raw scalar features over all given measurements."""
    scalars = torch.cat([m.raw_scalars() for m in measurements], dim=0)
    mean = scalars.mean(dim=0)
    std = scalars.std(dim=0).clamp(min=1e-6)
    return mean, std


def save_scaler(mean: torch.Tensor, std: torch.Tensor, path: pathlib.Path) -> None:
    with open(path, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)


def load_scaler(path: pathlib.Path) -> tuple[torch.Tensor, torch.Tensor]:
    with open(path) as f:
        stats = json.load(f)
    return (
        torch.tensor(stats["mean"], dtype=torch.float32),
        torch.tensor(stats["std"], dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Token construction
# ---------------------------------------------------------------------------

def build_model_tokens(
    measurements: HeliostatMeasurements,
    k: int,
    scaler_mean: torch.Tensor,
    scaler_std: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Subsample or pad the measurement set to ``k`` tokens.

    Returns
    -------
    flux    : [k, 256, 256] — zero rows for padding
    scalars : [k, 5]        — standardized; zero rows for padding
    mask    : [k] bool      — True for real measurements, False for padding

    When the heliostat has more than ``k`` measurements, a random subset is
    drawn (deterministic when ``generator`` is seeded; pass None for the first
    ``k`` in file order, e.g. for validation).
    """
    n = measurements.n
    device = measurements.flux.device
    if n >= k:
        if generator is None:
            idx = torch.arange(k, device=device)
        else:
            idx = torch.randperm(n, generator=generator, device=device)[:k]
        mask = torch.ones(k, dtype=torch.bool, device=device)
        flux = measurements.flux[idx]
        scalars = measurements.raw_scalars()[idx]
    else:
        mask = torch.zeros(k, dtype=torch.bool, device=device)
        mask[:n] = True
        flux = torch.zeros(k, *measurements.flux.shape[1:], device=device)
        flux[:n] = measurements.flux
        scalars = torch.zeros(k, N_SCALARS, device=device)
        scalars[:n] = measurements.raw_scalars()

    scalars = (scalars - scaler_mean.to(device)) / scaler_std.to(device)
    # Keep padding rows exactly zero after standardization.
    scalars = scalars * mask.unsqueeze(-1)
    return flux, scalars, mask
