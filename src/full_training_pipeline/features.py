from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class HeliostatCalibrationInput:
    heliostat_name: str
    kinematic_params: torch.Tensor      # (20,)  from coarse checkpoint
    heliostat_position: torch.Tensor    # (3,)   ENU absolute position
    sun_directions: torch.Tensor        # (N_meas, 3)  unit vectors pointing TO sun
    motor_positions: torch.Tensor       # (N_meas, 2)
    centroids: torch.Tensor             # (N_meas, 3)  ENU centroid on receiver
    flux_images: torch.Tensor | None    # (N_meas, H, W)  unused by linear model


@dataclass
class CalibrationNormStats:
    """Z-score normalisation statistics computed from the training split."""
    sun_mean: torch.Tensor          # (3,)
    sun_std: torch.Tensor           # (3,)
    motor_mean: torch.Tensor        # (2,)
    motor_std: torch.Tensor         # (2,)
    centroid_mean: torch.Tensor     # (3,)
    centroid_std: torch.Tensor      # (3,)
    heliostat_pos_mean: torch.Tensor  # (3,)
    heliostat_pos_std: torch.Tensor   # (3,)
    kinematic_mean: torch.Tensor    # (20,)
    kinematic_std: torch.Tensor     # (20,)

    def to_dict(self) -> dict[str, list[float]]:
        return {k: v.tolist() for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(d: dict[str, list[float]]) -> CalibrationNormStats:
        return CalibrationNormStats(**{k: torch.tensor(v, dtype=torch.float32) for k, v in d.items()})


def sun_angles_to_unit_vector(sun_elevation_deg: float, sun_azimuth_deg: float) -> np.ndarray:
    elevation_rad = math.radians(sun_elevation_deg)
    azimuth_rad = math.radians(sun_azimuth_deg)
    cos_elevation = math.cos(elevation_rad)
    return np.asarray(
        [
            cos_elevation * math.cos(azimuth_rad),
            cos_elevation * math.sin(azimuth_rad),
            math.sin(elevation_rad),
        ],
        dtype=np.float32,
    )
