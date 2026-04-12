from __future__ import annotations

import math
from typing import Iterable

import numpy as np

SAMPLE_FEATURE_NAMES: tuple[str, ...] = (
    "sun_x",
    "sun_y",
    "sun_z",
    "axis_1_motor_position",
    "axis_2_motor_position",
)

SUMMARY_FEATURE_NAMES: tuple[str, ...] = (
    "mean_sun_x",
    "mean_sun_y",
    "mean_sun_z",
    "mean_axis_1_motor_position",
    "mean_axis_2_motor_position",
    "std_sun_x",
    "std_sun_y",
    "std_sun_z",
    "std_axis_1_motor_position",
    "std_axis_2_motor_position",
    "sample_count",
)


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


def build_sample_feature_vector(calibration_metadata: dict[str, object]) -> np.ndarray:
    motor_position = calibration_metadata["motor_position"]
    sun_direction = sun_angles_to_unit_vector(
        sun_elevation_deg=float(calibration_metadata["sun_elevation"]),
        sun_azimuth_deg=float(calibration_metadata["sun_azimuth"]),
    )
    return np.asarray(
        [
            sun_direction[0],
            sun_direction[1],
            sun_direction[2],
            float(motor_position["axis_1_motor_position"]),
            float(motor_position["axis_2_motor_position"]),
        ],
        dtype=np.float32,
    )


def build_heliostat_feature_summary(
    calibration_metadata_items: Iterable[dict[str, object]],
) -> np.ndarray:
    sample_vectors = np.stack(
        [build_sample_feature_vector(metadata) for metadata in calibration_metadata_items],
        axis=0,
    )
    mean_features = sample_vectors.mean(axis=0)
    std_features = sample_vectors.std(axis=0)
    sample_count = np.asarray([sample_vectors.shape[0]], dtype=np.float32)
    return np.concatenate([mean_features, std_features, sample_count], axis=0).astype(np.float32)