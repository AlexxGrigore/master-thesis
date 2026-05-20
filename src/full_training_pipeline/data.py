from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
from PIL import Image
import paint.util.paint_mappings as paint_mappings
import torch
from artist.util.utils import convert_wgs84_coordinates_to_local_enu

from full_training_pipeline.features import (
    CalibrationNormStats,
    HeliostatCalibrationInput,
    sun_angles_to_unit_vector,
)
from full_training_pipeline.pipeline import GroupParameterState, flatten_wortberg_parameter_vector
from utils.evaluation import build_heliostat_data_mapping


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------

def _build_heliostat_lookup(
    scenario,
    group_states: list[GroupParameterState],
) -> dict[str, tuple[int, int]]:
    """Map each heliostat name to (group_index, local_index_within_group)."""
    lookup: dict[str, tuple[int, int]] = {}
    for group_idx, group in enumerate(scenario.heliostat_field.heliostat_groups):
        for local_idx, name in enumerate(group.names):
            lookup[name] = (group_idx, local_idx)
    return lookup


def _get_heliostat_position(scenario, group_idx: int, local_idx: int) -> torch.Tensor:
    """Return the absolute ENU position (3,) of a heliostat."""
    group = scenario.heliostat_field.heliostat_groups[group_idx]
    return group.positions[local_idx, :3].detach().cpu().float()


def _get_kinematic_params(
    group_states: list[GroupParameterState], group_idx: int, local_idx: int
) -> torch.Tensor:
    """Return the flattened 20-D Wortberg parameter vector for one heliostat."""
    flat = flatten_wortberg_parameter_vector(group_states[group_idx])  # (N_heliostats, 20)
    return flat[local_idx].detach().cpu()


def _load_flux_images(flux_paths: list[pathlib.Path]) -> torch.Tensor:
    """Load grayscale PNG flux images; returns (N, H, W) float32 tensor with values in [0, 1]."""
    imgs = []
    for path in flux_paths:
        img = Image.open(path).convert("L")
        imgs.append(torch.from_numpy(np.array(img, dtype=np.float32)) / 255.0)
    return torch.stack(imgs, dim=0)


# ---------------------------------------------------------------------------
# Real PAINT data builder
# ---------------------------------------------------------------------------

def build_calibration_inputs_real(
    *,
    mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    sample_limit_per_heliostat: int,
    centroid_method: str,
    scenario,
    group_states: list[GroupParameterState],
    load_flux_images: bool = False,
) -> dict[str, HeliostatCalibrationInput]:
    lookup = _build_heliostat_lookup(scenario, group_states)
    power_plant_pos = scenario.power_plant_position  # (3,) WGS84 [lat, lon, alt]

    inputs: dict[str, HeliostatCalibrationInput] = {}
    for heliostat_name, cal_paths, flux_paths in mapping:
        if heliostat_name not in lookup:
            continue
        group_idx, local_idx = lookup[heliostat_name]
        paths = cal_paths[:sample_limit_per_heliostat]
        if not paths:
            continue

        sun_list, motor_list, centroid_wgs84_list = [], [], []
        for path in paths:
            with open(path) as f:
                meta = json.load(f)
            sun_list.append(
                sun_angles_to_unit_vector(
                    float(meta[paint_mappings.SUN_ELEVATION]),
                    float(meta[paint_mappings.SUN_AZIMUTH]),
                )
            )
            mp = meta[paint_mappings.MOTOR_POS_KEY]
            motor_list.append([
                float(mp[paint_mappings.AXIS1_MOTOR_SAVE]),
                float(mp[paint_mappings.AXIS2_MOTOR_SAVE]),
            ])
            centroid_wgs84_list.append(
                meta[paint_mappings.FOCAL_SPOT_KEY][centroid_method]
            )

        sun_directions = torch.tensor(np.stack(sun_list, axis=0), dtype=torch.float32)
        motor_positions = torch.tensor(motor_list, dtype=torch.float32)

        centroid_wgs84 = torch.tensor(centroid_wgs84_list, dtype=torch.float64)
        centroids = convert_wgs84_coordinates_to_local_enu(
            centroid_wgs84, power_plant_pos
        ).float()  # (N_meas, 3)

        flux_images = _load_flux_images(flux_paths) if load_flux_images and flux_paths else None
        inputs[heliostat_name] = HeliostatCalibrationInput(
            heliostat_name=heliostat_name,
            kinematic_params=_get_kinematic_params(group_states, group_idx, local_idx),
            heliostat_position=_get_heliostat_position(scenario, group_idx, local_idx),
            sun_directions=sun_directions,
            motor_positions=motor_positions,
            centroids=centroids,
            flux_images=flux_images,
        )
    return inputs


# ---------------------------------------------------------------------------
# Synthetic data builder
# ---------------------------------------------------------------------------

def _build_synth_mapping(
    split_dir: pathlib.Path,
    sample_limit_per_heliostat: int,
) -> list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]:
    mapping = []
    for hid_dir in sorted(split_dir.iterdir()):
        if not hid_dir.is_dir():
            continue
        cal_paths = sorted(hid_dir.glob("*/calibration_properties.json"))[:sample_limit_per_heliostat]
        flux_paths = sorted(hid_dir.glob("*/flux_image.png"))[:sample_limit_per_heliostat]
        n = min(len(cal_paths), len(flux_paths))
        if n > 0:
            mapping.append((hid_dir.name, cal_paths[:n], flux_paths[:n]))
    return mapping


def build_calibration_inputs_synth(
    *,
    mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    scenario,
    group_states: list[GroupParameterState],
    load_flux_images: bool = False,
) -> dict[str, HeliostatCalibrationInput]:
    lookup = _build_heliostat_lookup(scenario, group_states)

    inputs: dict[str, HeliostatCalibrationInput] = {}
    for heliostat_name, cal_paths, flux_paths in mapping:
        if heliostat_name not in lookup:
            continue
        group_idx, local_idx = lookup[heliostat_name]

        sun_list, motor_list, centroid_list = [], [], []
        for path in cal_paths:
            meta = json.loads(path.read_text())
            ray_dir = meta["incident_ray_direction"]  # [x, y, z, 0.0] — points FROM sun
            sun_list.append([-ray_dir[0], -ray_dir[1], -ray_dir[2]])  # flip → points TO sun
            motor_list.append([float(meta["motor_position"][0]), float(meta["motor_position"][1])])
            fse = meta["focal_spot_enu"]  # [E, N, U, 1.0]
            centroid_list.append([fse[0], fse[1], fse[2]])

        flux_images = _load_flux_images(flux_paths) if load_flux_images and flux_paths else None
        inputs[heliostat_name] = HeliostatCalibrationInput(
            heliostat_name=heliostat_name,
            kinematic_params=_get_kinematic_params(group_states, group_idx, local_idx),
            heliostat_position=_get_heliostat_position(scenario, group_idx, local_idx),
            sun_directions=torch.tensor(sun_list, dtype=torch.float32),
            motor_positions=torch.tensor(motor_list, dtype=torch.float32),
            centroids=torch.tensor(centroid_list, dtype=torch.float32),
            flux_images=flux_images,
        )
    return inputs


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def compute_norm_stats(
    calibration_inputs: dict[str, HeliostatCalibrationInput],
) -> CalibrationNormStats:
    def _safe_std(t: torch.Tensor) -> torch.Tensor:
        s = t.std(dim=0)
        return torch.where(s < 1e-6, torch.ones_like(s), s)

    all_sun = torch.cat([inp.sun_directions for inp in calibration_inputs.values()], dim=0)
    all_motor = torch.cat([inp.motor_positions for inp in calibration_inputs.values()], dim=0)
    all_centroid = torch.cat([inp.centroids for inp in calibration_inputs.values()], dim=0)
    all_helpos = torch.stack([inp.heliostat_position for inp in calibration_inputs.values()], dim=0)
    all_kp = torch.stack([inp.kinematic_params for inp in calibration_inputs.values()], dim=0)

    return CalibrationNormStats(
        sun_mean=all_sun.mean(dim=0),
        sun_std=_safe_std(all_sun),
        motor_mean=all_motor.mean(dim=0),
        motor_std=_safe_std(all_motor),
        centroid_mean=all_centroid.mean(dim=0),
        centroid_std=_safe_std(all_centroid),
        heliostat_pos_mean=all_helpos.mean(dim=0),
        heliostat_pos_std=_safe_std(all_helpos),
        kinematic_mean=all_kp.mean(dim=0),
        kinematic_std=_safe_std(all_kp),
    )


def normalize_calibration_inputs(
    calibration_inputs: dict[str, HeliostatCalibrationInput],
    norm_stats: CalibrationNormStats,
) -> dict[str, HeliostatCalibrationInput]:
    ns = norm_stats
    normalized: dict[str, HeliostatCalibrationInput] = {}
    for name, inp in calibration_inputs.items():
        normalized[name] = HeliostatCalibrationInput(
            heliostat_name=inp.heliostat_name,
            kinematic_params=(inp.kinematic_params - ns.kinematic_mean) / ns.kinematic_std,
            heliostat_position=(inp.heliostat_position - ns.heliostat_pos_mean) / ns.heliostat_pos_std,
            sun_directions=(inp.sun_directions - ns.sun_mean) / ns.sun_std,
            motor_positions=(inp.motor_positions - ns.motor_mean) / ns.motor_std,
            centroids=(inp.centroids - ns.centroid_mean) / ns.centroid_std,
            flux_images=inp.flux_images,
        )
    return normalized


# ---------------------------------------------------------------------------
# Group-level structure (for pipeline)
# ---------------------------------------------------------------------------

def build_group_calibration_inputs(
    scenario,
    calibration_inputs: dict[str, HeliostatCalibrationInput],
) -> list[list[HeliostatCalibrationInput | None]]:
    """Return one list per heliostat group; None for heliostats with no calibration data."""
    result: list[list[HeliostatCalibrationInput | None]] = []
    for group in scenario.heliostat_field.heliostat_groups:
        result.append([calibration_inputs.get(name) for name in group.names])
    return result


# ---------------------------------------------------------------------------
# SplitDataBundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitDataBundle:
    split: str
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
    calibration_inputs: dict[str, HeliostatCalibrationInput]  # normalised
    norm_stats: CalibrationNormStats


# ---------------------------------------------------------------------------
# Top-level bundle builders
# ---------------------------------------------------------------------------

def build_split_bundle(
    *,
    benchmark_csv: pathlib.Path,
    calibration_properties_dir: pathlib.Path,
    flux_image_dir: pathlib.Path,
    split: str,
    sample_limit_per_heliostat: int,
    centroid_method: str,
    scenario,
    group_states: list[GroupParameterState],
    norm_stats: CalibrationNormStats | None = None,
    load_flux_images: bool = False,
) -> SplitDataBundle:
    mapping = build_heliostat_data_mapping(
        benchmark_csv=benchmark_csv,
        calibration_properties_dir=calibration_properties_dir,
        flux_image_dir=flux_image_dir,
        split=split,
    )
    mapping = _truncate_mapping(mapping, sample_limit_per_heliostat)
    raw_inputs = build_calibration_inputs_real(
        mapping=mapping,
        sample_limit_per_heliostat=sample_limit_per_heliostat,
        centroid_method=centroid_method,
        scenario=scenario,
        group_states=group_states,
        load_flux_images=load_flux_images,
    )
    if norm_stats is None:
        norm_stats = compute_norm_stats(raw_inputs)
    normalized = normalize_calibration_inputs(raw_inputs, norm_stats)
    return SplitDataBundle(
        split=split,
        heliostat_data_mapping=mapping,
        calibration_inputs=normalized,
        norm_stats=norm_stats,
    )


def build_split_bundle_synth(
    *,
    split_dir: pathlib.Path,
    sample_limit_per_heliostat: int,
    scenario,
    group_states: list[GroupParameterState],
    norm_stats: CalibrationNormStats | None = None,
    load_flux_images: bool = False,
) -> SplitDataBundle:
    mapping = _build_synth_mapping(split_dir, sample_limit_per_heliostat)
    raw_inputs = build_calibration_inputs_synth(
        mapping=mapping,
        scenario=scenario,
        group_states=group_states,
        load_flux_images=load_flux_images,
    )
    if norm_stats is None:
        norm_stats = compute_norm_stats(raw_inputs)
    normalized = normalize_calibration_inputs(raw_inputs, norm_stats)
    return SplitDataBundle(
        split=split_dir.name,
        heliostat_data_mapping=mapping,
        calibration_inputs=normalized,
        norm_stats=norm_stats,
    )


def _truncate_mapping(
    mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    sample_limit_per_heliostat: int,
) -> list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]:
    if sample_limit_per_heliostat <= 0:
        return mapping
    return [
        (name, cal[:sample_limit_per_heliostat], flux[:sample_limit_per_heliostat])
        for name, cal, flux in mapping
    ]
