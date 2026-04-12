from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
import torch

from full_training_pipeline.features import (
    SUMMARY_FEATURE_NAMES,
    build_heliostat_feature_summary,
)
from utils.evaluation import build_heliostat_data_mapping


@dataclass(frozen=True)
class SplitDataBundle:
    split: str
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
    raw_feature_summaries: dict[str, np.ndarray]
    normalized_feature_summaries: dict[str, torch.Tensor]
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    feature_names: tuple[str, ...] = SUMMARY_FEATURE_NAMES

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)


def _read_calibration_metadata(calibration_path: pathlib.Path) -> dict[str, object]:
    with open(calibration_path) as handle:
        return json.load(handle)


def _truncate_mapping(
    mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    sample_limit_per_heliostat: int,
) -> list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]:
    if sample_limit_per_heliostat <= 0:
        return mapping
    return [
        (heliostat_name, calibration_paths[:sample_limit_per_heliostat], flux_paths[:sample_limit_per_heliostat])
        for heliostat_name, calibration_paths, flux_paths in mapping
    ]


def build_split_mapping(
    *,
    benchmark_csv: pathlib.Path,
    calibration_properties_dir: pathlib.Path,
    flux_image_dir: pathlib.Path,
    split: str,
    sample_limit_per_heliostat: int,
) -> list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]:
    mapping = build_heliostat_data_mapping(
        benchmark_csv=benchmark_csv,
        calibration_properties_dir=calibration_properties_dir,
        flux_image_dir=flux_image_dir,
        split=split,
    )
    return _truncate_mapping(mapping, sample_limit_per_heliostat=sample_limit_per_heliostat)


def build_raw_feature_summaries(
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
) -> dict[str, np.ndarray]:
    summaries: dict[str, np.ndarray] = {}
    for heliostat_name, calibration_paths, _ in heliostat_data_mapping:
        if not calibration_paths:
            continue
        metadata_items = [_read_calibration_metadata(path) for path in calibration_paths]
        summaries[heliostat_name] = build_heliostat_feature_summary(metadata_items)
    return summaries


def compute_feature_normalization(
    raw_feature_summaries: dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not raw_feature_summaries:
        raise ValueError("Cannot normalize empty feature summaries.")
    feature_matrix = np.stack(list(raw_feature_summaries.values()), axis=0)
    mean = torch.tensor(feature_matrix.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(feature_matrix.std(axis=0), dtype=torch.float32)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def normalize_feature_summaries(
    raw_feature_summaries: dict[str, np.ndarray],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for heliostat_name, summary in raw_feature_summaries.items():
        normalized[heliostat_name] = (
            torch.tensor(summary, dtype=torch.float32) - feature_mean
        ) / feature_std
    return normalized


def build_split_bundle(
    *,
    benchmark_csv: pathlib.Path,
    calibration_properties_dir: pathlib.Path,
    flux_image_dir: pathlib.Path,
    split: str,
    sample_limit_per_heliostat: int,
    feature_mean: torch.Tensor | None = None,
    feature_std: torch.Tensor | None = None,
) -> SplitDataBundle:
    mapping = build_split_mapping(
        benchmark_csv=benchmark_csv,
        calibration_properties_dir=calibration_properties_dir,
        flux_image_dir=flux_image_dir,
        split=split,
        sample_limit_per_heliostat=sample_limit_per_heliostat,
    )
    raw_feature_summaries = build_raw_feature_summaries(mapping)
    if feature_mean is None or feature_std is None:
        feature_mean, feature_std = compute_feature_normalization(raw_feature_summaries)
    normalized_feature_summaries = normalize_feature_summaries(
        raw_feature_summaries,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    return SplitDataBundle(
        split=split,
        heliostat_data_mapping=mapping,
        raw_feature_summaries=raw_feature_summaries,
        normalized_feature_summaries=normalized_feature_summaries,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def build_group_feature_tensors(
    scenario,
    feature_summaries: dict[str, torch.Tensor],
    feature_dim: int,
    device: torch.device,
) -> list[torch.Tensor]:
    group_feature_tensors: list[torch.Tensor] = []
    zero_feature = torch.zeros(feature_dim, dtype=torch.float32, device=device)

    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        rows = []
        for heliostat_name in heliostat_group.names:
            feature_row = feature_summaries.get(heliostat_name)
            rows.append(feature_row.to(device) if feature_row is not None else zero_feature)
        group_feature_tensors.append(torch.stack(rows, dim=0))

    return group_feature_tensors