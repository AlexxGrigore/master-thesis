from __future__ import annotations

import torch

from full_training_pipeline.pipeline import (
    FineErrorLearningPipeline,
    GroupParameterState,
    apply_wortberg_parameter_vector,
)
from utils.evaluation import evaluate_flux_accuracy


@torch.no_grad()
def apply_model_to_scenario(
    *,
    scenario,
    residual_model: torch.nn.Module,
    group_parameter_states: list[GroupParameterState],
    group_feature_tensors: list[torch.Tensor],
) -> None:
    pipeline = FineErrorLearningPipeline(
        scenario=scenario,
        residual_model=residual_model,
        group_parameter_states=group_parameter_states,
        bitmap_resolution=torch.tensor([256, 256]),
        ray_tracing_batch_size=32,
    )
    for group_index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
        corrected_vector, _ = pipeline.predict_group_parameter_vector(
            group_index=group_index,
            group_features=group_feature_tensors[group_index],
        )
        apply_wortberg_parameter_vector(
            heliostat_group=heliostat_group,
            group_state=group_parameter_states[group_index],
            parameter_vector=corrected_vector,
        )


@torch.no_grad()
def evaluate_model_tracking_accuracy(
    *,
    scenario,
    residual_model: torch.nn.Module,
    group_parameter_states: list[GroupParameterState],
    group_feature_tensors: list[torch.Tensor],
    heliostat_data_mapping,
    data_parser,
    device: torch.device,
    bitmap_resolution: torch.Tensor,
    ray_tracing_batch_size: int,
) -> dict[str, object]:
    apply_model_to_scenario(
        scenario=scenario,
        residual_model=residual_model,
        group_parameter_states=group_parameter_states,
        group_feature_tensors=group_feature_tensors,
    )
    return evaluate_flux_accuracy(
        scenario=scenario,
        heliostat_data_mapping=heliostat_data_mapping,
        data_parser=data_parser,
        device=device,
        bitmap_resolution=bitmap_resolution,
        ray_tracing_batch_size=ray_tracing_batch_size,
    )