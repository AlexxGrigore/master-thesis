from __future__ import annotations

import torch

from full_training_pipeline.pipeline import (
    GroupParameterState,
    apply_wortberg_parameter_vector,
    flatten_wortberg_parameter_vector,
)
from utils.evaluation import evaluate_flux_accuracy


@torch.no_grad()
def apply_model_to_scenario(
    *,
    scenario,
    residual_model: torch.nn.Module,
    group_parameter_states: list[GroupParameterState],
    group_calibration_inputs: list[list],
) -> None:
    for group_index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
        group_state = group_parameter_states[group_index]
        base_vector = flatten_wortberg_parameter_vector(group_state)
        residual_vector = residual_model(group_calibration_inputs[group_index])
        corrected_vector = base_vector + residual_vector
        apply_wortberg_parameter_vector(
            heliostat_group=heliostat_group,
            group_state=group_state,
            parameter_vector=corrected_vector,
        )


@torch.no_grad()
def evaluate_model_tracking_accuracy(
    *,
    scenario,
    residual_model: torch.nn.Module,
    group_parameter_states: list[GroupParameterState],
    group_calibration_inputs: list[list],
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
        group_calibration_inputs=group_calibration_inputs,
    )
    return evaluate_flux_accuracy(
        scenario=scenario,
        heliostat_data_mapping=heliostat_data_mapping,
        data_parser=data_parser,
        device=device,
        bitmap_resolution=bitmap_resolution,
        ray_tracing_batch_size=ray_tracing_batch_size,
    )
