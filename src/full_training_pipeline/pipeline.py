from __future__ import annotations

from dataclasses import dataclass

import torch
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.util import index_mapping
from artist_extensions.kinematic_reconstructors import WortbergPixelReconstructor
from artist_extensions.loss_functions_ext import AlignmentLoss


@dataclass(frozen=True)
class GroupParameterState:
    heliostat_names: tuple[str, ...]
    base_translation: torch.Tensor
    base_rotation: torch.Tensor
    base_actuator_optimizable: torch.Tensor
    base_actuator_nonoptimizable: torch.Tensor
    base_position: torch.Tensor


def _remove_registered_parameter(module: torch.nn.Module, parameter_name: str) -> None:
    if parameter_name in module._parameters:
        module._parameters.pop(parameter_name)


def make_group_parameters_functional(heliostat_group) -> None:
    kinematic = heliostat_group.kinematics
    _remove_registered_parameter(kinematic, "translation_deviation_parameters")
    _remove_registered_parameter(kinematic, "rotation_deviation_parameters")
    _remove_registered_parameter(kinematic.actuators, "optimizable_parameters")
    _remove_registered_parameter(kinematic.actuators, "non_optimizable_parameters")


def capture_group_parameter_state(heliostat_group, device: torch.device) -> GroupParameterState:
    kinematic = heliostat_group.kinematics
    base_position = getattr(kinematic, "_base_position_deviation", None)
    if base_position is None:
        base_position = torch.zeros(kinematic.number_of_heliostats, 3, device=device)
    return GroupParameterState(
        heliostat_names=tuple(heliostat_group.names),
        base_translation=kinematic.translation_deviation_parameters.detach().to(device).clone(),
        base_rotation=kinematic.rotation_deviation_parameters.detach().to(device).clone(),
        base_actuator_optimizable=kinematic.actuators.optimizable_parameters.detach().to(device).clone(),
        base_actuator_nonoptimizable=kinematic.actuators.non_optimizable_parameters.detach().to(device).clone(),
        base_position=base_position.detach().to(device).clone(),
    )


def capture_all_group_parameter_states(scenario, device: torch.device) -> list[GroupParameterState]:
    return [
        capture_group_parameter_state(heliostat_group, device=device)
        for heliostat_group in scenario.heliostat_field.heliostat_groups
    ]


def flatten_wortberg_parameter_vector(group_state: GroupParameterState) -> torch.Tensor:
    actuator_initial_angles = group_state.base_actuator_optimizable[
        :, index_mapping.actuator_initial_angle, :
    ]
    actuator_offsets = group_state.base_actuator_nonoptimizable[:, index_mapping.actuator_offset, :]
    return torch.cat(
        [
            group_state.base_translation,
            group_state.base_rotation,
            actuator_initial_angles,
            actuator_offsets,
            group_state.base_position,
        ],
        dim=1,
    )


def apply_wortberg_parameter_vector(
    heliostat_group,
    group_state: GroupParameterState,
    parameter_vector: torch.Tensor,
) -> None:
    make_group_parameters_functional(heliostat_group)

    translation = parameter_vector[:, :9]
    rotation = parameter_vector[:, 9:13]
    actuator_angles = parameter_vector[:, 13:15]
    actuator_offsets = parameter_vector[:, 15:17]
    base_position = parameter_vector[:, 17:20]

    actuator_optimizable = group_state.base_actuator_optimizable.clone()
    actuator_nonoptimizable = group_state.base_actuator_nonoptimizable.clone()
    actuator_optimizable[:, index_mapping.actuator_initial_angle, :] = actuator_angles
    actuator_nonoptimizable[:, index_mapping.actuator_offset, :] = actuator_offsets

    kinematic = heliostat_group.kinematics
    kinematic.translation_deviation_parameters = translation
    kinematic.rotation_deviation_parameters = rotation
    kinematic.actuators.optimizable_parameters = actuator_optimizable
    kinematic.actuators.non_optimizable_parameters = actuator_nonoptimizable
    kinematic._base_position_deviation = base_position


def inject_base_position_deviation_if_present(
    heliostat_group,
    active_heliostats_mask: torch.Tensor,
    device: torch.device,
) -> None:
    kinematic = heliostat_group.kinematics
    if not hasattr(kinematic, "_base_position_deviation"):
        return
    repeat_counts = active_heliostats_mask.to(dtype=torch.int64)
    active_base_position = kinematic._base_position_deviation.repeat_interleave(repeat_counts, dim=0)
    pad = torch.zeros(active_base_position.shape[0], 1, device=device)
    kinematic.active_heliostat_positions = (
        kinematic.active_heliostat_positions + torch.cat([active_base_position, pad], dim=1)
    )


class FineErrorLearningPipeline(torch.nn.Module):
    def __init__(
        self,
        *,
        scenario,
        residual_model: torch.nn.Module,
        group_parameter_states: list[GroupParameterState],
        bitmap_resolution: torch.Tensor,
        ray_tracing_batch_size: int,
        loss_fn,
        loss_type: str,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.residual_model = residual_model
        self.group_parameter_states = group_parameter_states
        self.bitmap_resolution = bitmap_resolution
        self.ray_tracing_batch_size = ray_tracing_batch_size
        self.loss_fn = loss_fn
        self.loss_type = loss_type

    def predict_group_parameter_vector(
        self,
        *,
        group_index: int,
        group_calibration_inputs: list,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group_state = self.group_parameter_states[group_index]
        base_vector = flatten_wortberg_parameter_vector(group_state)
        residual_vector = self.residual_model(group_calibration_inputs)
        return base_vector + residual_vector, residual_vector

    def compute_dataset_loss(
        self,
        *,
        heliostat_data_mapping,
        data_parser,
        group_calibration_inputs: list[list],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        total_residual_penalty = torch.zeros((), dtype=torch.float32, device=device)
        contributing_groups = 0

        for group_index, heliostat_group in enumerate(self.scenario.heliostat_field.heliostat_groups):
            corrected_vector, residual_vector = self.predict_group_parameter_vector(
                group_index=group_index,
                group_calibration_inputs=group_calibration_inputs[group_index],
            )
            apply_wortberg_parameter_vector(
                heliostat_group=heliostat_group,
                group_state=self.group_parameter_states[group_index],
                parameter_vector=corrected_vector,
            )

            (
                measured_flux,
                focal_spots_measured,
                incident_ray_directions,
                motor_positions_measured,
                active_heliostats_mask,
                target_area_mask,
            ) = data_parser.parse_data_for_reconstruction(
                heliostat_data_mapping=heliostat_data_mapping,
                heliostat_group=heliostat_group,
                scenario=self.scenario,
                bitmap_resolution=self.bitmap_resolution,
                device=device,
            )

            if active_heliostats_mask.sum() == 0:
                continue

            heliostat_group.activate_heliostats(
                active_heliostats_mask=active_heliostats_mask,
                device=device,
            )
            inject_base_position_deviation_if_present(
                heliostat_group=heliostat_group,
                active_heliostats_mask=active_heliostats_mask,
                device=device,
            )
            heliostat_group.align_surfaces_with_incident_ray_directions(
                aim_points=self.scenario.solar_tower.get_centers_of_target_areas(
                    target_area_mask, device=device
                ),
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_heliostats_mask,
                device=device,
            )

            if self.loss_type == "alignment":
                kinematic = heliostat_group.kinematics
                loss_per_sample = self.loss_fn(
                    predicted_motor_positions=kinematic.active_motor_positions,
                    measured_motor_positions=motor_positions_measured,
                    actuators=kinematic.actuators,
                    device=device,
                )
            else:
                ray_tracer = HeliostatRayTracer(
                    scenario=self.scenario,
                    heliostat_group=heliostat_group,
                    blocking_active=False,
                    batch_size=min(
                        heliostat_group.number_of_active_heliostats,
                        self.ray_tracing_batch_size,
                    ),
                    bitmap_resolution=self.bitmap_resolution.to(device),
                )
                predicted_flux, _, _, _ = ray_tracer.trace_rays(
                    incident_ray_directions=incident_ray_directions,
                    active_heliostats_mask=active_heliostats_mask,
                    target_area_indices=target_area_mask,
                    device=device,
                )

                if self.loss_type == "pixel":
                    prediction = WortbergPixelReconstructor._peak_normalize(
                        WortbergPixelReconstructor._gaussian_blur(predicted_flux, sigma=1.0)
                    )
                    ground_truth = WortbergPixelReconstructor._peak_normalize(measured_flux)
                else:  # focal_spot
                    prediction = predicted_flux
                    ground_truth = focal_spots_measured

                loss_per_sample = self.loss_fn(
                    prediction=prediction,
                    ground_truth=ground_truth,
                    reduction_dimensions=(1, 2),
                    target_area_indices=target_area_mask,
                    device=device,
                )

            total_loss = total_loss + loss_per_sample.mean()
            total_residual_penalty = total_residual_penalty + residual_vector.pow(2).mean()
            contributing_groups += 1

        if contributing_groups == 0:
            raise RuntimeError("No active heliostat groups were found for the provided mapping.")

        normalizer = float(contributing_groups)
        return total_loss / normalizer, total_residual_penalty / normalizer