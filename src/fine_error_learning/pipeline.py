"""
Differentiable forward pass through ARTIST for fine_error_learning.

Ports the functional-injection pattern of
``src/too_old/full_training_pipeline/pipeline.py`` to the 24-parameter set of
the live two-stage pipeline (``src/one_heliostat_demo/single_heliostat/train.py``):

    θ_final = θ_KR + Δθ

is written into the kinematic/actuator tensors as PLAIN tensors (not
nn.Parameters), so gradients flow into Δθ. In the ARTIST version in use the
kinematic parameters are plain attribute tensors that are re-read on every
``activate_heliostats`` call; the ``_parameters`` cleanup below is kept as a
safeguard for versions where they are registered parameters.

24-D parameter vector ordering (matches stage1_checkpoint.pt content):
    [0:4]   rotation_deviation_parameters            (rad)
    [4:13]  translation_deviation_parameters         (m)
    [13:15] actuator initial_angle    (per actuator) (rad)
    [15:17] actuator initial_stroke_length           (m)
    [17:19] actuator offset           (non-opt idx 5) (m)
    [19:21] actuator pivot_radius     (non-opt idx 6) (m)
    [21:24] base_position_deviation                  (m)
"""
from __future__ import annotations

import logging

import torch
from artist.flux import get_center_of_mass
from artist.geometry import bitmap_coordinates_to_target_coordinates
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
from artist.util import indices

log = logging.getLogger(__name__)

N_PARAMS = 24

PARAMETER_NAMES: tuple[str, ...] = (
    "rotation_0", "rotation_1", "rotation_2", "rotation_3",
    "translation_0", "translation_1", "translation_2",
    "translation_3", "translation_4", "translation_5",
    "translation_6", "translation_7", "translation_8",
    "actuator_initial_angle_0", "actuator_initial_angle_1",
    "actuator_initial_stroke_0", "actuator_initial_stroke_1",
    "actuator_offset_0", "actuator_offset_1",
    "actuator_pivot_radius_0", "actuator_pivot_radius_1",
    "base_position_e", "base_position_n", "base_position_u",
)

# Per-parameter scale, used (a) to normalize the θ_KR model input and
# (b) as the tanh bounds when BOUNDED_HEAD is enabled. Angles ~0.1 rad,
# lengths ~5-10 cm — comfortably above the residuals the model should learn.
PARAMETER_SCALE = torch.tensor(
    [0.1] * 4        # rotation deviations [rad]
    + [0.05] * 9     # translation deviations [m]
    + [0.1] * 2      # actuator initial angles [rad]
    + [0.05] * 2     # actuator initial stroke lengths [m]
    + [0.05] * 2     # actuator offsets [m]
    + [0.1] * 2      # actuator pivot radii [m]
    + [0.1] * 3,     # base position deviations [m]
    dtype=torch.float32,
)
RESIDUAL_BOUNDS = PARAMETER_SCALE.clone()

# Heliostat position normalization for the model input (positions ~hundreds of m).
POSITION_SCALE_M = 100.0


def flatten_checkpoint_theta(checkpoint: dict) -> torch.Tensor:
    """Flatten a stage1_checkpoint.pt dict to the 24-D θ_KR vector (detached)."""
    return torch.cat(
        [
            checkpoint["rotation"][0].detach(),                                    # 4
            checkpoint["translation"][0].detach(),                                 # 9
            checkpoint["act_angle"][0, indices.actuator_initial_angle, :].detach(),          # 2
            checkpoint["act_angle"][0, indices.actuator_initial_stroke_length, :].detach(),  # 2
            checkpoint["act_offset"][0, indices.actuator_offset, :].detach(),                # 2
            checkpoint["act_offset"][0, indices.actuator_pivot_radius, :].detach(),          # 2
            checkpoint["base_pos"][0].detach(),                                    # 3
        ]
    ).float()


def _remove_registered_parameter(module: torch.nn.Module, parameter_name: str) -> None:
    if parameter_name in module._parameters:
        module._parameters.pop(parameter_name)


def apply_parameter_vector(state, theta_final: torch.Tensor) -> None:
    """Write a 24-D vector into the kinematics of ``state`` as plain tensors.

    ``theta_final`` may carry a grad_fn — gradients flow through into the
    caller's graph (that is the whole point of the functional injection).
    Constant actuator rows (type, clockwise flag, motor limits, increment) are
    restored from ``state.act_nonopt_template``.
    """
    kinematic = state.heliostat_group.kinematics
    _remove_registered_parameter(kinematic, "translation_deviation_parameters")
    _remove_registered_parameter(kinematic, "rotation_deviation_parameters")
    _remove_registered_parameter(kinematic.actuators, "optimizable_parameters")
    _remove_registered_parameter(kinematic.actuators, "non_optimizable_parameters")

    kinematic.rotation_deviation_parameters = theta_final[0:4].unsqueeze(0)
    kinematic.translation_deviation_parameters = theta_final[4:13].unsqueeze(0)
    kinematic.actuators.optimizable_parameters = torch.stack(
        [theta_final[13:15], theta_final[15:17]], dim=0
    ).unsqueeze(0)

    act_nonopt = state.act_nonopt_template.clone()
    act_nonopt[0, indices.actuator_offset, :] = theta_final[17:19]
    act_nonopt[0, indices.actuator_pivot_radius, :] = theta_final[19:21]
    kinematic.actuators.non_optimizable_parameters = act_nonopt

    # Repo-specific extension (train.py:825-828 of the live pipeline): the base
    # position deviation is applied to the active heliostat positions after
    # activation — see predict_flux below.
    kinematic._base_position_deviation = theta_final[21:24].unsqueeze(0)


def predict_flux(
    state,
    theta_final: torch.Tensor,
    incident_rays: torch.Tensor,
    motor_positions: torch.Tensor,
    target_indices: torch.Tensor,
    device: torch.device,
    random_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply θ, orient from the recorded motor positions, and trace rays.

    Returns (predicted_flux [N, 256, 256], bitmap_resolution [2],
    sampler_indices [N]).
    """
    apply_parameter_vector(state, theta_final)

    heliostat_group = state.heliostat_group
    kinematic = heliostat_group.kinematics
    n = incident_rays.shape[0]

    active_mask = torch.zeros(
        heliostat_group.number_of_heliostats, dtype=torch.long, device=device
    )
    active_mask[state.hel_idx] = n

    heliostat_group.activate_heliostats(active_heliostats_mask=active_mask, device=device)
    # Base-position deviation: shift the active instance origins (same idiom as
    # the live pipeline, train.py:2855-2861).
    repeated = kinematic._base_position_deviation.repeat_interleave(active_mask, dim=0)
    pad = torch.zeros(repeated.shape[0], 1, device=device)
    kinematic.active_heliostat_positions = (
        kinematic.active_heliostat_positions + torch.cat([repeated, pad], dim=1)
    )

    heliostat_group.align_surfaces_with_motor_positions(
        motor_positions=motor_positions,
        active_heliostats_mask=active_mask,
        device=device,
    )

    ray_tracer = HeliostatRayTracer(
        scenario=state.scenario,
        heliostat_group=heliostat_group,
        blocking_active=False,
        world_size=1,
        rank=0,
        batch_size=max(8, n),
        random_seed=random_seed,
    )
    flux, _, _, _ = ray_tracer.trace_rays(
        incident_ray_directions=incident_rays,
        active_heliostats_mask=active_mask,
        target_area_indices=target_indices,
        device=device,
    )
    return flux, ray_tracer.bitmap_resolution, ray_tracer.get_sampler_indices()


def focal_spot_centroid_loss(
    predicted_flux: torch.Tensor,
    focal_spots: torch.Tensor,
    target_indices: torch.Tensor,
    bitmap_resolution: torch.Tensor,
    scenario,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage-2 focal-spot loss of the live pipeline (train.py:2713-2722).

    Squared distance between the predicted-flux center of mass and the measured
    centroid c_gt on the target plane. Returns (per-sample squared distance [N]
    in m², predicted centroid coordinates [N, 4]).
    """
    bitmap_coords = get_center_of_mass(bitmaps=predicted_flux, device=device)
    pred_coords = bitmap_coordinates_to_target_coordinates(
        bitmap_coordinates=bitmap_coords,
        bitmap_resolution=bitmap_resolution,
        solar_tower=scenario.solar_tower,
        target_area_indices=target_indices,
        device=device,
    )
    return ((pred_coords[:, :3] - focal_spots[:, :3]) ** 2).sum(dim=-1), pred_coords


def pixelwise_flux_loss(
    predicted_flux: torch.Tensor,
    gt_flux: torch.Tensor,
    out_size: int = 32,
) -> torch.Tensor:
    """Auxiliary distribution loss between predicted and measured flux images.

    Both images are downsized with average pooling to ``out_size × out_size``
    (robust to the sparse, noisy 10-ray training bitmaps) and normalized to
    unit sum, so the loss compares the flux *distribution* — spot shape and
    spread — not absolute power. Returns per-sample MSE [N] (dimensionless).
    """
    gt = gt_flux.to(predicted_flux.device, predicted_flux.dtype)
    kernel = predicted_flux.shape[-1] // out_size
    pred = torch.nn.functional.avg_pool2d(predicted_flux.unsqueeze(1), kernel).squeeze(1)
    ref = torch.nn.functional.avg_pool2d(gt.unsqueeze(1), kernel).squeeze(1)
    pred = pred / pred.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    ref = ref / ref.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    return ((pred - ref) ** 2).mean(dim=(-2, -1))
