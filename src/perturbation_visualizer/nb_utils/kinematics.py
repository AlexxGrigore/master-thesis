"""Kinematic parameter management: snapshot/restore, optimizer setup, bounds clamping."""

import torch
from artist.util import indices

from .defaults import WORTBERG_BOUNDS


def snapshot_kinematics(k) -> dict:
    """Deep-copy all trainable deviation parameters (data only, no grad)."""
    return {
        "translation":   k.translation_deviation_parameters.data.clone(),
        "rotation":      k.rotation_deviation_parameters.data.clone(),
        "opt_params":    k.actuators.optimizable_parameters.data.clone(),
        "nonopt_params": k.actuators.non_optimizable_parameters.data.clone(),
    }


def restore_kinematics(k, snap: dict) -> None:
    """In-place restore from a snapshot produced by snapshot_kinematics."""
    with torch.no_grad():
        k.translation_deviation_parameters.data.copy_(snap["translation"])
        k.rotation_deviation_parameters.data.copy_(snap["rotation"])
        k.actuators.optimizable_parameters.data.copy_(snap["opt_params"])
        k.actuators.non_optimizable_parameters.data.copy_(snap["nonopt_params"])


def setup_optimizer(k, base_lr: float, device: torch.device):
    """Enable gradients, wire up hooks, create _base_position_deviation, return Adam.

    Stores k._initial_actuator_angle and k._initial_translation for use by
    apply_bounds.  Returns (optimizer, handles).
    """
    k.translation_deviation_parameters.requires_grad_(True)
    k.rotation_deviation_parameters.requires_grad_(True)
    k.actuators.optimizable_parameters.requires_grad_(True)

    handles = []

    def _freeze_stroke(grad):
        m = torch.ones_like(grad)
        m[:, indices.actuator_initial_stroke_length, :] = 0.0
        return grad * m

    handles.append(k.actuators.optimizable_parameters.register_hook(_freeze_stroke))

    k._base_position_deviation = torch.zeros(1, 3, device=device, requires_grad=True)
    k._initial_actuator_angle  = (
        k.actuators.optimizable_parameters[:, indices.actuator_initial_angle, :]
        .detach().clone()
    )
    k._initial_translation = k.translation_deviation_parameters.detach().clone()

    optimizer = torch.optim.Adam(
        [
            {"params": k.translation_deviation_parameters, "lr": base_lr * 5.0},
            {"params": k.rotation_deviation_parameters,    "lr": base_lr},
            {"params": k.actuators.optimizable_parameters, "lr": base_lr},
            {"params": k._base_position_deviation,         "lr": base_lr * 5.0},
        ],
        lr=base_lr,
    )
    return optimizer, handles


def apply_bounds(k, bounds: dict | None = None) -> None:
    """Clamp all optimised parameters to their deviation bounds after each step.

    bounds defaults to WORTBERG_BOUNDS.  Pass a custom dict to override (e.g.
    tighter bounds for a specific experiment).  Actuator-offset clamping is
    applied only when k._initial_actuator_offset exists.
    """
    if bounds is None:
        bounds = WORTBERG_BOUNDS

    with torch.no_grad():
        k.translation_deviation_parameters.data.clamp_(
            k._initial_translation - bounds["translation_m"],
            k._initial_translation + bounds["translation_m"],
        )
        k.rotation_deviation_parameters.data.clamp_(
            -bounds["rotation_rad"], bounds["rotation_rad"]
        )
        k.actuators.optimizable_parameters.data[
            :, indices.actuator_initial_angle, :
        ].clamp_(
            k._initial_actuator_angle - bounds["actuator_angle_rad"],
            k._initial_actuator_angle + bounds["actuator_angle_rad"],
        )
        if hasattr(k, "_initial_actuator_offset"):
            k.actuators.non_optimizable_parameters.data[
                :, indices.actuator_offset, :
            ].clamp_(
                k._initial_actuator_offset - bounds["actuator_offset_m"],
                k._initial_actuator_offset + bounds["actuator_offset_m"],
            )
        if hasattr(k, "_base_position_deviation"):
            k._base_position_deviation.data.clamp_(
                -bounds["base_position_m"], bounds["base_position_m"]
            )


def clip_and_step(optimizer: torch.optim.Optimizer, k, bounds: dict | None = None) -> None:
    """Gradient clip (norm=1) → optimizer step → bounds clamp."""
    params = [
        k.translation_deviation_parameters,
        k.rotation_deviation_parameters,
        k.actuators.optimizable_parameters,
    ]
    if hasattr(k, "_base_position_deviation"):
        params.append(k._base_position_deviation)
    if hasattr(k, "_initial_actuator_offset"):
        params.append(k.actuators.non_optimizable_parameters)
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()
    apply_bounds(k, bounds=bounds)


def reset_for_new_run(k, initial_snap: dict, hook_handles: list) -> None:
    """Tear down hooks, delete transient attrs, restore kinematics from snapshot."""
    for h in hook_handles:
        h.remove()
    hook_handles.clear()
    for attr in (
        "_initial_actuator_angle",
        "_initial_translation",
        "_initial_actuator_offset",
        "_base_position_deviation",
    ):
        if hasattr(k, attr):
            delattr(k, attr)
    with torch.no_grad():
        k.translation_deviation_parameters.requires_grad_(False)
        k.rotation_deviation_parameters.requires_grad_(False)
        k.actuators.optimizable_parameters.requires_grad_(False)
    restore_kinematics(k, initial_snap)
