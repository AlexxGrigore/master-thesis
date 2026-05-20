"""Extended loss functions for kinematic reconstruction experiments."""
from __future__ import annotations

import torch


class AlignmentLoss:
    """Motor-position MSE in angle space.

    Converts both predicted and measured motor positions to joint angles via
    the actuator model, then computes the per-sample squared difference.
    This makes the loss invariant to the motor-position scaling of individual
    actuator families.

    Returns
    -------
    torch.Tensor
        Shape ``[N_active_samples]`` — squared angle error summed over
        the two actuators, one value per calibration sample.
    """

    def __call__(
        self,
        predicted_motor_positions: torch.Tensor,
        measured_motor_positions: torch.Tensor,
        actuators,
        device: torch.device,
    ) -> torch.Tensor:
        pred_angles = actuators.motor_positions_to_angles(
            motor_positions=predicted_motor_positions, device=device
        )
        meas_angles = actuators.motor_positions_to_angles(
            motor_positions=measured_motor_positions.to(device), device=device
        )
        return ((pred_angles - meas_angles) ** 2).sum(dim=-1)
