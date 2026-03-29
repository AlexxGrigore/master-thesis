"""
Custom loss functions that extend ARTIST's built-in loss_functions.py.
"""

import torch
from artist.core.loss_functions import PixelLoss


class PixelLossL1(PixelLoss):
    """
    L1 (MAE) variant of PixelLoss.

    Identical to PixelLoss but uses L1Loss instead of MSELoss, making it less
    sensitive to outlier pixels (e.g. bright specular spots in flux images).
    Peak normalization and kwargs handling are inherited from PixelLoss.
    """

    def __init__(self, scenario) -> None:
        """Initialize with L1Loss instead of MSELoss."""
        super().__init__(scenario=scenario)
        self.loss_function = torch.nn.L1Loss(reduction="none")


class AlignmentLoss:
    """
    MSE between predicted and measured motor positions, computed in joint-angle space (radians).

    The kinematic forward pass produces predicted motor positions as a byproduct of inverse
    kinematics (stored in ``kinematic.active_motor_positions``). The measured motor positions
    come from the calibration properties JSON files. Computing the loss in angle space (radians)
    rather than raw encoder counts gives physically meaningful, well-conditioned gradients.

    Usage
    -----
    loss_fn = AlignmentLoss()
    loss_per_sample = loss_fn(
        predicted_motor_positions=kinematic.active_motor_positions,
        measured_motor_positions=motor_positions_measured,
        actuators=kinematic.actuators,
        device=device,
    )
    # loss_per_sample: [N_active], one scalar per active heliostat-sample pair
    """

    def __call__(
        self,
        predicted_motor_positions: torch.Tensor,
        measured_motor_positions: torch.Tensor,
        actuators,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute per-sample alignment loss.

        Parameters
        ----------
        predicted_motor_positions : torch.Tensor
            Motor positions predicted by the kinematic model.
            Tensor of shape [N_active, 2].
        measured_motor_positions : torch.Tensor
            Motor positions measured during calibration (from PAINT JSON files).
            Tensor of shape [N_active, 2].
        actuators : LinearActuators
            The actuator model (provides motor_positions_to_angles conversion).
        device : torch.device
            Compute device.

        Returns
        -------
        torch.Tensor
            Per-sample MSE loss in angle space.
            Tensor of shape [N_active].
        """
        pred_angles = actuators.motor_positions_to_angles(
            motor_positions=predicted_motor_positions, device=device
        )
        meas_angles = actuators.motor_positions_to_angles(
            motor_positions=measured_motor_positions.to(device), device=device
        )
        return ((pred_angles - meas_angles) ** 2).sum(dim=-1)
