"""
Custom loss functions that extend ARTIST's built-in loss_functions.py.
"""

import torch
import torch.nn.functional as F
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


def _gaussian_blur_batch(flux: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply a separable Gaussian blur to a batch of 2-D flux images [N, H, W]."""
    kernel_size = int(4 * sigma + 0.5) * 2 + 1
    coords = torch.arange(kernel_size, device=flux.device, dtype=flux.dtype) - kernel_size // 2
    gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel = (gauss_1d[:, None] * gauss_1d[None, :]).view(1, 1, kernel_size, kernel_size)
    return F.conv2d(flux.unsqueeze(1), kernel, padding=kernel_size // 2).squeeze(1)


class BlurredPixelLoss(PixelLoss):
    """
    Pixel-wise MSE loss: Gaussian blur (sigma=1) → peak-normalize to [0, 1] → MSE.

    Matches the evaluation metric in utils/evaluation.py (which uses L1; this uses
    MSE for smoother gradients during training). Scale-invariant: each image is
    divided by its own peak value after blurring, so absolute intensity differences
    between predicted (physical units) and measured ([0, 1]) don't affect the loss.
    """

    SIGMA = 1.0

    def __call__(self, prediction, ground_truth, **kwargs):
        N = prediction.shape[0]

        blurred_pred = _gaussian_blur_batch(prediction,   self.SIGMA)
        blurred_gt   = _gaussian_blur_batch(ground_truth, self.SIGMA)

        pred_peak = blurred_pred.view(N, -1).max(dim=1).values.clamp(min=1e-12)
        gt_peak   = blurred_gt.view(N, -1).max(dim=1).values.clamp(min=1e-12)

        pred_norm = blurred_pred / pred_peak.view(N, 1, 1)
        gt_norm   = blurred_gt   / gt_peak.view(N, 1, 1)

        loss = self.loss_function(pred_norm, gt_norm)
        return loss.sum(dim=kwargs["reduction_dimensions"])


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
