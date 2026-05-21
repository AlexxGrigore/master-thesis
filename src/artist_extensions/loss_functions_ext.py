"""Extended loss functions for kinematic reconstruction experiments."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F


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


class ContourLoss:
    """Contour-based loss matching the upper edge of the focal spot.

    Based on Tristan Wortberg (2025). Instead of collapsing each flux image to
    a single COM point, this loss extracts a 2-D soft contour image by detecting
    the *upper* edge of the focal spot (which is unaffected by blocking/shading)
    and compares predicted vs. measured contours via three complementary terms:

    Coarse (soft distance field)
        Each predicted contour pixel is penalised by its distance to the nearest
        GT contour pixel.  Provides gradients even when contours don't overlap.

    Fine (DICE coefficient)
        1 − DICE between predicted and GT contour images.  Sensitive to precise
        pixel-level alignment but needs initial overlap to produce gradients.

    Gravity (COM distance)
        Euclidean distance between the COMs of the two contour images.  Acts as a
        smooth global gradient, preventing stalls when coarse/fine are flat.

    The contour-extraction pipeline (applied identically to both images):
        1. Per-image min-max normalisation → [0, 1]
        2. q rounds of bilinear up/down-sampling + Gaussian blur (noise removal)
        3. Soft thresholding via sigmoid (centre τ, sharpness η)
        4. Soft erosion via 3×3 mean convolution (suppress isolated pixels)
        5. Vertical Sobel convolution + ReLU → upper-edge contour image C

    Parameters
    ----------
    smoothing_rounds : int
        Number of bilinear up/down passes before Gaussian blur (q).
    gaussian_kernel_size : int
        Kernel size for Gaussian blur (odd integer).
    gaussian_sigma : float
        Standard deviation for Gaussian blur.
    threshold_tau : float
        Sigmoid centre threshold τ.  Wortberg default: 0.58.
    threshold_eta : float
        Sigmoid sharpness η.  Wortberg default: 70.0.
    weight_coarse : float
        Weight β for the coarse distance-field term.
    weight_gravity : float
        Weight γ for the gravity (COM-distance) term.
        The fine (DICE) term receives weight 1 − β − γ.

    Notes
    -----
    The distance transform (coarse term) is computed via
    ``scipy.ndimage.distance_transform_edt`` on the CPU.  It is applied only to
    the binarised GT contour (no gradients needed), so it does not appear in the
    autograd graph.  Pre-computing and caching D_G before training starts would
    eliminate this per-epoch CPU cost.
    """

    def __init__(
        self,
        smoothing_rounds: int = 2,
        gaussian_kernel_size: int = 5,
        gaussian_sigma: float = 1.0,
        threshold_tau: float = 0.58,
        threshold_eta: float = 70.0,
        weight_coarse: float = 0.3,
        weight_gravity: float = 0.2,
    ) -> None:
        self.smoothing_rounds       = smoothing_rounds
        self.gaussian_kernel_size   = gaussian_kernel_size
        self.gaussian_sigma         = gaussian_sigma
        self.threshold_tau          = threshold_tau
        self.threshold_eta          = threshold_eta
        self.weight_coarse          = weight_coarse
        self.weight_gravity         = weight_gravity
        self.weight_fine            = 1.0 - weight_coarse - weight_gravity

    def __call__(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        target_area_indices=None,
        reduction_dimensions=None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Compute per-sample contour loss.

        Parameters
        ----------
        prediction : torch.Tensor, shape [N, H, W]
            Predicted flux images from the ray tracer.
        ground_truth : torch.Tensor, shape [N, H, W]
            Measured flux images from the dataset.

        Returns
        -------
        torch.Tensor, shape [N]
            Per-sample scalar loss values.
        """
        if device is not None:
            ground_truth = ground_truth.to(device)

        c_pred = self._to_contour(prediction)
        c_gt   = self._to_contour(ground_truth)

        coarse  = self._coarse_loss(c_pred, c_gt)
        fine    = self._fine_loss(c_pred, c_gt)
        gravity = self._gravity_loss(c_pred, c_gt)

        return self.weight_coarse * coarse + self.weight_fine * fine + self.weight_gravity * gravity

    # ------------------------------------------------------------------
    # Preprocessing pipeline
    # ------------------------------------------------------------------

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Per-image min-max normalisation to [0, 1]."""
        N = x.shape[0]
        mn = x.view(N, -1).min(dim=1).values
        mx = x.view(N, -1).max(dim=1).values
        return (x - mn.view(N, 1, 1)) / (mx - mn).view(N, 1, 1).clamp(min=1e-12)

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        """q rounds of bilinear up/down-sampling followed by Gaussian blur."""
        H, W = x.shape[-2], x.shape[-1]
        for _ in range(self.smoothing_rounds):
            x = F.interpolate(
                x.unsqueeze(1), scale_factor=2, mode="bilinear", align_corners=False
            ).squeeze(1)
            x = F.interpolate(
                x.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(1)
        ks = self.gaussian_kernel_size
        coords = torch.arange(ks, device=x.device, dtype=x.dtype) - ks // 2
        g = torch.exp(-0.5 * (coords / self.gaussian_sigma) ** 2)
        g = g / g.sum()
        kernel = (g[:, None] * g[None, :]).view(1, 1, ks, ks)
        return F.conv2d(x.unsqueeze(1), kernel, padding=ks // 2).squeeze(1)

    def _soft_threshold(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable sigmoid-based soft threshold."""
        return torch.sigmoid(self.threshold_eta * (x - self.threshold_tau))

    def _soft_erosion(self, x: torch.Tensor) -> torch.Tensor:
        """3×3 mean convolution to suppress isolated noise pixels."""
        kernel = torch.ones(1, 1, 3, 3, device=x.device, dtype=x.dtype) / 9.0
        return F.conv2d(x.unsqueeze(1), kernel, padding=1).squeeze(1)

    def _sobel_upper_edge(self, x: torch.Tensor) -> torch.Tensor:
        """Vertical Sobel filter detecting the upper edge of bright regions.

        Gives a positive response where pixels below are brighter than pixels
        above (the upper boundary of the focal spot).  ReLU suppresses the lower
        edge and any negative artefacts.
        """
        kernel = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [ 0.0,  0.0,  0.0],
             [ 1.0,  2.0,  1.0]],
            device=x.device, dtype=x.dtype,
        ).view(1, 1, 3, 3)
        return F.relu(F.conv2d(x.unsqueeze(1), kernel, padding=1).squeeze(1))

    def _to_contour(self, x: torch.Tensor) -> torch.Tensor:
        """Full preprocessing pipeline → soft contour image [N, H, W]."""
        x = self._normalize(x)
        x = self._smooth(x)
        x = self._soft_threshold(x)
        x = self._soft_erosion(x)
        return self._sobel_upper_edge(x)

    # ------------------------------------------------------------------
    # Loss terms
    # ------------------------------------------------------------------

    def _coarse_loss(self, c_pred: torch.Tensor, c_gt: torch.Tensor) -> torch.Tensor:
        """Soft distance-field loss: predicted contour weighted by distance to GT contour."""
        N = c_pred.shape[0]
        binary_gt = (c_gt.detach() > 0.5).cpu().numpy()  # [N, H, W] bool
        # distance_transform_edt: each non-zero pixel → distance to nearest zero pixel.
        # Passing ~binary_gt gives each non-contour pixel its distance to the nearest
        # contour pixel; contour pixels themselves get 0.
        d_gt = np.stack([
            distance_transform_edt(~binary_gt[i]).astype(np.float32)
            for i in range(N)
        ], axis=0)
        d_gt_t = torch.from_numpy(d_gt).to(device=c_pred.device, dtype=c_pred.dtype)
        return (c_pred * d_gt_t).sum(dim=(-2, -1))

    def _fine_loss(self, c_pred: torch.Tensor, c_gt: torch.Tensor) -> torch.Tensor:
        """1 − DICE coefficient between predicted and GT contour images."""
        eps = 1e-6
        intersection = (c_pred * c_gt).sum(dim=(-2, -1))
        union = c_pred.sum(dim=(-2, -1)) + c_gt.sum(dim=(-2, -1))
        dice = 2.0 * intersection / (union + eps)
        return 1.0 - dice

    def _gravity_loss(self, c_pred: torch.Tensor, c_gt: torch.Tensor) -> torch.Tensor:
        """Euclidean distance between the COMs of predicted and GT contour images."""
        eps = 1e-6
        N, H, W = c_pred.shape
        ys = torch.arange(H, device=c_pred.device, dtype=c_pred.dtype)
        xs = torch.arange(W, device=c_pred.device, dtype=c_pred.dtype)

        def _com(c: torch.Tensor) -> torch.Tensor:
            total = c.sum(dim=(-2, -1)).clamp(min=eps)
            cy = (c * ys[None, :, None]).sum(dim=(-2, -1)) / total
            cx = (c * xs[None, None, :]).sum(dim=(-2, -1)) / total
            return torch.stack([cy, cx], dim=-1)  # [N, 2]

        # Detach GT: we only backpropagate through the predicted side.
        return torch.norm(_com(c_pred) - _com(c_gt).detach(), dim=-1)
