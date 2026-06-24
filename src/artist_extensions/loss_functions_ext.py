"""Extended loss functions for kinematic reconstruction experiments."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F

from artist.util import indices


class NormalAlignmentLoss:
    """Stage-1 loss expressed directly in milliradians.

    Both the predicted and measured motor positions are run through the full
    kinematic chain (_compute_orientations_from_motor_positions) to obtain the
    concentrator normal vector for each calibration sample.  The loss is the
    geodesic angle between the two normals in mrad.

    Unlike the motor-position MSE (AlignmentLoss), this accounts for the
    non-isotropic Jacobian of the kinematics: an actuator error that barely
    moves the beam at noon counts less than the same error at a sun angle
    where that actuator has high leverage.

    The measured-side normal is detached from the autograd graph so that
    gradients only flow through the predicted side, mirroring the behaviour
    of AlignmentLoss.

    Returns
    -------
    torch.Tensor
        Shape ``[N_active_samples]`` — angular error in **mrad**, one value
        per calibration sample.
    """

    # Concentrator normal direction in ARTIST's homogeneous frame.
    # From kinematics_rigid_body.py line 386-388:
    #   concentrator_normals = orientations @ [0, -1, 0, 0]
    _NORMAL_VEC = torch.tensor([0.0, -1.0, 0.0, 0.0])

    def __call__(
        self,
        predicted_motor_positions: torch.Tensor,
        measured_motor_positions: torch.Tensor,
        kinematic,
        device: torch.device,
    ) -> torch.Tensor:
        nv = self._NORMAL_VEC.to(device=device, dtype=torch.float32)

        def _normal(motor_pos: torch.Tensor) -> torch.Tensor:
            O = kinematic._compute_orientations_from_motor_positions(
                motor_pos.to(device=device, dtype=torch.float32), device
            )
            return torch.nn.functional.normalize((O @ nv)[:, :3], dim=-1)

        n_pred = _normal(predicted_motor_positions)
        with torch.no_grad():
            n_meas = _normal(measured_motor_positions)

        cos_sim = (n_pred * n_meas).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        return torch.acos(cos_sim) * 1000.0  # mrad


class AlignmentLoss:
    """Motor-position alignment loss in milliradians.

    Converts both predicted and measured motor positions to joint angles via
    the actuator model, then returns the Euclidean distance between the two
    angle vectors scaled to mrad:

        loss = ||pred_angles - meas_angles||₂ × 1000   [mrad]

    This is an isotropic approximation of the true normal-vector angular
    error (see NormalAlignmentLoss for the exact version).

    Returns
    -------
    torch.Tensor
        Shape ``[N_active_samples]`` — alignment error in **mrad**, one
        value per calibration sample.
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
        return (pred_angles - meas_angles).norm(dim=-1) * 1000.0


class MotorStepLoss:
    """Stage-1 loss in motor-step space, optionally increment-normalized.

    Compares the predicted motor positions against the recorded ones directly,
    without ever calling ``motor_positions_to_angles``. This keeps the *optimized*
    actuator parameters (initial angle a_i, offset c_i) out of the loss
    computation entirely: gradients reach theta only through the prediction
    ``m_pred = align(theta, sun, c_gt)``, while the recorded motors ``m_c`` are a
    fixed, theta-free target. The result is real-data-valid and free of the
    self-referential coupling present in AlignmentLoss/NormalAlignmentLoss.

    With ``normalize_by_increment=True`` (default) each motor's residual is
    divided by that actuator's step increment (steps per unit stroke length),
    turning raw step counts into a comparable physical (stroke-length) scale so
    the two actuators are balanced regardless of their differing gear ratios.
    The increment is frozen in the Wortberg setup, so it acts as a fixed scaling
    constant; it is detached to guarantee no gradient flows through the
    normalization.

    Returns
    -------
    torch.Tensor
        Shape ``[N_active_samples]`` — per-sample residual norm. Units are
        stroke length when normalized, raw motor steps otherwise.
    """

    def __init__(self, normalize_by_increment: bool = True) -> None:
        self.normalize_by_increment = normalize_by_increment

    def __call__(
        self,
        predicted_motor_positions: torch.Tensor,
        measured_motor_positions: torch.Tensor,
        actuators,
        device: torch.device,
    ) -> torch.Tensor:
        pred = predicted_motor_positions.to(device)
        meas = measured_motor_positions.to(device)
        diff = pred - meas

        if self.normalize_by_increment:
            # Use the same physics-informed (post-softplus) increment the
            # kinematics use, detached so it is a pure scaling constant.
            non_optimizable_parameters, _ = actuators._physics_informed_parameters(
                device=device
            )
            increment = non_optimizable_parameters[:, indices.actuator_increment].detach()
            diff = diff / increment

        return diff.norm(dim=-1)


class ForwardAimLoss:
    """Forward-consistent Stage-1 alignment loss.

    Compares the FORWARD concentrator normal at the recorded motor positions m_c
    against the geometric desired normal — the bisector of the directions toward
    the sun and toward the observed centroid c_gt. It uses only the forward
    kinematics (``_compute_orientations_from_motor_positions``), never the inverse,
    so it is consistent with the forward model and has its minimum at the true
    parameters.

    This fixes the motor-position alignment loss, whose inverse map
    (``incident_ray_directions_to_orientations``) does not invert the rotation
    deviations and therefore places the loss minimum away from theta* (see
    STAGE1_ALIGNMENT_LOSS_FINDINGS.md).

    The optimized per-sample loss is the squared chord distance
    ``||n_fwd - n_desired||^2`` (smooth gradient everywhere — no ``sqrt``/``arccos``
    near convergence). The geometric target ``n_desired`` is detached, so gradients
    flow only through ``n_fwd`` into theta.

    With ``return_mrad=True`` the per-sample geodesic angle in mrad is returned
    instead, for display only: ``2 * arcsin(||n_fwd - n_desired|| / 2) * 1000``.

    Parameters
    ----------
    motor_positions : torch.Tensor
        Recorded motor positions m_c. Shape ``[N_active, 2]``.
    incident_rays : torch.Tensor
        Incident ray directions (sun -> heliostat). Shape ``[N_active, 4]``.
    aim_points : torch.Tensor
        Observed centroids c_gt in world coordinates. Shape ``[N_active, 4]``.
    origins : torch.Tensor
        Mirror (concentrator) origins. Shape ``[N_active, 3]``.

    Returns
    -------
    torch.Tensor
        Shape ``[N_active]`` — squared chord distance (loss), or mrad if
        ``return_mrad=True``.
    """

    _NORMAL_VEC = torch.tensor([0.0, -1.0, 0.0, 0.0])

    def __call__(
        self,
        motor_positions: torch.Tensor,
        incident_rays: torch.Tensor,
        aim_points: torch.Tensor,
        origins: torch.Tensor,
        kinematic,
        device: torch.device,
        return_mrad: bool = False,
    ) -> torch.Tensor:
        nv = self._NORMAL_VEC.to(device=device, dtype=torch.float32)
        orientations = kinematic._compute_orientations_from_motor_positions(
            motor_positions.to(device=device, dtype=torch.float32), device
        )
        n_fwd = torch.nn.functional.normalize((orientations @ nv)[:, :3], dim=-1)

        to_sun = torch.nn.functional.normalize(-incident_rays[:, :3].to(device), dim=-1)
        to_aim = torch.nn.functional.normalize(
            aim_points[:, :3].to(device) - origins.to(device), dim=-1
        )
        # Geometric desired normal = bisector of (toward sun) and (toward c_gt).
        # Detached: a fixed target, gradients flow only through n_fwd -> theta.
        n_desired = torch.nn.functional.normalize(to_sun + to_aim, dim=-1).detach()

        diff = n_fwd - n_desired
        chord2 = (diff * diff).sum(dim=-1)
        if return_mrad:
            chord = chord2.clamp(min=0.0).sqrt()
            return 2.0 * torch.arcsin((chord * 0.5).clamp(-1.0, 1.0)) * 1000.0
        return chord2


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
        # Cache: hash(binary_gt[i].tobytes()) -> float32 distance-transform array.
        # GT images are fixed across epochs, so the cache fills in epoch 1 and
        # gives 100% hits from epoch 2 onward, eliminating the per-epoch CPU cost.
        self._dt_cache: dict[int, np.ndarray] = {}

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
        d_gt_arrays = []
        for i in range(N):
            key = hash(binary_gt[i].tobytes())
            if key not in self._dt_cache:
                self._dt_cache[key] = distance_transform_edt(~binary_gt[i]).astype(np.float32)
            d_gt_arrays.append(self._dt_cache[key])
        d_gt_t = torch.from_numpy(np.stack(d_gt_arrays, axis=0)).to(
            device=c_pred.device, dtype=c_pred.dtype
        )
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

    # ------------------------------------------------------------------
    # Component-aware forward (for logging)
    # ------------------------------------------------------------------

    def forward_with_components(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        target_area_indices=None,
        reduction_dimensions=None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, float, float, float]:
        """Like __call__ but also returns unweighted per-term means for logging.

        Returns
        -------
        total : torch.Tensor, shape [N]  — weighted sum (same as __call__)
        mean_coarse : float              — unweighted coarse term mean over N
        mean_fine   : float              — unweighted fine term mean over N
        mean_gravity: float              — unweighted gravity term mean over N
        """
        if device is not None:
            ground_truth = ground_truth.to(device)
        c_pred = self._to_contour(prediction)
        c_gt   = self._to_contour(ground_truth)
        coarse  = self._coarse_loss(c_pred, c_gt)
        fine    = self._fine_loss(c_pred, c_gt)
        gravity = self._gravity_loss(c_pred, c_gt)
        total   = self.weight_coarse * coarse + self.weight_fine * fine + self.weight_gravity * gravity
        return (
            total,
            coarse.detach().mean().item(),
            fine.detach().mean().item(),
            gravity.detach().mean().item(),
        )

    # ------------------------------------------------------------------
    # Pipeline introspection (for step-by-step visualization)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_intermediate_steps(
        self, x: torch.Tensor
    ) -> list[tuple[str, np.ndarray]]:
        """Run the contour pipeline on a single image and return each intermediate.

        Parameters
        ----------
        x : torch.Tensor, shape [1, H, W]  — one flux image (batch dim required)

        Returns
        -------
        list of (step_name, H×W float32 numpy array) in pipeline order:
            Raw, Normalized, Smoothed, Thresholded, Eroded, Contour
        """
        def _np(t: torch.Tensor) -> np.ndarray:
            return t[0].cpu().float().numpy()

        steps = [("Raw", _np(x))]
        x = self._normalize(x);    steps.append(("Normalized",  _np(x)))
        x = self._smooth(x);       steps.append(("Smoothed",    _np(x)))
        x = self._soft_threshold(x); steps.append(("Thresholded", _np(x)))
        x = self._soft_erosion(x); steps.append(("Eroded",      _np(x)))
        x = self._sobel_upper_edge(x); steps.append(("Contour",  _np(x)))
        return steps
