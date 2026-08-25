"""Extended loss functions for kinematic reconstruction experiments."""
from __future__ import annotations

import torch

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



# ---------------------------------------------------------------------------
# Robust reductions for the Stage-1 objective
# ---------------------------------------------------------------------------

def robust_reduce(
    chord_squared: torch.Tensor,
    mode: str = "l2",
    delta_mrad: float = 3.0,
    trim_fraction: float = 0.25,
) -> torch.Tensor:
    """Aggregate ForwardAimLoss per-sample residuals with a robust estimator.

    This changes only HOW the per-sample residuals are combined, not what is
    measured. The plain mean of the squared chord (``"l2"``) is a least-squares
    fit: it weights every sample by the square of its error, so a handful of
    bad calibration samples dominate the solution and pull the fit off-centre.
    Robust reductions cap or discard that influence.

    The residual is expressed in mrad, ``r = ||n_fwd - n_desired|| * 1000``
    (the small-angle chord, within 0.1% of the geodesic angle over the whole
    working range), so ``delta_mrad`` is directly interpretable.

    Modes
    -----
    ``"l2"``      mean(r²)·1e-6 — identical to the historical ``lps.mean()``.
    ``"huber"``   quadratic within δ, linear beyond: bounds each outlier's
                  gradient to a constant instead of letting it grow with r.
    ``"soft_l1"`` pseudo-Huber ``2δ²(sqrt(1+(r/δ)²)−1)`` — the same behaviour
                  without the branch, smooth everywhere.
    ``"trimmed"`` least-trimmed-squares: drop the worst ``trim_fraction`` of
                  samples and take the mean of the rest.

    Trade-off (measured on AA23): robust modes improve the MEDIAN pointing
    error but cost mean and centroid accuracy, because they buy the bulk of the
    distribution by sacrificing the tail — and the centroid metric is
    tail-sensitive. Always report both.

    Parameters
    ----------
    chord_squared : torch.Tensor
        Per-sample squared chord distance from ``ForwardAimLoss``. Shape ``[N]``.
    mode : str
        One of ``"l2"``, ``"huber"``, ``"soft_l1"``, ``"trimmed"``.
    delta_mrad : float
        Huber / soft-L1 transition point δ, in mrad.
    trim_fraction : float
        Fraction of the largest residuals discarded in ``"trimmed"`` mode.

    Returns
    -------
    torch.Tensor
        Scalar loss. Scaled to the chord² convention so that ``"l2"`` reproduces
        the previous objective exactly and learning rates stay comparable.
    """
    r2_mrad = chord_squared * 1e6                     # (mrad)²
    scale = 1e-6                                      # back to the chord² scale

    if mode == "l2":
        return r2_mrad.mean() * scale

    if mode == "trimmed":
        if not 0.0 <= trim_fraction < 1.0:
            raise ValueError(f"trim_fraction must be in [0, 1), got {trim_fraction}")
        n_keep = max(1, int(round(r2_mrad.numel() * (1.0 - trim_fraction))))
        kept, _ = torch.topk(r2_mrad, n_keep, largest=False)
        return kept.mean() * scale

    # sqrt is safe: the +1e-12 keeps the gradient finite at r = 0, and both
    # remaining modes are quadratic there anyway.
    r = torch.sqrt(chord_squared + 1e-12) * 1000.0
    d = float(delta_mrad)

    if mode == "huber":
        quad = 0.5 * r2_mrad
        lin = d * (r - 0.5 * d)
        return torch.where(r <= d, quad, lin).mean() * (2.0 * scale)

    if mode == "soft_l1":
        return (2.0 * d * d * (torch.sqrt(1.0 + (r / d) ** 2) - 1.0)).mean() * scale

    raise ValueError(
        f"Unknown reduction {mode!r}. Choose from 'l2', 'huber', 'soft_l1', 'trimmed'."
    )


def robust_reduce_squared(
    squared_residual: torch.Tensor,
    mode: str = "l2",
    delta: float = 1.0,
    trim_fraction: float = 0.25,
) -> torch.Tensor:
    """Robustly aggregate per-sample SQUARED residuals, in the residual's own units.

    The Stage-2 counterpart of :func:`robust_reduce`. Stage 2 measures a
    focal-spot miss distance on the target plane, so the residual is in metres
    and ``delta`` must be given in metres too (convert an angular tolerance with
    ``delta_m = delta_mrad * heliostat_distance_m / 1000``).

    ``"l2"`` returns ``squared_residual.mean()`` unchanged, so it reproduces the
    historical objective exactly. All modes operate on the same residual scale,
    which keeps the arms of a comparison on equal footing under gradient
    clipping (a rescaled loss would otherwise clip differently).

    Parameters
    ----------
    squared_residual : torch.Tensor
        Per-sample squared residual, e.g. squared metres. Shape ``[N]``.
    mode : str
        ``"l2"``, ``"huber"``, ``"soft_l1"`` or ``"trimmed"``.
    delta : float
        Transition point, in the SAME units as the (unsquared) residual.
    trim_fraction : float
        Fraction of the largest residuals discarded in ``"trimmed"`` mode.

    Returns
    -------
    torch.Tensor
        Scalar loss, scaled to the squared-residual convention so learning rates
        stay comparable across modes.
    """
    if mode == "l2":
        return squared_residual.mean()

    if mode == "trimmed":
        if not 0.0 <= trim_fraction < 1.0:
            raise ValueError(f"trim_fraction must be in [0, 1), got {trim_fraction}")
        n_keep = max(1, int(round(squared_residual.numel() * (1.0 - trim_fraction))))
        kept, _ = torch.topk(squared_residual, n_keep, largest=False)
        return kept.mean()

    # Safe sqrt: both remaining modes are quadratic at 0, and the epsilon keeps
    # the gradient finite there.
    r = torch.sqrt(squared_residual + 1e-12)
    d = float(delta)

    if mode == "huber":
        # x2 so the quadratic branch matches mean(r^2) — same scale as "l2".
        return torch.where(
            r <= d, 0.5 * squared_residual, d * (r - 0.5 * d)
        ).mean() * 2.0

    if mode == "soft_l1":
        return (2.0 * d * d * (torch.sqrt(1.0 + (r / d) ** 2) - 1.0)).mean()

    raise ValueError(
        f"Unknown reduction {mode!r}. Choose from 'l2', 'huber', 'soft_l1', 'trimmed'."
    )
