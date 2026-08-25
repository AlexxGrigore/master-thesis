"""Upper-contour loss for image-based heliostat calibration.

Fresh implementation of the contour loss from Wortberg (2025), *Image Based
Heliostat-Calibration in Solar Tower Power Plants with Differentiable
Kinematics and Artificial Intelligence* (RWTH Aachen / DLR), §4.2.3–4.2.4.

Instead of collapsing each flux image to its centre of mass (which occlusion
by neighbouring heliostats corrupts — flux is eaten from the LOWER part of the
spot, pulling the COM upward), this loss extracts the *upper* contour of the
focal spot from both the predicted and the measured flux image and drives the
predicted contour onto the measured one via three complementary terms:

Coarse (soft distance field, eq. 4.36)
    Predicted contour mass weighted by the Euclidean distance to the nearest
    ground-truth contour pixel. Supplies a gradient even with zero overlap.
Fine (soft DICE, eq. 4.38–4.39)
    1 − DICE between the soft contour images. Precise once contours overlap.
Gravity (COM distance in metres, eq. 4.40)
    Distance between the contour centres of mass mapped onto the target plane
    (ENU). A smooth, minimum-free global anchor.

Weighted sum (eq. 4.41): (1 − β − γ)·Fine + β·Coarse + γ·Gravity.

Thesis typo corrections applied throughout (each marked in place):
  * the vertical Sobel runs on the ERODED field E (Fig. 4.7), not on the
    pre-erosion mask B of eq. 4.33;
  * only the POSITIVE Sobel response is kept (ReLU) — the upper edge;
  * the distance transform runs on the INVERTED binary contour so it measures
    distance TO the contour (eq. 4.35 prints the constraint backwards);
  * the flux is re-normalized after denoising, since τ is defined on [0, 1].

Differentiability contract: the predicted branch stays on the autograd graph
all the way back to the kinematic parameters; the ground-truth branch is
precomputed once per split under ``no_grad`` and cached (`ContourGroundTruth`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from artist.flux import get_center_of_mass
from artist.geometry.coordinates import bitmap_coordinates_to_target_coordinates
from artist.util import indices


class ContourExtractor(torch.nn.Module):
    """Flux image → soft upper-contour image (differentiable, §4.2.3, Fig. 4.7).

    Pipeline: per-image min-max normalize → q rounds of bilinear up/down
    interpolation + Gaussian blur (denoise) → re-normalize → sigmoid soft
    threshold → 3×3 mean soft erosion → vertical Sobel → ReLU (upper edge).

    Parameters
    ----------
    tau : float
        Soft-threshold centre τ on the [0, 1] normalized flux (thesis: 0.58,
        Bayesian-optimized on simulated STJ flux — retune on other data).
    eta : float
        Sigmoid sharpness η (thesis: 70.0).
    smoothing_rounds : int
        Number q of bilinear up/down-sampling passes.
    gaussian_sigma : float
        Standard deviation of the Gaussian denoising kernel.
    gaussian_kernel_size : int
        Side length of the Gaussian kernel (odd).
    band_sigma : float
        Std dev of an OPTIONAL Gaussian blur applied to the final upper-contour
        image (default 0.0 — a no-op, exact prior behavior). The Sobel-derived
        contour is inherently a 1-2 pixel-wide rim, which is a much noisier
        optimization target than a full-image centroid (few pixels contribute
        gradient at any step, and small kinematic changes cause the
        discretized edge to jump around). Blurring the extracted contour
        widens it into a several-pixel soft band, keeping the same "ignore
        the lower/corrupted region" idea while giving Fine/DICE far more
        pixels to compute a gradient from. Applied AFTER Sobel+ReLU, so it
        only affects band width/softness, not which region survives
        thresholding.
    """

    def __init__(
        self,
        tau: float = 0.58,
        eta: float = 70.0,
        smoothing_rounds: int = 2,
        gaussian_sigma: float = 1.0,
        gaussian_kernel_size: int = 5,
        band_sigma: float = 0.0,
    ) -> None:
        super().__init__()
        if gaussian_kernel_size % 2 != 1:
            raise ValueError("gaussian_kernel_size must be odd")
        self.tau = tau
        self.eta = eta
        self.smoothing_rounds = smoothing_rounds
        self.band_sigma = band_sigma

        # Fixed kernels as buffers so device/dtype follow the module.
        ax = torch.arange(gaussian_kernel_size, dtype=torch.float32)
        ax = ax - (gaussian_kernel_size - 1) / 2
        g1 = torch.exp(-(ax**2) / (2 * gaussian_sigma**2))
        g1 = g1 / g1.sum()
        self.register_buffer("_gauss", torch.outer(g1, g1).view(1, 1, *([gaussian_kernel_size] * 2)))
        # eq. 2.29 / 4.33 — (bottom neighbours) − (top neighbours): positive at
        # dark→bright transitions moving down the rows, i.e. the upper rim.
        self.register_buffer(
            "_sobel_y",
            torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
            ).view(1, 1, 3, 3),
        )
        self.register_buffer("_mean3", torch.full((1, 1, 3, 3), 1.0 / 9.0))

        if band_sigma > 0:
            bk = int(np.ceil(6 * band_sigma))
            bk = bk + 1 if bk % 2 == 0 else bk
            bk = max(bk, 3)
            bax = torch.arange(bk, dtype=torch.float32) - (bk - 1) / 2
            bg1 = torch.exp(-(bax**2) / (2 * band_sigma**2))
            bg1 = bg1 / bg1.sum()
            self.register_buffer("_band_gauss", torch.outer(bg1, bg1).view(1, 1, bk, bk))
        else:
            self.register_buffer("_band_gauss", None)

    def _widen_band(self, c: torch.Tensor) -> torch.Tensor:
        """Optional post-hoc blur widening the extracted contour (band_sigma > 0)."""
        if self.band_sigma <= 0 or self._band_gauss is None:
            return c
        pad = self._band_gauss.shape[-1] // 2
        return F.conv2d(c, self._band_gauss, padding=pad)

    @staticmethod
    def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Per-image min-max normalization to [0, 1] (eq. 3.3)."""
        mn = x.amin(dim=(-2, -1), keepdim=True)
        mx = x.amax(dim=(-2, -1), keepdim=True)
        return (x - mn) / (mx - mn + eps)

    def _denoise(self, x: torch.Tensor) -> torch.Tensor:
        """q rounds of bilinear up/down-sampling, then Gaussian blur (§3.4)."""
        H, W = x.shape[-2], x.shape[-1]
        for _ in range(self.smoothing_rounds):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        pad = self._gauss.shape[-1] // 2
        return F.conv2d(x, self._gauss, padding=pad)

    def forward(self, flux: torch.Tensor) -> torch.Tensor:
        """Extract the soft upper contour.

        Parameters
        ----------
        flux : torch.Tensor
            Flux images, any intensity range. Shape ``[N, H, W]``.

        Returns
        -------
        torch.Tensor
            Soft upper-contour images C(i, j) ≥ 0. Shape ``[N, H, W]``.
        """
        x = flux.unsqueeze(1)                       # [N,1,H,W]
        x = self._normalize(x)
        x = self._denoise(x)
        x = self._normalize(x)                      # τ is defined on [0,1] — re-normalize
        b = torch.sigmoid(self.eta * (x - self.tau))  # soft threshold (eq. 4.29–4.31)
        e = F.conv2d(b, self._mean3, padding=1)       # soft erosion (eq. 4.32)
        c = F.conv2d(e, self._sobel_y, padding=1)     # Sobel on E, per Fig. 4.7
        c = F.relu(c)                                 # upper (dark→bright) edge only
        c = self._widen_band(c)                       # optional post-hoc band widening
        return c.squeeze(1)

    @torch.no_grad()
    def intermediate_steps(self, flux: torch.Tensor) -> list[tuple[str, np.ndarray]]:
        """Run the pipeline on ONE image, returning every intermediate for plots.

        Parameters
        ----------
        flux : torch.Tensor
            A single flux image. Shape ``[H, W]``.

        Returns
        -------
        list of (name, H×W float32 array), in pipeline order.
        """
        x = flux.view(1, 1, *flux.shape[-2:]).float()

        def _np(t: torch.Tensor) -> np.ndarray:
            return t[0, 0].detach().cpu().float().numpy()

        steps = [("Raw", _np(x))]
        x = self._normalize(x);                        steps.append(("Normalized", _np(x)))
        x = self._denoise(x);                          steps.append(("Denoised", _np(x)))
        x = self._normalize(x);                        steps.append(("Renormalized", _np(x)))
        x = torch.sigmoid(self.eta * (x - self.tau));  steps.append(("Soft mask", _np(x)))
        x = F.conv2d(x, self._mean3, padding=1);       steps.append(("Eroded", _np(x)))
        x = F.relu(F.conv2d(x, self._sobel_y, padding=1))
        steps.append(("Upper contour (raw)", _np(x)))
        x = self._widen_band(x)
        steps.append(("Upper contour", _np(x)))
        return steps


def contour_center_of_mass(contours: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable COM of contour images, in bitmap pixel coordinates.

    Mirrors the conventions of ``artist.flux.get_center_of_mass``: the tensor
    is laid out ``[N, u(row), e(col)]`` and the result is (e_px, u_px) pairs,
    directly consumable by ``bitmap_coordinates_to_target_coordinates``.

    Parameters
    ----------
    contours : torch.Tensor
        Soft contour images. Shape ``[N, H, W]``.

    Returns
    -------
    torch.Tensor
        Pixel COM as (e, u) pairs. Shape ``[N, 2]``.
    """
    N, H, W = contours.shape
    u_coords = torch.arange(H, device=contours.device, dtype=contours.dtype)
    e_coords = torch.arange(W, device=contours.device, dtype=contours.dtype)
    mass = contours.sum(dim=(-2, -1)) + eps
    u_com = (contours * u_coords.view(1, H, 1)).sum(dim=(-2, -1)) / mass
    e_com = (contours * e_coords.view(1, 1, W)).sum(dim=(-2, -1)) / mass
    return torch.stack([e_com, u_com], dim=1)


def gt_distance_maps(gt_contours: torch.Tensor, tau: float) -> torch.Tensor:
    """Euclidean distance map D_G of the binarized GT contour (eq. 4.34–4.35).

    Hard threshold is fine here: no gradient is needed on the target side.
    The mask is INVERTED before the transform so each pixel stores its distance
    to the nearest CONTOUR pixel (eq. 4.35 prints the constraint backwards).

    Parameters
    ----------
    gt_contours : torch.Tensor
        Ground-truth soft contours. Shape ``[N, H, W]``.
    tau : float
        Binarization threshold applied to the contour values (eq. 4.34).

    Returns
    -------
    torch.Tensor
        Detached distance maps on the input's device/dtype. Shape ``[N, H, W]``.
        Zero on the contour, growing away from it; all-zero if the contour is
        empty (no pull rather than an arbitrary one).
    """
    binary = (gt_contours.detach() > tau).cpu().numpy()
    maps = np.stack(
        [
            distance_transform_edt(~b).astype(np.float32)
            if b.any()
            else np.zeros_like(b, dtype=np.float32)
            for b in binary
        ]
    )
    return torch.from_numpy(maps).to(device=gt_contours.device, dtype=gt_contours.dtype).detach()


@dataclass
class ContourGroundTruth:
    """Precomputed, fully detached ground-truth side of the contour loss.

    Built once per data split — the measured flux never changes during
    training, so neither do these.
    """

    contours: torch.Tensor       # C_G      [N, H, W]
    distance_maps: torch.Tensor  # D_G      [N, H, W]
    com_enu: torch.Tensor        # COM_ENU  [N, 4] homogeneous world coords


def build_contour_ground_truth(
    measured_flux: torch.Tensor,
    extractor: ContourExtractor,
    bitmap_resolution: torch.Tensor,
    solar_tower,
    target_area_indices: torch.Tensor,
    device: torch.device,
    chunk_size: int = 32,
) -> ContourGroundTruth:
    """Precompute the constant GT contours, distance maps and ENU COMs.

    Parameters
    ----------
    measured_flux : torch.Tensor
        Measured flux images in [0, 1]. Shape ``[N, H, W]``. Resized to the
        ray tracer's bitmap resolution if it differs.
    extractor : ContourExtractor
        The SAME extractor instance used on the predicted side.
    bitmap_resolution : torch.Tensor
        Ray-tracer bitmap resolution (width, height). Shape ``[2]``.
    solar_tower : SolarTower
        Tower with all target-area definitions (for the ENU COM mapping).
    target_area_indices : torch.Tensor
        Per-sample target area index. Shape ``[N]``.
    chunk_size : int
        Samples per extraction chunk (bounds peak memory).

    Returns
    -------
    ContourGroundTruth
        Everything detached, on ``device``.
    """
    res_hw = (
        int(bitmap_resolution[indices.unbatched_bitmap_u]),
        int(bitmap_resolution[indices.unbatched_bitmap_e]),
    )
    with torch.no_grad():
        flux = measured_flux.to(device=device, dtype=torch.float32)
        if tuple(flux.shape[-2:]) != res_hw:
            flux = F.interpolate(
                flux.unsqueeze(1), size=res_hw, mode="bilinear", align_corners=False
            ).squeeze(1)
        contours = torch.cat(
            [extractor(flux[i : i + chunk_size]) for i in range(0, flux.shape[0], chunk_size)]
        )
        dmaps = gt_distance_maps(contours, extractor.tau)
        com_enu = bitmap_coordinates_to_target_coordinates(
            bitmap_coordinates=contour_center_of_mass(contours),
            bitmap_resolution=bitmap_resolution,
            solar_tower=solar_tower,
            target_area_indices=target_area_indices.to(device),
            device=device,
        )
    return ContourGroundTruth(
        contours=contours.detach(),
        distance_maps=dmaps.detach(),
        com_enu=com_enu.detach(),
    )


class WortbergContourLoss:
    """Three-term weighted contour loss (eq. 4.36–4.41).

    Parameters
    ----------
    extractor : ContourExtractor
        Contour extraction pipeline, applied to the predicted flux.
    weight_coarse : float
        β — coarse (soft distance field) weight. NOTE: the raw coarse term is
        an unnormalized pixel sum (contour mass × pixel distance, magnitude
        ~1e3–1e5 on 256² images), so β must be small for Fine/Gravity to
        matter.
    weight_gravity : float
        γ — gravity (COM distance, metres) weight.
        Fine (soft DICE, dimensionless in [0, 1]) receives 1 − β − γ.
    dice_eps : float
        ε in the soft DICE (eq. 4.38).
    coarse_scale : float
        Divides the raw Coarse term before weighting (default 1.0, a no-op —
        preserves exact prior behavior for any existing caller). Raw Coarse is
        an unnormalized pixel-mass-times-distance sum, ~1e3 on 256² images, so
        with coarse_scale=1.0 β must stay tiny (~1e-4) for Fine/Gravity to
        matter at all. Passing a representative magnitude here (e.g. the
        empirical mean Coarse value from a prior run) turns β into a genuine
        0–1 mixing weight, comparable to weight_fine and weight_gravity —
        the normalization the contour-loss retuning plan calls "strongly
        preferred" before doing a β/γ hyperparameter search.
    gravity_scale : float
        Divides the raw Gravity term before weighting (default 1.0, a no-op).
        Raw Gravity is a metres-scale COM distance, ~0.05–0.15 m in this
        project's runs — same purpose as coarse_scale, for γ.
    """

    def __init__(
        self,
        extractor: ContourExtractor,
        weight_coarse: float,
        weight_gravity: float,
        dice_eps: float = 1e-6,
        coarse_scale: float = 1.0,
        gravity_scale: float = 1.0,
    ) -> None:
        if weight_coarse < 0 or weight_gravity < 0 or weight_coarse + weight_gravity > 1:
            raise ValueError("Require β, γ ≥ 0 and β + γ ≤ 1")
        if coarse_scale <= 0 or gravity_scale <= 0:
            raise ValueError("coarse_scale and gravity_scale must be > 0")
        self.extractor = extractor
        self.weight_coarse = weight_coarse
        self.weight_gravity = weight_gravity
        self.weight_fine = 1.0 - weight_coarse - weight_gravity
        self.dice_eps = dice_eps
        self.coarse_scale = coarse_scale
        self.gravity_scale = gravity_scale

    def __call__(
        self,
        prediction: torch.Tensor,
        gt_contours: torch.Tensor,
        gt_distance_maps: torch.Tensor,
        gt_com_enu: torch.Tensor,
        target_area_indices: torch.Tensor,
        bitmap_resolution: torch.Tensor,
        solar_tower,
        device: torch.device,
        gt_centroid_full: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Per-sample contour loss.

        ``gt_centroid_full`` is accepted and ignored — present only so this
        class shares an identical call signature with
        :class:`HybridFocalContourLoss`, letting callers use either
        interchangeably without branching.

        Parameters
        ----------
        prediction : torch.Tensor
            LIVE predicted flux from the ray tracer (never detach before the
            loss). Shape ``[N, H, W]``.
        gt_contours, gt_distance_maps : torch.Tensor
            Slices of a ``ContourGroundTruth``, aligned to the prediction
            (i.e. already reordered by the ray tracer's sampler indices).
            Shape ``[N, H, W]`` each.
        gt_com_enu : torch.Tensor
            GT contour COM in homogeneous world coords. Shape ``[N, 4]``.
        target_area_indices : torch.Tensor
            Per-sample target area index, aligned to the prediction. ``[N]``.
        bitmap_resolution : torch.Tensor
            Ray-tracer bitmap resolution (width, height). Shape ``[2]``.

        Returns
        -------
        (loss, components)
            ``loss`` — per-sample weighted total, shape ``[N]``;
            ``components`` — detached unweighted per-term means
            ``{"coarse", "fine", "gravity"}`` for logging.
        """
        c_pred = self.extractor(prediction)

        # Coarse (eq. 4.36): predicted contour mass priced by distance to C_G.
        coarse = (c_pred * gt_distance_maps).sum(dim=(-2, -1))

        # Fine (eq. 4.38–4.39): 1 − soft DICE.
        inter = (c_pred * gt_contours).sum(dim=(-2, -1))
        denom = c_pred.sum(dim=(-2, -1)) + gt_contours.sum(dim=(-2, -1))
        fine = 1.0 - (2.0 * inter + self.dice_eps) / (denom + self.dice_eps)

        # Gravity (eq. 4.40): COM distance in metres on the target surface.
        com_pred_enu = bitmap_coordinates_to_target_coordinates(
            bitmap_coordinates=contour_center_of_mass(c_pred),
            bitmap_resolution=bitmap_resolution,
            solar_tower=solar_tower,
            target_area_indices=target_area_indices,
            device=device,
        )
        gravity = torch.linalg.norm(com_pred_enu[:, :3] - gt_com_enu[:, :3], dim=-1)

        total = (
            self.weight_fine * fine
            + self.weight_coarse * (coarse / self.coarse_scale)
            + self.weight_gravity * (gravity / self.gravity_scale)
        )
        components = {
            "coarse": float(coarse.detach().mean()),
            "fine": float(fine.detach().mean()),
            "gravity": float(gravity.detach().mean()),
        }
        return total, components


class HybridFocalContourLoss:
    """Blend FocalSpotLoss (whole-image centroid) with WortbergContourLoss.

    total = focal_weight * (focal_term / focal_scale) + (1 - focal_weight) * contour_total

    Motivation: the pure contour loss deliberately discards everything but a
    thin upper rim of the flux image, which is a pure information cost when
    occlusion is mild (there is nothing in the discarded region to be robust
    to). This blend keeps the WHOLE image in play via the focal-spot term at
    all times, and adds the contour term on top purely as an occlusion-
    robustness supplement, rather than a full replacement — the loss should
    then track focal-spot when blocking is light and gain contour's
    robustness only where it's actually needed.

    Parameters
    ----------
    contour_loss : WortbergContourLoss
        The wrapped contour loss (already carries its own beta/gamma and
        coarse_scale/gravity_scale).
    focal_weight : float
        Mixing weight on the focal-spot term, in [0, 1]. 0 reproduces pure
        WortbergContourLoss; 1 reproduces pure FocalSpotLoss.
    focal_scale : float
        Divides the raw focal term (squared metres) before weighting, same
        normalization purpose as WortbergContourLoss's coarse_scale/
        gravity_scale — pick a representative magnitude from a prior run so
        focal_weight is a genuine mixing weight, not swamped by scale.
    """

    def __init__(
        self,
        contour_loss: WortbergContourLoss,
        focal_weight: float,
        focal_scale: float = 1.0,
    ) -> None:
        if not 0.0 <= focal_weight <= 1.0:
            raise ValueError("focal_weight must be in [0, 1]")
        if focal_scale <= 0:
            raise ValueError("focal_scale must be > 0")
        self.contour_loss = contour_loss
        self.focal_weight = focal_weight
        self.focal_scale = focal_scale

    def __call__(
        self,
        prediction: torch.Tensor,
        gt_contours: torch.Tensor,
        gt_distance_maps: torch.Tensor,
        gt_com_enu: torch.Tensor,
        target_area_indices: torch.Tensor,
        bitmap_resolution: torch.Tensor,
        solar_tower,
        device: torch.device,
        gt_centroid_full: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Per-sample hybrid loss. ``gt_centroid_full`` (the TRUE, whole-image
        ground-truth centroid, not the contour-restricted one) is required.
        """
        if gt_centroid_full is None:
            raise ValueError("HybridFocalContourLoss requires gt_centroid_full")

        contour_total, comps = self.contour_loss(
            prediction, gt_contours, gt_distance_maps, gt_com_enu,
            target_area_indices, bitmap_resolution, solar_tower, device,
        )

        bitmap_coords = get_center_of_mass(bitmaps=prediction, device=device)
        pred_coords = bitmap_coordinates_to_target_coordinates(
            bitmap_coordinates=bitmap_coords,
            bitmap_resolution=bitmap_resolution,
            solar_tower=solar_tower,
            target_area_indices=target_area_indices,
            device=device,
        )
        focal_raw = ((pred_coords[:, :3] - gt_centroid_full[:, :3]) ** 2).sum(dim=-1)
        focal_term = focal_raw / self.focal_scale

        total = self.focal_weight * focal_term + (1.0 - self.focal_weight) * contour_total
        comps["focal"] = float(focal_raw.detach().mean())
        return total, comps
