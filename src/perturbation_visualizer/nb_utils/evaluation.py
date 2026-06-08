"""Evaluation helpers: mrad error, centroid computation, active-pixel filter."""

import numpy as np
import torch


def bitmap_centroid(flux_img: torch.Tensor) -> tuple[float | None, float | None]:
    """Centre-of-mass (col, row) of a [H, W] flux tensor, or (None, None) if empty."""
    f = flux_img.cpu().float().numpy()
    total = f.sum()
    if total < 1e-12:
        return None, None
    h, w = f.shape
    cols = np.arange(w, dtype=np.float32).reshape(1, -1)
    rows = np.arange(h, dtype=np.float32).reshape(-1, 1)
    return float((f * cols).sum() / total), float((f * rows).sum() / total)


def active_pixel_percent(flux_img: torch.Tensor) -> float:
    """Percentage of pixels above 0.01 flux threshold in a [H, W] tensor."""
    return float((flux_img > 0.01).sum().item()) / float(flux_img.numel()) * 100.0


def compute_mrad(
    pred_cents: torch.Tensor,
    gt_cents: torch.Tensor,
    hel_dist_m: float,
) -> np.ndarray:
    """L2 centroid error in ENU coords (first 3 dims), converted to mrad."""
    return (
        torch.norm(pred_cents[:, :3] - gt_cents[:, :3], dim=1) / hel_dist_m * 1000
    ).cpu().numpy()


def current_base_pos(heliostat_group, device: torch.device) -> torch.Tensor:
    """Return k._base_position_deviation [1, 3], or zeros if not yet created."""
    k = heliostat_group.kinematics
    if hasattr(k, "_base_position_deviation"):
        return k._base_position_deviation.detach()
    return torch.zeros(1, 3, device=device)


@torch.no_grad()
def eval_mrad(
    rays: torch.Tensor,
    active_mask: torch.Tensor,
    target_mask: torch.Tensor,
    gt_cents: torch.Tensor,
    *,
    scenario,
    heliostat_group,
    forward_pass_fn,
    hel_dist_m: float,
    device: torch.device,
) -> tuple[np.ndarray, float, float]:
    """Forward pass then centroid error. Returns (errs_array, mean_mrad, median_mrad)."""
    base_pos = current_base_pos(heliostat_group, device)
    pred_cents, _ = forward_pass_fn(
        scenario, heliostat_group,
        rays, active_mask, target_mask,
        base_pos, device,
    )
    errs = compute_mrad(pred_cents, gt_cents, hel_dist_m)
    return errs, float(np.nanmean(errs)), float(np.nanmedian(errs))
