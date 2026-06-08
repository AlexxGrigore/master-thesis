"""Flux display helpers and sun-position utilities for demo notebooks."""

import numpy as np
import torch


def normalize_for_display(
    flux_img: torch.Tensor,
    reference: torch.Tensor | None = None,
) -> np.ndarray:
    """Return a [H, W] float64 array in [0, 1] suitable for imshow.

    flux_img   : raw flux tensor (either GT normalized PNG or ray-tracer output).
    reference  : when provided, energy-rescale flux_img to match reference total
                 before peak-normalising.  Use this to make ray-tracer output
                 (physical intensity units) visible on the same colour scale as
                 a [0,1]-normalised GT image.
    """
    arr = flux_img.cpu().numpy().astype(np.float64)
    if reference is not None:
        pred_sum = arr.sum()
        ref_sum = reference.cpu().numpy().sum()
        if pred_sum > 1e-12 and ref_sum > 1e-12:
            arr = arr * (ref_sum / pred_sum)
    peak = arr.max()
    return arr / peak if peak > 1e-12 else arr


def rays_to_polar(rays: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Convert incident-ray direction tensor to polar plot coordinates.

    Returns (az_plot_rad, zenith_deg) suitable for matplotlib polar axes using
    the European solar convention (South at top, clockwise).
    """
    sun = -rays.cpu().numpy()
    az = np.arctan2(sun[:, 0], sun[:, 1])
    az_plot = 3 * np.pi / 2 - az
    el = np.degrees(np.arcsin(np.clip(sun[:, 2], -1.0, 1.0)))
    zen = 90.0 - el
    return az_plot, zen
