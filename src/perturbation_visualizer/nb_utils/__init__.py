"""Shared utilities for the perturbation-visualiser demo notebooks."""

from .defaults import DEFAULT_TRAIN_CFG, WORTBERG_BOUNDS
from .evaluation import (
    active_pixel_percent,
    bitmap_centroid,
    compute_mrad,
    current_base_pos,
    eval_mrad,
)
from .kinematics import (
    apply_bounds,
    clip_and_step,
    reset_for_new_run,
    restore_kinematics,
    setup_optimizer,
    snapshot_kinematics,
)
from .losses import val_alignment_loss, val_focal_loss
from .visualization import normalize_for_display, rays_to_polar

__all__ = [
    # defaults
    "WORTBERG_BOUNDS",
    "DEFAULT_TRAIN_CFG",
    # kinematics
    "snapshot_kinematics",
    "restore_kinematics",
    "setup_optimizer",
    "apply_bounds",
    "clip_and_step",
    "reset_for_new_run",
    # evaluation
    "bitmap_centroid",
    "active_pixel_percent",
    "compute_mrad",
    "current_base_pos",
    "eval_mrad",
    # losses
    "val_alignment_loss",
    "val_focal_loss",
    # visualization
    "normalize_for_display",
    "rays_to_polar",
]
