"""Shared constants for the perturbation-visualiser demo notebooks."""

# Wortberg (2025) Table 5.3 deviation bounds — maximum clamp range for each
# kinematic parameter group during training.
WORTBERG_BOUNDS = {
    "translation_m":      0.05,
    "rotation_rad":       0.005,
    "actuator_angle_rad": 0.005,
    "actuator_offset_m":  0.005,
    "base_position_m":    0.05,
}

# Sensible defaults shared across the training notebooks.  Override only the
# keys you need to change in each notebook's config cell.
DEFAULT_TRAIN_CFG = {
    "surface_points_per_facet": 25,
    "train_rays":               10,
    "display_rays":             50,
    "stage1_epochs":            20,
    "stage2_epochs":            100,
    "mini_batch_size":          25,
    "base_lr":                  1e-4,
    "min_active_pixel_pct":     2.0,
}
