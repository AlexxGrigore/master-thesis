"""
Configuration for the perturbation visualizer experiment.

Edit this file to control which heliostat to trace, the sun position,
and the kinematic perturbations you want to apply.  The run script
generates a side-by-side flux comparison (clean vs. perturbed) and
reports the resulting centroid shift.

Sun position convention (ARTIST / PAINT)
-----------------------------------------
Azimuth is south-oriented: 0° = south, 90° = west, -90°/270° = east.
Elevation is degrees above the horizon.

Perturbation parameters
-----------------------
rotation_rad       : 4 joint-rotation deviations (rad)
actuator_angle_rad : 2 actuator initial-angle deviations a_i (rad)
actuator_stroke_m  : 2 actuator initial-stroke-length deviations b_i (m)
actuator_offset_m  : 2 actuator offset deviations c_i (m)
translation_m      : 9 joint+concentrator translation deviations (m)
base_position_m    : 3 heliostat base-position offset (east, north, up) (m)

Set all values to 0.0 for the clean (unperturbed) reference.
"""
import pathlib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]  # master-thesis/

# The 63-heliostat scenario used in the KR experiment.
SCENARIO_PATH = BASE_DIR / "scenarios" / "full_field_200_samples_scenario" / "scenario.h5"

OUTPUT_DIR = BASE_DIR / "outputs" / "perturbation_visualizer"

# ---------------------------------------------------------------------------
# Heliostat & target
# ---------------------------------------------------------------------------

# ID of the heliostat to trace (must exist in the scenario).
HELIOSTAT_ID = "AA23"

# Index of the receiver target area (0-based; typically 0 for the main target).
TARGET_AREA_INDEX = 0

# ---------------------------------------------------------------------------
# Sun position
# ---------------------------------------------------------------------------

# South-oriented azimuth (0° = south) and elevation above horizon, in degrees.
SUN_AZIMUTH_DEG   = 0.0    # sun due south
SUN_ELEVATION_DEG = 40.0   # 40° above horizon

# ---------------------------------------------------------------------------
# Ray-tracer settings
# ---------------------------------------------------------------------------

N_RAYS = 500                 # rays per light source; increase for cleaner images
SURFACE_POINTS_PER_FACET = 10  # (10×10 = 100 pts/facet); increase for more detail

# ---------------------------------------------------------------------------
# Perturbations to apply
# ---------------------------------------------------------------------------
# Each list contains delta values that are ADDED to the clean kinematic
# parameters.  Set an entry to 0.0 to leave that parameter unperturbed.

PERTURBATIONS = {
    # Tilt (rotation) deviations of the two kinematic joints (rad).
    # Each value rotates one joint axis, changing the mirror normal and
    # therefore tilting the reflected beam on the target.
    # [0] first_joint_tilt_n  — tilt of joint 1 around the north axis
    # [1] first_joint_tilt_u  — tilt of joint 1 around the up axis
    # [2] second_joint_tilt_e — tilt of joint 2 around the east axis
    # [3] second_joint_tilt_n — tilt of joint 2 around the north axis
    "rotation_rad":        [0.05, 0, 0.0, 0.0],

    # Actuator initial-angle deviation a_i (rad).
    # Shifts the zero-point of each actuator's angular range; effectively
    # biases how far each actuator extends for a given motor-position command,
    # causing a tracking offset that grows with sun elevation.
    # [0] actuator 1 (azimuth axis)   [1] actuator 2 (elevation axis)
    "actuator_angle_rad":  [0.0,    0.0],

    # Actuator initial-stroke-length deviation b_i (m).
    # Scales the effective stroke of each linear actuator; a non-zero value
    # stretches or compresses the whole motor-to-angle mapping, introducing
    # a tracking error that varies across the day.
    # [0] actuator 1   [1] actuator 2
    "actuator_stroke_m":   [0.0,    0.0],

    # Actuator offset deviation c_i (m).
    # Translates the mounting point of each actuator arm, shifting the
    # mechanical zero of that axis and producing a constant pointing bias.
    # [0] actuator 1   [1] actuator 2
    "actuator_offset_m":   [0.0,    0.0],

    # Translation deviations of the two joints and the concentrator (m).
    # Each value shifts a rigid-body frame origin in one ENU direction,
    # misplacing the rotation axis and distorting the beam direction.
    # [0] first_joint_translation_e   [1] first_joint_translation_n   [2] first_joint_translation_u
    # [3] second_joint_translation_e  [4] second_joint_translation_n  [5] second_joint_translation_u
    # [6] concentrator_translation_e  [7] concentrator_translation_n  [8] concentrator_translation_u
    "translation_m":       [0.0] * 9,

    # Global offset of the heliostat base position in ENU (m).
    # Moves the entire heliostat (both joints + mirror) from its nominal
    # position; the tracking geometry is then computed relative to the wrong
    # origin, shifting the focal spot on the target.
    # [0] east   [1] north   [2] up
    "base_position_m":     [0.0,    0.0,   0.0],
}
