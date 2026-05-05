from __future__ import annotations

import torch

from full_training_pipeline.features import HeliostatCalibrationInput

SHARED_WORTBERG_PARAMETER_NAMES: tuple[str, ...] = (
    "translation_0",
    "translation_1",
    "translation_2",
    "translation_3",
    "translation_4",
    "translation_5",
    "translation_6",
    "translation_7",
    "translation_8",
    "rotation_0",
    "rotation_1",
    "rotation_2",
    "rotation_3",
    "actuator_initial_angle_0",
    "actuator_initial_angle_1",
    "actuator_offset_0",
    "actuator_offset_1",
    "base_position_e",
    "base_position_n",
    "base_position_u",
)

SHARED_WORTBERG_RESIDUAL_BOUNDS = torch.tensor(
    [0.05] * 9 + [0.005] * 4 + [0.005] * 2 + [0.005] * 2 + [0.05] * 3,
    dtype=torch.float32,
)

_N_KINEMATIC = 20
_N_HELIOSTAT_POS = 3
_N_PER_MEASUREMENT = 8  # sun(3) + motor(2) + centroid(3)


class SharedLinearResidualModel(torch.nn.Module):
    """
    Linear residual model shared across all heliostats.

    Input: HeliostatCalibrationInput (or a list thereof for batch prediction).
    The model sorts measurements by sun elevation (ascending) internally so the
    linear weights have a canonical, permutation-stable interpretation.

    Input layout (after _select_and_flatten):
        heliostat_position  (3,)
        kinematic_params    (20,)
        [sun_x, sun_y, sun_z, motor_1, motor_2, cen_e, cen_n, cen_u] × n_measurements

    Total: 3 + 20 + n_measurements × 8
    """

    def __init__(self, n_measurements: int) -> None:
        super().__init__()
        self.n_measurements = n_measurements
        input_dim = _N_HELIOSTAT_POS + _N_KINEMATIC + n_measurements * _N_PER_MEASUREMENT
        self.linear = torch.nn.Linear(input_dim, len(SHARED_WORTBERG_PARAMETER_NAMES))
        self.register_buffer("residual_bounds", SHARED_WORTBERG_RESIDUAL_BOUNDS.clone())
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    # ------------------------------------------------------------------
    # Internal feature extractor (where sorting lives)
    # ------------------------------------------------------------------

    def _select_and_flatten(self, inp: HeliostatCalibrationInput) -> torch.Tensor:
        device = self.residual_bounds.device

        sun = inp.sun_directions.to(device)    # (N_meas, 3)
        motor = inp.motor_positions.to(device)  # (N_meas, 2)
        centroid = inp.centroids.to(device)     # (N_meas, 3)

        # Sort by sun elevation ascending: sun[:, 2] = sin(elevation).
        sort_idx = torch.argsort(sun[:, 2])
        sun = sun[sort_idx]
        motor = motor[sort_idx]
        centroid = centroid[sort_idx]

        per_meas = torch.cat([sun, motor, centroid], dim=1)  # (N_meas, 8)

        # Pad or truncate to exactly n_measurements.
        n = per_meas.shape[0]
        if n < self.n_measurements:
            pad = torch.zeros(self.n_measurements - n, _N_PER_MEASUREMENT, device=device)
            per_meas = torch.cat([per_meas, pad], dim=0)
        else:
            per_meas = per_meas[: self.n_measurements]

        helpos = inp.heliostat_position.to(device)   # (3,)
        kp = inp.kinematic_params.to(device)         # (20,)
        return torch.cat([helpos, kp, per_meas.flatten()])  # (input_dim,)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, inputs: list[HeliostatCalibrationInput | None]
    ) -> torch.Tensor:
        device = self.residual_bounds.device
        zero = torch.zeros(
            _N_HELIOSTAT_POS + _N_KINEMATIC + self.n_measurements * _N_PER_MEASUREMENT,
            dtype=torch.float32,
            device=device,
        )
        rows = [
            self._select_and_flatten(inp) if inp is not None else zero
            for inp in inputs
        ]
        features = torch.stack(rows, dim=0)  # (N_heliostats, input_dim)
        return torch.tanh(self.linear(features)) * self.residual_bounds
