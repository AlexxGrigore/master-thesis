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

# Aggregated measurement features (fixed size regardless of number of measurements):
#   mean_centroid   (3)  — mean ENU centroid across all measurements
#   std_centroid    (3)  — spread of centroid positions
#   range_centroid  (3)  — max - min centroid per axis
#   mean_sun        (3)  — mean sun direction
#   std_sun         (3)  — spread of sun directions (measurement coverage quality)
#   mean_motor      (2)  — mean motor encoder readings
#   std_motor       (2)  — spread of motor readings
_N_AGG = 19

INPUT_DIM = _N_HELIOSTAT_POS + _N_KINEMATIC + _N_AGG  # 42


class SharedLinearResidualModel(torch.nn.Module):
    """
    Linear residual model shared across all heliostats.

    Replaces per-measurement flattening (which caused severe overfitting with
    ~16K weights vs ~63 training heliostats) with 19 fixed aggregate statistics
    computed from all available measurements, giving a 42-D input and 880
    learnable parameters total.

    Input layout (after _select_and_flatten):
        heliostat_position  (3,)   — absolute ENU position
        kinematic_params    (20,)  — coarse checkpoint parameters
        aggregated_meas     (19,)  — summary statistics over all measurements
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(INPUT_DIM, len(SHARED_WORTBERG_PARAMETER_NAMES))
        self.register_buffer("residual_bounds", SHARED_WORTBERG_RESIDUAL_BOUNDS.clone())
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    # ------------------------------------------------------------------
    # Internal feature extractor
    # ------------------------------------------------------------------

    def _aggregate_measurements(self, inp: HeliostatCalibrationInput) -> torch.Tensor:
        """Compute 19 fixed-size summary statistics from variable-length measurements."""
        device = self.residual_bounds.device
        sun = inp.sun_directions.to(device)      # (N, 3)
        motor = inp.motor_positions.to(device)   # (N, 2)
        centroid = inp.centroids.to(device)      # (N, 3)

        mean_cen = centroid.mean(dim=0)                                          # (3,)
        std_cen = centroid.std(dim=0).nan_to_num(0.0)                            # (3,)
        range_cen = centroid.max(dim=0).values - centroid.min(dim=0).values      # (3,)
        mean_sun = sun.mean(dim=0)                                               # (3,)
        std_sun = sun.std(dim=0).nan_to_num(0.0)                                 # (3,)
        mean_motor = motor.mean(dim=0)                                           # (2,)
        std_motor = motor.std(dim=0).nan_to_num(0.0)                             # (2,)

        return torch.cat([mean_cen, std_cen, range_cen, mean_sun, std_sun, mean_motor, std_motor])

    def _select_and_flatten(self, inp: HeliostatCalibrationInput) -> torch.Tensor:
        device = self.residual_bounds.device
        helpos = inp.heliostat_position.to(device)    # (3,)
        kp = inp.kinematic_params.to(device)          # (20,)
        agg = self._aggregate_measurements(inp)       # (19,)
        return torch.cat([helpos, kp, agg])           # (42,)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, inputs: list[HeliostatCalibrationInput | None]
    ) -> torch.Tensor:
        device = self.residual_bounds.device
        zero = torch.zeros(INPUT_DIM, dtype=torch.float32, device=device)
        rows = [
            self._select_and_flatten(inp) if inp is not None else zero
            for inp in inputs
        ]
        features = torch.stack(rows, dim=0)  # (N_heliostats, 42)
        return torch.tanh(self.linear(features)) * self.residual_bounds


class SharedPolyResidualModel(SharedLinearResidualModel):
    """
    Polynomial residual model: expands the 42-D base features with power terms
    [x, x², ..., x^degree] (no cross-terms) before the linear output layer.

    Parameter counts vs SharedLinearResidualModel (860 params):
        degree=2 → input 84-D  → 1,700 params  (27:1 ratio for 63 heliostats)
        degree=3 → input 126-D → 2,540 params  (40:1 ratio for 63 heliostats)
    """

    def __init__(self, degree: int = 2) -> None:
        super().__init__()
        if degree < 2:
            raise ValueError("Use SharedLinearResidualModel for degree=1.")
        self.degree = degree
        expanded_dim = INPUT_DIM * degree
        self.linear = torch.nn.Linear(expanded_dim, len(SHARED_WORTBERG_PARAMETER_NAMES))
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(
        self, inputs: list[HeliostatCalibrationInput | None]
    ) -> torch.Tensor:
        device = self.residual_bounds.device
        zero = torch.zeros(INPUT_DIM, dtype=torch.float32, device=device)
        rows = [
            self._select_and_flatten(inp) if inp is not None else zero
            for inp in inputs
        ]
        features = torch.stack(rows, dim=0)  # (N_heliostats, 42)
        poly = torch.cat(
            [features ** k for k in range(1, self.degree + 1)], dim=-1
        )  # (N_heliostats, 42 * degree)
        return torch.tanh(self.linear(poly)) * self.residual_bounds


def build_residual_model(model_type: str) -> torch.nn.Module:
    """Factory — returns the correct model for a given model_type string."""
    if model_type == "linear":
        return SharedLinearResidualModel()
    if model_type == "poly2":
        return SharedPolyResidualModel(degree=2)
    if model_type == "poly3":
        return SharedPolyResidualModel(degree=3)
    if model_type == "poly4":
        return SharedPolyResidualModel(degree=4)
    if model_type == "transformer":
        from full_training_pipeline.model_transformer import SharedTransformerResidualModel
        return SharedTransformerResidualModel()
    raise ValueError(
        f"Unknown model_type {model_type!r}. Choose from: linear, poly2, poly3, poly4, transformer"
    )
