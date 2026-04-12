from __future__ import annotations

import torch

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


class SharedLinearResidualModel(torch.nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, len(SHARED_WORTBERG_PARAMETER_NAMES))
        self.register_buffer("residual_bounds", SHARED_WORTBERG_RESIDUAL_BOUNDS.clone())
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(features)) * self.residual_bounds