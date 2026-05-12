from __future__ import annotations

import torch

from full_training_pipeline.model import (
    INPUT_DIM,
    SHARED_WORTBERG_PARAMETER_NAMES,
    SHARED_WORTBERG_RESIDUAL_BOUNDS,
    SharedLinearResidualModel,
)


class SharedSNNResidualModel(SharedLinearResidualModel):
    """
    Self-Normalizing MLP residual model shared across all heliostats.

    Architecture:
        42D input → [Linear(in→h) → SELU] × n_hidden → Linear(h→20) → tanh × bounds

    Inherits feature extraction (_select_and_flatten, _aggregate_measurements) from
    SharedLinearResidualModel but replaces the single linear layer with a deep SELU
    network. torch.nn.Module.__init__ is called directly to avoid creating the unused
    self.linear from the parent class.

    Init strategy:
        - Hidden layers: Lecun normal (recommended for SELU, equivalent to kaiming_normal
          with fan_in and linear nonlinearity).
        - Output layer: zero weights and bias so corrections start at zero, identical to
          the linear and polynomial models.
    """

    def __init__(self, hidden_size: int = 16, n_hidden: int = 4) -> None:
        torch.nn.Module.__init__(self)
        self.register_buffer("residual_bounds", SHARED_WORTBERG_RESIDUAL_BOUNDS.clone())

        layers: list[torch.nn.Module] = []
        in_dim = INPUT_DIM
        for _ in range(n_hidden):
            layers.append(torch.nn.Linear(in_dim, hidden_size))
            layers.append(torch.nn.SELU())
            in_dim = hidden_size
        layers.append(torch.nn.Linear(in_dim, len(SHARED_WORTBERG_PARAMETER_NAMES)))
        self.net = torch.nn.Sequential(*layers)

        for module in self.net:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
                torch.nn.init.zeros_(module.bias)

        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, inputs: list) -> torch.Tensor:
        device = self.residual_bounds.device
        zero = torch.zeros(INPUT_DIM, dtype=torch.float32, device=device)
        rows = [
            self._select_and_flatten(inp) if inp is not None else zero
            for inp in inputs
        ]
        features = torch.stack(rows, dim=0)  # (N_heliostats, 42)
        return torch.tanh(self.net(features)) * self.residual_bounds
