"""
Tests for the fine_error_learning package.

Run with:
    cd master-thesis
    pytest tests/test_fine_error_learning.py -v

Covers:
  1. Permutation invariance — shuffling the measurement set (with its mask)
     leaves Δθ identical (no positional encoding + masked mean pooling).
  2. Zero-init head — Δθ == 0 at initialization (training starts at θ_KR).
  3. Shapes — Δθ is [B, 24]; masked pooling ignores padding tokens (pad
     content does not change the output).
  4. Gradient flow — one end-to-end step through the ARTIST ray tracer on a
     real per-heliostat scenario with its stage-1 checkpoint: forward +
     focal-spot loss + backward reaches the model weights.

Test 4 uses AB43 with the synthetic stage-1 checkpoints (config
STAGE1_CHECKPOINT_DIR). The stage-1 foundation brings the beam on target, so
the focal-spot gradient is nonzero. (With off-target warm starts — e.g. the
old real-data checkpoints on synthetic data — the gradient is exactly zero;
see warm_start.measure_on_target_fraction.)
"""

import pathlib
import sys

import pytest
import torch

# Make src/ importable so fine_error_learning can be found.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from fine_error_learning import config as fel_config
from fine_error_learning import data as fel_data
from fine_error_learning import pipeline as fel_pipeline
from fine_error_learning import warm_start as fel_warm_start
from fine_error_learning.model import FelTransformerModel

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_GRAD_HELIOSTAT = "AB43"
_GRAD_SCENARIO = (
    _REPO_ROOT / "scenarios" / "one_heliostat_scenarios" / _GRAD_HELIOSTAT / "scenario.h5"
)
_GRAD_CHECKPOINT = (
    _REPO_ROOT / "outputs" / "new_mapping_function" / "all63_stage1"
    / _GRAD_HELIOSTAT / "stage1_checkpoint.pt"
)
_GRAD_DATA = (
    _REPO_ROOT / "datasets" / "synthetic" / "balanced_dataset" / "dataset" / "train"
)
_GRAD_FILES_PRESENT = (
    _GRAD_SCENARIO.exists() and _GRAD_CHECKPOINT.exists() and (_GRAD_DATA / _GRAD_HELIOSTAT).exists()
)


def _small_model(**kwargs) -> FelTransformerModel:
    kwargs.setdefault("d_model", 32)
    kwargs.setdefault("n_heads", 2)
    kwargs.setdefault("n_layers", 1)
    kwargs.setdefault("d_ff", 64)
    kwargs.setdefault("dropout", 0.0)
    kwargs.setdefault("d_img", 16)
    return FelTransformerModel(**kwargs)


def _random_inputs(batch_size: int, k: int, n_real: int, device: torch.device):
    """Random token inputs with n_real real tokens followed by k-n_real pads."""
    flux = torch.rand(batch_size, k, 32, 32, device=device)
    scalars = torch.randn(batch_size, k, fel_data.N_SCALARS, device=device)
    mask = torch.zeros(batch_size, k, dtype=torch.bool, device=device)
    mask[:, :n_real] = True
    theta_kr = torch.randn(batch_size, fel_pipeline.N_PARAMS, device=device) * 0.01
    positions = torch.randn(batch_size, 3, device=device) * 100.0
    return flux, scalars, mask, theta_kr, positions


def test_zero_init_head_gives_zero_delta(device: torch.device) -> None:
    model = _small_model().to(device)
    model.eval()
    flux, scalars, mask, theta_kr, positions = _random_inputs(2, 8, 8, device)
    with torch.no_grad():
        delta = model(flux, scalars, mask, theta_kr, positions)
    assert torch.all(delta == 0.0)


def test_output_shape(device: torch.device) -> None:
    model = _small_model().to(device)
    model.eval()
    flux, scalars, mask, theta_kr, positions = _random_inputs(3, 10, 10, device)
    with torch.no_grad():
        delta = model(flux, scalars, mask, theta_kr, positions)
    assert delta.shape == (3, fel_pipeline.N_PARAMS)


def test_masked_pooling_ignores_pad_tokens(device: torch.device) -> None:
    """Changing the CONTENT of padded tokens must not change Δθ."""
    torch.manual_seed(0)
    model = _small_model().to(device)
    model.eval()
    flux, scalars, mask, theta_kr, positions = _random_inputs(2, 8, 5, device)
    flux_dirty = flux.clone()
    scalars_dirty = scalars.clone()
    flux_dirty[:, 5:] = 1.0
    scalars_dirty[:, 5:] = 1000.0
    with torch.no_grad():
        delta_clean = model(flux, scalars, mask, theta_kr, positions)
        delta_dirty = model(flux_dirty, scalars_dirty, mask, theta_kr, positions)
    assert torch.allclose(delta_clean, delta_dirty)


def test_permutation_invariance(device: torch.device) -> None:
    """Shuffling the measurement set (tokens + mask) leaves Δθ identical."""
    torch.manual_seed(1)
    model = _small_model().to(device)
    model.eval()
    flux, scalars, mask, theta_kr, positions = _random_inputs(2, 8, 8, device)
    permutation = torch.randperm(8)
    with torch.no_grad():
        delta = model(flux, scalars, mask, theta_kr, positions)
        delta_shuffled = model(
            flux[:, permutation], scalars[:, permutation], mask[:, permutation],
            theta_kr, positions,
        )
    assert torch.allclose(delta, delta_shuffled, atol=1e-6)


def test_use_flux_ablation(device: torch.device) -> None:
    """use_flux=False must ignore the images entirely."""
    torch.manual_seed(2)
    model = _small_model(use_flux=False).to(device)
    model.eval()
    flux, scalars, mask, theta_kr, positions = _random_inputs(2, 8, 8, device)
    with torch.no_grad():
        delta = model(None, scalars, mask, theta_kr, positions)
        delta_other_images = model(
            torch.rand_like(flux), scalars, mask, theta_kr, positions
        )
    assert delta.shape == (2, fel_pipeline.N_PARAMS)
    assert torch.allclose(delta, delta_other_images)


@pytest.mark.skipif(not _GRAD_FILES_PRESENT, reason="AB43 scenario/checkpoint/data not present")
def test_gradient_flow_end_to_end(device: torch.device) -> None:
    """Forward + focal-spot loss + backward through ARTIST reaches the model."""
    import types

    cfg = types.SimpleNamespace(
        SCENARIO_PATH_TEMPLATE=fel_config.SCENARIO_PATH_TEMPLATE,
        STAGE1_CHECKPOINT_DIR=fel_config.STAGE1_CHECKPOINT_DIR,
        WARM_START="stage1",
        SURFACE_POINTS_PER_FACET=10,
    )
    state = fel_warm_start.load_warm_start_state(_GRAD_HELIOSTAT, cfg, device)

    measurements = fel_data.load_measurements(
        _GRAD_DATA, _GRAD_HELIOSTAT, state.heliostat_group, state.scenario, device
    ).capped(4)
    scaler = fel_data.compute_scaler([measurements])
    state.scenario.set_number_of_rays(2)

    model = _small_model().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(2):
        flux_tokens, scalars, mask = fel_data.build_model_tokens(
            measurements, k=4, scaler_mean=scaler[0], scaler_std=scaler[1]
        )
        delta = model(
            flux_tokens.unsqueeze(0),
            scalars.unsqueeze(0),
            mask.unsqueeze(0),
            state.theta_kr.unsqueeze(0),
            state.heliostat_position.unsqueeze(0),
        ).squeeze(0)

        flux, bitmap_resolution, sampler_indices = fel_pipeline.predict_flux(
            state=state,
            theta_final=state.theta_kr + delta,
            incident_rays=measurements.incident_rays,
            motor_positions=measurements.motor_positions,
            target_indices=measurements.target_indices,
            device=device,
            random_seed=step,
        )
        lps, _ = fel_pipeline.focal_spot_centroid_loss(
            predicted_flux=flux,
            focal_spots=measurements.focal_spots[sampler_indices],
            target_indices=measurements.target_indices[sampler_indices],
            bitmap_resolution=bitmap_resolution,
            scenario=state.scenario,
            device=device,
        )
        loss = lps.mean() + 1e-4 * delta.pow(2).mean()
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # The zero-init head learns first; after one step it is nonzero and the
    # second backward reaches the encoder/token projection as well.
    head_weight_grad = model.head[-1].weight.grad
    assert head_weight_grad is not None and (head_weight_grad != 0).any()
    encoder_grads = [
        p.grad for n, p in model.named_parameters() if "token_proj" in n or "encoder" in n
    ]
    assert any(g is not None and (g != 0).any() for g in encoder_grads)
