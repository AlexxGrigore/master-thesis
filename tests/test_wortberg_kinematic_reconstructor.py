"""
Tests for WortbergKinematicReconstructor.

These tests use the same toy 4-heliostat ARTIST scenario that ARTIST's own
test_kinematic_reconstructor.py uses, so they run quickly on CPU.

The goal is to verify that WortbergKinematicReconstructor:
  1. Can be constructed (basic smoke test).
  2. Registers the correct parameters as learnable and freezes b_i.
  3. Restricts non_optimizable gradients to c_i (actuator_offset) only.
  4. Creates _base_position_deviation only when train_position_deviation=True.
  5. Keeps all parameters within their deviation bounds after optimizer steps.
  6. Actually reduces the training loss over a short run (the optimizer is doing work).
  7. Handles flat optimization-config dicts (the config-restructuring path in __init__).

Run with:
    cd master-thesis
    pytest tests/test_wortberg_kinematic_reconstructor.py -v
"""

import pathlib
import sys

import h5py
import paint.util.paint_mappings as paint_mappings
import pytest
import torch

# Make src/ importable so artist_extensions can be found.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from artist import ARTIST_ROOT
from artist.core.loss_functions import FocalSpotLoss
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, index_mapping

from artist_extensions.kinematic_reconstructors import WortbergKinematicReconstructor

# ---------------------------------------------------------------------------
# Paths to the ARTIST toy data (reused from ARTIST's own tests)
# ---------------------------------------------------------------------------
_ARTIST_DATA = pathlib.Path(ARTIST_ROOT) / "tests" / "data"
_SCENARIO_PATH = _ARTIST_DATA / "scenarios" / "test_scenario_paint_four_heliostats.h5"
_FIELD_DATA = _ARTIST_DATA / "field_data"

# Two heliostats from the toy scenario, each with 2 calibration images.
_HELIOSTAT_DATA_MAPPING = [
    (
        "AA39",
        [
            _FIELD_DATA / "AA39-calibration-properties_1.json",
            _FIELD_DATA / "AA39-calibration-properties_2.json",
        ],
        [
            _FIELD_DATA / "AA39-flux_1.png",
            _FIELD_DATA / "AA39-flux_2.png",
        ],
    ),
    (
        "AA31",
        [
            _FIELD_DATA / "AA31-calibration-properties_1.json",
            _FIELD_DATA / "AA31-calibration-properties_2.json",
        ],
        [
            _FIELD_DATA / "AA31-flux_1.png",
            _FIELD_DATA / "AA31-flux_2.png",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _find_active_group(scenario, device):
    """Return the (group_index, heliostat_group) that contains AA39/AA31.

    The toy 4-heliostat scenario splits heliostats into two groups.  Group 0 is
    an "ideal prototype" with a degenerate 2D actuator tensor; group 1 holds the
    real heliostats with 3D actuator tensors.  We identify the active group by
    running the data parser and finding the first group with a non-zero active
    mask — this is the group WortbergKinematicReconstructor is designed to work
    with.
    """
    parser = PaintCalibrationDataParser(centroid_extraction_method=paint_mappings.UTIS_KEY)
    for i, group in enumerate(scenario.heliostat_field.heliostat_groups):
        _, _, _, _, mask, _ = parser.parse_data_for_reconstruction(
            heliostat_data_mapping=_HELIOSTAT_DATA_MAPPING,
            heliostat_group=group,
            scenario=scenario,
            device=device,
        )
        if mask.sum() > 0:
            return i, group
    raise RuntimeError("No active heliostats found in any group for the test data mapping.")


def _make_optimization_config(max_epoch: int = 5, early_stopping_patience: int = 2):
    """Return an optimization config dict in the nested form WortbergKinematicReconstructor expects."""
    return {
        config_dictionary.optimization: {
            config_dictionary.initial_learning_rate: 1e-3,
            config_dictionary.tolerance: 0.0,
            config_dictionary.max_epoch: max_epoch,
            config_dictionary.batch_size: 50,
            config_dictionary.log_step: 1,
            config_dictionary.early_stopping_delta: 1.0,
            config_dictionary.early_stopping_patience: early_stopping_patience,
            config_dictionary.early_stopping_window: 50,
        },
        config_dictionary.scheduler: {
            config_dictionary.scheduler_type: config_dictionary.reduce_on_plateau,
            config_dictionary.gamma: 0.99,
            config_dictionary.min: 1e-6,
            config_dictionary.reduce_factor: 0.9,
            config_dictionary.patience: 100,
            config_dictionary.threshold: 1e-3,
            config_dictionary.cooldown: 20,
        },
    }


def _make_flat_optimization_config(max_epoch: int = 5):
    """Return the same config in the *flat* form (as experiment.py passes it)."""
    return {
        config_dictionary.initial_learning_rate: 1e-3,
        config_dictionary.tolerance: 0.0,
        config_dictionary.max_epoch: max_epoch,
        config_dictionary.batch_size: 50,
        config_dictionary.log_step: 1,
        config_dictionary.early_stopping_delta: 1.0,
        config_dictionary.early_stopping_patience: 2,
        config_dictionary.early_stopping_window: 50,
        config_dictionary.scheduler: config_dictionary.reduce_on_plateau,
        "scheduler_parameters": {
            config_dictionary.reduce_factor: 0.9,
            config_dictionary.patience: 100,
            config_dictionary.threshold: 1e-3,
            config_dictionary.cooldown: 20,
            config_dictionary.min: 1e-6,
        },
    }


def _load_scenario(device: torch.device) -> Scenario:
    with h5py.File(_SCENARIO_PATH, "r") as f:
        return Scenario.load_scenario_from_hdf5(scenario_file=f, device=device)


def _make_reconstructor(
    ddp_setup: dict,
    device: torch.device,
    scenario: Scenario,
    train_position_deviation: bool = True,
    max_epoch: int = 5,
) -> WortbergKinematicReconstructor:
    ddp_setup[config_dictionary.device] = device
    ddp_setup[config_dictionary.groups_to_ranks_mapping] = {0: [0, 1]}
    ddp_setup[config_dictionary.ranks_to_groups_mapping] = {0: [0], 1: [0]}

    data = {
        config_dictionary.data_parser: PaintCalibrationDataParser(
            centroid_extraction_method=paint_mappings.UTIS_KEY
        ),
        config_dictionary.heliostat_data_mapping: _HELIOSTAT_DATA_MAPPING,
    }

    return WortbergKinematicReconstructor(
        ddp_setup=ddp_setup,
        scenario=scenario,
        data=data,
        optimization_configuration=_make_optimization_config(max_epoch=max_epoch),
        reconstruction_method=config_dictionary.kinematics_reconstruction_raytracing,
        train_position_deviation=train_position_deviation,
    )


# ---------------------------------------------------------------------------
# Test 1 — construction smoke test
# ---------------------------------------------------------------------------

def test_construction(ddp_setup_for_testing, device):
    """WortbergKinematicReconstructor can be instantiated without error."""
    torch.manual_seed(7)
    scenario = _load_scenario(device)
    reconstructor = _make_reconstructor(ddp_setup_for_testing, device, scenario)
    assert isinstance(reconstructor, WortbergKinematicReconstructor)


# ---------------------------------------------------------------------------
# Test 2 — correct parameters are learnable after _setup_optimizer
# ---------------------------------------------------------------------------

def test_learnable_parameters(ddp_setup_for_testing, device):
    """
    After _setup_optimizer:
      - translation_deviation_parameters  has requires_grad=True
      - rotation_deviation_parameters     has requires_grad=True
      - actuators.optimizable_parameters  has requires_grad=True
      - actuators.non_optimizable_params  has requires_grad=True
      - _base_position_deviation          has requires_grad=True
    """
    torch.manual_seed(7)
    scenario = _load_scenario(device)
    reconstructor = _make_reconstructor(ddp_setup_for_testing, device, scenario)

    _, group = _find_active_group(scenario, device)
    reconstructor._setup_optimizer(group, device)
    kin = group.kinematics
    assert kin.translation_deviation_parameters.requires_grad, \
        "translation_deviation_parameters must require gradients"
    assert kin.rotation_deviation_parameters.requires_grad, \
        "rotation_deviation_parameters must require gradients"
    assert kin.actuators.optimizable_parameters.requires_grad, \
        "actuators.optimizable_parameters must require gradients"
    assert kin.actuators.non_optimizable_parameters.requires_grad, \
        "actuators.non_optimizable_parameters must require gradients"
    assert hasattr(kin, "_base_position_deviation"), \
        "_base_position_deviation must be created when train_position_deviation=True"
    assert kin._base_position_deviation.requires_grad, \
        "_base_position_deviation must require gradients"


# ---------------------------------------------------------------------------
# Test 3 — b_i (initial_stroke_length) gradient is zeroed by the freeze hook
# ---------------------------------------------------------------------------

def test_stroke_length_gradient_is_frozen(ddp_setup_for_testing, device):
    """
    The _freeze_stroke_length hook must zero the gradient on the
    actuator_initial_stroke_length slice of optimizable_parameters.
    """
    torch.manual_seed(7)
    scenario = _load_scenario(device)
    reconstructor = _make_reconstructor(ddp_setup_for_testing, device, scenario)

    _, group = _find_active_group(scenario, device)
    optimizer, _, _ = reconstructor._setup_optimizer(group, device)
    kin = group.kinematics

    # Synthetic backward pass so the hooks fire.
    loss = kin.actuators.optimizable_parameters.sum()
    loss.backward()

    b_i_grad = kin.actuators.optimizable_parameters.grad[
        :, index_mapping.actuator_initial_stroke_length, :
    ]
    assert b_i_grad.abs().max().item() == pytest.approx(0.0, abs=1e-9), \
        "b_i (initial_stroke_length) gradient must be exactly zero"

    # The a_i slice should still have non-zero gradient.
    a_i_grad = kin.actuators.optimizable_parameters.grad[
        :, index_mapping.actuator_initial_angle, :
    ]
    assert a_i_grad.abs().max().item() > 0, \
        "a_i (initial_angle) gradient must be non-zero"


# ---------------------------------------------------------------------------
# Test 4 — non_optimizable gradient restricted to c_i (actuator_offset)
# ---------------------------------------------------------------------------

def test_only_actuator_offset_gradient(ddp_setup_for_testing, device):
    """
    The _only_actuator_offset hook must zero all slices of
    non_optimizable_parameters except actuator_offset (c_i).
    """
    torch.manual_seed(7)
    scenario = _load_scenario(device)
    reconstructor = _make_reconstructor(ddp_setup_for_testing, device, scenario)

    _, group = _find_active_group(scenario, device)
    optimizer, _, _ = reconstructor._setup_optimizer(group, device)
    kin = group.kinematics

    loss = kin.actuators.non_optimizable_parameters.sum()
    loss.backward()

    full_grad = kin.actuators.non_optimizable_parameters.grad  # [N, P, 2]
    for param_idx in range(full_grad.shape[1]):
        if param_idx == index_mapping.actuator_offset:
            assert full_grad[:, param_idx, :].abs().max().item() > 0, \
                f"c_i (actuator_offset, index {param_idx}) must have non-zero gradient"
        else:
            assert full_grad[:, param_idx, :].abs().max().item() == pytest.approx(0.0, abs=1e-9), \
                f"non_optimizable_parameters index {param_idx} must have zero gradient (only c_i is optimised)"


# ---------------------------------------------------------------------------
# Test 5 — no _base_position_deviation when train_position_deviation=False
# ---------------------------------------------------------------------------

def test_no_base_position_deviation_when_disabled(ddp_setup_for_testing, device):
    """
    When train_position_deviation=False, _base_position_deviation must NOT be
    created on the kinematic object.
    """
    torch.manual_seed(7)
    scenario = _load_scenario(device)
    reconstructor = _make_reconstructor(
        ddp_setup_for_testing, device, scenario, train_position_deviation=False
    )

    _, group = _find_active_group(scenario, device)
    kin = group.kinematics
    if hasattr(kin, "_base_position_deviation"):
        del kin._base_position_deviation

    reconstructor._setup_optimizer(group, device)

    assert not hasattr(kin, "_base_position_deviation"), \
        "_base_position_deviation must not be created when train_position_deviation=False"


# ---------------------------------------------------------------------------
# Test 6 — deviation bounds are respected after clamping
# ---------------------------------------------------------------------------

def test_deviation_bounds_respected(ddp_setup_for_testing, device):
    """
    After _apply_deviation_bounds, all optimised parameters must lie within
    the Table 5.3 bounds relative to their initial values.
    """
    torch.manual_seed(7)
    scenario = _load_scenario(device)
    reconstructor = _make_reconstructor(ddp_setup_for_testing, device, scenario)

    _, group = _find_active_group(scenario, device)
    optimizer, init_angle, init_offset = reconstructor._setup_optimizer(group, device)
    kin = group.kinematics

    # Manually push parameters far outside the bounds.
    with torch.no_grad():
        kin.translation_deviation_parameters.fill_(1.0)
        kin.rotation_deviation_parameters.fill_(1.0)
        kin.actuators.optimizable_parameters[:, index_mapping.actuator_initial_angle, :] += 1.0
        kin.actuators.non_optimizable_parameters[:, index_mapping.actuator_offset, :] += 1.0
        if hasattr(kin, "_base_position_deviation"):
            kin._base_position_deviation.fill_(1.0)

    reconstructor._apply_deviation_bounds(group, init_angle, init_offset)

    # translation: within ±_BOUND_TRANSLATION_M of the snapshot
    deviation = (
        kin.translation_deviation_parameters - kin._initial_translation_deviation
    ).abs()
    assert deviation.max().item() <= reconstructor._BOUND_TRANSLATION_M + 1e-6, \
        "translation_deviation_parameters exceeded bound after clamping"

    # rotation: within ±_BOUND_ROTATION_RAD of zero
    assert kin.rotation_deviation_parameters.abs().max().item() \
        <= reconstructor._BOUND_ROTATION_RAD + 1e-6, \
        "rotation_deviation_parameters exceeded bound after clamping"

    # a_i: within ±_BOUND_ACTUATOR_ANGLE_RAD of snapshot
    a_i = kin.actuators.optimizable_parameters[:, index_mapping.actuator_initial_angle, :]
    assert (a_i - init_angle).abs().max().item() \
        <= reconstructor._BOUND_ACTUATOR_ANGLE_RAD + 1e-6, \
        "actuator_initial_angle (a_i) exceeded bound after clamping"

    # c_i: within ±_BOUND_ACTUATOR_OFFSET_M of snapshot
    c_i = kin.actuators.non_optimizable_parameters[:, index_mapping.actuator_offset, :]
    assert (c_i - init_offset).abs().max().item() \
        <= reconstructor._BOUND_ACTUATOR_OFFSET_M + 1e-6, \
        "actuator_offset (c_i) exceeded bound after clamping"

    # base position: within ±_BOUND_BASE_POSITION_M of zero
    if hasattr(kin, "_base_position_deviation"):
        assert kin._base_position_deviation.abs().max().item() \
            <= reconstructor._BOUND_BASE_POSITION_M + 1e-6, \
            "_base_position_deviation exceeded bound after clamping"


# ---------------------------------------------------------------------------
# Test 7 — training loss decreases (the optimizer is doing real work)
# ---------------------------------------------------------------------------

def test_loss_decreases_over_training(ddp_setup_for_testing, device):
    """
    After a short training run the mean loss must be lower than the initial
    loss.  This is the core sanity check: if the optimizer is broken or
    gradients are not flowing, the loss will not decrease.
    """
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    scenario = _load_scenario(device)
    reconstructor = _make_reconstructor(
        ddp_setup_for_testing, device, scenario, max_epoch=15
    )

    loss_fn = FocalSpotLoss(scenario=scenario)
    reconstructor.reconstruct_kinematics(loss_definition=loss_fn, device=device)

    history = reconstructor._convergence_history
    assert len(history) >= 2, "convergence history must have at least 2 entries"

    first_loss = history[0]["loss"]
    last_loss = history[-1]["loss"]
    assert last_loss < first_loss, (
        f"Loss did not decrease: initial={first_loss:.6f}, final={last_loss:.6f}. "
        "Gradients may not be flowing through WortbergKinematicReconstructor."
    )


# ---------------------------------------------------------------------------
# Test 8 — flat optimization config is restructured correctly
# ---------------------------------------------------------------------------

def test_flat_config_restructuring(ddp_setup_for_testing, device):
    """
    WortbergKinematicReconstructor.__init__ accepts the flat config dict that
    experiment.py passes (where scheduler type and scheduler_parameters are
    top-level keys) and restructures it into the nested form expected by the
    parent class.
    """
    torch.manual_seed(7)
    scenario = _load_scenario(device)

    ddp_setup_for_testing[config_dictionary.device] = device
    ddp_setup_for_testing[config_dictionary.groups_to_ranks_mapping] = {0: [0, 1]}
    ddp_setup_for_testing[config_dictionary.ranks_to_groups_mapping] = {0: [0], 1: [0]}

    data = {
        config_dictionary.data_parser: PaintCalibrationDataParser(
            centroid_extraction_method=paint_mappings.UTIS_KEY
        ),
        config_dictionary.heliostat_data_mapping: _HELIOSTAT_DATA_MAPPING,
    }

    reconstructor = WortbergKinematicReconstructor(
        ddp_setup=ddp_setup_for_testing,
        scenario=scenario,
        data=data,
        optimization_configuration=_make_flat_optimization_config(),
        reconstruction_method=config_dictionary.kinematics_reconstruction_raytracing,
        train_position_deviation=True,
    )

    # After restructuring, the nested keys must exist.
    assert config_dictionary.max_epoch in reconstructor.optimization_configuration, \
        "max_epoch missing from restructured optimization_configuration"
    assert config_dictionary.initial_learning_rate in reconstructor.optimization_configuration, \
        "initial_learning_rate missing from restructured optimization_configuration"
