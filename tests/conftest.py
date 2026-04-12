"""
Shared pytest fixtures for master-thesis tests.

Mirrors the setup used in ARTIST/tests/conftest.py so that our custom
reconstructors are tested under the same deterministic conditions.
"""

import os
import pathlib
import platform
import random

import numpy as np
import pytest

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

from artist.util import config_dictionary

# Resolve the ARTIST test-data root so we can reuse the toy scenarios and
# calibration files that ship with ARTIST.
ARTIST_TESTS_DATA = pathlib.Path(__file__).parent.parent.parent / "ARTIST" / "tests" / "data"


@pytest.fixture(params=["cpu"])
def device(request: pytest.FixtureRequest) -> torch.device:
    """
    Return the device for tests.  GPU is included when not in CI.
    """
    param = request.param
    if param == "gpu":
        if os.environ.get("CI", "false").lower() == "true":
            pytest.skip("Skipping GPU test in CI environment")
        os_name = platform.system()
        if os_name in {"Linux", "Windows"}:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")
    return torch.device("cpu")


@pytest.fixture
def ddp_setup_for_testing() -> dict:
    """Single-process DDP setup used in all reconstructor tests."""
    return {
        config_dictionary.device: None,
        config_dictionary.is_distributed: False,
        config_dictionary.is_nested: False,
        config_dictionary.rank: 0,
        config_dictionary.world_size: 1,
        config_dictionary.process_subgroup: None,
        config_dictionary.groups_to_ranks_mapping: None,
        config_dictionary.heliostat_group_rank: 0,
        config_dictionary.heliostat_group_world_size: 1,
        config_dictionary.ranks_to_groups_mapping: None,
    }


@pytest.fixture(scope="session", autouse=True)
def enforce_determinism():
    """Fix all random seeds and enable deterministic PyTorch algorithms."""
    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    yield
