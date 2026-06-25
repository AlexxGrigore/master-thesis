"""Central path resolution for the profiling experiment (local vs DAIC).

On DAIC the PAINT dataset does NOT live in the repo — it sits on the CVlab umbrella
share. Locally it's in ``<repo>/datasets/paint``. Everything else (scenarios, outputs,
the deflectometry-availability JSON shipped in git) is repo-relative and resolves
correctly on either machine via ``REPO`` below.

Pass ``daic=True`` to point the PAINT-derived paths at the umbrella share.
"""

from __future__ import annotations

import pathlib

# master-thesis/  (this file is master-thesis/src/profiling_experiment/paths.py)
REPO = pathlib.Path(__file__).resolve().parents[2]

# Benchmark whose train split supplies the calibration samples for training.
BENCHMARK_NAME = "benchmark_split-balanced_train-100_validation-50_deflectometry"

_LOCAL_PAINT = REPO / "datasets" / "paint"
_DAIC_PAINT = pathlib.Path(
    "/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint"
)


def paint_dir(daic: bool) -> pathlib.Path:
    """Root of the PAINT dataset (umbrella share on DAIC, repo locally)."""
    return _DAIC_PAINT if daic else _LOCAL_PAINT


def heliostats_dir(daic: bool) -> pathlib.Path:
    """Folder containing one sub-folder per heliostat (Properties/, Deflectometry/)."""
    return paint_dir(daic) / "heliostats"


def tower_file(daic: bool) -> pathlib.Path:
    return heliostats_dir(daic) / "WRI1030197-tower-measurements.json"


def benchmark_csv(daic: bool) -> pathlib.Path:
    return paint_dir(daic) / "splits" / f"{BENCHMARK_NAME}.csv"


def calibration_dir(daic: bool) -> pathlib.Path:
    return paint_dir(daic) / BENCHMARK_NAME / "calibration_properties"


def flux_dir(daic: bool) -> pathlib.Path:
    return paint_dir(daic) / BENCHMARK_NAME / "flux_image"


def availability_json() -> pathlib.Path:
    """Deflectometry-availability map (shipped in git, same path both machines)."""
    return REPO / "src" / "utils" / "deflectometry_availability.json"


def scenario_dir() -> pathlib.Path:
    """Where per-N scenarios are written / read."""
    return REPO / "scenarios" / "profiling"


def output_dir() -> pathlib.Path:
    """Where profiling JSON / CSV results are written."""
    return REPO / "outputs" / "new_mapping_function" / "profiling_experiment"
