"""Single source of truth for which heliostats enter the profiling experiment.

The whole experiment hinges on creation and training operating on *identical*
heliostat sets: the scenario built for N heliostats must contain exactly the N
heliostats that training then runs on. Both ``create_scenarios.py`` and
``run_training.py`` therefore derive their list from here.

A heliostat is eligible only if it BOTH:

  * has fitted-deflectometry data and is in the benchmark — required to build a NURBS
    surface (reuses ``create_all_scenarios._build_fitted_heliostat_list``), AND
  * has at least ``min_train_samples`` calibration measurements in the train split —
    required for joint kinematics reconstruction, which needs a uniform sample count
    across all heliostats.

The eligible list is sorted by name, so ``select(10)`` is always a strict subset of
``select(20)`` — the N sweep is nested, not a fresh random draw each time.
"""

from __future__ import annotations

import pathlib
import sys

# Make the sibling project modules importable: src/ (for create_all_scenarios and the
# utils package) and the one_heliostat_demo config that holds the benchmark paths.
_SRC = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_SRC), str(_SRC / "one_heliostat_demo" / "single_heliostat")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as cfg  # noqa: E402  (one_heliostat_demo/single_heliostat/config.py)
import create_all_scenarios as cas  # noqa: E402

from utils.evaluation import build_heliostat_data_mapping  # noqa: E402

# A (name, properties_path, deflectometry_path) triple, as consumed by
# ``extract_paint_heliostats_fitted_surface``.
FittingEntry = tuple[str, pathlib.Path, pathlib.Path]


def _paint_dir(daic: bool) -> pathlib.Path:
    return cas.DAIC_PAINT_DIR if daic else cas.LOCAL_PAINT_DIR


def eligible_fitting_list(
    daic: bool = False, min_train_samples: int = 10
) -> list[FittingEntry]:
    """Return all eligible heliostats as fitting triples, sorted by name."""
    paint_dir = _paint_dir(daic)

    # Heliostats with deflectometry data (and existing Properties/Deflectometry files).
    fitting = cas._build_fitted_heliostat_list(
        paint_dir, cas.DEFLECTOMETRY_AVAILABILITY_JSON
    )

    # How many train-split calibration samples each heliostat has.
    train_map = build_heliostat_data_mapping(
        pathlib.Path(cfg.BENCHMARK_CSV),
        pathlib.Path(cfg.CALIBRATION_DIR),
        pathlib.Path(cfg.REAL_FLUX_DIR),
        "train",
    )
    sample_counts = {name: len(calib_paths) for name, calib_paths, _ in train_map}

    eligible = [
        entry
        for entry in fitting
        if sample_counts.get(entry[0], 0) >= min_train_samples
    ]
    eligible.sort(key=lambda entry: entry[0])
    return eligible


def select_fitting_list(
    n: int, daic: bool = False, min_train_samples: int = 10
) -> list[FittingEntry]:
    """First ``n`` eligible heliostats as fitting triples (for scenario creation)."""
    eligible = eligible_fitting_list(daic=daic, min_train_samples=min_train_samples)
    if len(eligible) < n:
        raise ValueError(
            f"Requested {n} heliostats but only {len(eligible)} are eligible "
            f"(deflectometry + >= {min_train_samples} train samples)."
        )
    return eligible[:n]


def select_names(
    n: int, daic: bool = False, min_train_samples: int = 10
) -> list[str]:
    """First ``n`` eligible heliostat names (for filtering the training mapping)."""
    return [entry[0] for entry in select_fitting_list(n, daic, min_train_samples)]


if __name__ == "__main__":
    # Quick sanity check: print the eligible pool size and the first 50 names.
    pool = eligible_fitting_list()
    print(f"Eligible heliostats: {len(pool)}")
    print("First 50:", [e[0] for e in pool[:50]])
