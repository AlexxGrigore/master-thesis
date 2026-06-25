"""Single source of truth for which heliostats enter the profiling experiment.

The experiment hinges on creation and training operating on *identical* heliostat sets,
so both ``create_scenarios.py`` and ``run_training.py`` derive their list from here.

A heliostat is eligible only if it BOTH:

  * has fitted-deflectometry data and existing Properties/Deflectometry files — required
    to build a NURBS surface, AND
  * has at least ``min_train_samples`` calibration measurements in the train split —
    required for joint kinematics reconstruction (uniform sample count across heliostats).

The eligible list is sorted by name, so ``select(10)`` is a strict subset of
``select(20)`` — the N sweep is nested, not a fresh random draw each time.

This module is self-contained: it does NOT import ``create_all_scenarios`` (which is stale
— it imports the pre-refactor ``data_parser`` API that no longer exists in the current
ARTIST). The deflectometry-folder scan is reimplemented here against the filesystem only.
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime

# src/ must be importable for the project's data-mapping helper.
_SRC = pathlib.Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import paths  # noqa: E402  (sibling module)

from utils.evaluation import build_heliostat_data_mapping  # noqa: E402

# A (name, properties_path, deflectometry_path) triple, as consumed by
# ``extract_paint_heliostats_fitted_surface``.
FittingEntry = tuple[str, pathlib.Path, pathlib.Path]


def _deflectometry_timestamp(filepath: pathlib.Path) -> datetime:
    """Parse the timestamp in a deflectometry filename; datetime.min on failure.

    Expected: {name}-filled-YYYY-MM-DDZHH-MM-SSZ-deflectometry.h5
    """
    parts = filepath.stem.split("-")
    for i, part in enumerate(parts):
        if len(part) == 4 and part.isdigit():  # year token
            try:
                date_str = f"{parts[i]}-{parts[i+1]}-{parts[i+2].split('Z')[0]}"
                time_str = (
                    f"{parts[i+2].split('Z')[1]}-{parts[i+3]}-{parts[i+4].split('Z')[0]}"
                )
                return datetime.strptime(
                    f"{date_str} {time_str.replace('-', ':')}", "%Y-%m-%d %H:%M:%S"
                )
            except (IndexError, ValueError):
                pass
    return datetime.min


def _latest_deflectometry_file(
    name: str, deflectometry_folder: pathlib.Path
) -> pathlib.Path | None:
    files = list(deflectometry_folder.glob(f"{name}-filled-*-deflectometry.h5"))
    return max(files, key=_deflectometry_timestamp) if files else None


def _build_fitted_heliostat_list(
    heliostats_dir: pathlib.Path, availability_json: pathlib.Path
) -> list[FittingEntry]:
    """(name, properties_path, deflectometry_path) for benchmark heliostats with
    deflectometry data and the files actually on disk. Sorted by name."""
    with open(availability_json) as f:
        availability: dict[str, dict] = json.load(f)

    target_names = sorted(
        name
        for name, info in availability.items()
        if info.get("has_deflectometry") and info.get("in_benchmark")
    )

    result: list[FittingEntry] = []
    for name in target_names:
        folder = heliostats_dir / name
        properties_folder = folder / "Properties"
        deflectometry_folder = folder / "Deflectometry"
        if not properties_folder.exists() or not deflectometry_folder.exists():
            continue
        props = list(properties_folder.glob(f"{name}-heliostat-properties.json"))
        if not props:
            continue
        defl = _latest_deflectometry_file(name, deflectometry_folder)
        if defl is None:
            continue
        result.append((name, props[0], defl))
    return result


def eligible_fitting_list(
    daic: bool = False, min_train_samples: int = 10
) -> list[FittingEntry]:
    """All eligible heliostats as fitting triples, sorted by name."""
    fitting = _build_fitted_heliostat_list(
        paths.heliostats_dir(daic), paths.availability_json()
    )
    train_map = build_heliostat_data_mapping(
        paths.benchmark_csv(daic),
        paths.calibration_dir(daic),
        paths.flux_dir(daic),
        "train",
    )
    sample_counts = {name: len(calib_paths) for name, calib_paths, _ in train_map}

    eligible = [e for e in fitting if sample_counts.get(e[0], 0) >= min_train_samples]
    eligible.sort(key=lambda e: e[0])
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


def select_names(n: int, daic: bool = False, min_train_samples: int = 10) -> list[str]:
    """First ``n`` eligible heliostat names (for filtering the training mapping)."""
    return [e[0] for e in select_fitting_list(n, daic, min_train_samples)]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--daic", action="store_true")
    args = ap.parse_args()
    pool = eligible_fitting_list(daic=args.daic)
    print(f"Eligible heliostats: {len(pool)}")
    print("First 50:", [e[0] for e in pool[:50]])
