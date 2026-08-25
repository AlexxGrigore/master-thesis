"""Build ideal_1277_AY36_deflectometry.h5 for Experiment F (full-field blocking).

Experiment F needs the whole 1277-heliostat ideal field present as blockers, with
AY36 carrying its REAL deflectometry-fitted NURBS surface instead of the ideal one.

Rather than re-running the NURBS fit, this script copies
``scenarios/full_benchmark_ideal/ideal_1277.h5`` verbatim and then transplants the
``surface`` subtree of AY36 from the already-built neighbourhood scenario
``scenarios/neighbourhoods/AY36/scenario.h5`` (which holds exactly the
deflectometry-fitted surface named in that scenario's blockers.json). Positions,
kinematics, actuators and every other heliostat stay byte-identical to
ideal_1277.h5, which is never touched.

Usage
-----
    python build_fullfield_deflectometry_scenario.py AY36
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import h5py

_here = pathlib.Path(__file__).resolve().parent
_ROOT = _here.parents[2]

log = logging.getLogger(__name__)

BASE_SCENARIO = _ROOT / "scenarios" / "full_benchmark_ideal" / "ideal_1277.h5"
OUT_SCENARIO = (
    _ROOT / "scenarios" / "full_benchmark_ideal" / "ideal_1277_AY36_deflectometry.h5"
)
NEIGHBOURHOOD_SCENARIO = (
    _ROOT / "scenarios" / "neighbourhoods" / "AY36" / "scenario.h5"
)
_HELIOSTAT = "AY36"


def build(force: bool = False) -> pathlib.Path:
    if OUT_SCENARIO.exists() and not force:
        log.info(f"[SKIP] exists: {OUT_SCENARIO} — use --force")
        return OUT_SCENARIO
    for p in (BASE_SCENARIO, NEIGHBOURHOOD_SCENARIO):
        if not p.exists():
            raise FileNotFoundError(p)

    # 1. byte-copy the base scenario, then 2. swap AY36's surface subtree in r+
    #    mode (the copy is the only file opened for writing; the base stays read-only).
    import shutil

    shutil.copy2(BASE_SCENARIO, OUT_SCENARIO)

    with h5py.File(NEIGHBOURHOOD_SCENARIO, "r") as src:
        src_surface = src[f"heliostats/{_HELIOSTAT}/surface/facets"]
        facet_names = sorted(src_surface.keys())
        control_points = {
            facet: src_surface[f"{facet}/control_points"][()] for facet in facet_names
        }
        degrees = {facet: src_surface[f"{facet}/degrees"][()] for facet in facet_names}
        canting = {facet: src_surface[f"{facet}/canting"][()] for facet in facet_names}
        position = {facet: src_surface[f"{facet}/position"][()] for facet in facet_names}

    with h5py.File(OUT_SCENARIO, "r+") as dst:
        dst_surface = dst[f"heliostats/{_HELIOSTAT}/surface/facets"]
        dst_facets = sorted(dst_surface.keys())
        assert dst_facets == facet_names, (
            f"facet mismatch: neighbourhood {facet_names} vs full-field {dst_facets}"
        )
        n_changed_cp = 0
        for facet in facet_names:
            grp = dst_surface[facet]
            if (grp["degrees"][()] != degrees[facet]).any():
                raise ValueError(f"{facet}: NURBS degrees differ — refusing to transplant")
            if (grp["canting"][()] != canting[facet]).any():
                raise ValueError(f"{facet}: canting differs — refusing to transplant")
            if (grp["position"][()] != position[facet]).any():
                raise ValueError(f"{facet}: facet position differs — refusing to transplant")
            if (grp["control_points"][()] != control_points[facet]).any():
                n_changed_cp += 1
            grp["control_points"][...] = control_points[facet]
    log.info(
        f"{OUT_SCENARIO.name}: transplanted fitted control points on "
        f"{n_changed_cp}/{len(facet_names)} facets of {_HELIOSTAT} "
        f"(source: {NEIGHBOURHOOD_SCENARIO.name})"
    )

    # Sanity: everything except AY36's surface/control_points identical to base.
    import numpy as np

    diffs = []
    with h5py.File(BASE_SCENARIO, "r") as a, h5py.File(OUT_SCENARIO, "r") as b:
        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if name.startswith(f"heliostats/{_HELIOSTAT}/surface") and name.endswith(
                "control_points"
            ):
                return  # the intended difference
            da, db = obj[()], b[name][()]
            same = np.array_equal(da, db)
            if not same:
                diffs.append(name)
        a.visititems(visit)
    if diffs:
        raise RuntimeError(f"unexpected differences vs ideal_1277.h5: {diffs[:10]}")
    log.info("verified: only AY36 surface control_points differ from ideal_1277.h5")
    return OUT_SCENARIO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(force=args.force)


if __name__ == "__main__":
    main()
