"""Build the reduced full-field blocking scenario for Experiment F (AY36).

A pure-geometry census (vertical_shadow_blockers.py, results in
outputs/new_mapping_function/blocking_study/vertical_shadow/AY36/
vertical_shadow_blockers.json) determined that under a conservative
vertical-mirror assumption only 14 heliostats can ever shade AY36's incident
light for the 199 sampled sun positions. The full 1277-heliostat blocking run
proved infeasible on CPU (1-sample smoke test: ~280 s wall, ~23 GB peak RSS),
so Experiment F uses this REDUCED scenario: AY36 (real deflectometry surface)
plus those 14 ideal blockers, all in one heliostat group.

Source: scenarios/full_benchmark_ideal/ideal_1277_AY36_deflectometry.h5
(already verified: byte-identical to ideal_1277.h5 except AY36's fitted NURBS
control points). This script copies the file's top-level structure and only the
15 needed heliostat subgroups into a NEW file; nothing existing is modified.

Output: scenarios/neighbourhoods_fullfield/AY36/scenario.h5

Usage
-----
    python build_reduced_fullfield_scenario.py [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

import h5py

_here = pathlib.Path(__file__).resolve().parent
_ROOT = _here.parents[2]

log = logging.getLogger(__name__)

SOURCE_SCENARIO = (
    _ROOT / "scenarios" / "full_benchmark_ideal" / "ideal_1277_AY36_deflectometry.h5"
)
OUT_DIR = _ROOT / "scenarios" / "neighbourhoods_fullfield" / "AY36"
OUT_SCENARIO = OUT_DIR / "scenario.h5"
CENSUS_JSON = (
    _ROOT / "outputs" / "new_mapping_function" / "blocking_study" / "vertical_shadow"
    / "AY36" / "vertical_shadow_blockers.json"
)

STUDIED = "AY36"
# Union of the per-sample blocker sets from the vertical-shadow census
# (see outputs/.../vertical_shadow/AY36/vertical_shadow_blockers.json).
BLOCKERS = [
    "AY35", "AY37", "AX35", "AX36", "AX37", "AY34", "AX33",
    "AY33", "AW33", "AX31", "AX32", "AX38", "AY32", "AY38",
]


def build(force: bool = False) -> pathlib.Path:
    if OUT_SCENARIO.exists() and not force:
        log.info(f"[SKIP] exists: {OUT_SCENARIO} — use --force")
        return OUT_SCENARIO
    if not SOURCE_SCENARIO.exists():
        raise FileNotFoundError(SOURCE_SCENARIO)

    members = [STUDIED] + BLOCKERS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tmp_path = OUT_DIR / "scenario.h5.tmp"
    with h5py.File(SOURCE_SCENARIO, "r") as src, h5py.File(tmp_path, "w") as dst:
        for attr, value in src.attrs.items():
            dst.attrs[attr] = value
        for key in src.keys():
            if key == "heliostats":
                continue
            src.copy(key, dst)
        missing = [m for m in members if m not in src["heliostats"]]
        if missing:
            raise RuntimeError(f"heliostats missing from source scenario: {missing}")
        dst.create_group("heliostats")
        for m in members:
            src.copy(f"heliostats/{m}", dst["heliostats"], name=m)
        # Group assignment is implicit: all heliostats form the single group
        # (number_of_heliostat_groups == 1 was copied above).
    tmp_path.rename(OUT_SCENARIO)

    # Verify: exactly the 15 members; AY36 keeps the deflectometry surface.
    import numpy as np

    with h5py.File(OUT_SCENARIO, "r") as f:
        got = sorted(f["heliostats"].keys())
        assert got == sorted(members), f"member mismatch: {got}"
        assert int(f["number_of_heliostat_groups"][()]) == 1
        cp_red = f["heliostats/AY36/surface/facets/facet_1/control_points"][()]
    with h5py.File(
        _ROOT / "scenarios" / "neighbourhoods" / "AY36" / "scenario.h5", "r"
    ) as ref:
        cp_defl = ref["heliostats/AY36/surface/facets/facet_1/control_points"][()]
    with h5py.File(
        _ROOT / "scenarios" / "full_benchmark_ideal" / "ideal_1277.h5", "r"
    ) as ref:
        cp_ideal = ref["heliostats/AY36/surface/facets/facet_1/control_points"][()]
        cp_aw33_new = ref["heliostats/AW33/surface/facets/facet_1/control_points"][()]
    assert np.array_equal(cp_red, cp_defl) and not np.array_equal(cp_red, cp_ideal), \
        "AY36 does NOT carry its deflectometry surface — aborting"
    with h5py.File(OUT_SCENARIO, "r") as f:
        assert np.array_equal(
            f["heliostats/AW33/surface/facets/facet_1/control_points"][()], cp_aw33_new
        ), "blocker surface mismatch vs ideal_1277.h5"
    log.info(
        f"built {OUT_SCENARIO} with {len(members)} heliostats "
        f"(AY36 deflectometry + {len(BLOCKERS)} ideal blockers), verified"
    )

    (OUT_DIR / "blockers.json").write_text(
        json.dumps(
            {
                "heliostat_id": STUDIED,
                "members": members,
                "blockers": BLOCKERS,
                "source_scenario": str(SOURCE_SCENARIO),
                "census": str(CENSUS_JSON),
                "surfaces": "deflectometry (AY36, transplanted from "
                            "ideal_1277_AY36_deflectometry.h5) + 14 ideal",
                "note": "reduced full-field scenario for Experiment F; blocker set "
                        "from the vertical-shadow census (union over 199 samples)",
            },
            indent=2,
        )
    )
    return OUT_SCENARIO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(force=args.force)


if __name__ == "__main__":
    main()
