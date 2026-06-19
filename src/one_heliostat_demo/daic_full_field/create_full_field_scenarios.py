"""
Create per-heliostat scenario HDF5 files for the full solar field.

Reads deflectometry_availability.json (2,015 heliostats) and splits them into:
  - deflectometry: heliostats with has_deflectometry=True whose filled file exists
    → full_field_one_heliostat_scenarios/deflectometry/{hid}/scenario.h5
  - ideal: all remaining heliostats
    → full_field_one_heliostat_scenarios/ideal/{hid}/scenario_ideal.h5

Both output directories are already consumed by _find_scenario() in
generate_full_field_dataset.py without any further changes.

The script processes one heliostat at a time and skips existing files by default,
making it safely resumable and parallelisable (e.g. SLURM array jobs on DAIC).

Usage
-----
    # Smoke test (first 3 of each type, no --force):
    python create_full_field_scenarios.py --smoke-test

    # Full run on DAIC:
    python create_full_field_scenarios.py --daic

    # Specific heliostat subset:
    python create_full_field_scenarios.py --heliostat-ids AA23 AA24 BE35

    # Deflectometry only (skip ideal — useful for quick GPU jobs):
    python create_full_field_scenarios.py --only-deflectometry --daic

    # Overwrite existing files:
    python create_full_field_scenarios.py --force --daic
"""

import argparse
import json
import logging
import pathlib
import sys
import time
from datetime import datetime

import torch

_here = pathlib.Path(__file__).resolve().parent          # daic_full_field/
_src  = _here.parent.parent                              # src/
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_here.parent))                    # one_heliostat_demo/

from artist.io import paint_scenario_parser              # noqa: E402
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator  # noqa: E402
from artist.util import constants as config_dictionary, set_logger_config  # noqa: E402
from artist.util import get_device                       # noqa: E402
from artist.util.config import LightSourceConfig, LightSourceListConfig  # noqa: E402

log = logging.getLogger(__name__)

# NURBS fitting parameters — same values used in create_all_scenarios.py.
_N_CONTROL_POINTS  = torch.tensor([20, 20])
_FIT_METHOD        = config_dictionary.fit_nurbs_from_normals
_DEFL_STEP_SIZE    = 100
_FIT_TOLERANCE     = 1e-10
_FIT_MAX_EPOCH     = 400

_AVAILABILITY_JSON = _src / "utils" / "deflectometry_availability.json"


# ---------------------------------------------------------------------------
# Helpers (mirrored from create_all_scenarios.py)
# ---------------------------------------------------------------------------

def _find_latest_deflectometry_file(hid: str, defl_dir: pathlib.Path) -> pathlib.Path | None:
    """Return the most recent *-filled-*-deflectometry.h5 file, or None."""
    files = sorted(defl_dir.glob(f"{hid}-filled-*-deflectometry.h5"))
    return files[-1] if files else None


def _make_light_source_config() -> LightSourceListConfig:
    return LightSourceListConfig(
        light_source_list=[
            LightSourceConfig(
                light_source_key="sun_1",
                light_source_type=config_dictionary.sun_key,
                number_of_rays=10,
                distribution_type=config_dictionary.light_source_distribution_is_normal,
                mean=0.0,
                covariance=4.3681e-06,
            )
        ]
    )


def _make_nurbs_optimizer_scheduler():
    opt = torch.optim.Adam([torch.empty(1, requires_grad=True)], lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.2, patience=50,
        threshold=1e-7, threshold_mode="abs",
    )
    return opt, sched


# ---------------------------------------------------------------------------
# Per-heliostat scenario creator
# ---------------------------------------------------------------------------

def create_one_scenario(
    hid: str,
    props_path: pathlib.Path,
    defl_path: pathlib.Path | None,
    out_path: pathlib.Path,
    power_plant_config,
    target_area_list_planar_config,
    target_area_list_cylindrical_config,
    device: torch.device,
) -> None:
    """
    Create and save a single-heliostat scenario HDF5 file.

    If defl_path is given, fits NURBS surfaces from deflectometry.
    Otherwise uses ideal (flat) surfaces.
    """
    surface_label = "deflectometry" if defl_path is not None else "ideal"
    log.info(f"  {hid}  ({surface_label})  → {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if defl_path is not None:
        opt, sched = _make_nurbs_optimizer_scheduler()
        heliostat_list_config, prototype_config = (
            paint_scenario_parser.extract_paint_heliostats_fitted_surface(
                paths=[(hid, props_path, defl_path)],
                power_plant_position=power_plant_config.power_plant_position,
                number_of_nurbs_control_points=_N_CONTROL_POINTS,
                deflectometry_step_size=_DEFL_STEP_SIZE,
                nurbs_fit_method=_FIT_METHOD,
                nurbs_fit_tolerance=_FIT_TOLERANCE,
                nurbs_fit_max_epoch=_FIT_MAX_EPOCH,
                nurbs_fit_optimizer=opt,
                nurbs_fit_scheduler=sched,
                device=device,
            )
        )
    else:
        heliostat_list_config, prototype_config = (
            paint_scenario_parser.extract_paint_heliostats_ideal_surface(
                paths=[(hid, props_path)],
                power_plant_position=power_plant_config.power_plant_position,
                number_of_nurbs_control_points=_N_CONTROL_POINTS,
                device=device,
            )
        )

    H5ScenarioGenerator(
        file_path=out_path,
        power_plant_config=power_plant_config,
        target_area_list_planar_config=target_area_list_planar_config,
        target_area_list_cylindrical_config=target_area_list_cylindrical_config,
        light_source_list_config=_make_light_source_config(),
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    ).generate_scenario()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classify_heliostats(
    availability_json: pathlib.Path,
    heliostats_dir: pathlib.Path,
    heliostat_ids: list[str] | None,
) -> tuple[list[tuple[str, pathlib.Path, pathlib.Path]],   # deflectometry: (hid, props, defl)
           list[tuple[str, pathlib.Path]],                  # ideal: (hid, props)
           list[str]]:                                      # skipped
    """
    Read the availability JSON and classify each heliostat.

    Returns (defl_list, ideal_list, skipped).
    """
    with open(availability_json) as fh:
        availability: dict[str, dict] = json.load(fh)

    # Filter out the "metadata" sentinel key.
    all_ids = sorted(k for k in availability if k != "metadata")
    if heliostat_ids is not None:
        all_ids = [h for h in all_ids if h in heliostat_ids]

    defl_list:  list[tuple[str, pathlib.Path, pathlib.Path]] = []
    ideal_list: list[tuple[str, pathlib.Path]]               = []
    skipped:    list[str]                                     = []

    for hid in all_ids:
        hel_dir   = heliostats_dir / hid
        props_dir = hel_dir / "Properties"
        props_files = sorted(props_dir.glob(f"{hid}-heliostat-properties.json")) if props_dir.exists() else []

        if not props_files:
            log.warning(f"  {hid}: properties JSON not found in {props_dir} — skipped")
            skipped.append(hid)
            continue

        props_path = props_files[0]
        has_defl   = availability[hid].get("has_deflectometry", False)

        if has_defl:
            defl_dir  = hel_dir / "Deflectometry"
            defl_file = _find_latest_deflectometry_file(hid, defl_dir) if defl_dir.exists() else None
            if defl_file is not None:
                defl_list.append((hid, props_path, defl_file))
                continue
            else:
                log.warning(f"  {hid}: has_deflectometry=True but no filled file found — using ideal")

        ideal_list.append((hid, props_path))

    return defl_list, ideal_list, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create full-field per-heliostat scenario HDF5 files."
    )
    p.add_argument(
        "--daic", action="store_true",
        help="Use DAIC cluster paths.",
    )
    p.add_argument(
        "--heliostat-ids", nargs="+", default=None, metavar="ID",
        help="Process only these heliostat IDs (default: all in availability JSON).",
    )
    p.add_argument(
        "--output-dir", type=pathlib.Path, default=None,
        help="Root for output scenarios (default: scenarios/full_field_one_heliostat_scenarios/).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing scenario files.",
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Process first 3 heliostats of each type (deflectometry and ideal) for quick validation.",
    )
    p.add_argument(
        "--only-deflectometry", action="store_true",
        help="Skip the ideal group.",
    )
    p.add_argument(
        "--only-ideal", action="store_true",
        help="Skip the deflectometry group.",
    )
    return p.parse_args()


def main() -> None:
    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    torch.manual_seed(7)

    args = _parse_args()

    _daic_base         = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
    _daic_paint_hels   = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint/heliostats")
    _local_base        = _src.parent
    _local_paint_hels  = _local_base / "datasets" / "paint" / "heliostats"

    if args.daic:
        base_dir      = _daic_base
        heliostats_dir = _daic_paint_hels
    else:
        base_dir      = _local_base
        heliostats_dir = _local_paint_hels

    tower_file = heliostats_dir / "WRI1030197-tower-measurements.json"
    output_dir = args.output_dir or (
        base_dir / "outputs" / "smoke" if args.smoke_test
        else base_dir / "scenarios" / "full_field_one_heliostat_scenarios"
    )

    # Set up logging to file.
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info(f"PAINT heliostats dir : {heliostats_dir}")
    log.info(f"Tower file           : {tower_file}")
    log.info(f"Output dir           : {output_dir}")
    log.info(f"Force                : {args.force}")
    log.info(f"Smoke test           : {args.smoke_test}")

    if not heliostats_dir.exists():
        log.error(f"PAINT heliostats directory not found: {heliostats_dir}")
        sys.exit(1)
    if not tower_file.exists():
        log.error(f"Tower measurements file not found: {tower_file}")
        sys.exit(1)
    if not _AVAILABILITY_JSON.exists():
        log.error(f"Availability JSON not found: {_AVAILABILITY_JSON}")
        sys.exit(1)

    # Classify heliostats.
    defl_list, ideal_list, skipped_no_props = _classify_heliostats(
        _AVAILABILITY_JSON, heliostats_dir, args.heliostat_ids
    )

    if args.smoke_test:
        defl_list  = defl_list[:3]
        ideal_list = ideal_list[:3]

    if args.only_deflectometry:
        ideal_list = []
    if args.only_ideal:
        defl_list = []

    log.info(
        f"Heliostats: deflectometry={len(defl_list)}  ideal={len(ideal_list)}  "
        f"skipped(no props)={len(skipped_no_props)}"
    )

    # Load tower config once — shared by all heliostats.
    device = get_device()
    log.info(f"Device: {device}")
    log.info("Loading tower measurements ...")
    power_plant_config, target_area_list_planar_config, target_area_list_cylindrical_config = (
        paint_scenario_parser.extract_paint_tower_measurements(
            tower_measurements_path=tower_file, device=device
        )
    )
    log.info("Tower measurements loaded.")

    t_total = time.time()
    results: list[dict] = []

    # ------------------------------------------------------------------ #
    # Process deflectometry heliostats                                     #
    # ------------------------------------------------------------------ #
    if defl_list:
        log.info(f"\n{'=' * 60}")
        log.info(f"  Deflectometry group ({len(defl_list)} heliostats)")
        log.info(f"{'=' * 60}")

    for hid, props_path, defl_path in defl_list:
        out_path = output_dir / "deflectometry" / hid / "scenario.h5"
        t0 = time.time()

        if out_path.exists() and not args.force:
            log.info(f"  [SKIP] {hid} — already exists")
            results.append({"heliostat_id": hid, "type": "deflectometry", "status": "skipped"})
            continue

        try:
            create_one_scenario(
                hid=hid,
                props_path=props_path,
                defl_path=defl_path,
                out_path=out_path,
                power_plant_config=power_plant_config,
                target_area_list_planar_config=target_area_list_planar_config,
                target_area_list_cylindrical_config=target_area_list_cylindrical_config,
                device=device,
            )
            elapsed = (time.time() - t0) / 60.0
            log.info(f"  [OK]   {hid}  ({elapsed:.1f} min)")
            results.append({
                "heliostat_id": hid, "type": "deflectometry",
                "status": "ok", "elapsed_min": round(elapsed, 2),
            })
        except Exception as exc:
            elapsed = (time.time() - t0) / 60.0
            log.error(f"  [ERR]  {hid}: {exc}", exc_info=True)
            results.append({
                "heliostat_id": hid, "type": "deflectometry",
                "status": "error", "error": str(exc), "elapsed_min": round(elapsed, 2),
            })

    # ------------------------------------------------------------------ #
    # Process ideal heliostats                                             #
    # ------------------------------------------------------------------ #
    if ideal_list:
        log.info(f"\n{'=' * 60}")
        log.info(f"  Ideal group ({len(ideal_list)} heliostats)")
        log.info(f"{'=' * 60}")

    for hid, props_path in ideal_list:
        out_path = output_dir / "ideal" / hid / "scenario_ideal.h5"
        t0 = time.time()

        if out_path.exists() and not args.force:
            log.info(f"  [SKIP] {hid} — already exists")
            results.append({"heliostat_id": hid, "type": "ideal", "status": "skipped"})
            continue

        try:
            create_one_scenario(
                hid=hid,
                props_path=props_path,
                defl_path=None,
                out_path=out_path,
                power_plant_config=power_plant_config,
                target_area_list_planar_config=target_area_list_planar_config,
                target_area_list_cylindrical_config=target_area_list_cylindrical_config,
                device=device,
            )
            elapsed = (time.time() - t0) / 60.0
            log.info(f"  [OK]   {hid}  ({elapsed:.1f} min)")
            results.append({
                "heliostat_id": hid, "type": "ideal",
                "status": "ok", "elapsed_min": round(elapsed, 2),
            })
        except Exception as exc:
            elapsed = (time.time() - t0) / 60.0
            log.error(f"  [ERR]  {hid}: {exc}", exc_info=True)
            results.append({
                "heliostat_id": hid, "type": "ideal",
                "status": "error", "error": str(exc), "elapsed_min": round(elapsed, 2),
            })

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    total_min = (time.time() - t_total) / 60.0
    n_ok      = sum(1 for r in results if r["status"] == "ok")
    n_skip    = sum(1 for r in results if r["status"] == "skipped")
    n_err     = sum(1 for r in results if r["status"] == "error")

    summary = {
        "timestamp":        datetime.now().strftime("%Y%m%d_%H%M%S"),
        "output_dir":       str(output_dir),
        "n_deflectometry":  len(defl_list),
        "n_ideal":          len(ideal_list),
        "n_skipped_no_props": len(skipped_no_props),
        "n_ok":             n_ok,
        "n_skipped":        n_skip,
        "n_error":          n_err,
        "total_min":        round(total_min, 2),
        "heliostats":       results,
    }
    with open(output_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print()
    print("=" * 65)
    print(
        f"  Done  |  ok={n_ok}  skipped={n_skip}  errors={n_err}  "
        f"|  {total_min:.1f} min"
    )
    print(f"  Output: {output_dir}")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
