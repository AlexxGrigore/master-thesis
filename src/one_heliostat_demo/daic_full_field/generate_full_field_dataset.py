"""
Generate a perturbed synthetic dataset for the full heliostat field.

Unlike generate_all.py, which borrows sun positions from PAINT benchmark CSVs
(and is therefore limited to heliostats that appear in PAINT), this script
samples sun positions uniformly from a pre-built universe of all unique
(azimuth, elevation) pairs found in the PAINT dataset.  Any heliostat that has
a single-heliostat scenario file can be processed, regardless of whether it
appears in PAINT.

Sun-position universe
---------------------
Build it once with build_sun_universe.py:
    python build_sun_universe.py
This writes datasets/sun_universe.json — a sorted list of [azimuth, elevation]
pairs (degrees).  The file is a static asset committed to the repository.

Output layout
-------------
    {output_dir}/
        dataset/
            perturbations.json
            train/{hid}/{idx:04d}/
                calibration_properties.json
                flux_image.png
        summary.json
        run.log

Everything is saved under train/ so the existing _pool_and_split in train.py
can pool and re-split at training time without any structural changes.

Active-pixel filter
-------------------
Both the fast-scan rejection AND the per-sample save filter use the same
normalised threshold (> 0.01 on [0,1] flux) that the training pipeline uses.
This closes the small discrepancy in the existing generate_all.py pipeline
where 19/12 113 samples were generated but later filtered at training time.

Usage
-----
    python generate_full_field_dataset.py
    python generate_full_field_dataset.py --smoke-test
    python generate_full_field_dataset.py --heliostat-ids AC36 BE35
    python generate_full_field_dataset.py --daic
    python generate_full_field_dataset.py --daic --n-pool 200 --sampling-seed 0
"""

import argparse
import gc
import json
import logging
import pathlib
import sys
import time
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

_here = pathlib.Path(__file__).resolve().parent          # daic_full_field/
_src  = _here.parent.parent                              # src/
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_here.parent))                    # one_heliostat_demo/

from single_heliostat import config as cfg               # noqa: E402

from artist.geometry import coordinates                  # noqa: E402
from artist.scenario.scenario import Scenario            # noqa: E402
from artist.util import constants as _const, get_device, set_logger_config  # noqa: E402
from artist.util import setup_distributed_environment    # noqa: E402

from utils.synth_data import (                           # noqa: E402
    _forward_pass,
    apply_perturbations,
    reset_perturbations,
    sample_perturbations,
)

log = logging.getLogger(__name__)

DEFAULT_N_POOL      = 200     # total samples to attempt per heliostat
DEFAULT_TARGET_IDX  = 1       # STJ_LOWER (matches the most common target in PAINT)
DEFAULT_SCAN_RAYS   = 10
DEFAULT_GENERATE_RAYS = 100
DEFAULT_MIN_ACTIVE_PCT = 2.0  # percent of pixels > 0.01 (normalised) required


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_universe(path: pathlib.Path) -> list[tuple[float, float]]:
    """Load sun_universe.json and return as list of (azimuth_deg, elevation_deg)."""
    with open(path) as fh:
        data = json.load(fh)
    return [(float(az), float(el)) for az, el in data]


def _sample_sun_positions(
    universe: list[tuple[float, float]],
    n: int,
    rng: np.random.Generator,
) -> tuple[list[float], list[float]]:
    """
    Draw n positions uniformly without replacement from the universe.
    Falls back to sampling with replacement if n > len(universe).
    """
    replace = n > len(universe)
    indices = rng.choice(len(universe), size=n, replace=replace)
    azimuths   = [universe[i][0] for i in indices]
    elevations = [universe[i][1] for i in indices]
    return azimuths, elevations


def _rays_from_sun_angles(
    azimuths_deg: list[float],
    elevations_deg: list[float],
    device: torch.device,
) -> torch.Tensor:
    """
    Convert (azimuth, elevation) in degrees to incident ray directions [N, 4].

    Replicates the formula used by PaintCalibrationDataParser:
        enu  = azimuth_elevation_to_enu(az, el, degree=True)   [N, 3]
        pos4 = convert_3d_points_to_4d_format(enu)             [N, 4]  (w=1)
        rays = [0,0,0,1] - pos4                                [N, 4]  (w=0)
    """
    az = torch.tensor(azimuths_deg,   dtype=torch.float32, device=device)
    el = torch.tensor(elevations_deg, dtype=torch.float32, device=device)

    enu  = coordinates.azimuth_elevation_to_enu(az, el, degree=True, device=device)
    pos4 = coordinates.convert_3d_points_to_4d_format(enu, device=device)
    rays = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device) - pos4
    return rays   # [N, 4]


def _active_pixel_pct_normalised(flux_img: torch.Tensor) -> float:
    """Fraction (%) of pixels > 0.01 on the [0,1]-normalised saved image."""
    fl    = flux_img.cpu().float().numpy()
    fmax  = fl.max()
    if fmax < 1e-12:
        return 0.0
    norm  = fl / fmax          # normalise to [0,1] as PNG save/load would do
    return float((norm > 0.01).sum()) / norm.size * 100.0


def _find_scenario(
    scenario_root: pathlib.Path, hid: str
) -> tuple[pathlib.Path, str] | None:
    """
    Locate the scenario file for a heliostat and return (path, type).

    Supports two directory layouts:

    DAIC layout (subdirectories per type):
      {scenario_root}/deflectometry/{hid}/scenario.h5
      {scenario_root}/ideal/{hid}/scenario_ideal.h5

    Local flat layout (all heliostats in one directory):
      {scenario_root}/{hid}/scenario.h5
      {scenario_root}/{hid}/scenario_ideal.h5

    Preference order within each layout: deflectometry before ideal.
    Returns None if no file is found in any location.
    """
    candidates = [
        (scenario_root / "deflectometry" / hid / "scenario.h5",       "deflectometry"),
        (scenario_root / hid / "scenario.h5",                          "deflectometry"),
        (scenario_root / "ideal"          / hid / "scenario_ideal.h5", "ideal"),
        (scenario_root / hid / "scenario_ideal.h5",                    "ideal"),
    ]
    for path, scen_type in candidates:
        if path.exists():
            return path, scen_type
    return None


def _generate_heliostat(
    heliostat_id: str,
    scenario_path: pathlib.Path,
    universe: list[tuple[float, float]],
    dataset_dir: pathlib.Path,
    cfg,
    device: torch.device,
    seed_offset: int = 0,
    n_pool: int = DEFAULT_N_POOL,
    target_index: int = DEFAULT_TARGET_IDX,
    scan_rays: int = DEFAULT_SCAN_RAYS,
    generate_rays: int = DEFAULT_GENERATE_RAYS,
    min_active_pct: float = DEFAULT_MIN_ACTIVE_PCT,
    max_attempts: int = 10,
) -> dict:
    """
    Generate and save a perturbed synthetic dataset for one heliostat.

    dataset_dir is the dataset root for this heliostat's deflectometry group
    (e.g. output_dir / "with_deflectometry" / "dataset").  Samples are saved
    under dataset_dir / "train" / heliostat_id / {idx:04d}/.

    Returns
    -------
    dict with "succeeded", "n_saved", "attempt_used", "elapsed_min"
    """
    log.info(f"Loading scenario: {scenario_path}")
    with h5py.File(scenario_path, "r") as fh:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=fh,
            device=device,
            number_of_surface_points_per_facet=torch.tensor(
                [cfg.SURFACE_POINTS_PER_FACET, cfg.SURFACE_POINTS_PER_FACET]
            ),
        )

    heliostat_group = scenario.heliostat_field.heliostat_groups[0]
    kinematic       = heliostat_group.kinematics
    old_n_rays      = scenario.light_sources.light_source_list[0].number_of_rays

    # Sample sun positions from the universe (fixed for this heliostat so that
    # all perturbation attempts use the same geometry pool).
    sampling_rng = np.random.default_rng(seed=cfg.RANDOM_SEED + seed_offset * 1000)
    azimuths, elevations = _sample_sun_positions(universe, n_pool, sampling_rng)

    pool_rays        = _rays_from_sun_angles(azimuths, elevations, device)  # [N, 4]
    pool_target_mask = torch.full(
        (n_pool,), fill_value=target_index, dtype=torch.long, device=device
    )
    pool_active_mask = torch.tensor([n_pool], device=device, dtype=torch.long)

    # Resample perturbation seeds until enough samples pass the active-pixel filter.
    chosen_pert      = None
    chosen_flux      = None
    chosen_centroids = None
    chosen_motor_pos = None
    attempt_used     = 0
    succeeded        = False

    for attempt in range(max_attempts):
        seed = cfg.RANDOM_SEED + seed_offset * (max_attempts + 1) + attempt

        pert_tensors = sample_perturbations(
            n_heliostats=1, ranges=cfg.RANDOM_PERT_BOUNDS, seed=seed
        )
        snap    = apply_perturbations(kinematic, pert_tensors, device)
        pert_bpd = kinematic._base_position_deviation.detach().clone()

        # Fast scan — use physical-units > 0 threshold here (cheap pre-filter).
        scenario.set_number_of_rays(scan_rays)
        with torch.no_grad():
            _, flux_scan = _forward_pass(
                scenario, heliostat_group,
                pool_rays, pool_active_mask, pool_target_mask, pert_bpd, device,
            )

        n_ok_scan = sum(
            1 for i in range(n_pool)
            if float((flux_scan[i] > 0).sum()) / flux_scan[i].numel() * 100.0
            >= min_active_pct
        )
        scan_pass = n_ok_scan >= n_pool // 2     # require at least 50% to pass
        log.info(
            f"  Attempt {attempt + 1}/{max_attempts}: "
            f"fast scan {n_ok_scan}/{n_pool}  "
            f"{'[pass]' if scan_pass else '[fail]'}"
        )

        if not scan_pass:
            reset_perturbations(kinematic, snap)
            continue

        # Full generation.
        scenario.set_number_of_rays(generate_rays)
        centroids, flux = _forward_pass(
            scenario, heliostat_group,
            pool_rays, pool_active_mask, pool_target_mask, pert_bpd, device,
        )
        motor_pos = heliostat_group.kinematics.active_motor_positions.detach().clone()

        # Count samples that pass the normalised threshold (matches training filter).
        n_ok_full = sum(
            1 for i in range(n_pool)
            if _active_pixel_pct_normalised(flux[i]) >= min_active_pct
        )
        ok = n_ok_full >= n_pool // 2
        log.info(
            f"  Attempt {attempt + 1}/{max_attempts}: "
            f"full gen  {n_ok_full}/{n_pool}  "
            f"{'[OK]' if ok else '[FAIL]'}"
        )

        reset_perturbations(kinematic, snap)

        if ok:
            chosen_pert      = pert_tensors
            chosen_flux      = flux
            chosen_centroids = centroids
            chosen_motor_pos = motor_pos
            attempt_used     = attempt
            succeeded        = True
            break

    scenario.set_number_of_rays(old_n_rays)

    if not succeeded:
        log.warning(f"  {heliostat_id}: no valid seed after {max_attempts} attempts — skipping.")
        return {"succeeded": False, "n_saved": 0, "attempt_used": max_attempts, "elapsed_min": 0.0}

    # Collect indices of all passing samples.
    passing = [i for i in range(n_pool) if _active_pixel_pct_normalised(chosen_flux[i]) >= min_active_pct]
    n_saved = len(passing)

    if n_saved == 0:
        log.warning(f"  {heliostat_id}: 0 samples passed the filter — skipping save.")
        return {"succeeded": False, "n_saved": 0, "attempt_used": attempt_used + 1}

    # Assign each sample to train / validation / test using a balanced split
    # (KMeans on sun positions so all splits cover the full sky uniformly).
    # The split is purely cosmetic — the training pipeline re-pools and re-splits
    # anyway — but it makes the folder structure easier to inspect.
    try:
        from paint.data.dataset_splits import DatasetSplitter
        import paint.util.paint_mappings as paint_mappings

        _sun = -pool_rays[passing].cpu().numpy()   # negate: ray points sun→hel
        azimuths   = np.degrees(np.arctan2(_sun[:, 0], _sun[:, 1])) % 360.0
        elevations = np.degrees(np.arcsin(np.clip(_sun[:, 2], -1.0, 1.0)))

        val_size   = min(50, n_saved // 4)
        train_size = n_saved - 2 * val_size

        if train_size >= 1:
            heliostat_df = pd.DataFrame({
                paint_mappings.HELIOSTAT_ID: heliostat_id,
                paint_mappings.AZIMUTH:      azimuths,
                paint_mappings.ELEVATION:    elevations,
                paint_mappings.DATETIME:     pd.Timestamp("2020-06-15 12:00:00"),
                paint_mappings.SPLIT_KEY:    "",
            }, index=range(n_saved))

            split_df = DatasetSplitter._get_balanced_splits(heliostat_df, train_size, val_size)
            # split labels are 'train', 'validation', 'test'
            split_labels = split_df[paint_mappings.SPLIT_KEY].tolist()
        else:
            split_labels = ["train"] * n_saved

    except ImportError:
        log.warning("PAINT library not available; saving all samples to train/.")
        split_labels = ["train"] * n_saved

    # Save each sample to its assigned split subfolder.
    split_counters: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    for j, i in enumerate(passing):
        split = split_labels[j]
        idx   = split_counters[split]
        split_counters[split] += 1

        sample_dir = dataset_dir / split / heliostat_id / f"{idx:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        cal = {
            "target_area_index":      target_index,
            "incident_ray_direction": pool_rays[i].cpu().tolist(),
            "focal_spot_enu":         chosen_centroids[i].cpu().tolist(),
            "motor_position":         chosen_motor_pos[i].cpu().tolist(),
        }
        with open(sample_dir / "calibration_properties.json", "w") as fh:
            json.dump(cal, fh, indent=2)

        fl   = chosen_flux[i].cpu().float().numpy()
        fmax = fl.max()
        if fmax > 1e-12:
            fl_uint8 = (fl / fmax * 255).clip(0, 255).astype(np.uint8)
        else:
            fl_uint8 = np.zeros_like(fl, dtype=np.uint8)
        Image.fromarray(fl_uint8, mode="L").save(sample_dir / "flux_image.png")

    log.info(
        f"  {heliostat_id}: saved {n_saved}/{n_pool}  "
        f"train={split_counters['train']}  "
        f"val={split_counters['validation']}  "
        f"test={split_counters['test']}  "
        f"→ {dataset_dir}"
    )
    return {
        "succeeded":    True,
        "n_saved":      n_saved,
        "attempt_used": attempt_used + 1,
        "pert_tensors": chosen_pert,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate full-field synthetic dataset using a shared sun-position universe."
    )
    p.add_argument(
        "--output-dir", type=pathlib.Path, default=None,
        help="Output root (default: outputs/full_field_{timestamp}/).",
    )
    p.add_argument(
        "--universe-file", type=pathlib.Path, default=None,
        help="Path to sun_universe.json (default: datasets/sun_universe.json).",
    )
    p.add_argument(
        "--scenario-dir", type=pathlib.Path, default=None,
        help="Root of one_heliostat_scenarios/ (default: cfg.SCENARIO_PATH_TEMPLATE parent).",
    )
    p.add_argument(
        "--heliostat-ids", nargs="+", default=None, metavar="ID",
        help="Subset of heliostat IDs to process (default: all found in scenario-dir).",
    )
    p.add_argument(
        "--n-pool", type=int, default=DEFAULT_N_POOL,
        help=f"Total samples to attempt per heliostat (default: {DEFAULT_N_POOL}).",
    )
    p.add_argument(
        "--target-index", type=int, default=DEFAULT_TARGET_IDX,
        help=f"Fixed target area index for all samples (default: {DEFAULT_TARGET_IDX} = STJ_LOWER).",
    )
    p.add_argument(
        "--sampling-seed", type=int, default=42,
        help="RNG seed for uniform sampling from the universe (default: 42).",
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Quick sanity check: first 3 heliostats, fast ray counts.",
    )
    p.add_argument(
        "--daic", action="store_true",
        help="Use DAIC cluster paths.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    _daic_base  = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
    _daic_paint = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")

    if args.daic:
        cfg.BASE_DIR  = _daic_base
        cfg.PAINT_DIR = _daic_paint

    # Timestamp is created once so the directory name is stable for the whole run.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve paths.
    universe_file = args.universe_file or (cfg.BASE_DIR / "datasets" / "sun_universe.json")
    scenario_root = args.scenario_dir  or (
        cfg.BASE_DIR / "scenarios" / "one_heliostat_scenarios"
    )
    output_dir    = args.output_dir    or (
        cfg.BASE_DIR / "outputs" / "smoke" if args.smoke_test
        else cfg.BASE_DIR / "outputs" / f"full_field_{run_timestamp}"
    )

    n_pool    = args.n_pool
    scan_rays = DEFAULT_SCAN_RAYS
    gen_rays  = DEFAULT_GENERATE_RAYS

    if args.smoke_test:
        scan_rays = 5
        gen_rays  = 10
        n_pool    = 20

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset roots for the two deflectometry groups.
    defl_dataset_dir  = output_dir / "with_deflectometry"  / "dataset"
    ideal_dataset_dir = output_dir / "without_deflectometry" / "dataset"

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info(f"Output dir     : {output_dir}")
    log.info(f"Universe file  : {universe_file}")
    log.info(f"Scenario root  : {scenario_root}")
    log.info(f"n_pool         : {n_pool}")
    log.info(f"target_index   : {args.target_index}")
    log.info(f"sampling_seed  : {args.sampling_seed}")
    log.info(f"smoke_test     : {args.smoke_test}")

    # Load universe.
    if not pathlib.Path(universe_file).exists():
        log.error(
            f"Sun universe file not found: {universe_file}\n"
            "Run build_sun_universe.py first to generate it."
        )
        sys.exit(1)
    universe = _load_universe(pathlib.Path(universe_file))
    log.info(f"Universe loaded: {len(universe)} unique (azimuth, elevation) positions")

    # Discover heliostat IDs and classify by scenario type.
    scenario_root = pathlib.Path(scenario_root)
    if args.heliostat_ids:
        candidate_ids = args.heliostat_ids
    else:
        found: set[str] = set()
        # DAIC layout: deflectometry/ and ideal/ subdirectories.
        for subdir_name, filename in [
            ("deflectometry", "scenario.h5"),
            ("ideal",         "scenario_ideal.h5"),
        ]:
            subdir = scenario_root / subdir_name
            if subdir.is_dir():
                found.update(
                    d.name for d in subdir.iterdir()
                    if d.is_dir() and (d / filename).exists()
                )
        # Local flat layout: heliostats directly under scenario_root.
        found.update(
            d.name for d in scenario_root.iterdir()
            if d.is_dir() and (
                (d / "scenario.h5").exists() or (d / "scenario_ideal.h5").exists()
            )
        )
        candidate_ids = sorted(found)
    if args.smoke_test:
        candidate_ids = candidate_ids[:3]

    # Map each heliostat to its scenario path and deflectometry type.
    heliostat_scenarios: list[tuple[str, pathlib.Path, str]] = []  # (hid, path, type)
    skipped_no_scenario: list[str] = []
    for hid in candidate_ids:
        found = _find_scenario(scenario_root, hid)
        if found is None:
            log.warning(f"{hid}: no scenario.h5 or scenario_ideal.h5 found — skipped")
            skipped_no_scenario.append(hid)
        else:
            heliostat_scenarios.append((hid, found[0], found[1]))

    n_defl  = sum(1 for _, _, t in heliostat_scenarios if t == "deflectometry")
    n_ideal = sum(1 for _, _, t in heliostat_scenarios if t == "ideal")
    log.info(
        f"Heliostats to process: {len(heliostat_scenarios)}  "
        f"(deflectometry={n_defl}, ideal={n_ideal})"
    )

    device = get_device()
    defl_perturbations:  dict = {}
    ideal_perturbations: dict = {}
    summary: list[dict] = []
    t_total = time.time()

    with setup_distributed_environment(
        number_of_heliostat_groups=1, device=device
    ) as ddp:
        device = ddp[_const.device]
        log.info(f"Device: {device}")

        for i, (hid, scenario_path, scen_type) in enumerate(
            tqdm(heliostat_scenarios, desc="Generating", unit="hel", dynamic_ncols=True)
        ):
            log.info(f"\n[{i + 1}/{len(heliostat_scenarios)}] {hid}  ({scen_type})")
            dataset_dir = defl_dataset_dir if scen_type == "deflectometry" else ideal_dataset_dir
            t_hel = time.time()

            try:
                result = _generate_heliostat(
                    heliostat_id=hid,
                    scenario_path=scenario_path,
                    universe=universe,
                    dataset_dir=dataset_dir,
                    cfg=cfg,
                    device=device,
                    seed_offset=i + args.sampling_seed * 10_000,
                    n_pool=n_pool,
                    target_index=args.target_index,
                    scan_rays=scan_rays,
                    generate_rays=gen_rays,
                )
                elapsed_min = (time.time() - t_hel) / 60.0

                summary.append({
                    "heliostat_id":  hid,
                    "scenario_type": scen_type,
                    "status":        "ok" if result["succeeded"] else "no_valid_seed",
                    "n_saved":       result["n_saved"],
                    "attempt_used":  result["attempt_used"],
                    "elapsed_min":   round(elapsed_min, 2),
                })

                if result["succeeded"] and result.get("pert_tensors") is not None:
                    pt  = result["pert_tensors"]
                    rec = {
                        "rotation_rad":       pt["rotation"][0].cpu().tolist(),
                        "actuator_angle_rad": pt["actuator_angle"][0].cpu().tolist(),
                        "actuator_stroke_m":  pt["actuator_stroke"][0].cpu().tolist(),
                        "actuator_offset_m":  pt["actuator_offset"][0].cpu().tolist(),
                        "translation_m":      pt["translation"][0].cpu().tolist(),
                        "base_position_m":    pt["base_position"][0].cpu().tolist(),
                    }
                    if scen_type == "deflectometry":
                        defl_perturbations[hid]  = rec
                    else:
                        ideal_perturbations[hid] = rec

            except Exception as exc:
                elapsed_min = (time.time() - t_hel) / 60.0
                log.error(f"  {hid} FAILED: {exc}", exc_info=True)
                summary.append({
                    "heliostat_id":  hid,
                    "scenario_type": scen_type,
                    "status":        "error",
                    "error":         str(exc),
                    "elapsed_min":   round(elapsed_min, 2),
                })
            finally:
                gc.collect()
                torch.cuda.empty_cache()

    # Write perturbations.json for each deflectometry group.
    for pert_dict, dset_dir in [
        (defl_perturbations,  defl_dataset_dir),
        (ideal_perturbations, ideal_dataset_dir),
    ]:
        if not pert_dict:
            continue
        pfile = dset_dir / "perturbations.json"
        pfile.parent.mkdir(parents=True, exist_ok=True)
        with open(pfile, "w") as fh:
            json.dump(pert_dict, fh, indent=2)
        log.info(f"perturbations.json → {pfile}  ({len(pert_dict)} heliostats)")

    # Write top-level summary.json.
    total_min = (time.time() - t_total) / 60.0
    n_ok      = sum(1 for s in summary if s["status"] == "ok")
    n_no_seed = sum(1 for s in summary if s["status"] == "no_valid_seed")
    n_error   = sum(1 for s in summary if s["status"] == "error")

    summary_doc = {
        "timestamp":      run_timestamp,
        "output_dir":     str(output_dir),
        "universe_file":  str(universe_file),
        "n_pool":         n_pool,
        "target_index":   args.target_index,
        "sampling_seed":  args.sampling_seed,
        "n_deflectometry": n_defl,
        "n_ideal":         n_ideal,
        "n_ok":           n_ok,
        "n_no_seed":      n_no_seed,
        "n_error":        n_error,
        "n_skipped_no_scenario": len(skipped_no_scenario),
        "total_min":      round(total_min, 2),
        "heliostats":     summary,
    }
    with open(output_dir / "summary.json", "w") as fh:
        json.dump(summary_doc, fh, indent=2)

    # Final table.
    print()
    print("=" * 75)
    print(
        f"  Generation complete  |  {n_ok}/{len(heliostat_scenarios)} succeeded  "
        f"|  {total_min:.1f} min"
    )
    print(f"  deflectometry={n_defl}  ideal={n_ideal}  skipped={len(skipped_no_scenario)}")
    print("=" * 75)
    print(f"  {'Heliostat':<12} {'Type':<16} {'Status':<14} {'Saved':>6} {'Attempt':>7} {'Min':>6}")
    print("  " + "-" * 63)
    for s in summary:
        stype = s.get("scenario_type", "?")
        if s["status"] == "ok":
            print(
                f"  {s['heliostat_id']:<12} {stype:<16} {'ok':<14} "
                f"{s['n_saved']:>6} {s['attempt_used']:>7} {s['elapsed_min']:>6.1f}"
            )
        elif s["status"] == "no_valid_seed":
            print(
                f"  {s['heliostat_id']:<12} {stype:<16} {'no valid seed':<14} "
                f"{'—':>6} {'—':>7} {s['elapsed_min']:>6.1f}"
            )
        else:
            print(
                f"  {s['heliostat_id']:<12} {stype:<16} {'error':<14}  "
                f"{s.get('error', '')[:35]}"
            )
    print("=" * 75)
    print(f"\n  Output: {output_dir}")
    print()


if __name__ == "__main__":
    main()
