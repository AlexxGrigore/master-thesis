"""
Generate a perturbed synthetic dataset for a single heliostat.

For each heliostat this script:
  1. Loads ray directions from the PAINT benchmark calibration dataset (all
     three PAINT splits are pooled into a single geometry pool).
  2. Samples random kinematic perturbations (or uses CUSTOM_PERTURBATIONS_SPEC).
  3. Fast scan: ray-traces the full pool with SCAN_RAYS to check whether
     enough samples pass the active-pixel filter (MIN_POOL = MIN_TRAIN +
     MIN_VAL + MIN_TEST).  Bad seeds are rejected cheaply before the
     expensive full-quality pass.
  4. Full generation: re-traces with GENERATE_RAYS for the accepted seed.
  5. Saves samples to output_dir/dataset/{split}/{HELIOSTAT_ID}/{idx:04d}/:
       calibration_properties.json  (incident_ray_direction, focal_spot_enu, motor_position, target_area_index)
       flux_image.png               (uint8 peak-normalised flux bitmap)
     The on-disk split folders mirror the originating PAINT splits (train /
     val / test) so that existing notebooks can pool them at read time.
  6. Saves output_dir/dataset/perturbations.json with GT perturbation values.

Resampling logic
----------------
If fewer than MIN_POOL pool samples survive the active-pixel filter after the
fast scan, the perturbation is resampled with seed = RANDOM_SEED + attempt
(up to MAX_RESAMPLE_ATTEMPTS tries).  The full generation pass is only run
for seeds that survive the fast scan.

Usage
-----
  python generate_dataset.py [--heliostat-id AC33] [--output-dir PATH]
"""

import argparse
import json
import logging
import pathlib
import sys

import h5py
import numpy as np
import torch
from PIL import Image

_here = pathlib.Path(__file__).resolve().parent   # single_heliostat/
_src  = _here.parent.parent                         # src/
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_here))

from artist.scenario.scenario import Scenario
from artist.util import constants as _const, get_device, set_logger_config
from artist.util import setup_distributed_environment

from artist.io.paint_calibration_parser import PaintCalibrationDataParser

from utils.evaluation import build_heliostat_data_mapping
from utils.synth_data import (
    _forward_pass,
    apply_perturbations,
    reset_perturbations,
    sample_perturbations,
)

log = logging.getLogger(__name__)


def _active_pixel_percent(flux_img: torch.Tensor) -> float:
    # Raw ray-tracer output is in physical intensity units (~1e-4 at 100m).
    # Unhit pixels are exactly 0, so > 0 is the correct threshold here.
    return float((flux_img > 0).sum().item()) / float(flux_img.numel()) * 100.0


def generate(
    heliostat_id: str,
    output_dir: pathlib.Path,
    cfg,
    device: torch.device,
    seed_offset: int = 0,
) -> dict:
    """
    Generate and save a perturbed synthetic dataset for one heliostat.

    Returns
    -------
    dict with:
        "pert_tensors"  : dict of torch.Tensor — perturbations that were used
        "dataset_dir"   : pathlib.Path to the saved dataset root
        "attempt_used"  : int — which attempt succeeded (0-based)
    """
    output_dir    = pathlib.Path(output_dir)
    scenario_path = pathlib.Path(cfg.SCENARIO_PATH_TEMPLATE.format(heliostat_id=heliostat_id))
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario not found: {scenario_path}")

    log.info(f"Loading scenario for {heliostat_id}: {scenario_path}")
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

    min_per_split = {
        "train": getattr(cfg, "MIN_TRAIN_SAMPLES",     50),
        "val":   getattr(cfg, "MIN_VAL_SAMPLES",       50),
        "test":  getattr(cfg, "MIN_TEST_SAMPLES",      50),
    }
    min_active_pct = getattr(cfg, "MIN_ACTIVE_PIXEL_PERCENT", 2.0)
    max_attempts   = getattr(cfg, "MAX_RESAMPLE_ATTEMPTS",    10)
    scan_rays      = getattr(cfg, "SCAN_RAYS",                10)
    min_pool       = sum(min_per_split.values())

    # -------------------------------------------------------------------------
    # Pool all PAINT splits into one geometry pool.
    # The split boundaries are recorded so results can be saved back to the
    # correct on-disk folders (train / val / test).
    # -------------------------------------------------------------------------
    old_n_rays = scenario.light_sources.light_source_list[0].number_of_rays

    _PAINT_SPLITS = {"train": "train", "validation": "val", "test": "test"}
    paint_parser  = PaintCalibrationDataParser()

    pool_rays_list        = []
    pool_target_mask_list = []
    split_boundaries: dict[str, tuple[int, int]] = {}  # our_split -> (start, end)
    offset = 0

    for paint_split, our_split in _PAINT_SPLITS.items():
        full_mapping = build_heliostat_data_mapping(
            benchmark_csv=cfg.BENCHMARK_CSV,
            calibration_properties_dir=cfg.CALIBRATION_DIR,
            flux_image_dir=cfg.REAL_FLUX_DIR,
            split=paint_split,
        )
        hel_mapping = [
            (hid, cal_paths, flux_paths)
            for hid, cal_paths, flux_paths in full_mapping
            if hid == heliostat_id
        ]
        if not hel_mapping:
            log.warning(
                f"Skipping {our_split}: {heliostat_id} not found in PAINT {paint_split} split"
            )
            continue
        n_samples = len(hel_mapping[0][1])
        _, _, rays, _, _, target_mask = paint_parser.parse_data_for_reconstruction(
            heliostat_data_mapping=hel_mapping,
            heliostat_group=heliostat_group,
            scenario=scenario,
            device=device,
        )
        pool_rays_list.append(rays)
        pool_target_mask_list.append(target_mask)
        split_boundaries[our_split] = (offset, offset + n_samples)
        offset += n_samples
        log.info(f"  Loaded {n_samples} ray directions for {our_split} from PAINT {paint_split}")

    if not split_boundaries:
        raise RuntimeError(
            f"No PAINT data found for {heliostat_id} in {cfg.BENCHMARK_CSV}"
        )

    N_POOL           = offset
    pool_rays        = torch.cat(pool_rays_list,        dim=0)  # [N_POOL, 4]
    pool_target_mask = torch.cat(pool_target_mask_list, dim=0)  # [N_POOL]
    pool_active_mask = torch.tensor([N_POOL], device=device, dtype=torch.long)

    log.info(
        f"Pool: {N_POOL} rays total  |  splits: "
        + ", ".join(f"{sp}={e-s}" for sp, (s, e) in split_boundaries.items())
        + f"  |  min_pool={min_pool}"
    )

    # -------------------------------------------------------------------------
    # Resample loop: fast scan first, then full-quality generation.
    # -------------------------------------------------------------------------
    # Fixed perturbations (committed perturbations.json replay): use exactly the
    # given values instead of sampling — keeps regenerated datasets identical to
    # the original across machines.
    fixed_pert_tensors = None
    fixed_json = getattr(cfg, "FIXED_PERTURBATIONS_JSON", None)
    if fixed_json:
        with open(fixed_json) as fh:
            spec = json.load(fh).get(heliostat_id)
        if spec is None:
            raise KeyError(
                f"{heliostat_id} not found in FIXED_PERTURBATIONS_JSON ({fixed_json})"
            )
        fixed_pert_tensors = {
            "rotation":        torch.tensor([spec["rotation_rad"]], dtype=torch.float32, device=device),
            "actuator_angle":  torch.tensor([spec["actuator_angle_rad"]], dtype=torch.float32, device=device),
            "actuator_stroke": torch.tensor([spec["actuator_stroke_m"]], dtype=torch.float32, device=device),
            "actuator_offset": torch.tensor([spec["actuator_offset_m"]], dtype=torch.float32, device=device),
            "translation":     torch.tensor([spec["translation_m"]], dtype=torch.float32, device=device),
            "base_position":   torch.tensor([spec["base_position_m"]], dtype=torch.float32, device=device),
        }
        log.info(f"Using fixed perturbations from {fixed_json}")

    chosen_pert      = None
    chosen_flux      = None
    chosen_centroids = None
    chosen_motor_pos = None
    attempt_used     = 0
    succeeded        = False

    for attempt in range(max_attempts):
        seed = cfg.RANDOM_SEED + seed_offset * (max_attempts + 1) + attempt

        if fixed_pert_tensors is not None:
            pert_tensors = fixed_pert_tensors
        elif cfg.DATA_MODE == "random_synthetic":
            pert_tensors = sample_perturbations(
                n_heliostats=1, ranges=cfg.RANDOM_PERT_BOUNDS, seed=seed
            )
        else:
            pert_tensors = {
                k: torch.tensor([v], dtype=torch.float32, device=device)
                for k, v in cfg.CUSTOM_PERTURBATIONS_SPEC.items()
            }

        snap    = apply_perturbations(kinematic, pert_tensors, device)
        pert_bpd = kinematic._base_position_deviation.detach().clone()  # [1, 3]

        # -- Fast scan --------------------------------------------------------
        scenario.set_number_of_rays(scan_rays)
        with torch.no_grad():
            _, flux_scan = _forward_pass(
                scenario, heliostat_group,
                pool_rays, pool_active_mask, pool_target_mask, pert_bpd, device,
            )

        n_ok_scan = sum(
            1 for i in range(N_POOL)
            if _active_pixel_percent(flux_scan[i]) >= min_active_pct
        )
        scan_pass = n_ok_scan >= min_pool
        log.info(
            f"Attempt {attempt + 1}/{max_attempts}: "
            f"fast scan {n_ok_scan}/{N_POOL} >= {min_pool}  "
            f"{'[pass]' if scan_pass else '[fail — skip]'}"
        )

        if not scan_pass and cfg.DATA_MODE == "random_synthetic":
            reset_perturbations(kinematic, snap)
            continue

        # -- Full-quality generation ------------------------------------------
        scenario.set_number_of_rays(cfg.GENERATE_RAYS)
        centroids, flux = _forward_pass(
            scenario, heliostat_group,
            pool_rays, pool_active_mask, pool_target_mask, pert_bpd, device,
        )
        motor_pos = heliostat_group.kinematics.active_motor_positions.detach().clone()

        n_ok_total = sum(
            1 for i in range(N_POOL)
            if _active_pixel_percent(flux[i]) >= min_active_pct
        )
        ok     = n_ok_total >= min_pool
        status = "[OK]" if ok else "[FAIL]"
        log.info(
            f"Attempt {attempt + 1}/{max_attempts}: "
            f"full gen  {n_ok_total}/{N_POOL} >= {min_pool}  {status}"
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

        if cfg.DATA_MODE != "random_synthetic":
            log.warning("DATA_MODE='synthetic' — cannot resample. Proceeding as-is.")
            break

    scenario.set_number_of_rays(old_n_rays)

    if not succeeded:
        log.warning(
            f"{heliostat_id}: no seed produced >= {min_pool} valid pool samples "
            f"after {max_attempts} attempts — skipping save."
        )
        return {
            "succeeded":    False,
            "pert_tensors": None,
            "dataset_dir":  output_dir / "dataset",
            "attempt_used": max_attempts,
        }

    # -------------------------------------------------------------------------
    # Save dataset to disk, split back along PAINT split boundaries.
    # -------------------------------------------------------------------------
    dataset_dir = output_dir / "dataset"

    for our_split, (start, end) in split_boundaries.items():
        out_split = dataset_dir / our_split / heliostat_id
        out_split.mkdir(parents=True, exist_ok=True)

        n                 = end - start
        split_rays        = pool_rays[start:end]
        split_flux        = chosen_flux[start:end]
        split_centroids   = chosen_centroids[start:end]
        split_motor_pos   = chosen_motor_pos[start:end]
        split_target_mask = pool_target_mask[start:end]

        saved_idx = 0
        for i in range(n):
            if _active_pixel_percent(split_flux[i]) < min_active_pct:
                continue

            sample_dir = out_split / f"{saved_idx:04d}"
            sample_dir.mkdir(exist_ok=True)

            cal = {
                "target_area_index":      int(split_target_mask[i].item()),
                "incident_ray_direction": split_rays[i].cpu().tolist(),
                "focal_spot_enu":         split_centroids[i].cpu().tolist(),
                "motor_position":         split_motor_pos[i].cpu().tolist(),
            }
            with open(sample_dir / "calibration_properties.json", "w") as fh:
                json.dump(cal, fh, indent=2)

            fl    = split_flux[i].cpu().float().numpy()
            fmax  = fl.max()
            if fmax > 1e-12:
                fl_uint8 = (fl / fmax * 255).clip(0, 255).astype(np.uint8)
            else:
                fl_uint8 = np.zeros_like(fl, dtype=np.uint8)
            Image.fromarray(fl_uint8, mode="L").save(sample_dir / "flux_image.png")

            saved_idx += 1

        log.info(
            f"Saved {saved_idx}/{n} samples (passed filter) → {out_split}"
        )

    # Save perturbations.json alongside the dataset.
    pert_json = {
        heliostat_id: {
            "rotation_rad":       chosen_pert["rotation"][0].cpu().tolist(),
            "actuator_angle_rad": chosen_pert["actuator_angle"][0].cpu().tolist(),
            "actuator_stroke_m":  chosen_pert["actuator_stroke"][0].cpu().tolist(),
            "actuator_offset_m":  chosen_pert["actuator_offset"][0].cpu().tolist(),
            "translation_m":      chosen_pert["translation"][0].cpu().tolist(),
            "base_position_m":    chosen_pert["base_position"][0].cpu().tolist(),
        }
    }
    with open(dataset_dir / "perturbations.json", "w") as fh:
        json.dump(pert_json, fh, indent=2)
    log.info(f"Saved perturbations.json → {dataset_dir / 'perturbations.json'}")

    return {
        "succeeded":    True,
        "pert_tensors": chosen_pert,
        "dataset_dir":  dataset_dir,
        "attempt_used": attempt_used,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate perturbed synthetic dataset.")
    parser.add_argument("--heliostat-id", default=None)
    parser.add_argument("--output-dir",   type=pathlib.Path, default=None)
    args = parser.parse_args()

    import config as cfg  # noqa: PLC0415

    heliostat_id = args.heliostat_id or cfg.HELIOSTAT_ID
    output_dir   = (
        args.output_dir
        or cfg.BASE_DIR / "outputs" / f"one_heliostat_demo_dataset_{heliostat_id}"
    )

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)

    n_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
        pathlib.Path(cfg.SCENARIO_PATH_TEMPLATE.format(heliostat_id=heliostat_id))
    )
    device = get_device()
    with setup_distributed_environment(
        number_of_heliostat_groups=n_groups, device=device
    ) as ddp:
        device = ddp[_const.device]
        result = generate(heliostat_id, output_dir, cfg, device)

    print(f"Dataset saved to: {result['dataset_dir']}")
    print(f"Attempt used:     {result['attempt_used'] + 1}")


if __name__ == "__main__":
    main()
