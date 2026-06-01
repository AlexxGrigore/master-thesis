"""
Generate a perturbed synthetic dataset for a single heliostat.

For each split (train / val / test) this script:
  1. Loads ray directions from the PAINT benchmark calibration dataset.
  2. Samples random kinematic perturbations (or uses CUSTOM_PERTURBATIONS_SPEC).
  3. Ray-traces under the perturbed kinematics to produce synthetic flux and centroids.
  4. Saves samples to output_dir/dataset/{split}/{HELIOSTAT_ID}/{idx:04d}/:
       calibration_properties.json  (incident_ray_direction, focal_spot_enu, motor_position, target_area_index)
       flux_image.png               (uint8 peak-normalised flux bitmap)
  5. Saves output_dir/dataset/perturbations.json with GT perturbation values.

Resampling logic
----------------
If, after applying the perturbations, fewer than MIN_{TRAIN,VAL,TEST}_SAMPLES pass
the active-pixel filter (MIN_ACTIVE_PIXEL_PERCENT), the perturbations are resampled
with seed = RANDOM_SEED + attempt (up to MAX_RESAMPLE_ATTEMPTS tries).

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
    output_dir  = pathlib.Path(output_dir)
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
        "val":   getattr(cfg, "MIN_VAL_SAMPLES",        20),
        "test":  getattr(cfg, "MIN_TEST_SAMPLES",       20),
    }
    min_active_pct = getattr(cfg, "MIN_ACTIVE_PIXEL_PERCENT", 2.0)
    max_attempts   = getattr(cfg, "MAX_RESAMPLE_ATTEMPTS",    10)

    # Pre-load ray directions from the PAINT benchmark dataset.
    # Only incident_ray_direction, active_mask, and target_mask are used;
    # flux and centroids will be re-generated from the perturbed kinematics.
    old_n_rays = scenario.light_sources.light_source_list[0].number_of_rays
    scenario.set_number_of_rays(cfg.GENERATE_RAYS)

    # PAINT splits use "validation"; our internal names use "val".
    _PAINT_SPLITS = {"train": "train", "validation": "val", "test": "test"}

    paint_parser = PaintCalibrationDataParser()
    splits_info: list[tuple[str, int]] = []
    split_rays: dict[str, tuple] = {}
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
        _, _, rays, _, active_mask, target_mask = paint_parser.parse_data_for_reconstruction(
            heliostat_data_mapping=hel_mapping,
            heliostat_group=heliostat_group,
            scenario=scenario,
            device=device,
        )
        splits_info.append((our_split, n_samples))
        split_rays[our_split] = (rays, active_mask, target_mask, n_samples)
        log.info(f"  Loaded {n_samples} ray directions for {our_split} from PAINT {paint_split}")

    if not splits_info:
        raise RuntimeError(
            f"No PAINT data found for {heliostat_id} in {cfg.BENCHMARK_CSV}"
        )

    # Resample loop: try different seeds until filtered counts meet minimums.
    chosen_pert   = None
    chosen_results = None
    attempt_used  = 0

    for attempt in range(max_attempts):
        seed = cfg.RANDOM_SEED + seed_offset * 20 + attempt

        if cfg.DATA_MODE == "random_synthetic":
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

        split_results: dict[str, dict] = {}
        for split, (rays, active_mask, target_mask, n_samples) in split_rays.items():
            centroids, flux = _forward_pass(
                scenario, heliostat_group,
                rays, active_mask, target_mask, pert_bpd, device,
            )
            motor_pos = heliostat_group.kinematics.active_motor_positions.detach().clone()

            n_ok = sum(
                1 for i in range(n_samples)
                if _active_pixel_percent(flux[i]) >= min_active_pct
            )
            split_results[split] = {
                "rays": rays, "active_mask": active_mask, "target_mask": target_mask,
                "centroids": centroids, "flux": flux, "motor_pos": motor_pos,
                "n_samples": n_samples, "n_ok": n_ok,
            }

        reset_perturbations(kinematic, snap)

        counts = {sp: split_results[sp]["n_ok"] for sp in split_results}
        ok     = all(
            counts.get(sp, 0) >= min_per_split.get(sp, 0)
            for sp, _ in splits_info
        )
        status = "[OK]" if ok else "[FAIL]"
        log.info(
            f"Attempt {attempt + 1}/{max_attempts}: "
            + "  ".join(f"{sp}={counts.get(sp, 0)}/{min_per_split.get(sp, 0)}" for sp, _ in splits_info)
            + f"  {status}"
        )

        chosen_pert    = pert_tensors
        chosen_results = split_results
        attempt_used   = attempt

        if ok:
            break
        if cfg.DATA_MODE != "random_synthetic":
            log.warning("DATA_MODE='synthetic' — cannot resample. Proceeding as-is.")
            break
    else:
        log.warning(
            f"Could not meet sample minimums after {max_attempts} attempts. "
            "Using last attempt."
        )

    scenario.set_number_of_rays(old_n_rays)

    # Save dataset to disk.
    dataset_dir = output_dir / "dataset"

    for split, sdata in chosen_results.items():
        out_split = dataset_dir / split / heliostat_id
        out_split.mkdir(parents=True, exist_ok=True)

        n           = sdata["n_samples"]
        rays        = sdata["rays"]
        centroids   = sdata["centroids"]
        motor_pos   = sdata["motor_pos"]
        target_mask = sdata["target_mask"]
        flux        = sdata["flux"]

        for i in range(n):
            sample_dir = out_split / f"{i:04d}"
            sample_dir.mkdir(exist_ok=True)

            cal = {
                "target_area_index":      int(target_mask[i].item()),
                "incident_ray_direction": rays[i].cpu().tolist(),
                "focal_spot_enu":         centroids[i].cpu().tolist(),
                "motor_position":         motor_pos[i].cpu().tolist(),
            }
            with open(sample_dir / "calibration_properties.json", "w") as fh:
                json.dump(cal, fh, indent=2)

            fl    = flux[i].cpu().float().numpy()
            fmax  = fl.max()
            if fmax > 1e-12:
                fl_uint8 = (fl / fmax * 255).clip(0, 255).astype(np.uint8)
            else:
                fl_uint8 = np.zeros_like(fl, dtype=np.uint8)
            Image.fromarray(fl_uint8, mode="L").save(sample_dir / "flux_image.png")

        log.info(
            f"Saved {n} samples ({sdata['n_ok']} pass filter) → {out_split}"
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
