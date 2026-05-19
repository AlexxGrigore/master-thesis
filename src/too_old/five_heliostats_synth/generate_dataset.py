"""
Pre-generate all synthetic calibration datasets for the 5-heliostat experiment.

Run this once before main.py. It saves files into:

    scenarios/five_heliostats_scenario/synthetic_data/{split}/{heliostat_id}/{idx:04d}/
        calibration_properties.json   — pre-computed geometry (no WGS84 needed)
        flux_image.png                — ray-traced flux bitmap (uint8, normalised)

Three splits are generated:
  train/  — TRAIN_SAMPLES (50) measurements per heliostat
  val/    — VAL_SAMPLES   (10) measurements per heliostat
  test/   — TEST_SAMPLES  (10) measurements per heliostat

At training time the SyntheticDatasetParser reads from these folders; the
heliostat_data_mapping passed to parse_data_for_reconstruction controls how
many measurements are loaded (e.g. n_train=10 loads only the first 10 per heliostat).

Usage
-----
    python generate_dataset.py
    python generate_dataset.py --force    # overwrite existing files
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
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment

_here = pathlib.Path(__file__).resolve().parent
_src  = _here.parent
sys.path.insert(0, str(_src))

import config as cfg
from data import _equalize_mapping, _forward_pass
from utils.evaluation import build_heliostat_data_mapping

log = logging.getLogger(__name__)


def synthetic_data_dir() -> pathlib.Path:
    return cfg.SCENARIO_PATH.parent / "synthetic_data"


def split_dir(split: str) -> pathlib.Path:
    return synthetic_data_dir() / split


def _split_complete(split: str, heliostat_ids: list, n_samples: int) -> bool:
    """Return True if every heliostat has at least n_samples measurement folders."""
    base = split_dir(split)
    for hid in heliostat_ids:
        hel_dir = base / hid
        if not hel_dir.exists():
            return False
        existing = sorted(hel_dir.iterdir())
        if len(existing) < n_samples:
            return False
    return True


def _build_mapping(split: str) -> list:
    return build_heliostat_data_mapping(
        benchmark_csv=cfg.BENCHMARK_CSV,
        calibration_properties_dir=cfg.CALIBRATION_PROPERTIES_DIR,
        flux_image_dir=cfg.FLUX_IMAGE_DIR,
        split=split,
    )


def _filter(mapping: list) -> list:
    ids = set(cfg.HELIOSTAT_IDS)
    filtered = [(hid, cal, flux) for hid, cal, flux in mapping if hid in ids]
    if not filtered:
        raise RuntimeError(f"None of {cfg.HELIOSTAT_IDS} found in mapping.")
    return filtered


def _generate_split(
    split: str,
    real_mapping: list,
    n_samples: int,
    scenario,
    heliostat_group,
    n_rays: int,
    device: torch.device,
) -> None:
    """
    Generate synthetic calibration files for one split.

    For each active heliostat (in heliostat-group order), saves n_samples
    subdirectories under split_dir(split)/{heliostat_id}/.
    """
    out_dir = split_dir(split)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_parser = PaintCalibrationDataParser(
        sample_limit=n_samples,
        centroid_extraction_method=cfg.CENTROID_METHOD,
    )
    equalized = _equalize_mapping(_filter(real_mapping), n_samples)

    with torch.no_grad():
        _, _, incident_rays, motor_pos, active_mask, target_mask = (
            real_parser.parse_data_for_reconstruction(
                heliostat_data_mapping=equalized,
                heliostat_group=heliostat_group,
                scenario=scenario,
                device=device,
            )
        )

    if active_mask.sum() == 0:
        raise RuntimeError(f"No active heliostats found for split '{split}'.")

    old_rays = scenario.light_sources.light_source_list[0].number_of_rays
    scenario.set_number_of_rays(n_rays)

    n_hels = active_mask.shape[0]
    zero_base_pos = torch.zeros(n_hels, 3, device=device)

    centroids, flux = _forward_pass(
        scenario, heliostat_group, incident_rays, active_mask, target_mask,
        zero_base_pos, device,
    )

    scenario.set_number_of_rays(old_rays)

    # Save per-measurement files, ordered by heliostat-group index.
    active_indices = torch.where(active_mask.bool())[0]
    samples_per    = active_mask[active_indices].long()

    offset = 0
    total_saved = 0
    for j, group_idx in enumerate(active_indices):
        n = samples_per[j].item()
        hid = heliostat_group.names[group_idx.item()]
        hel_dir = out_dir / hid

        for k in range(n):
            i = offset + k
            meas_dir = hel_dir / f"{k:04d}"
            meas_dir.mkdir(parents=True, exist_ok=True)

            cal = {
                "target_area_index":    int(target_mask[i].item()),
                "incident_ray_direction": incident_rays[i].tolist(),
                "focal_spot_enu":       centroids[i].tolist(),
                "motor_position":       motor_pos[i].tolist(),
            }
            with open(meas_dir / "calibration_properties.json", "w") as fh:
                json.dump(cal, fh, indent=2)

            flux_i = flux[i]  # [H, W]
            fmax = flux_i.max().item()
            if fmax > 1e-12:
                flux_uint8 = (flux_i / fmax * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            else:
                flux_uint8 = np.zeros(flux_i.shape, dtype=np.uint8)
            Image.fromarray(flux_uint8, mode="L").save(meas_dir / "flux_image.png")
            total_saved += 1

        offset += n

    log.info(
        f"  [{split}] saved {total_saved} measurements "
        f"({len(active_indices)} heliostats × {samples_per[0].item()} samples) → {out_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic calibration datasets.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)

    splits_needed: list[tuple[str, int]] = []
    for split, n_samples in [
        ("val",   cfg.VAL_SAMPLES),
        ("test",  cfg.TEST_SAMPLES),
        ("train", cfg.TRAIN_SAMPLES),
    ]:
        if args.force or not _split_complete(split, cfg.HELIOSTAT_IDS, n_samples):
            splits_needed.append((split, n_samples))
        else:
            log.info(f"[{split}] already complete — skipping (use --force to regenerate).")

    if not splits_needed:
        log.info("All splits already generated.")
        return

    log.info(f"Splits to generate: {[s for s, _ in splits_needed]}")

    device = get_device()
    log.info(f"Device: {device}")

    log.info("Building data mappings …")
    raw_mappings = {
        "val":   _build_mapping("validation"),
        "test":  _build_mapping("test"),
        "train": _build_mapping("train"),
    }

    n_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(cfg.SCENARIO_PATH)

    with setup_distributed_environment(
        number_of_heliostat_groups=n_groups, device=device
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        for split, n_samples in splits_needed:
            log.info(f"Generating split '{split}' ({n_samples} samples/heliostat) …")

            # Reload a fresh scenario for each split to avoid activation state leakage.
            with h5py.File(cfg.SCENARIO_PATH, "r") as f:
                scenario = Scenario.load_scenario_from_hdf5(
                    scenario_file=f, device=device,
                    number_of_surface_points_per_facet=torch.tensor(
                        [cfg.SURFACE_POINTS_PER_FACET, cfg.SURFACE_POINTS_PER_FACET]
                    ),
                )
            heliostat_group = scenario.heliostat_field.heliostat_groups[0]

            _generate_split(
                split=split,
                real_mapping=raw_mappings[split],
                n_samples=n_samples,
                scenario=scenario,
                heliostat_group=heliostat_group,
                n_rays=cfg.SYNTH_GEN_RAYS,
                device=device,
            )

    log.info(f"Done. Dataset root: {synthetic_data_dir()}")


if __name__ == "__main__":
    main()
