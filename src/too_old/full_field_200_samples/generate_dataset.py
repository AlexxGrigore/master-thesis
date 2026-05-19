"""
Pre-generate synthetic calibration datasets for the full-field 200-samples
experiment (100 train / 50 val / 50 test per heliostat).

Must be run after create_scenario.py. Saves files into:

    scenarios/full_field_200_samples_scenario/synthetic_data/{split}/{hid}/{idx:04d}/
        calibration_properties.json
        flux_image.png

OOM avoidance
-------------
Two complementary mitigations keep memory bounded regardless of field size:

  1. Heliostat chunking (GENERATION_CHUNK_SIZE): the mapping is processed
     in groups of 10 heliostats so only one chunk's PAINT files are in
     memory at a time.

  2. Fixed ray-tracer batch_size=8: regardless of how many instances are
     active in a chunk the ray tracer processes only 8 at a time, keeping
     peak GPU memory constant.

Usage
-----
    python generate_dataset.py
    python generate_dataset.py --force   # overwrite existing files
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
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment
from artist.util.utils import bitmap_coordinates_to_target_coordinates, get_center_of_mass

_here = pathlib.Path(__file__).resolve().parent
_src  = _here.parent
sys.path.insert(0, str(_src))

import paint.util.paint_mappings as paint_mappings
from five_heliostats_synth.data import _equalize_mapping
from utils.evaluation import build_heliostat_data_mapping

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR  = pathlib.Path(__file__).resolve().parents[2]
PAINT_DIR = BASE_DIR / "datasets" / "paint"

SCENARIO_PATH  = BASE_DIR / "scenarios" / "full_field_200_samples_scenario" / "scenario.h5"
BENCHMARK_NAME = "benchmark_split-balanced_train-100_validation-50_deflectometry"
BENCHMARK_CSV              = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
CALIBRATION_PROPERTIES_DIR = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
FLUX_IMAGE_DIR             = PAINT_DIR / BENCHMARK_NAME / "flux_image"

CENTROID_METHOD = paint_mappings.UTIS_KEY

TRAIN_SAMPLES = 100
VAL_SAMPLES   = 50
TEST_SAMPLES  = 50

SURFACE_POINTS_PER_FACET = 25   # 25×25 = 625 pts/facet
SYNTH_GEN_RAYS           = 100  # high ray count → clean centroids

GENERATION_CHUNK_SIZE = 10      # heliostats per forward-pass chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_data_dir() -> pathlib.Path:
    return SCENARIO_PATH.parent / "synthetic_data"


def _split_complete(split: str, heliostat_ids: list[str], n_samples: int) -> bool:
    base = _synthetic_data_dir() / split
    for hid in heliostat_ids:
        hel_dir = base / hid
        if not hel_dir.exists():
            return False
        if len(sorted(hel_dir.iterdir())) < n_samples:
            return False
    return True


def _build_mapping(paint_split: str) -> list:
    return build_heliostat_data_mapping(
        benchmark_csv=BENCHMARK_CSV,
        calibration_properties_dir=CALIBRATION_PROPERTIES_DIR,
        flux_image_dir=FLUX_IMAGE_DIR,
        split=paint_split,
    )


# ---------------------------------------------------------------------------
# OOM-safe forward pass: fixed batch_size=8
# ---------------------------------------------------------------------------

@torch.no_grad()
def _forward_pass_chunked(
    scenario,
    heliostat_group,
    incident_rays: torch.Tensor,
    active_mask: torch.Tensor,
    target_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    heliostat_group.activate_heliostats(active_heliostats_mask=active_mask, device=device)

    kinematic   = heliostat_group.kinematics
    n_instances = int(active_mask.sum().item())
    pad_pos     = torch.zeros(n_instances, 4, device=device)
    kinematic.active_heliostat_positions = kinematic.active_heliostat_positions + pad_pos

    heliostat_group.align_surfaces_with_incident_ray_directions(
        aim_points=scenario.solar_tower.get_centers_of_target_areas(target_mask, device=device),
        incident_ray_directions=incident_rays,
        active_heliostats_mask=active_mask,
        device=device,
    )

    ray_tracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=heliostat_group,
        blocking_active=False,
        world_size=1,
        rank=0,
        batch_size=8,
        random_seed=42,
    )
    flux_sampler, _, _, _ = ray_tracer.trace_rays(
        incident_ray_directions=incident_rays,
        active_heliostats_mask=active_mask,
        target_area_indices=target_mask,
        device=device,
    )
    sample_indices = ray_tracer.get_sampler_indices()

    bitmap_coords       = get_center_of_mass(bitmaps=flux_sampler, device=device)
    centroids_sampler   = bitmap_coordinates_to_target_coordinates(
        bitmap_coordinates=bitmap_coords,
        bitmap_resolution=ray_tracer.bitmap_resolution,
        solar_tower=scenario.solar_tower,
        target_area_indices=target_mask[sample_indices],
        device=device,
    )

    inverse_perm = torch.argsort(sample_indices)
    return centroids_sampler[inverse_perm], flux_sampler[inverse_perm]


# ---------------------------------------------------------------------------
# Split generation (chunked)
# ---------------------------------------------------------------------------

def _generate_split(
    split: str,
    full_mapping: list,
    n_samples: int,
    scenario,
    heliostat_group,
    n_rays: int,
    device: torch.device,
) -> None:
    out_dir = _synthetic_data_dir() / split
    out_dir.mkdir(parents=True, exist_ok=True)

    old_rays = scenario.light_sources.light_source_list[0].number_of_rays
    scenario.set_number_of_rays(n_rays)

    n_chunks    = (len(full_mapping) + GENERATION_CHUNK_SIZE - 1) // GENERATION_CHUNK_SIZE
    total_saved = 0

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * GENERATION_CHUNK_SIZE
        chunk       = full_mapping[chunk_start : chunk_start + GENERATION_CHUNK_SIZE]

        equalized   = _equalize_mapping(chunk, n_samples)
        active_chunk = [(hid, cal, flux) for hid, cal, flux in equalized if cal]
        if not active_chunk:
            continue

        real_parser = PaintCalibrationDataParser(
            sample_limit=n_samples,
            centroid_extraction_method=CENTROID_METHOD,
        )
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
            continue

        centroids, flux = _forward_pass_chunked(
            scenario, heliostat_group, incident_rays, active_mask, target_mask, device,
        )

        active_indices = torch.where(active_mask.bool())[0]
        samples_per    = active_mask[active_indices].long()

        offset      = 0
        chunk_saved = 0
        for j, group_idx in enumerate(active_indices):
            n   = samples_per[j].item()
            hid = heliostat_group.names[group_idx.item()]
            hel_dir = out_dir / hid

            for k in range(n):
                i        = offset + k
                meas_dir = hel_dir / f"{k:04d}"
                meas_dir.mkdir(parents=True, exist_ok=True)

                cal = {
                    "target_area_index":      int(target_mask[i].item()),
                    "incident_ray_direction": incident_rays[i].tolist(),
                    "focal_spot_enu":         centroids[i].tolist(),
                    "motor_position":         motor_pos[i].tolist(),
                }
                with open(meas_dir / "calibration_properties.json", "w") as fh:
                    json.dump(cal, fh, indent=2)

                flux_i    = flux[i]
                fmax      = flux_i.max().item()
                if fmax > 1e-12:
                    flux_uint8 = (flux_i / fmax * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                else:
                    flux_uint8 = np.zeros(flux_i.shape, dtype=np.uint8)
                Image.fromarray(flux_uint8, mode="L").save(meas_dir / "flux_image.png")
                chunk_saved += 1

            offset += n

        total_saved += chunk_saved
        chunk_hids = [hid for hid, _, _ in active_chunk]
        log.info(
            f"  [{split}] chunk {chunk_idx + 1}/{n_chunks}: "
            f"{chunk_saved} measurements for {chunk_hids}"
        )

        del centroids, flux, incident_rays, active_mask, target_mask, motor_pos
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scenario.set_number_of_rays(old_rays)
    log.info(f"  [{split}] total saved: {total_saved} → {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets for full-field 200-samples experiment."
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)

    if not SCENARIO_PATH.exists():
        sys.exit(
            f"Scenario not found: {SCENARIO_PATH}\n"
            "Run create_scenario.py first."
        )

    device   = get_device()
    n_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(SCENARIO_PATH)

    with setup_distributed_environment(
        number_of_heliostat_groups=n_groups, device=device
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        with h5py.File(SCENARIO_PATH, "r") as f:
            tmp_scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=f, device=device,
                number_of_surface_points_per_facet=torch.tensor(
                    [SURFACE_POINTS_PER_FACET, SURFACE_POINTS_PER_FACET]
                ),
            )
        scenario_hids = set(tmp_scenario.heliostat_field.heliostat_groups[0].names)
        del tmp_scenario
        log.info(f"Scenario heliostats: {len(scenario_hids)}")

        splits_to_run = []
        for split_name, paint_split, n_samples in [
            ("val",   "validation", VAL_SAMPLES),
            ("test",  "test",       TEST_SAMPLES),
            ("train", "train",      TRAIN_SAMPLES),
        ]:
            if not args.force and _split_complete(split_name, list(scenario_hids), n_samples):
                log.info(f"[{split_name}] already complete — skipping (use --force).")
            else:
                splits_to_run.append((split_name, paint_split, n_samples))

        if not splits_to_run:
            log.info("All splits already generated.")
            return

        log.info(f"Splits to generate: {[s for s, _, _ in splits_to_run]}")

        for split_name, paint_split, n_samples in splits_to_run:
            log.info(f"Generating split '{split_name}' ({n_samples} samples/heliostat) …")

            raw_mapping = _build_mapping(paint_split)
            mapping = [
                (hid, cal, flux)
                for hid, cal, flux in raw_mapping
                if hid in scenario_hids
            ]
            if not mapping:
                log.warning(
                    f"[{split_name}] No heliostats overlap between benchmark and scenario."
                )
                continue

            log.info(f"  Heliostats in mapping ∩ scenario: {len(mapping)}")

            with h5py.File(SCENARIO_PATH, "r") as f:
                scenario = Scenario.load_scenario_from_hdf5(
                    scenario_file=f, device=device,
                    number_of_surface_points_per_facet=torch.tensor(
                        [SURFACE_POINTS_PER_FACET, SURFACE_POINTS_PER_FACET]
                    ),
                )
            heliostat_group = scenario.heliostat_field.heliostat_groups[0]

            _generate_split(
                split=split_name,
                full_mapping=mapping,
                n_samples=n_samples,
                scenario=scenario,
                heliostat_group=heliostat_group,
                n_rays=SYNTH_GEN_RAYS,
                device=device,
            )

    log.info(f"Done. Dataset root: {_synthetic_data_dir()}")


if __name__ == "__main__":
    main()
