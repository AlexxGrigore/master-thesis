"""
Pre-generate synthetic calibration datasets for the full-63-heliostat
kinematic reconstruction experiment (100 train / 50 val / 50 test per heliostat).

Corrected pipeline vs full_field_200_samples
--------------------------------------------
Perturbations are applied to the scenario BEFORE ray-tracing, so the synthetic
dataset reflects the behaviour of a perturbed (real-world) heliostat field.
The KR then starts from a clean scenario and must recover the perturbations —
the real inverse problem.

Saves files into:
    scenarios/full_63_heli_kin_reconstruct/synthetic_data/{split}/{hid}/{idx:04d}/
        calibration_properties.json
        flux_image.png

Also saves:
    scenarios/full_63_heli_kin_reconstruct/synthetic_data/perturbations.json

Usage
-----
    python generate_dataset.py
    python generate_dataset.py --force
"""
import argparse
import json
import logging
import pathlib
import sys

from tqdm import tqdm

import h5py
import numpy as np
import torch
from PIL import Image
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
from artist.io.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import constants as config_dictionary, set_logger_config
from artist.util import get_device, setup_distributed_environment
from artist.geometry import bitmap_coordinates_to_target_coordinates
from artist.flux import get_center_of_mass

_here = pathlib.Path(__file__).resolve().parent
_src  = _here.parent
sys.path.insert(0, str(_src))

import paint.util.paint_mappings as paint_mappings
from utils.synth_data import (
    _equalize_mapping,
    apply_perturbations,
    perturbations_to_json,
    sample_perturbations,
)
from utils.evaluation import build_heliostat_data_mapping

import config as cfg

log = logging.getLogger(__name__)

GENERATION_CHUNK_SIZE = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_complete(split: str, heliostat_ids: list[str], n_samples: int) -> bool:
    base = cfg.SYNTHETIC_DATA_DIR / split
    for hid in heliostat_ids:
        hel_dir = base / hid
        if not hel_dir.exists():
            return False
        if len(sorted(hel_dir.iterdir())) < n_samples:
            return False
    return True


def _build_mapping(paint_split: str) -> list:
    return build_heliostat_data_mapping(
        benchmark_csv=cfg.BENCHMARK_CSV,
        calibration_properties_dir=cfg.CALIBRATION_PROPERTIES_DIR,
        flux_image_dir=cfg.FLUX_IMAGE_DIR,
        split=paint_split,
    )


# ---------------------------------------------------------------------------
# OOM-safe forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def _forward_pass_chunked(
    scenario,
    heliostat_group,
    incident_rays: torch.Tensor,
    active_mask: torch.Tensor,
    target_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    heliostat_group.activate_heliostats(active_heliostats_mask=active_mask, device=device)
    kinematic = heliostat_group.kinematics

    # Inject base position deviation when present (set by apply_perturbations).
    # active_mask[i] = n_samples for active heliostats, so repeat_interleave
    # expands [N_heliostats, 3] → [N_active_instances, 3].
    if hasattr(kinematic, "_base_position_deviation"):
        base_dev = kinematic._base_position_deviation.repeat_interleave(active_mask, dim=0)
        pad = torch.zeros(base_dev.shape[0], 1, device=device)
        kinematic.active_heliostat_positions = (
            kinematic.active_heliostat_positions + torch.cat([base_dev, pad], dim=1)
        )

    heliostat_group.align_surfaces_with_incident_ray_directions(
        aim_points=scenario.solar_tower.get_centers_of_target_areas(target_mask, device=device),
        incident_ray_directions=incident_rays,
        active_heliostats_mask=active_mask,
        device=device,
    )

    # Capture motor positions computed by the perturbed model during alignment.
    # These are the ground-truth motor positions for Stage-1 AlignmentLoss and
    # are in the same natural (incident_rays) order as incident_rays.
    perturbed_motor_positions = kinematic.active_motor_positions.detach().clone()

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

    bitmap_coords     = get_center_of_mass(bitmaps=flux_sampler, device=device)
    centroids_sampler = bitmap_coordinates_to_target_coordinates(
        bitmap_coordinates=bitmap_coords,
        bitmap_resolution=ray_tracer.bitmap_resolution,
        solar_tower=scenario.solar_tower,
        target_area_indices=target_mask[sample_indices],
        device=device,
    )

    inverse_perm = torch.argsort(sample_indices)
    return centroids_sampler[inverse_perm], flux_sampler[inverse_perm], perturbed_motor_positions


# ---------------------------------------------------------------------------
# Split generation
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
    out_dir = cfg.SYNTHETIC_DATA_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    old_rays = scenario.light_sources.light_source_list[0].number_of_rays
    scenario.set_number_of_rays(n_rays)

    n_chunks    = (len(full_mapping) + GENERATION_CHUNK_SIZE - 1) // GENERATION_CHUNK_SIZE
    total_saved = 0
    n_heliostats = len(full_mapping)

    pbar = tqdm(
        total=n_heliostats * n_samples,
        desc=f"{split:5s}",
        unit="img",
        dynamic_ncols=True,
    )

    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * GENERATION_CHUNK_SIZE
        chunk       = full_mapping[chunk_start : chunk_start + GENERATION_CHUNK_SIZE]

        equalized    = _equalize_mapping(chunk, n_samples)
        active_chunk = [(hid, cal, flux) for hid, cal, flux in equalized if cal]
        if not active_chunk:
            continue

        real_parser = PaintCalibrationDataParser(
            sample_limit=n_samples,
            centroid_extraction_method=cfg.CENTROID_METHOD,
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

        centroids, flux, perturbed_motor_pos = _forward_pass_chunked(
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
                    # Motor positions from the PERTURBED model (not real PAINT values).
                    # Stage-1 AlignmentLoss must match these, not real-measurement positions.
                    "motor_position":         perturbed_motor_pos[i].tolist(),
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

            pbar.update(n)
            offset += n

        total_saved += chunk_saved

        del centroids, flux, perturbed_motor_pos, incident_rays, active_mask, target_mask, motor_pos
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar.close()
    scenario.set_number_of_rays(old_rays)
    log.info(f"  [{split}] total saved: {total_saved} → {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate perturbed synthetic datasets for full-63-heliostat KR experiment."
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--daic",  action="store_true", help="Use DAIC cluster paths.")
    args = parser.parse_args()

    if args.daic:
        cfg.IS_ON_DAIC = True
        cfg.BASE_DIR   = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
        cfg.PAINT_DIR  = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")
        cfg.SCENARIO_PATH              = cfg.BASE_DIR / "scenarios" / "full_field_200_samples_scenario" / "scenario.h5"
        cfg.SYNTHETIC_DATA_DIR         = cfg.BASE_DIR / "scenarios" / "full_63_heli_kin_reconstruct" / "synthetic_data"
        cfg.BENCHMARK_CSV              = cfg.PAINT_DIR / "splits" / f"{cfg.BENCHMARK_NAME}.csv"
        cfg.CALIBRATION_PROPERTIES_DIR = cfg.PAINT_DIR / cfg.BENCHMARK_NAME / "calibration_properties"
        cfg.FLUX_IMAGE_DIR             = cfg.PAINT_DIR / cfg.BENCHMARK_NAME / "flux_image"

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)

    if not cfg.SCENARIO_PATH.exists():
        sys.exit(
            f"Scenario not found: {cfg.SCENARIO_PATH}\n"
            "Run full_field_200_samples/create_scenario.py first."
        )

    device   = get_device()
    n_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(cfg.SCENARIO_PATH)

    with setup_distributed_environment(
        number_of_heliostat_groups=n_groups, device=device
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        # Load scenario once and apply perturbations — all splits share the same
        # perturbed scenario so train/val/test data are consistent.
        with h5py.File(cfg.SCENARIO_PATH, "r") as f:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=f, device=device,
                number_of_surface_points_per_facet=torch.tensor(
                    [cfg.SURFACE_POINTS_PER_FACET, cfg.SURFACE_POINTS_PER_FACET]
                ),
            )
        heliostat_group = scenario.heliostat_field.heliostat_groups[0]
        heliostat_ids   = list(heliostat_group.names)
        scenario_hids   = set(heliostat_ids)
        log.info(f"Scenario heliostats: {len(heliostat_ids)}")

        # Sample perturbations (same seed as training) and apply to scenario.
        perturbations = sample_perturbations(
            n_heliostats=len(heliostat_ids),
            ranges=cfg.PERTURBATION_RANGES,
            seed=cfg.PERTURBATION_SEED,
        )
        apply_perturbations(heliostat_group.kinematics, perturbations, device)
        log.info("Perturbations applied to scenario kinematics.")

        # Persist perturbations alongside the data so main.py can load them for reporting.
        cfg.SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
        pert_json = perturbations_to_json(perturbations, heliostat_ids)
        with open(cfg.SYNTHETIC_DATA_DIR / "perturbations.json", "w") as f:
            json.dump(pert_json, f, indent=2)
        log.info(f"Perturbations saved → {cfg.SYNTHETIC_DATA_DIR / 'perturbations.json'}")

        splits_to_run = []
        for split_name, paint_split, n_samples in [
            ("val",   "validation", cfg.VAL_SAMPLES),
            ("test",  "test",       cfg.TEST_SAMPLES),
            ("train", "train",      cfg.TRAIN_SAMPLES),
        ]:
            if not args.force and _split_complete(split_name, heliostat_ids, n_samples):
                log.info(f"[{split_name}] already complete — skipping (use --force).")
            else:
                splits_to_run.append((split_name, paint_split, n_samples))

        if not splits_to_run:
            log.info("All splits already generated.")
            return

        log.info(f"Splits to generate: {[s for s, _, _ in splits_to_run]}")

        for split_name, paint_split, n_samples in splits_to_run:
            log.info(f"Generating split '{split_name}' ({n_samples} samples/heliostat) …")

            raw_mapping = build_heliostat_data_mapping(
                benchmark_csv=cfg.BENCHMARK_CSV,
                calibration_properties_dir=cfg.CALIBRATION_PROPERTIES_DIR,
                flux_image_dir=cfg.FLUX_IMAGE_DIR,
                split=paint_split,
            )
            mapping = [
                (hid, cal, flux)
                for hid, cal, flux in raw_mapping
                if hid in scenario_hids
            ]
            if not mapping:
                log.warning(f"[{split_name}] No heliostats overlap between benchmark and scenario.")
                continue

            log.info(f"  Heliostats in mapping ∩ scenario: {len(mapping)}")

            _generate_split(
                split=split_name,
                full_mapping=mapping,
                n_samples=n_samples,
                scenario=scenario,
                heliostat_group=heliostat_group,
                n_rays=cfg.SYNTH_GEN_RAYS,
                device=device,
            )

    log.info(f"Done. Dataset root: {cfg.SYNTHETIC_DATA_DIR}")


if __name__ == "__main__":
    main()
