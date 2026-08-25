"""Generate a synthetic dataset with blocking ON, neighbours aimed at the RECEIVER.

Sibling of generate_occlusion_dataset.py (which fixes blockers to a controlled
vertical->horizontal tilt sweep). This instead aims EVERY neighbour in the
scenario normally at a target under each sample's own real sun direction, but
overrides which target: the "receiver" (index resolved from
scenario.solar_tower.target_name_to_index), not the studied heliostat's own
per-sample recorded target -- via aimed_neighbour_surfaces's existing
target_index_override. This produces one occlusion geometry (not a sweep),
matching the real neighbour-pose assumption used for the field-wide
Stage-2-blocking-ON-with-receiver-target training arms.

Per sample: real PAINT sun direction + the STUDIED heliostat's own target/aim
point (pooled from --benchmark, default the 50/20/20 split) -- only the
NEIGHBOURS get the receiver override, the studied heliostat's own aim is
unaffected, matching generate_blocking_dataset.py's convention. Output format
matches generate_blocking_dataset.py exactly (dataset/{split}/{HID}/{idx:04d}/
calibration_properties.json + flux_image.png), so train.py reads it unchanged.

Usage
-----
    python generate_receiver_occlusion_dataset.py BE25 \\
        --benchmark benchmark_split-balanced_train-50_validation-20 \\
        --surface-points 100 --rays 10
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time

import h5py
import numpy as np
import torch
from PIL import Image

_here = pathlib.Path(__file__).resolve().parent          # blocking_study/
_src = _here.parents[1]                                   # src/
for _p in (str(_src), str(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from artist.flux import get_center_of_mass  # noqa: E402
from artist.geometry import bitmap_coordinates_to_target_coordinates  # noqa: E402
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer  # noqa: E402
from artist.scenario.scenario import Scenario  # noqa: E402
from artist.util import get_device, set_logger_config  # noqa: E402

from blocking_utils import (  # noqa: E402
    aimed_neighbour_surfaces,
    exact_blocking,
    one_hot_mask,
)
from utils.evaluation import build_heliostat_data_mapping  # noqa: E402
from utils.synth_data import apply_perturbations, reset_perturbations, sample_perturbations  # noqa: E402

log = logging.getLogger(__name__)

_ROOT = _here.parents[2]                                  # master-thesis/
PAINT_DIR = _ROOT / "datasets" / "paint"
SCENARIO_ROOT = _ROOT / "scenarios" / "neighbourhoods"

_PAINT_SPLITS = {"train": "train", "validation": "val", "test": "test"}
MIN_ACTIVE_PIXEL_PERCENT = 2.0
GENERATE_SEED = 7
RECEIVER_TARGET_NAME = "receiver"

RANDOM_PERT_BOUNDS = {
    "rotation_rad": 0.005,
    "actuator_angle_rad": 0.005,
    "actuator_stroke_m": 0.005,
    "actuator_offset_m": 0.005,
    "translation_m": 0.05,
    "base_position_m": 0.05,
}
PERTURBATION_SEED = 42


def _active_pixel_percent(flux_img: torch.Tensor) -> float:
    return float((flux_img > 0).sum().item()) / float(flux_img.numel()) * 100.0


def _load_pool(heliostat_id: str, benchmark: str, hg, scenario, device):
    from artist.io.paint_calibration_parser import PaintCalibrationDataParser
    parser = PaintCalibrationDataParser()
    pool_rays, pool_targets, boundaries = [], [], {}
    offset = 0
    for paint_split, our_split in _PAINT_SPLITS.items():
        mapping = build_heliostat_data_mapping(
            benchmark_csv=PAINT_DIR / "splits" / f"{benchmark}.csv",
            calibration_properties_dir=PAINT_DIR / benchmark / "calibration_properties",
            flux_image_dir=PAINT_DIR / benchmark / "flux_image",
            split=paint_split,
        )
        mapping = [m for m in mapping if m[0] == heliostat_id]
        if not mapping:
            raise RuntimeError(f"{heliostat_id} not in {benchmark} {paint_split} split")
        n = len(mapping[0][1])
        _, _, rays, _, _, target_mask = parser.parse_data_for_reconstruction(
            heliostat_data_mapping=mapping, heliostat_group=hg, scenario=scenario, device=device,
        )
        pool_rays.append(rays)
        pool_targets.append(target_mask)
        boundaries[our_split] = (offset, offset + n)
        offset += n
        log.info(f"  {our_split}: {n} samples from PAINT {paint_split}")
    return torch.cat(pool_rays), torch.cat(pool_targets), boundaries


def _load_or_sample_perturbation(occlusion_root: pathlib.Path, heliostat_id: str,
                                    n_hel: int, hel_idx: int, device) -> dict:
    pert_file = occlusion_root / "perturbations.json"
    all_pert = json.load(open(pert_file)) if pert_file.exists() else {}
    if heliostat_id not in all_pert:
        sampled = sample_perturbations(n_heliostats=1, ranges=RANDOM_PERT_BOUNDS, seed=PERTURBATION_SEED)
        all_pert[heliostat_id] = {
            "rotation_rad": sampled["rotation"][0].tolist(),
            "actuator_angle_rad": sampled["actuator_angle"][0].tolist(),
            "actuator_stroke_m": sampled["actuator_stroke"][0].tolist(),
            "actuator_offset_m": sampled["actuator_offset"][0].tolist(),
            "translation_m": sampled["translation"][0].tolist(),
            "base_position_m": sampled["base_position"][0].tolist(),
        }
        occlusion_root.mkdir(parents=True, exist_ok=True)
        pert_file.write_text(json.dumps(all_pert, indent=2))
        log.info(f"  sampled new perturbation for {heliostat_id} (seed={PERTURBATION_SEED}) -> {pert_file}")
    else:
        log.info(f"  reusing cached perturbation for {heliostat_id} from {pert_file}")

    src = all_pert[heliostat_id]
    key_map = {
        "rotation_rad": "rotation", "actuator_angle_rad": "actuator_angle",
        "actuator_stroke_m": "actuator_stroke", "actuator_offset_m": "actuator_offset",
        "translation_m": "translation", "base_position_m": "base_position",
    }
    pert = {}
    for src_key, key in key_map.items():
        values = torch.tensor(src[src_key], dtype=torch.float32, device=device)
        full = torch.zeros(n_hel, values.numel(), dtype=torch.float32, device=device)
        full[hel_idx] = values
        pert[key] = full
    return pert


def generate(
    heliostat_id: str,
    benchmark: str,
    occlusion_root: pathlib.Path,
    device: torch.device,
    n_rays: int,
    surface_points_per_facet: int,
    max_samples: int | None,
    paired: bool,
    force: bool,
) -> dict:
    scenario_path = SCENARIO_ROOT / heliostat_id / "scenario.h5"
    if not scenario_path.exists():
        raise FileNotFoundError(
            f"No neighbourhood scenario for {heliostat_id}: {scenario_path}. "
            "Run build_neighbourhood_scenario.py first."
        )
    dataset_root = occlusion_root / "dataset"
    marker = dataset_root / "train" / heliostat_id
    if marker.exists() and any(marker.iterdir()) and not force:
        log.info(f"[SKIP] dataset exists for {heliostat_id} ({marker}) -- use --force")
        return {"heliostat_id": heliostat_id, "skipped": True}

    with h5py.File(scenario_path) as fh:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=fh, device=device,
            number_of_surface_points_per_facet=torch.tensor(
                [surface_points_per_facet, surface_points_per_facet]
            ),
        )
    scenario.set_number_of_rays(n_rays)
    hg = scenario.heliostat_field.heliostat_groups[0]
    kinematic = hg.kinematics
    hel_idx = hg.names.index(heliostat_id)
    n_hel = hg.number_of_heliostats
    receiver_index = int(scenario.solar_tower.target_name_to_index[RECEIVER_TARGET_NAME])
    hel_pos = hg.positions[hel_idx, :3].float()
    tower_ref = scenario.solar_tower.target_areas[0].centers[:, :3].float().mean(dim=0)
    hel_dist_m = float(torch.norm(hel_pos - tower_ref.to(device)).item())
    log.info(
        f"{heliostat_id}: {n_hel} group members {hg.names}, row {hel_idx}, "
        f"neighbours aimed at receiver (index {receiver_index}), dist {hel_dist_m:.0f} m, "
        f"{n_rays} rays, {surface_points_per_facet}x{surface_points_per_facet} pts/facet"
    )

    pool_rays, pool_targets, boundaries = _load_pool(heliostat_id, benchmark, hg, scenario, device)
    n_pool = pool_rays.shape[0]
    if max_samples:
        keep = []
        for our_split, (s, e) in boundaries.items():
            n_take = max(1, int((e - s) * max_samples / n_pool))
            keep.extend(range(s, min(e, s + n_take)))
        pool_rays = pool_rays[keep]
        pool_targets = pool_targets[keep]
        new_b, offset = {}, 0
        for our_split, (s, e) in boundaries.items():
            n_take = max(1, int((e - s) * max_samples / n_pool))
            new_b[our_split] = (offset, offset + n_take)
            offset += n_take
        boundaries = new_b
        n_pool = len(keep)
        log.info(f"  smoke mode: truncated pool to {n_pool} samples")

    pert = _load_or_sample_perturbation(occlusion_root, heliostat_id, n_hel, hel_idx, device)
    snapshot = apply_perturbations(kinematic, pert, device)
    base_pos_delta = kinematic._base_position_deviation.detach().clone()

    index_to_name = {v: k for k, v in scenario.solar_tower.target_name_to_index.items()}

    records: list[dict] = []
    t0 = time.time()
    for i in range(n_pool):
        sun = pool_rays[i : i + 1]
        tgt = pool_targets[i : i + 1]
        tgt_index = int(tgt.item())
        aim = scenario.solar_tower.get_centers_of_target_areas(tgt, device=device)

        # Neighbours (all group members, studied heliostat's own plane excluded
        # from blocking later by self-exclusion) aimed at the RECEIVER, not the
        # studied heliostat's own per-sample target.
        surfaces = aimed_neighbour_surfaces(
            hg, scenario, pool_rays[i], tgt_index, device, target_index_override=receiver_index,
        )

        mask = one_hot_mask(hel_idx, 1, n_hel, device)
        hg.activate_heliostats(active_heliostats_mask=mask, device=device)
        pad = torch.zeros(1, 1, device=device)
        kinematic.active_heliostat_positions = (
            kinematic.active_heliostat_positions
            + torch.cat([base_pos_delta[hel_idx : hel_idx + 1, :3], pad], dim=1)
        )
        hg.align_surfaces_with_incident_ray_directions(
            aim_points=aim, incident_ray_directions=sun, active_heliostats_mask=mask, device=device,
        )
        m_c = kinematic.active_motor_positions.detach().clone()[0]

        def _centroid(flux):
            bc = get_center_of_mass(bitmaps=flux, device=device)
            return bitmap_coordinates_to_target_coordinates(
                bitmap_coordinates=bc, bitmap_resolution=ray_tracer.bitmap_resolution,
                solar_tower=scenario.solar_tower, target_area_indices=tgt, device=device,
            )[0]

        ray_tracer = HeliostatRayTracer(
            scenario=scenario, heliostat_group=hg, blocking_active=False,
            world_size=1, rank=0, batch_size=1, random_seed=GENERATE_SEED,
        )
        ray_tracer.blocking_active = True
        ray_tracer.blocking_heliostat_surfaces_active = surfaces
        with exact_blocking():
            flux_on, _, _, bf = ray_tracer.trace_rays(
                incident_ray_directions=sun, active_heliostats_mask=mask,
                target_area_indices=tgt, device=device,
            )
        c_on = _centroid(flux_on)

        rec = {
            "pool_index": i,
            "split": next(sp for sp, (s, e) in boundaries.items() if s <= i < e),
            "target_name": index_to_name.get(tgt_index, str(tgt_index)),
            "target_area_index": tgt_index,
            "incident_ray_direction": pool_rays[i].cpu().tolist(),
            "blocked_fraction": float(1.0 - bf.item()),
            "active_pixel_percent": _active_pixel_percent(flux_on[0]),
            "saved": False,
        }

        if paired:
            rt_off = HeliostatRayTracer(
                scenario=scenario, heliostat_group=hg, blocking_active=False,
                world_size=1, rank=0, batch_size=1, random_seed=GENERATE_SEED,
            )
            flux_off, _, _, _ = rt_off.trace_rays(
                incident_ray_directions=sun, active_heliostats_mask=mask,
                target_area_indices=tgt, device=device,
            )
            c_off = _centroid(flux_off)
            shift_m = c_on[:3] - c_off[:3]
            rec["centroid_shift_mrad"] = float(torch.norm(shift_m).item() / hel_dist_m * 1000.0)
            rec["flux_lost_fraction"] = float(
                1.0 - flux_on.sum().item() / max(flux_off.sum().item(), 1e-12)
            )

        if rec["active_pixel_percent"] >= MIN_ACTIVE_PIXEL_PERCENT:
            rec["saved"] = True
            rec["calibration"] = {
                "target_area_index": tgt_index,
                "incident_ray_direction": pool_rays[i].cpu().tolist(),
                "focal_spot_enu": c_on.cpu().tolist(),
                "motor_position": m_c.cpu().tolist(),
            }
            rec["_flux"] = flux_on[0].cpu().float().numpy()
        records.append(rec)

        if (i + 1) % 25 == 0 or i + 1 == n_pool:
            rate = (time.time() - t0) / (i + 1)
            log.info(
                f"  [{heliostat_id}] {i + 1}/{n_pool}  "
                f"blocked={rec['blocked_fraction'] * 100:.1f}%  "
                f"({rate:.1f} s/sample, ETA {rate * (n_pool - i - 1) / 60:.0f} min)"
            )

    reset_perturbations(kinematic, snapshot)

    n_saved = 0
    per_split_saved: dict[str, int] = {}
    for our_split, (s, e) in boundaries.items():
        out_split = dataset_root / our_split / heliostat_id
        out_split.mkdir(parents=True, exist_ok=True)
        idx = 0
        for rec in records[s:e]:
            if not rec["saved"]:
                continue
            sample_dir = out_split / f"{idx:04d}"
            sample_dir.mkdir(exist_ok=True)
            with open(sample_dir / "calibration_properties.json", "w") as fh:
                json.dump(rec["calibration"], fh, indent=2)
            fl = rec.pop("_flux")
            fmax = fl.max()
            fl8 = (fl / fmax * 255).clip(0, 255).astype(np.uint8) if fmax > 1e-12 \
                else np.zeros_like(fl, dtype=np.uint8)
            Image.fromarray(fl8, mode="L").save(sample_dir / "flux_image.png")
            idx += 1
        per_split_saved[our_split] = idx
        n_saved += idx
        log.info(f"  saved {idx}/{e - s} {our_split} samples -> {out_split}")

    shared_pert_file = occlusion_root / "perturbations.json"
    if shared_pert_file.exists():
        (dataset_root / "perturbations.json").write_text(shared_pert_file.read_text())

    blocked = torch.tensor([r["blocked_fraction"] for r in records])
    summary = {
        "heliostat_id": heliostat_id,
        "neighbour_pose": "aimed at receiver",
        "receiver_target_index": receiver_index,
        "scenario": str(scenario_path),
        "benchmark": benchmark,
        "group_members": hg.names,
        "hel_dist_m": hel_dist_m,
        "n_pool": n_pool,
        "n_saved": n_saved,
        "per_split_saved": per_split_saved,
        "n_rays": n_rays,
        "surface_points_per_facet": surface_points_per_facet,
        "paired": paired,
        "blocked_fraction": {
            "mean": float(blocked.mean()), "median": float(blocked.median()), "max": float(blocked.max()),
        },
        "minutes": (time.time() - t0) / 60.0,
    }
    if paired:
        shifts = torch.tensor([r.get("centroid_shift_mrad", float("nan")) for r in records])
        lost = torch.tensor([r.get("flux_lost_fraction", float("nan")) for r in records])

        def _nanmax(t: torch.Tensor) -> float:
            finite = t[~torch.isnan(t)]
            return float(finite.max()) if finite.numel() else float("nan")

        summary["centroid_shift_mrad"] = {"mean": float(shifts.nanmean()), "max": _nanmax(shifts)}
        summary["flux_lost_fraction"] = {"mean": float(lost.nanmean()), "max": _nanmax(lost)}
    report_path = occlusion_root / "generation_report.json"
    report_path.write_text(json.dumps(summary, indent=2))
    log.info(f"  report -> {report_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("heliostat_id")
    parser.add_argument("--benchmark", default="benchmark_split-balanced_train-50_validation-20")
    parser.add_argument("--occlusion-root", type=pathlib.Path, default=None,
                        help="default: datasets/synthetic/<heliostat_id>_receiver_occlusion/")
    parser.add_argument("--rays", type=int, default=10)
    parser.add_argument("--surface-points", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-paired", dest="paired", action="store_false")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("artist", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    occlusion_root = args.occlusion_root or (
        _ROOT / "datasets" / "synthetic" / f"{args.heliostat_id.lower()}_receiver_occlusion"
    )
    device = get_device()
    generate(
        heliostat_id=args.heliostat_id, benchmark=args.benchmark, occlusion_root=occlusion_root,
        device=device, n_rays=args.rays, surface_points_per_facet=args.surface_points,
        max_samples=args.max_samples, paired=args.paired, force=args.force,
    )


if __name__ == "__main__":
    main()
