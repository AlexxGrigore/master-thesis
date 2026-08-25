"""Generate a blocking-aware synthetic dataset for the Experiment-S training arms.

For each studied heliostat:
  1. Load its neighbourhood scenario (deflectometry surface for the studied
     heliostat, ideal blockers — built by build_neighbourhood_scenario.py).
  2. Reuse the perturbation already sampled in the balanced synthetic dataset
     (datasets/synthetic/balanced_dataset/dataset/perturbations.json), applied
     to the STUDIED heliostat only; neighbours stay nominal.
  3. For every real PAINT calibration sample of the 100-50 benchmark (sun
     direction + aim target from the real data): aim the perturbed heliostat
     at its target centre (producing the recorded motors m_c), aim every
     neighbour at the SAME target, and ray-trace with blocking ON.
  4. A paired trace with blocking OFF (identical ray seed) records the bias
     blocking injects — the quantity arm A0 is forced to absorb.
  5. Save in the exact format of generate_dataset.py, so the training pipeline
     reads it unchanged: dataset/{split}/{HID}/{idx:04d}/calibration_properties.json
     + flux_image.png, plus dataset/perturbations.json.

The per-sample report (blocked fraction, paired centroid shift, flux lost) goes
to outputs/new_mapping_function/blocking_study/experiment_s/{HID}/generation_report.json.

Usage
-----
    python generate_blocking_dataset.py AY36 BA35 BE35 AA27
    python generate_blocking_dataset.py AY36 --max-samples 8 --rays 10   # smoke
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

from artist.io.paint_calibration_parser import PaintCalibrationDataParser  # noqa: E402
from artist.scenario.scenario import Scenario  # noqa: E402
from artist.util import get_device, set_logger_config  # noqa: E402

from blocking_utils import (  # noqa: E402
    aimed_neighbour_surfaces,
    exact_blocking,
    one_hot_mask,
)
from utils.evaluation import build_heliostat_data_mapping  # noqa: E402
from utils.synth_data import apply_perturbations, reset_perturbations  # noqa: E402

log = logging.getLogger(__name__)

_ROOT = _here.parents[2]                                  # master-thesis/
PAINT_DIR = _ROOT / "datasets" / "paint"
BENCHMARK = "benchmark_split-balanced_train-100_validation-50_deflectometry"
SOURCE_PERTURBATIONS = (
    _ROOT / "datasets" / "synthetic" / "balanced_dataset" / "dataset" / "perturbations.json"
)
DATASET_ROOT = _ROOT / "datasets" / "synthetic" / "blocking_dataset" / "dataset"
REPORT_ROOT = (
    _ROOT / "outputs" / "new_mapping_function" / "blocking_study" / "experiment_s"
)
SCENARIO_ROOT = _ROOT / "scenarios" / "neighbourhoods"

_PAINT_SPLITS = {"train": "train", "validation": "val", "test": "test"}
_PERT_KEYS = {  # balanced_dataset key -> apply_perturbations key
    "rotation_rad": "rotation",
    "actuator_angle_rad": "actuator_angle",
    "actuator_stroke_m": "actuator_stroke",
    "actuator_offset_m": "actuator_offset",
    "translation_m": "translation",
    "base_position_m": "base_position",
}
MIN_ACTIVE_PIXEL_PERCENT = 2.0
SURFACE_POINTS_PER_FACET = 25
GENERATE_SEED = 7  # fixed: the paired off/on traces share identical rays


def _active_pixel_percent(flux_img: torch.Tensor) -> float:
    return float((flux_img > 0).sum().item()) / float(flux_img.numel()) * 100.0


def _load_pool(heliostat_id: str, hg, scenario, device):
    """Real sun directions + aim targets for one heliostat, all PAINT splits pooled."""
    parser = PaintCalibrationDataParser()
    pool_rays, pool_targets, boundaries = [], [], {}
    offset = 0
    for paint_split, our_split in _PAINT_SPLITS.items():
        mapping = build_heliostat_data_mapping(
            benchmark_csv=PAINT_DIR / "splits" / f"{BENCHMARK}.csv",
            calibration_properties_dir=PAINT_DIR / BENCHMARK / "calibration_properties",
            flux_image_dir=PAINT_DIR / BENCHMARK / "flux_image",
            split=paint_split,
        )
        mapping = [m for m in mapping if m[0] == heliostat_id]
        if not mapping:
            raise RuntimeError(f"{heliostat_id} not in {BENCHMARK} {paint_split} split")
        n = len(mapping[0][1])
        _, _, rays, _, _, target_mask = parser.parse_data_for_reconstruction(
            heliostat_data_mapping=mapping,
            heliostat_group=hg,
            scenario=scenario,
            device=device,
        )
        pool_rays.append(rays)
        pool_targets.append(target_mask)
        boundaries[our_split] = (offset, offset + n)
        offset += n
        log.info(f"  {our_split}: {n} samples from PAINT {paint_split}")
    return torch.cat(pool_rays), torch.cat(pool_targets), boundaries


def _load_perturbation(heliostat_id: str, n_hel: int, hel_idx: int, device) -> dict:
    """Existing balanced-dataset perturbation, embedded in a full-group zero dict."""
    with open(SOURCE_PERTURBATIONS) as fh:
        src = json.load(fh)[heliostat_id]
    pert = {}
    for src_key, key in _PERT_KEYS.items():
        values = torch.tensor(src[src_key], dtype=torch.float32, device=device)
        full = torch.zeros(n_hel, values.numel(), dtype=torch.float32, device=device)
        full[hel_idx] = values
        pert[key] = full
    return pert


def generate(
    heliostat_id: str,
    device: torch.device,
    n_rays: int = 100,
    max_samples: int | None = None,
    paired: bool = True,
    force: bool = False,
) -> dict:
    scenario_path = SCENARIO_ROOT / heliostat_id / "scenario.h5"
    if not scenario_path.exists():
        raise FileNotFoundError(
            f"No neighbourhood scenario for {heliostat_id}: {scenario_path}. "
            "Run build_neighbourhood_scenario.py first."
        )
    out_hel = DATASET_ROOT
    marker = out_hel / "train" / heliostat_id
    if marker.exists() and any(marker.iterdir()) and not force:
        log.info(f"[SKIP] dataset exists for {heliostat_id} ({marker}) — use --force")
        return {"heliostat_id": heliostat_id, "skipped": True}

    with h5py.File(scenario_path) as fh:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=fh,
            device=device,
            number_of_surface_points_per_facet=torch.tensor(
                [SURFACE_POINTS_PER_FACET, SURFACE_POINTS_PER_FACET]
            ),
        )
    scenario.set_number_of_rays(n_rays)
    hg = scenario.heliostat_field.heliostat_groups[0]
    kinematic = hg.kinematics
    hel_idx = hg.names.index(heliostat_id)
    n_hel = hg.number_of_heliostats
    hel_pos = hg.positions[hel_idx, :3].float()
    tower_ref = scenario.solar_tower.target_areas[0].centers[:, :3].float().mean(dim=0)
    hel_dist_m = float(torch.norm(hel_pos - tower_ref.to(device)).item())
    log.info(
        f"{heliostat_id}: {n_hel} group members {hg.names}, row {hel_idx}, "
        f"dist {hel_dist_m:.0f} m, {n_rays} rays"
    )

    pool_rays, pool_targets, boundaries = _load_pool(heliostat_id, hg, scenario, device)
    n_pool = pool_rays.shape[0]
    if max_samples:
        # Smoke mode: truncate each split proportionally from the front.
        keep = []
        for our_split, (s, e) in boundaries.items():
            n_take = max(1, int((e - s) * max_samples / n_pool))
            keep.extend(range(s, min(e, s + n_take)))
        pool_rays = pool_rays[keep]
        pool_targets = pool_targets[keep]
        # Rebuild boundaries on the truncated pool.
        new_b, offset = {}, 0
        for our_split, (s, e) in boundaries.items():
            n_take = max(1, int((e - s) * max_samples / n_pool))
            new_b[our_split] = (offset, offset + n_take)
            offset += n_take
        boundaries = new_b
        n_pool = len(keep)
        log.info(f"  smoke mode: truncated pool to {n_pool} samples")

    pert = _load_perturbation(heliostat_id, n_hel, hel_idx, device)
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

        surfaces = aimed_neighbour_surfaces(hg, scenario, pool_rays[i], tgt_index, device)

        mask = one_hot_mask(hel_idx, 1, n_hel, device)
        hg.activate_heliostats(active_heliostats_mask=mask, device=device)
        pad = torch.zeros(1, 1, device=device)
        kinematic.active_heliostat_positions = (
            kinematic.active_heliostat_positions
            + torch.cat([base_pos_delta[hel_idx : hel_idx + 1, :3], pad], dim=1)
        )
        hg.align_surfaces_with_incident_ray_directions(
            aim_points=aim,
            incident_ray_directions=sun,
            active_heliostats_mask=mask,
            device=device,
        )
        m_c = kinematic.active_motor_positions.detach().clone()[0]  # [2]

        from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
        from artist.flux import get_center_of_mass
        from artist.geometry import bitmap_coordinates_to_target_coordinates

        def _centroid(flux):
            bc = get_center_of_mass(bitmaps=flux, device=device)
            return bitmap_coordinates_to_target_coordinates(
                bitmap_coordinates=bc,
                bitmap_resolution=ray_tracer.bitmap_resolution,
                solar_tower=scenario.solar_tower,
                target_area_indices=tgt,
                device=device,
            )[0]

        # Blocking ON (the dataset the arms train on).
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
            rec["centroid_shift_mrad"] = float(
                torch.norm(shift_m).item() / hel_dist_m * 1000.0
            )
            rec["centroid_shift_up_m"] = float(shift_m[2].item())
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
                f"  {i + 1}/{n_pool}  blocked={rec['blocked_fraction'] * 100:.1f}%  "
                f"({rate:.1f} s/sample, ETA {rate * (n_pool - i - 1) / 60:.0f} min)"
            )

    reset_perturbations(kinematic, snapshot)

    # ------------------------------------------------------------------ save
    n_saved = 0
    per_split_saved: dict[str, int] = {}
    for our_split, (s, e) in boundaries.items():
        out_split = DATASET_ROOT / our_split / heliostat_id
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

    # Merge this heliostat into the shared perturbations.json.
    pert_file = DATASET_ROOT / "perturbations.json"
    pert_all = json.load(open(pert_file)) if pert_file.exists() else {}
    src_pert = json.load(open(SOURCE_PERTURBATIONS))[heliostat_id]
    pert_all[heliostat_id] = src_pert
    with open(pert_file, "w") as fh:
        json.dump(pert_all, fh, indent=2)

    blocked = torch.tensor([r["blocked_fraction"] for r in records])
    summary = {
        "heliostat_id": heliostat_id,
        "scenario": str(scenario_path),
        "group_members": hg.names,
        "hel_dist_m": hel_dist_m,
        "n_pool": n_pool,
        "n_saved": n_saved,
        "per_split_saved": per_split_saved,
        "n_rays": n_rays,
        "paired": paired,
        "blocked_fraction": {
            "mean": float(blocked.mean()),
            "median": float(blocked.median()),
            "max": float(blocked.max()),
        },
        "minutes": (time.time() - t0) / 60.0,
        "samples": [{k: v for k, v in r.items() if k not in ("calibration", "_flux")}
                    for r in records],
    }
    if paired:
        shifts = torch.tensor(
            [r.get("centroid_shift_mrad", float("nan")) for r in records]
        )
        summary["centroid_shift_mrad"] = {
            "mean": float(shifts.nanmean()),
            "median": float(shifts.nanmedian()),
            "max": float(shifts.max()),
        }
    report_dir = REPORT_ROOT / heliostat_id
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "generation_report.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"  report -> {report_dir / 'generation_report.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("heliostat_ids", nargs="+")
    parser.add_argument("--rays", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-paired", dest="paired", action="store_false")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("artist", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    device = get_device()
    for hid in args.heliostat_ids:
        generate(hid, device, n_rays=args.rays, max_samples=args.max_samples,
                 paired=args.paired, force=args.force)


if __name__ == "__main__":
    main()
