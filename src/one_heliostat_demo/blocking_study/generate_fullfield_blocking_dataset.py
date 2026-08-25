"""Generate the Experiment-F full-field blocking dataset (AY36, 1277 heliostats).

Identical protocol to generate_blocking_dataset.py (Experiment S) except:

  1. Scenario: scenarios/neighbourhoods_fullfield/AY36/scenario.h5 — AY36 with
     its real deflectometry surface plus the 14 ideal blockers identified by
     the vertical-shadow census (build_reduced_fullfield_scenario.py). The
     naive full-1277-field variant was measured infeasible on CPU (1-sample
     smoke: ~280 s wall, ~23 GB peak RSS; see the smoke note in the report).
  2. Neighbour aim convention: per sample, AY36 aims at ITS OWN recorded target;
     all 1276 passive blockers aim at ``solar_tower_juelich_lower`` under that
     sample's sun direction (BLOCKER_TARGET_NAME below), via the additive
     ``target_index_override`` of blocking_utils.aimed_neighbour_surfaces.
  3. Same 199 pooled real samples and same 99/50/50 train/val/test split as
     Experiment S (identical pooling/split code path below).

Output: datasets/synthetic/fullfield_blocking_dataset/dataset/{train,val,test}/AY36/
plus outputs/new_mapping_function/blocking_study/experiment_full_field/AY36/
generation_report.json (blocked fractions, centroid shifts, per-sample timing
and peak memory).

Usage
-----
    python generate_fullfield_blocking_dataset.py AY36
    python generate_fullfield_blocking_dataset.py AY36 --max-samples 1   # smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import resource
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
    _ROOT / "datasets" / "synthetic" / "blocking_dataset" / "dataset" / "perturbations.json"
)
DATASET_ROOT = _ROOT / "datasets" / "synthetic" / "fullfield_blocking_dataset" / "dataset"
REPORT_ROOT = (
    _ROOT / "outputs" / "new_mapping_function" / "blocking_study" / "experiment_full_field"
)
SCENARIO_PATH = (
    _ROOT / "scenarios" / "neighbourhoods_fullfield" / "AY36" / "scenario.h5"
)
CENSUS_JSON = (
    _ROOT / "outputs" / "new_mapping_function" / "blocking_study" / "vertical_shadow"
    / "AY36" / "vertical_shadow_blockers.json"
)
BLOCKER_TARGET_NAME = "solar_tower_juelich_lower"

_PAINT_SPLITS = {"train": "train", "validation": "val", "test": "test"}
_PERT_KEYS = {  # perturbations.json key -> apply_perturbations key
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


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def _load_pool(heliostat_id: str, hg, scenario, device):
    """Real sun directions + aim targets for one heliostat, all PAINT splits pooled.

    Byte-identical code path to generate_blocking_dataset.py so the 199 pooled
    samples and the 99/50/50 split assignment match Experiment S exactly.
    """
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
    """AY36's already-sampled blocking-dataset perturbation, verbatim, embedded in
    a full-group (1277-row) zero dict — all blockers stay nominal."""
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
    dataset_root: pathlib.Path = DATASET_ROOT,
    report_root: pathlib.Path = REPORT_ROOT,
    scenario_path: pathlib.Path = SCENARIO_PATH,
) -> dict:
    if not scenario_path.exists():
        raise FileNotFoundError(
            f"Missing reduced full-field scenario: {scenario_path}. "
            "Run build_reduced_fullfield_scenario.py first."
        )
    out_hel = dataset_root
    marker = out_hel / "train" / heliostat_id
    if marker.exists() and any(marker.iterdir()) and not force:
        log.info(f"[SKIP] dataset exists for {heliostat_id} ({marker}) — use --force")
        return {"heliostat_id": heliostat_id, "skipped": True}

    t_load = time.time()
    with h5py.File(scenario_path) as fh:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=fh,
            device=device,
            number_of_surface_points_per_facet=torch.tensor(
                [SURFACE_POINTS_PER_FACET, SURFACE_POINTS_PER_FACET]
            ),
        )
    log.info(f"scenario loaded in {time.time() - t_load:.1f} s")
    scenario.set_number_of_rays(n_rays)
    hg = scenario.heliostat_field.heliostat_groups[0]
    kinematic = hg.kinematics
    hel_idx = hg.names.index(heliostat_id)
    n_hel = hg.number_of_heliostats
    hel_pos = hg.positions[hel_idx, :3].float()
    tower_ref = scenario.solar_tower.target_areas[0].centers[:, :3].float().mean(dim=0)
    hel_dist_m = float(torch.norm(hel_pos - tower_ref.to(device)).item())
    blocker_index = int(scenario.solar_tower.target_name_to_index[BLOCKER_TARGET_NAME])
    log.info(
        f"{heliostat_id}: {n_hel} group members, row {hel_idx}, dist {hel_dist_m:.0f} m, "
        f"{n_rays} rays, blockers aim at '{BLOCKER_TARGET_NAME}' (index {blocker_index})"
    )

    pool_rays, pool_targets, boundaries = _load_pool(heliostat_id, hg, scenario, device)
    n_pool = pool_rays.shape[0]
    orig_indices = list(range(n_pool))  # pool indices into the vertical-shadow census
    if max_samples:
        # Smoke mode: truncate each split proportionally from the front.
        keep = []
        for our_split, (s, e) in boundaries.items():
            n_take = max(1, int((e - s) * max_samples / n_pool))
            keep.extend(range(s, min(e, s + n_take)))
        pool_rays = pool_rays[keep]
        pool_targets = pool_targets[keep]
        orig_indices = keep
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

    # Vertical-shadow census (sanity cross-check): per-pool-sample blocker lists.
    census_blockers: list[list[str]] | None = None
    if CENSUS_JSON.exists():
        census = json.load(open(CENSUS_JSON))
        census_blockers = census["per_sample_blockers"]
        log.info(f"  census loaded: {len(census_blockers)} per-sample blocker lists")
    else:
        log.warning(f"  census JSON not found ({CENSUS_JSON}) — no cross-check")

    from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
    from artist.flux import get_center_of_mass
    from artist.geometry import bitmap_coordinates_to_target_coordinates

    records: list[dict] = []
    t0 = time.time()
    for i in range(n_pool):
        t_sample = time.time()
        sun = pool_rays[i : i + 1]
        tgt = pool_targets[i : i + 1]
        tgt_index = int(tgt.item())
        aim = scenario.solar_tower.get_centers_of_target_areas(tgt, device=device)

        # Experiment F: AY36 aims at ITS OWN target; all passive blockers aim at
        # solar_tower_juelich_lower under this sample's sun direction.
        surfaces = aimed_neighbour_surfaces(
            hg, scenario, pool_rays[i], tgt_index, device,
            target_index_override=blocker_index,
        )

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
            "pool_index": orig_indices[i],
            "split": next(sp for sp, (s, e) in boundaries.items() if s <= i < e),
            "target_name": index_to_name.get(tgt_index, str(tgt_index)),
            "target_area_index": tgt_index,
            "blocker_target_name": BLOCKER_TARGET_NAME,
            "census_blockers": (
                census_blockers[orig_indices[i]] if census_blockers is not None else None
            ),
            "incident_ray_direction": pool_rays[i].cpu().tolist(),
            "blocked_fraction": float(1.0 - bf.item()),
            "active_pixel_percent": _active_pixel_percent(flux_on[0]),
            "seconds": time.time() - t_sample,
            "peak_rss_mb": _peak_rss_mb(),
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

        if (i + 1) % 10 == 0 or i + 1 == n_pool:
            rate = (time.time() - t0) / (i + 1)
            log.info(
                f"  {i + 1}/{n_pool}  blocked={rec['blocked_fraction'] * 100:.1f}%  "
                f"({rate:.1f} s/sample, ETA {rate * (n_pool - i - 1) / 60:.0f} min, "
                f"peak RSS {_peak_rss_mb():.0f} MB)"
            )

    reset_perturbations(kinematic, snapshot)

    # ------------------------------------------------------------------ save
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

    # Merge this heliostat into the shared perturbations.json.
    pert_file = dataset_root / "perturbations.json"
    pert_all = json.load(open(pert_file)) if pert_file.exists() else {}
    src_pert = json.load(open(SOURCE_PERTURBATIONS))[heliostat_id]
    pert_all[heliostat_id] = src_pert
    with open(pert_file, "w") as fh:
        json.dump(pert_all, fh, indent=2)

    blocked = torch.tensor([r["blocked_fraction"] for r in records])
    secs = [r["seconds"] for r in records]

    # Census consistency: blocked_fraction > 0 should only happen when the
    # vertical-shadow census lists at least one blocker for that sample
    # (necessary, not sufficient — the census is a conservative superset).
    census_check = None
    if census_blockers is not None:
        listed = [bool(r["census_blockers"]) for r in records]
        hit = [r["blocked_fraction"] > 1e-9 for r in records]
        census_check = {
            "n_samples": len(records),
            "n_blocked_when_listed": int(sum(h and l for h, l in zip(hit, listed))),
            "n_blocked_when_unlisted": int(sum(h and not l for h, l in zip(hit, listed))),
            "n_unblocked_when_listed": int(sum(not h and l for h, l in zip(hit, listed))),
            "n_unblocked_when_unlisted": int(
                sum(not h and not l for h, l in zip(hit, listed))
            ),
            "note": "blocked_when_unlisted must be 0 — otherwise the reduced "
                    "scenario misses a blocker the census did not predict",
        }

    summary = {
        "heliostat_id": heliostat_id,
        "scenario": str(scenario_path),
        "n_heliostats": n_hel,
        "group_members": hg.names,
        "full_field_attempt": {
            "scenario": "scenarios/full_benchmark_ideal/ideal_1277_AY36_deflectometry.h5 "
                        "(1277 heliostats)",
            "result": "infeasible on CPU — 1-sample smoke test: 281.8 s wall, "
                      "23.1 GB peak memory footprint (/usr/bin/time -l, 2026-07-28)",
            "fallback": "reduced scenario with the 14 vertical-shadow census blockers "
                        "(scenarios/neighbourhoods_fullfield/AY36/scenario.h5); the "
                        "census proves no other heliostat can shade AY36's incident "
                        "light for these 199 sun positions, so the reduction is "
                        "physically exact for incident-side blocking",
        },
        "census_consistency": census_check,
        "blocker_aim": {
            "convention": "all passive blockers aimed at a fixed target",
            "blocker_target_name": BLOCKER_TARGET_NAME,
            "blocker_target_index": blocker_index,
        },
        "hel_dist_m": hel_dist_m,
        "n_pool": n_pool,
        "n_saved": n_saved,
        "per_split_saved": per_split_saved,
        "n_rays": n_rays,
        "paired": paired,
        "blocking_filter": "exact (brute_blocking.exact_blocking) — no pre-gate used",
        "blocked_fraction": {
            "mean": float(blocked.mean()),
            "median": float(blocked.median()),
            "max": float(blocked.max()),
        },
        "timing": {
            "seconds_per_sample_mean": float(np.mean(secs)),
            "seconds_per_sample_max": float(np.max(secs)),
            "peak_rss_mb": float(max(r["peak_rss_mb"] for r in records)),
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
    report_dir = report_root / heliostat_id
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "generation_report.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"  report -> {report_dir / 'generation_report.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("heliostat_ids", nargs="+", default=["AY36"])
    parser.add_argument("--rays", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-paired", dest="paired", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-root", type=pathlib.Path, default=DATASET_ROOT,
                        help="Dataset output root (override for smoke tests).")
    parser.add_argument("--report-root", type=pathlib.Path, default=REPORT_ROOT,
                        help="Generation-report root (override for smoke tests).")
    parser.add_argument("--scenario", type=pathlib.Path, default=SCENARIO_PATH,
                        help="Scenario file (default: reduced full-field scenario).")
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("artist", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    device = get_device()
    log.info(f"device: {device}")
    for hid in args.heliostat_ids:
        generate(hid, device, n_rays=args.rays, max_samples=args.max_samples,
                 paired=args.paired, force=args.force,
                 dataset_root=args.output_root, report_root=args.report_root,
                 scenario_path=args.scenario)


if __name__ == "__main__":
    main()
