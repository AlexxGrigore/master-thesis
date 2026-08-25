"""Sunshape-covariance crossing point on 10 random field heliostats.

Follow-up to ba72_sunshape_sweep_all_samples.py, which found a consistent
sunshape std ~3.7-3.8 mrad crosses the simulated-vs-measured size gap across
BA72's real samples. This checks whether that holds on other heliostats.

IMPORTANT CAVEAT: unlike BA72 (deflectometry-fitted neighbourhood scenario),
this uses the full-field Stage-1-only run's kinematics and scenarios
(`outputs/full_field_1277/stage1_only_ideal_surfaces/`, single-heliostat
"ideal" scenarios per `FULL_FIELD_STAGE1_RESULTS.md`) -- IDEAL (nominal)
mirror surfaces, zero canting/shape error, for all 1277 heliostats. A missing
surface-induced spread could shift the crossing point vs. BA72's
deflectometry-based result; this script does not disentangle sunshape from
surface-model effects, it only checks whether *some* consistent crossing
point exists per heliostat under this checkpoint/scenario combination.

Usage
-----
    python field_sunshape_sweep_10_heliostats.py
"""

from __future__ import annotations

import csv
import json
import logging
import pathlib
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_here = pathlib.Path(__file__).resolve().parent          # blocking_study/
_src = _here.parents[1]                                   # src/
_sh = _src / "one_heliostat_demo" / "single_heliostat"    # single_heliostat/
for _p in (str(_src), str(_sh), str(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from artist.scene.sun import Sun  # noqa: E402
from artist.util import get_device, set_logger_config  # noqa: E402

import config as cfg  # noqa: E402
from train import _bitmap_centroid, _load_fixed_split_real, _load_scenario, _one_hot_active  # noqa: E402
from utils.synth_data import _forward_pass  # noqa: E402
from artist_extensions.contour_loss import ContourExtractor  # noqa: E402

log = logging.getLogger(__name__)

_ROOT = _here.parents[2]  # master-thesis/
BENCHMARK_NAME = "benchmark_split-balanced_train-50_validation-20"
PAINT_DIR = _ROOT / "datasets" / "paint"
CKPT_ROOT = _ROOT / "outputs" / "full_field_1277" / "stage1_only_ideal_surfaces"
SCEN_ROOT = _ROOT / "scenarios" / "full_field_one_heliostat_scenarios" / "ideal"
OUT_DIR = _ROOT / "outputs" / "new_mapping_function" / "field_sunshape_sweep_10_heliostats"
EXCLUDE_IDS = {"BA72"}  # already analyzed separately
RANDOM_SEED = 42
N_HELIOSTATS = 10

RENDER_SURFACE_POINTS = 50
RENDER_RAYS = 50
STD_MRAD_SWEEP = [2.09, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0]
PIXEL_MISS_THRESHOLD = 40.0


def pick_heliostats() -> list[str]:
    ids = set()
    with open(PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            ids.add(row[1])
    ids -= EXCLUDE_IDS
    valid = [h for h in sorted(ids)
             if (CKPT_ROOT / h / "stage1_checkpoint.pt").exists()
             and (SCEN_ROOT / h / "scenario_ideal.h5").exists()]
    random.seed(RANDOM_SEED)
    return random.sample(valid, N_HELIOSTATS)


def load_stage1_kinematics(kinematic, heliostat_id: str, device: torch.device):
    ckpt_path = CKPT_ROOT / heliostat_id / "stage1_checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    if ckpt.get("heliostat_id") not in (None, heliostat_id):
        raise ValueError(f"Checkpoint belongs to {ckpt.get('heliostat_id')!r}, not {heliostat_id!r}")
    kinematic.translation_deviation_parameters.data.copy_(ckpt["translation"].to(device))
    kinematic.rotation_deviation_parameters.data.copy_(ckpt["rotation"].to(device))
    kinematic.actuators.optimizable_parameters.data.copy_(ckpt["act_angle"].to(device))
    kinematic.actuators.non_optimizable_parameters.data.copy_(ckpt["act_offset"].to(device))
    kinematic._base_position_deviation = ckpt["base_pos"].clone().to(device)


def bbox_extent(mask: np.ndarray, level: float = 0.5):
    ys, xs = np.nonzero(mask > level)
    if len(xs) == 0:
        return 0.0, 0.0, 0.0
    return float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1), float((mask > level).sum())


def render_at_std(scenario, hg, ray_i, target_i, motor_i, active_mask, zero_bpd, device, std_mrad):
    cov = (std_mrad / 1000.0) ** 2
    new_sun = Sun(
        number_of_rays=RENDER_RAYS,
        distribution_parameters={"distribution_type": "normal", "mean": 0.0, "covariance": cov},
        device=device,
    )
    scenario.light_sources.light_source_list[0] = new_sun
    scenario.set_number_of_rays(RENDER_RAYS)
    with torch.no_grad():
        _cent, pred_flux = _forward_pass(
            scenario, hg, ray_i, active_mask, target_i, zero_bpd, device, motor_positions=motor_i,
        )
    return pred_flux[0].float().cpu()


def analyze_heliostat(heliostat_id: str, device, extractor):
    scenario_path = SCEN_ROOT / heliostat_id / "scenario_ideal.h5"
    cfg.SURFACE_POINTS_PER_FACET = RENDER_SURFACE_POINTS
    scenario, hg, hel_dist_m, hel_idx = _load_scenario(heliostat_id, cfg, device, scenario_path=scenario_path)
    n_hel = hg.number_of_heliostats
    load_stage1_kinematics(hg.kinematics, heliostat_id, device)

    train_data, val_data, test_data, _ = _load_fixed_split_real(
        heliostat_id, cfg, hg, scenario, device, hel_idx=hel_idx, n_hel=n_hel
    )
    all_samples = []
    for split_name, data in (("train", train_data), ("val", val_data), ("test", test_data)):
        if data is None:
            continue
        flux, centroids, rays, motor_pos, _am, target_mask = data
        for k in range(flux.shape[0]):
            all_samples.append((split_name, k, flux[k].float().cpu(), rays[k : k + 1],
                                 motor_pos[k : k + 1], target_mask[k : k + 1]))

    active_mask = _one_hot_active(hel_idx, 1, n_hel, device)
    zero_bpd = torch.zeros(n_hel, 3, device=device)

    results, excluded = [], []
    for split_name, k, gt_flux, ray_i, motor_i, target_i in all_samples:
        gt_steps = dict(extractor.intermediate_steps(gt_flux))
        gt_w, gt_h, gt_area = bbox_extent(gt_steps["Soft mask"])
        gt_col, gt_row = _bitmap_centroid(gt_flux)

        pred_default = render_at_std(scenario, hg, ray_i, target_i, motor_i, active_mask, zero_bpd, device, 2.09)
        if pred_default.shape != gt_flux.shape:
            pred_default = F.interpolate(
                pred_default[None, None], size=tuple(gt_flux.shape), mode="bilinear", align_corners=False,
            )[0, 0]
        pred_col, pred_row = _bitmap_centroid(pred_default)
        if gt_col is None or pred_col is None or gt_area == 0:
            excluded.append({"split": split_name, "index": k, "reason": "empty flux"})
            continue
        miss_px = float(np.hypot(gt_col - pred_col, gt_row - pred_row))
        if miss_px > PIXEL_MISS_THRESHOLD:
            excluded.append({"split": split_name, "index": k, "reason": "aim miss", "miss_px": miss_px})
            continue

        ratios = []
        for std_mrad in STD_MRAD_SWEEP:
            pred_flux = pred_default if std_mrad == 2.09 else render_at_std(
                scenario, hg, ray_i, target_i, motor_i, active_mask, zero_bpd, device, std_mrad)
            if pred_flux.shape != gt_flux.shape:
                pred_flux = F.interpolate(
                    pred_flux[None, None], size=tuple(gt_flux.shape), mode="bilinear", align_corners=False,
                )[0, 0]
            pred_steps = dict(extractor.intermediate_steps(pred_flux))
            _pr_w, _pr_h, pr_area = bbox_extent(pred_steps["Soft mask"])
            ratios.append(pr_area / gt_area if gt_area else 0.0)

        ratios_arr = np.array(ratios)
        stds_arr = np.array(STD_MRAD_SWEEP)
        if ratios_arr[-1] < 1.0:
            crossing_std = None
        elif ratios_arr[0] > 1.0:
            crossing_std = float(stds_arr[0])
        else:
            crossing_std = float(np.interp(1.0, ratios_arr, stds_arr))

        results.append({"split": split_name, "index": k, "miss_px": miss_px,
                         "crossing_std_mrad": crossing_std, "ratios_area": ratios})

    return {
        "heliostat_id": heliostat_id, "n_total": len(all_samples),
        "n_excluded": len(excluded), "excluded": excluded, "results": results,
    }


def main() -> None:
    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    device = get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg.DATA_MODE = "real"
    cfg.USE_FIXED_SPLIT = True
    cfg.BENCHMARK_NAME = BENCHMARK_NAME
    cfg.BENCHMARK_CSV = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
    cfg.CALIBRATION_DIR = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
    cfg.REAL_FLUX_DIR = PAINT_DIR / BENCHMARK_NAME / "flux_image"

    heliostat_ids = pick_heliostats()
    log.info(f"Selected heliostats (seed={RANDOM_SEED}): {heliostat_ids}")

    extractor = ContourExtractor(
        tau=cfg.CONTOUR_TAU, eta=cfg.CONTOUR_ETA, smoothing_rounds=cfg.CONTOUR_SMOOTHING_ROUNDS,
        gaussian_sigma=cfg.CONTOUR_GAUSS_SIGMA, gaussian_kernel_size=cfg.CONTOUR_GAUSS_KSIZE,
    )

    all_heliostat_results = {}
    for hid in heliostat_ids:
        log.info(f"=== {hid} ===")
        res = analyze_heliostat(hid, device, extractor)
        all_heliostat_results[hid] = res
        crossings = [r["crossing_std_mrad"] for r in res["results"] if r["crossing_std_mrad"] is not None]
        log.info(f"{hid}: n_total={res['n_total']} n_excluded={res['n_excluded']} "
                 f"n_analyzed={len(res['results'])} n_crossings={len(crossings)} "
                 f"{'median=' + f'{np.median(crossings):.2f}' if crossings else 'NO CROSSINGS'}")

    with open(OUT_DIR / "field_sweep_results.json", "w") as fh:
        json.dump({"heliostat_ids": heliostat_ids, "std_mrad_sweep": STD_MRAD_SWEEP,
                    "pixel_miss_threshold": PIXEL_MISS_THRESHOLD,
                    "per_heliostat": all_heliostat_results}, fh, indent=2)

    # ------------------------------------------------------------------
    # Summary figure: per-heliostat crossing-std boxplot/points + combined hist.
    # ------------------------------------------------------------------
    per_hel_crossings = {}
    for hid, res in all_heliostat_results.items():
        crossings = [r["crossing_std_mrad"] for r in res["results"] if r["crossing_std_mrad"] is not None]
        per_hel_crossings[hid] = crossings

    all_crossings = [v for lst in per_hel_crossings.values() for v in lst]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    labels = [hid for hid in heliostat_ids if per_hel_crossings[hid]]
    data = [per_hel_crossings[hid] for hid in labels]
    if data:
        ax1.boxplot(data, labels=labels, showmeans=True)
    ax1.axhline(3.81, color="red", linestyle="--", linewidth=1, label="BA72 median (3.81 mrad)")
    ax1.set_ylabel("per-sample crossing std [mrad]")
    ax1.set_title("Crossing std by heliostat (10 random field heliostats)")
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(fontsize=8)

    if all_crossings:
        ax2.hist(all_crossings, bins=20, color="#33a02c", edgecolor="black")
        ax2.axvline(np.median(all_crossings), color="red", linestyle="--",
                     label=f"median={np.median(all_crossings):.2f} mrad")
        ax2.axvline(3.81, color="0.4", linestyle=":", label="BA72 median (3.81 mrad)")
    ax2.set_xlabel("crossing std [mrad]")
    ax2.set_ylabel("count")
    ax2.set_title(f"All samples pooled across {len(labels)} heliostats (n={len(all_crossings)})")
    ax2.legend(fontsize=8)

    fig.suptitle("Sunshape-covariance crossing point: 10 random field heliostats "
                 "(ideal surfaces, full-field Stage-1 checkpoint)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "field_crossing_std_summary.png", dpi=150)
    plt.close(fig)

    log.info("=== SUMMARY ===")
    for hid in heliostat_ids:
        res = all_heliostat_results[hid]
        crossings = per_hel_crossings[hid]
        log.info(f"{hid}: total={res['n_total']} excluded={res['n_excluded']} analyzed={len(res['results'])} "
                 f"median_crossing={np.median(crossings) if crossings else None}")
    if all_crossings:
        arr = np.array(all_crossings)
        log.info(f"POOLED (n={len(arr)}): mean={arr.mean():.2f} median={np.median(arr):.2f} "
                  f"std={arr.std():.2f} min={arr.min():.2f} max={arr.max():.2f}")
    log.info(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
