"""Sunshape calibration via an IDEAL-AIM heliostat, on ALL real BA72 train samples.

Improves on ba72_sunshape_variance_calibration.py / ba72_sunshape_sweep_all_samples.py,
both of which relied on the (imperfect, ~8 mrad mean error) Stage-1-trained
kinematic model to orient the simulated heliostat, then had to EXCLUDE any
sample where that residual aim error was too large to compare spot sizes at
all (55%+ of BA72's real samples were thrown out this way).

This removes that confound entirely: instead of orienting from Stage-1's
learned motors, the mirror is aimed EXACTLY at the real measured centroid
(the calibration parser's own `focal_spots` field, i.e. c_gt) via
`align_surfaces_with_incident_ray_directions(aim_points=c_gt, ...)` -- a pure
geometric forward-aim that is exact regardless of any kinematic-model
quality. "Ideal" here means ideal POINTING (no Stage-1 deviation parameters
applied, plain nominal kinematics), NOT an idealized mirror surface -- the
scenario's real deflectometry-fitted surface is still used, since that's a
property of the scenario file, not of the kinematic deviation parameters.
This isolates the spot-SIZE question from Stage-1's pointing-accuracy
question, and lets every one of the 49 real training samples be used (no
aim-quality gate needed).

Two metrics computed per sample, matching the two earlier methods so the
comparison is apples-to-apples:
  - bbox-threshold area (tau=0.58 after denoise, i.e. what contour loss's own
    thresholding pipeline actually sees)
  - flux-weighted variance (full second moment, no thresholding)

Usage
-----
    python ba72_sunshape_ideal_aim_calibration.py
"""

from __future__ import annotations

import json
import logging
import pathlib
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

from artist.flux import get_center_of_mass  # noqa: E402
from artist.geometry import bitmap_coordinates_to_target_coordinates  # noqa: E402
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer  # noqa: E402
from artist.scene.sun import Sun  # noqa: E402
from artist.util import get_device, set_logger_config  # noqa: E402

import config as cfg  # noqa: E402
from train import _bitmap_centroid, _load_fixed_split_real, _load_scenario, _one_hot_active  # noqa: E402
from artist_extensions.contour_loss import ContourExtractor  # noqa: E402

log = logging.getLogger(__name__)

_ROOT = _here.parents[2]  # master-thesis/
HELIOSTAT_ID = "BA72"
BENCHMARK_NAME = "benchmark_split-balanced_train-50_validation-20"
PAINT_DIR = _ROOT / "datasets" / "paint"
SCENARIO_PATH = _ROOT / "scenarios" / "neighbourhoods" / HELIOSTAT_ID / "scenario.h5"
OUT_DIR = _ROOT / "outputs" / "new_mapping_function" / "ba72_real_data" / "sunshape_ideal_aim_calibration"

RENDER_SURFACE_POINTS = 50
RENDER_RAYS = 50
CALIB_STD_MRAD = [2.09, 4.0, 6.0, 8.0]
CENTROID_SANITY_PX_WARN = 5.0  # ideal aim should land within a few px of c_gt


def bbox_extent(mask: np.ndarray, level: float = 0.5):
    ys, xs = np.nonzero(mask > level)
    if len(xs) == 0:
        return 0.0, 0.0, 0.0
    return float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1), float((mask > level).sum())


def flux_variance_trace(flux: torch.Tensor) -> float | None:
    f = flux.cpu().float().numpy()
    fsum = f.sum()
    if fsum < 1e-12:
        return None
    h, w = f.shape
    cols = np.arange(w, dtype=np.float64).reshape(1, -1)
    rows = np.arange(h, dtype=np.float64).reshape(-1, 1)
    col_bar = (f * cols).sum() / fsum
    row_bar = (f * rows).sum() / fsum
    var_x = (f * (cols - col_bar) ** 2).sum() / fsum
    var_y = (f * (rows - row_bar) ** 2).sum() / fsum
    return float(var_x + var_y)


def ideal_aim_render(scenario, hg, active_mask, aim_point, sun_dir, target_i, device, std_mrad):
    """Aim a nominal (no Stage-1 deviation) heliostat exactly at aim_point, then
    ray-trace at the given sunshape std. Returns (flux[H,W], centroid_enu[4])."""
    cov = (std_mrad / 1000.0) ** 2
    new_sun = Sun(
        number_of_rays=RENDER_RAYS,
        distribution_parameters={"distribution_type": "normal", "mean": 0.0, "covariance": cov},
        device=device,
    )
    scenario.light_sources.light_source_list[0] = new_sun
    scenario.set_number_of_rays(RENDER_RAYS)
    with torch.no_grad():
        hg.activate_heliostats(active_heliostats_mask=active_mask, device=device)
        hg.align_surfaces_with_incident_ray_directions(
            aim_points=aim_point, incident_ray_directions=sun_dir,
            active_heliostats_mask=active_mask, device=device,
        )
        ray_tracer = HeliostatRayTracer(
            scenario=scenario, heliostat_group=hg, blocking_active=False,
            world_size=1, rank=0, batch_size=1, random_seed=7,
        )
        flux, _, _, _ = ray_tracer.trace_rays(
            incident_ray_directions=sun_dir, active_heliostats_mask=active_mask,
            target_area_indices=target_i, device=device,
        )
        bc = get_center_of_mass(bitmaps=flux, device=device)
        cent = bitmap_coordinates_to_target_coordinates(
            bitmap_coordinates=bc, bitmap_resolution=ray_tracer.bitmap_resolution,
            solar_tower=scenario.solar_tower, target_area_indices=target_i, device=device,
        )[0]
    return flux[0].detach().cpu(), cent.detach().cpu()


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
    cfg.SURFACE_POINTS_PER_FACET = RENDER_SURFACE_POINTS

    # NOTE: no Stage-1 checkpoint loaded -- nominal (ideal-pointing) kinematics,
    # real deflectometry surface still baked into the scenario file itself.
    scenario, hg, hel_dist_m, hel_idx = _load_scenario(HELIOSTAT_ID, cfg, device, scenario_path=SCENARIO_PATH)
    n_hel = hg.number_of_heliostats
    idx2name = {v: k for k, v in scenario.solar_tower.target_name_to_index.items()}

    train_data, _val_data, _test_data, _ = _load_fixed_split_real(
        HELIOSTAT_ID, cfg, hg, scenario, device, hel_idx=hel_idx, n_hel=n_hel
    )
    flux_all, centroids, rays, motor_pos, _am, target_mask = train_data
    n_train = flux_all.shape[0]
    log.info(f"Train samples: {n_train} (ALL used -- no aim-quality gate needed)")

    active_mask = _one_hot_active(hel_idx, 1, n_hel, device)
    extractor = ContourExtractor(
        tau=cfg.CONTOUR_TAU, eta=cfg.CONTOUR_ETA, smoothing_rounds=cfg.CONTOUR_SMOOTHING_ROUNDS,
        gaussian_sigma=cfg.CONTOUR_GAUSS_SIGMA, gaussian_kernel_size=cfg.CONTOUR_GAUSS_KSIZE,
    )
    std_arr = np.array(CALIB_STD_MRAD)
    x_design = std_arr ** 2

    results = []
    bad_aim_sanity = []
    for k in range(n_train):
        gt_flux = flux_all[k].float().cpu()
        sun_i = rays[k : k + 1]
        aim_i = centroids[k : k + 1]
        target_i = target_mask[k : k + 1]
        target_name = idx2name[int(target_i.item())]

        gt_col, gt_row = _bitmap_centroid(gt_flux)
        gt_steps = dict(extractor.intermediate_steps(gt_flux))
        gt_w, gt_h, gt_area = bbox_extent(gt_steps["Soft mask"])
        var_measured = flux_variance_trace(gt_flux)
        if gt_col is None or var_measured is None or gt_area == 0:
            continue

        areas, variances = [], []
        pred_centroid_px_first = None
        for i, std_mrad in enumerate(CALIB_STD_MRAD):
            pred_flux, _pred_cent_enu = ideal_aim_render(
                scenario, hg, active_mask, aim_i, sun_i, target_i, device, std_mrad
            )
            if pred_flux.shape != gt_flux.shape:
                pred_flux = F.interpolate(
                    pred_flux[None, None], size=tuple(gt_flux.shape), mode="bilinear", align_corners=False,
                )[0, 0]
            pred_col, pred_row = _bitmap_centroid(pred_flux)
            if i == 0:
                pred_centroid_px_first = (pred_col, pred_row)
            pred_steps = dict(extractor.intermediate_steps(pred_flux))
            _pw, _ph, pr_area = bbox_extent(pred_steps["Soft mask"])
            areas.append(pr_area)
            variances.append(flux_variance_trace(pred_flux))
        areas = np.array(areas, dtype=np.float64)
        variances = np.array(variances, dtype=np.float64)

        # Sanity check: ideal aim should land within a few px of the real centroid.
        if pred_centroid_px_first[0] is not None:
            aim_miss_px = float(np.hypot(gt_col - pred_centroid_px_first[0], gt_row - pred_centroid_px_first[1]))
            if aim_miss_px > CENTROID_SANITY_PX_WARN:
                bad_aim_sanity.append({"index": k, "aim_miss_px": aim_miss_px})
        else:
            aim_miss_px = None

        # --- bbox-threshold crossing (interpolate on the area ratio grid) ---
        ratio_area = areas / gt_area
        if ratio_area[-1] < 1.0:
            crossing_bbox = None
        elif ratio_area[0] > 1.0:
            crossing_bbox = float(std_arr[0])
        else:
            crossing_bbox = float(np.interp(1.0, ratio_area, std_arr))

        # --- variance linear fit (Var = a + k*std^2), solve for match ---
        A = np.vstack([np.ones_like(x_design), x_design]).T
        (a_fit, k_fit), *_ = np.linalg.lstsq(A, variances, rcond=None)
        pred_fit = a_fit + k_fit * x_design
        ss_res = float(np.sum((variances - pred_fit) ** 2))
        ss_tot = float(np.sum((variances - variances.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        if k_fit <= 0 or var_measured < a_fit:
            crossing_var = None
        else:
            crossing_var = float(np.sqrt((var_measured - a_fit) / k_fit))

        results.append({
            "index": k, "target_area_name": target_name, "aim_miss_px": aim_miss_px,
            "gt_area": gt_area, "areas_sim": areas.tolist(), "crossing_std_bbox_mrad": crossing_bbox,
            "var_measured": var_measured, "variances_sim": variances.tolist(),
            "fit_r2": r2, "crossing_std_variance_mrad": crossing_var,
        })
        log.info(f"[{k}] target={target_name}  aim_miss_px={aim_miss_px:.2f}  "
                 f"crossing_bbox={crossing_bbox}  crossing_var={crossing_var}  R2={r2:.4f}")

    with open(OUT_DIR / "ideal_aim_calibration_results.json", "w") as fh:
        json.dump({
            "heliostat_id": HELIOSTAT_ID, "split": "train", "n_train_total": n_train,
            "n_analyzed": len(results), "n_aim_sanity_warnings": len(bad_aim_sanity),
            "aim_sanity_warnings": bad_aim_sanity,
            "calib_std_mrad": CALIB_STD_MRAD, "results": results,
        }, fh, indent=2)

    bbox_vals = [r["crossing_std_bbox_mrad"] for r in results if r["crossing_std_bbox_mrad"] is not None]
    var_vals = [r["crossing_std_variance_mrad"] for r in results if r["crossing_std_variance_mrad"] is not None]
    log.info(f"n_train={n_train}  n_analyzed={len(results)}  n_aim_sanity_warnings={len(bad_aim_sanity)}")
    if bbox_vals:
        arr = np.array(bbox_vals)
        log.info(f"BBOX method:     n={len(arr)} mean={arr.mean():.3f} median={np.median(arr):.3f} std={arr.std():.3f}")
    if var_vals:
        arr = np.array(var_vals)
        log.info(f"VARIANCE method: n={len(arr)} mean={arr.mean():.3f} median={np.median(arr):.3f} std={arr.std():.3f}")

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    if bbox_vals:
        axes[0].hist(bbox_vals, bins=12, color="#1f78b4", edgecolor="black", alpha=0.7, label="bbox-threshold method")
        axes[0].axvline(np.median(bbox_vals), color="#1f78b4", linestyle="--")
    if var_vals:
        axes[0].hist(var_vals, bins=12, color="#e31a1c", edgecolor="black", alpha=0.5, label="variance method")
        axes[0].axvline(np.median(var_vals), color="#e31a1c", linestyle="--")
    axes[0].set_xlabel("crossing std [mrad]")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Ideal-aim calibration, ALL {len(results)}/{n_train} train samples")
    axes[0].legend(fontsize=8)

    miss_px = [r["aim_miss_px"] for r in results if r["aim_miss_px"] is not None]
    axes[1].hist(miss_px, bins=15, color="0.5", edgecolor="black")
    axes[1].axvline(CENTROID_SANITY_PX_WARN, color="red", linestyle="--", label=f"{CENTROID_SANITY_PX_WARN} px")
    axes[1].set_xlabel("ideal-aim centroid miss [px] (sanity check)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Aim-quality sanity check\n(should be ~0, unlike the Stage-1-based method)")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"{HELIOSTAT_ID}: sunshape calibration via ideal-aim heliostat (train split)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "ideal_aim_calibration_summary.png", dpi=150)
    plt.close(fig)

    log.info(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
