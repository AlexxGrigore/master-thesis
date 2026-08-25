"""Determine BA72's sunshape covariance from TRAINING samples, properly.

Replaces the earlier bounding-box "crossing point" heuristic
(ba72_sunshape_sweep_all_samples.py) with a threshold-free, physically
motivated method: flux-weighted second moments (variance) add in quadrature
under convolution, independent of the shape of each contributing blur
(this is the standard "convolution method" used in solar-tower flux
modeling, e.g. HFLCAL-style combined effective sunshape). Concretely, fit

    Var_simulated(std) = Var_optics_only + k * std^2

from a few clean renders at different sunshape std (no thresholding, no
contour extraction -- just the raw flux-weighted image variance), then solve
directly for the std at which Var_simulated(std) = Var_measured. This needs
no arbitrary threshold and gives an exact per-sample answer from a linear
fit instead of interpolating a coarse ratio-vs-std grid.

Uses ONLY the training split (49 real BA72 samples, minus the aim-quality
gate) -- the validation/test splits are left untouched for later checks.

Usage
-----
    python ba72_sunshape_variance_calibration.py
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

from artist.scene.sun import Sun  # noqa: E402
from artist.util import get_device, set_logger_config  # noqa: E402

import config as cfg  # noqa: E402
from train import _bitmap_centroid, _load_fixed_split_real, _load_scenario, _one_hot_active  # noqa: E402
from utils.synth_data import _forward_pass  # noqa: E402

log = logging.getLogger(__name__)

_ROOT = _here.parents[2]  # master-thesis/
HELIOSTAT_ID = "BA72"
BENCHMARK_NAME = "benchmark_split-balanced_train-50_validation-20"
PAINT_DIR = _ROOT / "datasets" / "paint"
SCENARIO_PATH = _ROOT / "scenarios" / "neighbourhoods" / HELIOSTAT_ID / "scenario.h5"
STAGE1_CHECKPOINT = (
    _ROOT / "outputs" / "new_mapping_function" / "ba72_real_data" / "stage1_only"
    / "stage1_checkpoint.pt"
)
OUT_DIR = _ROOT / "outputs" / "new_mapping_function" / "ba72_real_data" / "sunshape_variance_calibration"

RENDER_SURFACE_POINTS = 50
RENDER_RAYS = 50
# Calibration renders: only need enough points to fit a clean line in
# (std^2, Var) space and sanity-check linearity via R^2 -- no coarse-grid
# interpolation needed like the old crossing-point method.
CALIB_STD_MRAD = [2.09, 4.0, 6.0, 8.0]
PIXEL_MISS_THRESHOLD = 40.0  # same aim-quality gate as before


def load_stage1_kinematics(kinematic, device: torch.device) -> None:
    ckpt = torch.load(STAGE1_CHECKPOINT, map_location=device)
    if ckpt.get("heliostat_id") not in (None, HELIOSTAT_ID):
        raise ValueError(f"Checkpoint belongs to {ckpt.get('heliostat_id')!r}, not {HELIOSTAT_ID!r}")
    kinematic.translation_deviation_parameters.data.copy_(ckpt["translation"].to(device))
    kinematic.rotation_deviation_parameters.data.copy_(ckpt["rotation"].to(device))
    kinematic.actuators.optimizable_parameters.data.copy_(ckpt["act_angle"].to(device))
    kinematic.actuators.non_optimizable_parameters.data.copy_(ckpt["act_offset"].to(device))
    kinematic._base_position_deviation = ckpt["base_pos"].clone().to(device)


def flux_variance_trace(flux: torch.Tensor) -> float | None:
    """Flux-weighted (Var_x + Var_y), in px^2 -- no thresholding at all."""
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

    scenario, hg, hel_dist_m, hel_idx = _load_scenario(HELIOSTAT_ID, cfg, device, scenario_path=SCENARIO_PATH)
    n_hel = hg.number_of_heliostats
    load_stage1_kinematics(hg.kinematics, device)
    idx2name = {v: k for k, v in scenario.solar_tower.target_name_to_index.items()}

    train_data, _val_data, _test_data, _ = _load_fixed_split_real(
        HELIOSTAT_ID, cfg, hg, scenario, device, hel_idx=hel_idx, n_hel=n_hel
    )
    flux_all, centroids, rays, motor_pos, _am, target_mask = train_data
    n_train = flux_all.shape[0]
    log.info(f"Train samples: {n_train}")

    active_mask = _one_hot_active(hel_idx, 1, n_hel, device)
    zero_bpd = torch.zeros(n_hel, 3, device=device)
    std_arr = np.array(CALIB_STD_MRAD)
    x_design = std_arr ** 2  # regress Var on std^2

    results, excluded = [], []
    for k in range(n_train):
        gt_flux = flux_all[k].float().cpu()
        ray_i = rays[k : k + 1]
        motor_i = motor_pos[k : k + 1]
        target_i = target_mask[k : k + 1]
        target_name = idx2name[int(target_i.item())]

        gt_col, gt_row = _bitmap_centroid(gt_flux)
        var_measured = flux_variance_trace(gt_flux)

        # Aim-quality gate at the default sunshape (same convention as before).
        pred_default = render_at_std(scenario, hg, ray_i, target_i, motor_i, active_mask, zero_bpd, device, CALIB_STD_MRAD[0])
        if pred_default.shape != gt_flux.shape:
            pred_default = F.interpolate(
                pred_default[None, None], size=tuple(gt_flux.shape), mode="bilinear", align_corners=False,
            )[0, 0]
        pred_col, pred_row = _bitmap_centroid(pred_default)
        if gt_col is None or pred_col is None or var_measured is None:
            excluded.append({"index": k, "reason": "empty flux"})
            continue
        miss_px = float(np.hypot(gt_col - pred_col, gt_row - pred_row))
        if miss_px > PIXEL_MISS_THRESHOLD:
            excluded.append({"index": k, "reason": "aim miss", "miss_px": miss_px, "target": target_name})
            continue

        # Calibration renders (reuse the default-std one already computed).
        variances = [flux_variance_trace(pred_default)]
        for std_mrad in CALIB_STD_MRAD[1:]:
            pred_flux = render_at_std(scenario, hg, ray_i, target_i, motor_i, active_mask, zero_bpd, device, std_mrad)
            if pred_flux.shape != gt_flux.shape:
                pred_flux = F.interpolate(
                    pred_flux[None, None], size=tuple(gt_flux.shape), mode="bilinear", align_corners=False,
                )[0, 0]
            variances.append(flux_variance_trace(pred_flux))
        variances = np.array(variances, dtype=np.float64)

        # Linear fit Var = a + k * std^2, plus R^2 as a linearity sanity check.
        A = np.vstack([np.ones_like(x_design), x_design]).T
        (a_fit, k_fit), *_ = np.linalg.lstsq(A, variances, rcond=None)
        pred_fit = a_fit + k_fit * x_design
        ss_res = float(np.sum((variances - pred_fit) ** 2))
        ss_tot = float(np.sum((variances - variances.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        if k_fit <= 0 or var_measured < a_fit:
            # degenerate: measured spot narrower than the optics-only limit,
            # or a non-physical (non-increasing) fit -- report but don't solve.
            std_needed = None
        else:
            std_needed = float(np.sqrt((var_measured - a_fit) / k_fit))

        results.append({
            "index": k, "target_area_name": target_name, "miss_px": miss_px,
            "var_measured_px2": var_measured, "variances_sim_px2": variances.tolist(),
            "fit_intercept_a": float(a_fit), "fit_slope_k": float(k_fit), "fit_r2": r2,
            "std_needed_mrad": std_needed,
        })
        log.info(f"[{k}] target={target_name}  miss_px={miss_px:.1f}  R2={r2:.4f}  "
                 f"std_needed={std_needed}")

    with open(OUT_DIR / "variance_calibration_results.json", "w") as fh:
        json.dump({
            "heliostat_id": HELIOSTAT_ID, "split": "train", "n_train_total": n_train,
            "n_excluded": len(excluded), "excluded": excluded,
            "calib_std_mrad": CALIB_STD_MRAD, "pixel_miss_threshold": PIXEL_MISS_THRESHOLD,
            "results": results,
        }, fh, indent=2)

    solved = [r for r in results if r["std_needed_mrad"] is not None]
    n_bad_fit = sum(1 for r in results if r["fit_r2"] < 0.99)
    log.info(f"n_train={n_train}  n_excluded={len(excluded)}  n_analyzed={len(results)}  "
             f"n_solved={len(solved)}  n_fits_with_R2<0.99={n_bad_fit}")
    if solved:
        arr = np.array([r["std_needed_mrad"] for r in solved])
        log.info(f"std_needed (mrad): mean={arr.mean():.3f} median={np.median(arr):.3f} "
                  f"std={arr.std():.3f} min={arr.min():.3f} max={arr.max():.3f}")

    # ------------------------------------------------------------------
    # Figure: distribution + linearity check + one example fit.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    if solved:
        vals = [r["std_needed_mrad"] for r in solved]
        axes[0].hist(vals, bins=12, color="#1f78b4", edgecolor="black")
        axes[0].axvline(np.median(vals), color="red", linestyle="--",
                          label=f"median={np.median(vals):.2f} mrad")
        axes[0].axvline(3.80, color="0.4", linestyle=":", label="earlier bbox-method estimate (3.80)")
    axes[0].set_xlabel("std_needed [mrad]")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Variance-matching calibration\n(train split, n={len(solved)}/{n_train})")
    axes[0].legend(fontsize=8)

    r2_vals = [r["fit_r2"] for r in results]
    axes[1].hist(r2_vals, bins=12, color="#33a02c", edgecolor="black")
    axes[1].set_xlabel("linear-fit R^2 (Var vs std^2)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Linearity check per sample")

    if results:
        ex = results[len(results) // 2]
        axes[2].scatter(x_design, ex["variances_sim_px2"], color="#1f78b4", label="rendered points")
        xx = np.linspace(0, max(x_design) * 1.1, 50)
        axes[2].plot(xx, ex["fit_intercept_a"] + ex["fit_slope_k"] * xx, "k--", label="linear fit")
        axes[2].axhline(ex["var_measured_px2"], color="red", linestyle=":", label="measured Var")
        if ex["std_needed_mrad"] is not None:
            axes[2].axvline(ex["std_needed_mrad"] ** 2, color="red", linestyle=":")
        axes[2].set_xlabel("std^2 [mrad^2]")
        axes[2].set_ylabel("image variance [px^2]")
        axes[2].set_title(f"Example fit (sample idx {ex['index']}, R2={ex['fit_r2']:.4f})")
        axes[2].legend(fontsize=8)

    fig.suptitle(f"{HELIOSTAT_ID}: sunshape std via variance matching (train split)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "variance_calibration_summary.png", dpi=150)
    plt.close(fig)

    log.info(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
