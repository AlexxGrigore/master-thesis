"""One real BA72 sample: measured flux vs. simulated flux (real-data Stage-1
kinematics), then the CURRENT contour-loss pipeline applied to both.

Directly answers "is the predicted spot really smaller than the measured
one, or was that a simulation-only artifact?" using the actual real-data
Stage-1 checkpoint (run_ba72_real_data_stage1.py) and the CURRENT project
contour hyperparameters (config.py's CONTOUR_* block: tau=0.58, eta=70,
sigma=3, ksize=13 -- NOT the ContourExtractor class defaults, which are the
unpatched thesis values sigma=1/ksize=5 and would misrepresent the current
setup).

Sample: one real PAINT test-split sample for BA72 (any sun position -- first
available). Render: 50x50 surface points/facet, 50 rays/point (DISPLAY_RAYS-
style, low speckle) so the size comparison isn't confounded by Monte-Carlo
noise. No blocking (plain forward pass) -- this is about the flux-size
mismatch itself, not occlusion.

Usage
-----
    python ba72_contour_real_sample.py
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

from artist.util import get_device, set_logger_config  # noqa: E402

import config as cfg  # noqa: E402
from train import _load_fixed_split_real, _load_scenario, _one_hot_active  # noqa: E402
from utils.synth_data import _forward_pass  # noqa: E402
from artist_extensions.contour_loss import ContourExtractor  # noqa: E402

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
OUT_DIR = _ROOT / "outputs" / "new_mapping_function" / "ba72_real_data" / "contour_sample_check"

RENDER_SURFACE_POINTS = 50
RENDER_RAYS = 50
SAMPLE_INDEX = 8  # a well-aimed test sample (s1_centroid_mrad ~0.26, per arm_results.json
                  # per_sample_test) -- avoids picking a badly-misaimed sample where the
                  # simulated beam misses the frame, which would confound the size comparison


def load_stage1_kinematics(kinematic, device: torch.device) -> None:
    ckpt = torch.load(STAGE1_CHECKPOINT, map_location=device)
    if ckpt.get("heliostat_id") not in (None, HELIOSTAT_ID):
        raise ValueError(f"Checkpoint belongs to {ckpt.get('heliostat_id')!r}, not {HELIOSTAT_ID!r}")
    kinematic.translation_deviation_parameters.data.copy_(ckpt["translation"].to(device))
    kinematic.rotation_deviation_parameters.data.copy_(ckpt["rotation"].to(device))
    kinematic.actuators.optimizable_parameters.data.copy_(ckpt["act_angle"].to(device))
    kinematic.actuators.non_optimizable_parameters.data.copy_(ckpt["act_offset"].to(device))
    kinematic._base_position_deviation = ckpt["base_pos"].clone().to(device)


def contour_pixels(contour_img: np.ndarray, threshold: float = 0.05):
    rows, cols = np.nonzero(contour_img > threshold)
    return cols, rows


def main() -> None:
    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    device = get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not STAGE1_CHECKPOINT.exists():
        raise FileNotFoundError(f"{STAGE1_CHECKPOINT} missing -- run run_ba72_real_data_stage1.py first.")

    # 1. Load scenario + real-data Stage-1 kinematics.
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
    log.info(f"Loaded real-data Stage-1 checkpoint {STAGE1_CHECKPOINT}")

    # 2. Real 50/20/20 split, pick one sample from the (post-swap) test set.
    train_data, val_data, test_data, _ = _load_fixed_split_real(
        HELIOSTAT_ID, cfg, hg, scenario, device, hel_idx=hel_idx, n_hel=n_hel
    )
    flux, centroids, rays, motor_pos, _active_mask, target_mask = test_data
    i = SAMPLE_INDEX
    gt_flux = flux[i].float().cpu()
    ray_i = rays[i : i + 1]
    motor_i = motor_pos[i : i + 1]
    target_i = target_mask[i : i + 1]
    target_name = {v: k for k, v in scenario.solar_tower.target_name_to_index.items()}[int(target_i.item())]
    log.info(f"Sample {i}: target={target_name!r}, incident_ray={ray_i[0].tolist()}, motors={motor_i[0].tolist()}")

    # 3. Render the simulated flux: plain forward pass (no blocking), current
    #    real-data Stage-1 kinematics, dense low-speckle render.
    active_mask = _one_hot_active(hel_idx, 1, n_hel, device)
    zero_bpd = torch.zeros(n_hel, 3, device=device)
    old_n_rays = scenario.light_sources.light_source_list[0].number_of_rays
    scenario.set_number_of_rays(RENDER_RAYS)
    with torch.no_grad():
        _pred_cent, pred_flux = _forward_pass(
            scenario, hg, ray_i, active_mask, target_i, zero_bpd, device, motor_positions=motor_i,
        )
    scenario.set_number_of_rays(old_n_rays)
    pred_flux = pred_flux[0].float().cpu()
    log.info(f"Rendered {RENDER_SURFACE_POINTS}x{RENDER_SURFACE_POINTS} pts/facet, {RENDER_RAYS} rays/pt, "
             f"no blocking. GT shape={tuple(gt_flux.shape)}, pred shape={tuple(pred_flux.shape)}")

    resized = False
    if pred_flux.shape != gt_flux.shape:
        resized = True
        pred_flux = F.interpolate(
            pred_flux[None, None], size=tuple(gt_flux.shape), mode="bilinear", align_corners=False,
        )[0, 0]
        log.info(f"Resized predicted flux to GT shape {tuple(gt_flux.shape)}")

    # 4. Contour extraction with the CURRENT project hyperparameters (config.py),
    #    not the ContourExtractor class defaults (those are the unpatched thesis
    #    values sigma=1/ksize=5, which would misrepresent the actual setup).
    extractor = ContourExtractor(
        tau=cfg.CONTOUR_TAU,
        eta=cfg.CONTOUR_ETA,
        smoothing_rounds=cfg.CONTOUR_SMOOTHING_ROUNDS,
        gaussian_sigma=cfg.CONTOUR_GAUSS_SIGMA,
        gaussian_kernel_size=cfg.CONTOUR_GAUSS_KSIZE,
    )
    gt_steps = dict(extractor.intermediate_steps(gt_flux))
    pred_steps = dict(extractor.intermediate_steps(pred_flux))
    gt_contour = gt_steps["Upper contour"]
    pred_contour = pred_steps["Upper contour"]

    # Size diagnostics (soft-mask footprint, same convention as the tau sweep).
    def _bbox_extent(mask: np.ndarray, level: float = 0.5):
        ys, xs = np.nonzero(mask > level)
        if len(xs) == 0:
            return 0.0, 0.0, 0.0
        return float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1), float((mask > level).sum())

    gt_w, gt_h, gt_area = _bbox_extent(gt_steps["Soft mask"])
    pr_w, pr_h, pr_area = _bbox_extent(pred_steps["Soft mask"])
    diag = {
        "gt_softmask_width_px": gt_w, "pred_softmask_width_px": pr_w,
        "ratio_width": (pr_w / gt_w) if gt_w else None,
        "gt_softmask_height_px": gt_h, "pred_softmask_height_px": pr_h,
        "ratio_height": (pr_h / gt_h) if gt_h else None,
        "gt_softmask_area_px": gt_area, "pred_softmask_area_px": pr_area,
        "ratio_area": (pr_area / gt_area) if gt_area else None,
    }
    log.info(f"Size diagnostics: {json.dumps(diag, indent=2)}")

    # ------------------------------------------------------------------
    # Figure 1: pipeline comparison grid (shared scale per column).
    # ------------------------------------------------------------------
    col_names = ["Raw", "Denoised", "Soft mask", "Eroded", "Upper contour"]
    fig, axes = plt.subplots(2, len(col_names), figsize=(4.2 * len(col_names), 8.6))
    # "Raw" is in genuinely different units per row (measured flux is stored
    # pre-normalized to [0,1] from the PNG; simulated flux is unnormalized
    # physical ray-tracer intensity, here up to ~73) -- a shared scale would
    # make the measured row invisible. Every later column is unit-consistent
    # (the extractor's first internal step is a per-image min-max normalize),
    # so only "Raw" gets a per-row scale; everything else stays shared.
    for c, name in enumerate(col_names):
        per_row_scale = name == "Raw"
        vmax = max(float(np.max(gt_steps[name])), float(np.max(pred_steps[name]))) or 1.0
        for r, (label, d) in enumerate((("Measured (real)", gt_steps), ("Simulated (predicted)", pred_steps))):
            ax = axes[r, c]
            this_vmax = (float(np.max(d[name])) or 1.0) if per_row_scale else vmax
            ax.imshow(d[name], cmap="hot", vmin=0, vmax=this_vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{name}\n({'own scale per row' if per_row_scale else f'shared scale 0-{vmax:.2g}'})",
                              fontsize=11)
            if c == 0:
                ax.set_ylabel(label, fontsize=12)
            if per_row_scale:
                ax.text(0.02, 0.02, f"vmax={this_vmax:.2g}", transform=ax.transAxes,
                        color="white", fontsize=8, va="bottom")
    fig.suptitle(
        f"{HELIOSTAT_ID} real test sample {i} (target={target_name}) -- contour pipeline, "
        f"measured vs. simulated\nreal-data Stage-1 kinematics, current contour hyperparameters "
        f"(tau={cfg.CONTOUR_TAU}, eta={cfg.CONTOUR_ETA}, sigma={cfg.CONTOUR_GAUSS_SIGMA}, "
        f"ksize={cfg.CONTOUR_GAUSS_KSIZE}), render {RENDER_SURFACE_POINTS}^2 pts/facet x {RENDER_RAYS} rays"
        + (", resized to GT" if resized else ""),
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT_DIR / "contour_pipeline_comparison.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: side-by-side flux with upper-contour overlay + size summary.
    # ------------------------------------------------------------------
    gt_cols, gt_rows = contour_pixels(gt_contour)
    pr_cols, pr_rows = contour_pixels(pred_contour)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6))
    for ax, bg, title in (
        (axes[0], gt_flux.numpy(), "Measured (real) flux"),
        (axes[1], pred_flux.numpy(), "Simulated (predicted) flux"),
    ):
        ax.imshow(bg / (bg.max() or 1.0), cmap="gray", vmin=0, vmax=1)
        ax.scatter(gt_cols, gt_rows, s=5, c="lime", label="Measured upper contour")
        ax.scatter(pr_cols, pr_rows, s=5, c="red", label="Simulated upper contour")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="lower right", fontsize=9)

    axes[2].imshow(gt_flux.numpy() / (gt_flux.numpy().max() or 1.0), cmap="gray", vmin=0, vmax=1)
    axes[2].scatter(gt_cols, gt_rows, s=6, c="lime", marker="x", label="Measured upper contour")
    axes[2].scatter(pr_cols, pr_rows, s=6, c="red", marker="+", label="Simulated upper contour")
    axes[2].set_title("Both contours, on the measured-flux background", fontsize=12)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    axes[2].legend(loc="lower right", fontsize=9)

    annotation = (
        f"sample {i}, target={target_name}\n"
        f"soft-mask bbox: measured {gt_w:.0f}x{gt_h:.0f}px (area {gt_area:.0f}) | "
        f"simulated {pr_w:.0f}x{pr_h:.0f}px (area {pr_area:.0f})\n"
        f"ratio (sim/measured): width={diag['ratio_width']:.2f}  height={diag['ratio_height']:.2f}  "
        f"area={diag['ratio_area']:.2f}"
    )
    fig.suptitle(f"{HELIOSTAT_ID} real test sample -- upper contours & size comparison\n{annotation}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(OUT_DIR / "contour_overlay_size_comparison.png", dpi=150)
    plt.close(fig)

    summary = {
        "heliostat_id": HELIOSTAT_ID,
        "sample_index": i,
        "target_area_name": target_name,
        "incident_ray_direction": ray_i[0].tolist(),
        "motor_position": motor_i[0].tolist(),
        "stage1_checkpoint": str(STAGE1_CHECKPOINT.relative_to(_ROOT)),
        "render_settings": {"surface_points_per_facet": RENDER_SURFACE_POINTS, "rays_per_point": RENDER_RAYS},
        "contour_hyperparameters": {
            "tau": cfg.CONTOUR_TAU, "eta": cfg.CONTOUR_ETA,
            "smoothing_rounds": cfg.CONTOUR_SMOOTHING_ROUNDS,
            "gaussian_sigma": cfg.CONTOUR_GAUSS_SIGMA, "gaussian_kernel_size": cfg.CONTOUR_GAUSS_KSIZE,
        },
        "gt_flux_minmax": [float(gt_flux.min()), float(gt_flux.max())],
        "pred_flux_minmax": [float(pred_flux.min()), float(pred_flux.max())],
        "predicted_resized_to_gt": resized,
        "size_diagnostics": diag,
    }
    with open(OUT_DIR / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
