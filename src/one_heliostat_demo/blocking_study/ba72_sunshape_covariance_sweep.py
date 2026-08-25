"""Sweep ARTIST's sunshape covariance and check whether the simulated flux
footprint on a real BA72 sample converges toward the measured footprint size.

Follow-up to ba72_contour_real_sample.py, which found the simulated spot is
systematically smaller than the measured one (soft-mask area ratio 0.65 at
ARTIST's default sunshape std ~2.1 mrad) after ruling out camera saturation
and mirror-mesh resolution as causes. The leading remaining hypothesis is
that ARTIST's plain isotropic-Gaussian sunshape model (no circumsolar aura,
no atmospheric scattering, no camera PSF) is simply narrower than the real
sun's effective angular spread as captured by a real camera. This script
tests that directly: re-render the SAME real sample at a range of enlarged
sunshape covariances and see where the simulated footprint size crosses the
measured one.

Usage
-----
    python ba72_sunshape_covariance_sweep.py
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
OUT_DIR = _ROOT / "outputs" / "new_mapping_function" / "ba72_real_data" / "sunshape_covariance_sweep"

RENDER_SURFACE_POINTS = 50
RENDER_RAYS = 50
SAMPLE_INDEX = 8  # same well-aimed sample as ba72_contour_real_sample.py

# Sweep in std (mrad); ARTIST's default is ~2.09 mrad (covariance 4.3681e-06).
STD_MRAD_SWEEP = [2.09, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]


def load_stage1_kinematics(kinematic, device: torch.device) -> None:
    ckpt = torch.load(STAGE1_CHECKPOINT, map_location=device)
    if ckpt.get("heliostat_id") not in (None, HELIOSTAT_ID):
        raise ValueError(f"Checkpoint belongs to {ckpt.get('heliostat_id')!r}, not {HELIOSTAT_ID!r}")
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


def contour_pixels(contour_img: np.ndarray, threshold: float = 0.05):
    rows, cols = np.nonzero(contour_img > threshold)
    return cols, rows


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

    train_data, val_data, test_data, _ = _load_fixed_split_real(
        HELIOSTAT_ID, cfg, hg, scenario, device, hel_idx=hel_idx, n_hel=n_hel
    )
    flux, centroids, rays, motor_pos, _active_mask, target_mask = test_data
    i = SAMPLE_INDEX
    gt_flux = flux[i].float().cpu()
    ray_i = rays[i : i + 1]
    motor_i = motor_pos[i : i + 1]
    target_i = target_mask[i : i + 1]

    extractor = ContourExtractor(
        tau=cfg.CONTOUR_TAU, eta=cfg.CONTOUR_ETA, smoothing_rounds=cfg.CONTOUR_SMOOTHING_ROUNDS,
        gaussian_sigma=cfg.CONTOUR_GAUSS_SIGMA, gaussian_kernel_size=cfg.CONTOUR_GAUSS_KSIZE,
    )
    gt_steps = dict(extractor.intermediate_steps(gt_flux))
    gt_w, gt_h, gt_area = bbox_extent(gt_steps["Soft mask"])
    log.info(f"Measured soft-mask bbox: {gt_w:.0f}x{gt_h:.0f}px, area={gt_area:.0f}")

    active_mask = _one_hot_active(hel_idx, 1, n_hel, device)
    zero_bpd = torch.zeros(n_hel, 3, device=device)

    rows_data = []
    frames = {}
    for std_mrad in STD_MRAD_SWEEP:
        cov = (std_mrad / 1000.0) ** 2
        new_sun = Sun(
            number_of_rays=RENDER_RAYS,
            distribution_parameters={"distribution_type": "normal", "mean": 0.0, "covariance": cov},
            device=device,
        )
        scenario.light_sources.light_source_list[0] = new_sun
        scenario.set_number_of_rays(RENDER_RAYS)
        with torch.no_grad():
            _pred_cent, pred_flux = _forward_pass(
                scenario, hg, ray_i, active_mask, target_i, zero_bpd, device, motor_positions=motor_i,
            )
        pred_flux = pred_flux[0].float().cpu()
        if pred_flux.shape != gt_flux.shape:
            pred_flux = F.interpolate(
                pred_flux[None, None], size=tuple(gt_flux.shape), mode="bilinear", align_corners=False,
            )[0, 0]
        pred_steps = dict(extractor.intermediate_steps(pred_flux))
        pr_w, pr_h, pr_area = bbox_extent(pred_steps["Soft mask"])
        row = {
            "std_mrad": std_mrad, "covariance": cov,
            "pred_w": pr_w, "pred_h": pr_h, "pred_area": pr_area,
            "ratio_width": pr_w / gt_w if gt_w else None,
            "ratio_height": pr_h / gt_h if gt_h else None,
            "ratio_area": pr_area / gt_area if gt_area else None,
        }
        rows_data.append(row)
        frames[std_mrad] = (pred_flux.numpy(), pred_steps)
        log.info(f"std={std_mrad:5.2f} mrad: pred {pr_w:.0f}x{pr_h:.0f}px area={pr_area:.0f}  "
                 f"ratio(w/h/area)={row['ratio_width']:.2f}/{row['ratio_height']:.2f}/{row['ratio_area']:.2f}")

    with open(OUT_DIR / "sweep_results.json", "w") as fh:
        json.dump({"heliostat_id": HELIOSTAT_ID, "sample_index": i,
                    "measured": {"width_px": gt_w, "height_px": gt_h, "area_px": gt_area},
                    "rows": rows_data}, fh, indent=2)

    # ------------------------------------------------------------------
    # Figure 1: ratio-vs-std curves, with the crossing at ratio=1 marked.
    # ------------------------------------------------------------------
    stds = [r["std_mrad"] for r in rows_data]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for key, label, color in (("ratio_width", "width", "#1f78b4"),
                                ("ratio_height", "height", "#33a02c"),
                                ("ratio_area", "area", "#e31a1c")):
        vals = [r[key] for r in rows_data]
        ax.plot(stds, vals, "o-", label=f"{label} ratio (sim/measured)", color=color)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", label="ratio = 1 (match)")
    ax.axvline(2.09, color="0.5", linewidth=1, linestyle=":", label="ARTIST default (2.09 mrad)")
    ax.set_xlabel("sunshape std dev [mrad]")
    ax.set_ylabel("simulated / measured soft-mask size ratio")
    ax.set_title(f"{HELIOSTAT_ID} real sample {i}: simulated footprint size vs. sunshape std")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ratio_vs_sunshape_std.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: soft-mask panels across the sweep, plus the measured one.
    # ------------------------------------------------------------------
    n_panels = len(STD_MRAD_SWEEP) + 1
    ncols = 4
    nrows = -(-n_panels // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows))
    axes = np.atleast_2d(axes)
    ax0 = axes.flat[0]
    ax0.imshow(gt_steps["Soft mask"], cmap="hot", vmin=0, vmax=1)
    ax0.set_title(f"MEASURED\n{gt_w:.0f}x{gt_h:.0f}px, area={gt_area:.0f}", fontsize=9)
    ax0.set_xticks([]); ax0.set_yticks([])
    for k, std_mrad in enumerate(STD_MRAD_SWEEP, start=1):
        ax = axes.flat[k]
        _flux, steps = frames[std_mrad]
        row = rows_data[k - 1]
        ax.imshow(steps["Soft mask"], cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"std={std_mrad:.2f} mrad\narea ratio={row['ratio_area']:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(n_panels, nrows * ncols):
        axes.flat[k].axis("off")
    fig.suptitle(f"{HELIOSTAT_ID} sample {i}: simulated soft mask across the sunshape sweep", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "softmask_panels_across_sweep.png", dpi=140)
    plt.close(fig)

    log.info(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
