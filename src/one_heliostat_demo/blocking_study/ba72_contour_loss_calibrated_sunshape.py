"""GT vs. predicted (Stage-1 kinematics) contour loss, with the CALIBRATED
sunshape std (3.36 mrad, from ba72_sunshape_ideal_aim_calibration.py's
bbox-threshold method -- the metric consistent with what contour loss's own
thresholding pipeline actually operates on) instead of ARTIST's default
(2.09 mrad).

Same real BA72 test sample used throughout this study (test index 8, the
best-aimed real sample, s1_centroid_mrad ~0.26). Produces a 2x2 grid (row 1 =
measured/GT: flux + upper contour; row 2 = predicted, Stage-1 kinematics +
calibrated sunshape: flux + upper contour) plus the actual WortbergContourLoss
value (coarse/fine/gravity/weighted total) between the two.

Usage
-----
    python ba72_contour_loss_calibrated_sunshape.py
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
from artist_extensions.contour_loss import (  # noqa: E402
    ContourExtractor,
    WortbergContourLoss,
    build_contour_ground_truth,
)

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
SAMPLE_INDEX = 8  # same well-aimed sample used throughout
CALIBRATED_SUNSHAPE_STD_MRAD = 3.36  # ba72_sunshape_ideal_aim_calibration.py, bbox-threshold method
DEFAULT_SUNSHAPE_STD_MRAD = 2.09  # ARTIST's default, kept for the side-by-side comparison


def load_stage1_kinematics(kinematic, device: torch.device) -> None:
    ckpt = torch.load(STAGE1_CHECKPOINT, map_location=device)
    if ckpt.get("heliostat_id") not in (None, HELIOSTAT_ID):
        raise ValueError(f"Checkpoint belongs to {ckpt.get('heliostat_id')!r}, not {HELIOSTAT_ID!r}")
    kinematic.translation_deviation_parameters.data.copy_(ckpt["translation"].to(device))
    kinematic.rotation_deviation_parameters.data.copy_(ckpt["rotation"].to(device))
    kinematic.actuators.optimizable_parameters.data.copy_(ckpt["act_angle"].to(device))
    kinematic.actuators.non_optimizable_parameters.data.copy_(ckpt["act_offset"].to(device))
    kinematic._base_position_deviation = ckpt["base_pos"].clone().to(device)


def render_with_sunshape(scenario, hg, active_mask, ray_i, motor_i, target_i, zero_bpd, device, std_mrad):
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


def contour_pixels(contour_img: np.ndarray, threshold: float = 0.05):
    rows, cols = np.nonzero(contour_img > threshold)
    return cols, rows


def compute_loss(gt_flux, pred_flux, extractor, scenario, target_i, device):
    bitmap_resolution = torch.tensor([pred_flux.shape[1], pred_flux.shape[0]], dtype=torch.long, device=device)
    gt_side = build_contour_ground_truth(
        gt_flux[None].to(device), extractor, bitmap_resolution, scenario.solar_tower, target_i, device,
    )
    loss_fn = WortbergContourLoss(extractor, weight_coarse=cfg.CONTOUR_BETA, weight_gravity=cfg.CONTOUR_GAMMA)
    with torch.no_grad():
        total, comp = loss_fn(
            pred_flux[None].to(device), gt_side.contours, gt_side.distance_maps,
            gt_side.com_enu, target_i, bitmap_resolution, scenario.solar_tower, device,
        )
    return float(total.item()), comp


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
    target_name = {v: k for k, v in scenario.solar_tower.target_name_to_index.items()}[int(target_i.item())]

    active_mask = _one_hot_active(hel_idx, 1, n_hel, device)
    zero_bpd = torch.zeros(n_hel, 3, device=device)

    pred_flux_default = render_with_sunshape(
        scenario, hg, active_mask, ray_i, motor_i, target_i, zero_bpd, device, DEFAULT_SUNSHAPE_STD_MRAD)
    pred_flux_calib = render_with_sunshape(
        scenario, hg, active_mask, ray_i, motor_i, target_i, zero_bpd, device, CALIBRATED_SUNSHAPE_STD_MRAD)
    for name, pf in (("default", pred_flux_default), ("calibrated", pred_flux_calib)):
        if pf.shape != gt_flux.shape:
            log.info(f"resizing predicted ({name}) {tuple(pf.shape)} -> GT {tuple(gt_flux.shape)}")
    if pred_flux_default.shape != gt_flux.shape:
        pred_flux_default = F.interpolate(
            pred_flux_default[None, None], size=tuple(gt_flux.shape), mode="bilinear", align_corners=False)[0, 0]
    if pred_flux_calib.shape != gt_flux.shape:
        pred_flux_calib = F.interpolate(
            pred_flux_calib[None, None], size=tuple(gt_flux.shape), mode="bilinear", align_corners=False)[0, 0]

    extractor = ContourExtractor(
        tau=cfg.CONTOUR_TAU, eta=cfg.CONTOUR_ETA, smoothing_rounds=cfg.CONTOUR_SMOOTHING_ROUNDS,
        gaussian_sigma=cfg.CONTOUR_GAUSS_SIGMA, gaussian_kernel_size=cfg.CONTOUR_GAUSS_KSIZE,
    )
    gt_steps = dict(extractor.intermediate_steps(gt_flux))
    pred_steps_default = dict(extractor.intermediate_steps(pred_flux_default))
    pred_steps_calib = dict(extractor.intermediate_steps(pred_flux_calib))

    total_default, comp_default = compute_loss(gt_flux, pred_flux_default, extractor, scenario, target_i, device)
    total_calib, comp_calib = compute_loss(gt_flux, pred_flux_calib, extractor, scenario, target_i, device)
    log.info(f"DEFAULT sunshape ({DEFAULT_SUNSHAPE_STD_MRAD} mrad): total={total_default:.4f}  {comp_default}")
    log.info(f"CALIBRATED sunshape ({CALIBRATED_SUNSHAPE_STD_MRAD} mrad): total={total_calib:.4f}  {comp_calib}")

    with open(OUT_DIR / "contour_loss_calibrated_vs_default.json", "w") as fh:
        json.dump({
            "heliostat_id": HELIOSTAT_ID, "sample_index": i, "target_area_name": target_name,
            "default_sunshape_std_mrad": DEFAULT_SUNSHAPE_STD_MRAD, "default_loss": {"total": total_default, **comp_default},
            "calibrated_sunshape_std_mrad": CALIBRATED_SUNSHAPE_STD_MRAD, "calibrated_loss": {"total": total_calib, **comp_calib},
        }, fh, indent=2)

    # ------------------------------------------------------------------
    # 2x2 grid: row 1 = GT (flux, upper contour); row 2 = predicted with the
    # CALIBRATED sunshape (flux, upper contour). Loss value annotated below.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10.5))

    axes[0, 0].imshow(gt_flux.numpy(), cmap="inferno", vmin=0, vmax=float(gt_flux.max()))
    axes[0, 0].set_title("GT (measured) flux", fontsize=12)
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])

    gt_contour_vmax = float(gt_steps["Upper contour"].max()) or 1.0
    axes[0, 1].imshow(gt_steps["Upper contour"], cmap="hot", vmin=0, vmax=gt_contour_vmax)
    axes[0, 1].set_title("GT upper contour", fontsize=12)
    axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])

    axes[1, 0].imshow(pred_flux_calib.numpy(), cmap="inferno", vmin=0, vmax=float(pred_flux_calib.max()))
    axes[1, 0].set_title(f"Predicted flux (Stage-1 kinematics,\nsunshape std={CALIBRATED_SUNSHAPE_STD_MRAD} mrad)", fontsize=12)
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])

    pred_contour_vmax = float(pred_steps_calib["Upper contour"].max()) or 1.0
    axes[1, 1].imshow(pred_steps_calib["Upper contour"], cmap="hot", vmin=0, vmax=pred_contour_vmax)
    axes[1, 1].set_title("Predicted upper contour", fontsize=12)
    axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])

    loss_text = (
        f"WortbergContourLoss (beta={cfg.CONTOUR_BETA}, gamma={cfg.CONTOUR_GAMMA}):  "
        f"coarse={comp_calib['coarse']:.3f}   fine={comp_calib['fine']:.4f}   "
        f"gravity={comp_calib['gravity']:.4f} m   |   weighted total = {total_calib:.4f}"
    )
    fig.suptitle(
        f"{HELIOSTAT_ID} real test sample {i} (target={target_name}), calibrated sunshape\n{loss_text}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out1 = OUT_DIR / "contour_loss_calibrated_sunshape.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    log.info(f"wrote {out1}")

    # ------------------------------------------------------------------
    # Companion figure: same grid but with the DEFAULT (uncalibrated)
    # sunshape, for a direct before/after comparison.
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(10.5, 10.5))
    axes2[0, 0].imshow(gt_flux.numpy(), cmap="inferno", vmin=0, vmax=float(gt_flux.max()))
    axes2[0, 0].set_title("GT (measured) flux", fontsize=12)
    axes2[0, 0].set_xticks([]); axes2[0, 0].set_yticks([])
    axes2[0, 1].imshow(gt_steps["Upper contour"], cmap="hot", vmin=0, vmax=gt_contour_vmax)
    axes2[0, 1].set_title("GT upper contour", fontsize=12)
    axes2[0, 1].set_xticks([]); axes2[0, 1].set_yticks([])
    axes2[1, 0].imshow(pred_flux_default.numpy(), cmap="inferno", vmin=0, vmax=float(pred_flux_default.max()))
    axes2[1, 0].set_title(f"Predicted flux (Stage-1 kinematics,\nsunshape std={DEFAULT_SUNSHAPE_STD_MRAD} mrad, ARTIST default)", fontsize=12)
    axes2[1, 0].set_xticks([]); axes2[1, 0].set_yticks([])
    pred_default_contour_vmax = float(pred_steps_default["Upper contour"].max()) or 1.0
    axes2[1, 1].imshow(pred_steps_default["Upper contour"], cmap="hot", vmin=0, vmax=pred_default_contour_vmax)
    axes2[1, 1].set_title("Predicted upper contour", fontsize=12)
    axes2[1, 1].set_xticks([]); axes2[1, 1].set_yticks([])
    loss_text_default = (
        f"WortbergContourLoss (beta={cfg.CONTOUR_BETA}, gamma={cfg.CONTOUR_GAMMA}):  "
        f"coarse={comp_default['coarse']:.3f}   fine={comp_default['fine']:.4f}   "
        f"gravity={comp_default['gravity']:.4f} m   |   weighted total = {total_default:.4f}"
    )
    fig2.suptitle(
        f"{HELIOSTAT_ID} real test sample {i} (target={target_name}), DEFAULT (uncalibrated) sunshape\n{loss_text_default}",
        fontsize=11,
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    out2 = OUT_DIR / "contour_loss_default_sunshape.png"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    log.info(f"wrote {out2}")


if __name__ == "__main__":
    main()
