"""Sunshape-covariance crossing point, computed across EVERY real BA72 sample.

Follow-up to ba72_sunshape_covariance_sweep.py, which found a std ~4.0 mrad
closes the simulated-vs-measured spot-size gap on ONE well-aimed sample. This
checks whether that crossing point is stable across BA72's full real dataset
(train+val+test, 89 samples) rather than a one-sample fluke.

Samples where the simulated beam simply misses the frame (large aim error at
Stage-1 alone) are excluded from the crossing-point statistics -- a size
comparison is meaningless when the two blobs aren't even in the same place --
but the exclusion count and threshold are reported explicitly, not silently
dropped.

Usage
-----
    python ba72_sunshape_sweep_all_samples.py
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
OUT_DIR = _ROOT / "outputs" / "new_mapping_function" / "ba72_real_data" / "sunshape_sweep_all_samples"

RENDER_SURFACE_POINTS = 50
RENDER_RAYS = 50
STD_MRAD_SWEEP = [2.09, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0]
# Aim-quality gate: exclude samples whose default-sunshape predicted centroid
# lands more than this many pixels from the measured centroid (out of a
# 256x256 frame) -- these are Stage-1 aim failures, not a size-comparable pair.
PIXEL_MISS_THRESHOLD = 40.0


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

    train_data, val_data, test_data, _ = _load_fixed_split_real(
        HELIOSTAT_ID, cfg, hg, scenario, device, hel_idx=hel_idx, n_hel=n_hel
    )
    all_samples = []  # (split_name, global_index, flux, ray, motor, target)
    for split_name, data in (("train", train_data), ("val", val_data), ("test", test_data)):
        flux, centroids, rays, motor_pos, _am, target_mask = data
        for k in range(flux.shape[0]):
            all_samples.append((split_name, k, flux[k].float().cpu(), rays[k : k + 1],
                                 motor_pos[k : k + 1], target_mask[k : k + 1]))
    log.info(f"Total real samples: {len(all_samples)} "
             f"(train {train_data[0].shape[0]}, val {val_data[0].shape[0]}, test {test_data[0].shape[0]})")

    extractor = ContourExtractor(
        tau=cfg.CONTOUR_TAU, eta=cfg.CONTOUR_ETA, smoothing_rounds=cfg.CONTOUR_SMOOTHING_ROUNDS,
        gaussian_sigma=cfg.CONTOUR_GAUSS_SIGMA, gaussian_kernel_size=cfg.CONTOUR_GAUSS_KSIZE,
    )
    active_mask = _one_hot_active(hel_idx, 1, n_hel, device)
    zero_bpd = torch.zeros(n_hel, 3, device=device)

    results = []
    excluded = []
    for split_name, k, gt_flux, ray_i, motor_i, target_i in all_samples:
        gt_steps = dict(extractor.intermediate_steps(gt_flux))
        gt_w, gt_h, gt_area = bbox_extent(gt_steps["Soft mask"])
        gt_col, gt_row = _bitmap_centroid(gt_flux)

        # Aim-quality gate at ARTIST's default sunshape.
        pred_default = render_at_std(scenario, hg, ray_i, target_i, motor_i, active_mask, zero_bpd, device, 2.09)
        if pred_default.shape != gt_flux.shape:
            pred_default = F.interpolate(
                pred_default[None, None], size=tuple(gt_flux.shape), mode="bilinear", align_corners=False,
            )[0, 0]
        pred_col, pred_row = _bitmap_centroid(pred_default)
        if gt_col is None or pred_col is None:
            excluded.append({"split": split_name, "index": k, "reason": "empty flux"})
            continue
        miss_px = float(np.hypot(gt_col - pred_col, gt_row - pred_row))
        if miss_px > PIXEL_MISS_THRESHOLD or gt_area == 0:
            excluded.append({"split": split_name, "index": k, "reason": "aim miss", "miss_px": miss_px})
            continue

        # Full sweep for this sample (reusing the default-std render for std=2.09).
        ratios = []
        for std_mrad in STD_MRAD_SWEEP:
            if std_mrad == 2.09:
                pred_flux = pred_default
            else:
                pred_flux = render_at_std(scenario, hg, ray_i, target_i, motor_i, active_mask, zero_bpd, device, std_mrad)
                if pred_flux.shape != gt_flux.shape:
                    pred_flux = F.interpolate(
                        pred_flux[None, None], size=tuple(gt_flux.shape), mode="bilinear", align_corners=False,
                    )[0, 0]
            pred_steps = dict(extractor.intermediate_steps(pred_flux))
            _pr_w, _pr_h, pr_area = bbox_extent(pred_steps["Soft mask"])
            ratios.append(pr_area / gt_area if gt_area else 0.0)

        # Interpolate the std at which ratio_area crosses 1.0 (ratios assumed
        # increasing in std, as established on the single-sample sweep).
        ratios_arr = np.array(ratios)
        stds_arr = np.array(STD_MRAD_SWEEP)
        if ratios_arr[-1] < 1.0:
            crossing_std = None  # never reaches parity even at the top of the sweep
        elif ratios_arr[0] > 1.0:
            crossing_std = float(stds_arr[0])  # already past parity at the default
        else:
            crossing_std = float(np.interp(1.0, ratios_arr, stds_arr))

        results.append({
            "split": split_name, "index": k, "miss_px": miss_px,
            "gt_w": gt_w, "gt_h": gt_h, "gt_area": gt_area,
            "ratios_area": ratios, "crossing_std_mrad": crossing_std,
        })
        log.info(f"[{split_name}:{k}] miss_px={miss_px:.1f}  crossing_std={crossing_std}")

    with open(OUT_DIR / "all_samples_results.json", "w") as fh:
        json.dump({
            "heliostat_id": HELIOSTAT_ID,
            "n_total_samples": len(all_samples),
            "n_excluded": len(excluded),
            "pixel_miss_threshold": PIXEL_MISS_THRESHOLD,
            "std_mrad_sweep": STD_MRAD_SWEEP,
            "excluded": excluded,
            "results": results,
        }, fh, indent=2)

    crossings = [r["crossing_std_mrad"] for r in results if r["crossing_std_mrad"] is not None]
    n_no_cross = sum(1 for r in results if r["crossing_std_mrad"] is None)
    log.info(f"n_total={len(all_samples)}  n_excluded(aim miss)={len(excluded)}  "
             f"n_analyzed={len(results)}  n_no_crossing_in_range={n_no_cross}  n_with_crossing={len(crossings)}")
    if crossings:
        arr = np.array(crossings)
        log.info(f"crossing std (mrad): mean={arr.mean():.2f} median={np.median(arr):.2f} "
                  f"std={arr.std():.2f} min={arr.min():.2f} max={arr.max():.2f}")

    # ------------------------------------------------------------------
    # Figure: histogram of per-sample crossing std, plus all ratio curves.
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    if crossings:
        ax1.hist(crossings, bins=15, color="#1f78b4", edgecolor="black")
        ax1.axvline(np.median(crossings), color="red", linestyle="--",
                     label=f"median={np.median(crossings):.2f} mrad")
        ax1.axvline(4.0, color="0.4", linestyle=":", label="single-sample result (4.0 mrad)")
    ax1.set_xlabel("per-sample crossing std [mrad]")
    ax1.set_ylabel("count")
    ax1.set_title(f"Crossing-std distribution (n={len(crossings)}/{len(results)} analyzed, "
                   f"{len(excluded)} excluded as aim-miss)")
    ax1.legend(fontsize=9)

    for r in results:
        color = "#e31a1c" if r["crossing_std_mrad"] is None else "#1f78b4"
        alpha = 0.8 if r["crossing_std_mrad"] is None else 0.25
        ax2.plot(STD_MRAD_SWEEP, r["ratios_area"], color=color, alpha=alpha, linewidth=1)
    ax2.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax2.set_xlabel("sunshape std dev [mrad]")
    ax2.set_ylabel("area ratio (sim/measured)")
    ax2.set_title("Every analyzed sample's ratio-vs-std curve\n(red = never reaches parity in this range)")
    fig.suptitle(f"{HELIOSTAT_ID}: sunshape-covariance crossing point across all real samples")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "crossing_std_distribution.png", dpi=150)
    plt.close(fig)

    log.info(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
