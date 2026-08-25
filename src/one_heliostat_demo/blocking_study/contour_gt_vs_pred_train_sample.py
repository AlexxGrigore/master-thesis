"""Contour loss on a TRAINING sample, GT vs prediction, with STAGE-1 kinematics.

Companion to ``contour_gt_vs_pred.py`` (which used the fully trained B0
kinematics on a test sample). Here we render one train-split sample of AY36
with the post-Stage-1 kinematics from the B1 (blocking-on) long run, dense
(100×100 surface points per facet, 10 rays per point), and inspect whether the
contour extraction itself behaves well.

Sample selection: train samples whose recorded target is solar_tower_juelich_lower;
blocked fractions are estimated cheaply with the 15-heliostat neighbourhood
scenario (the scenario the dataset and the experiment actually use) and the
most-blocked candidate is used for the dense render.

Outputs (new folder, nothing overwritten):
    outputs/new_mapping_function/blocking_study/contour_loss_analysis/AY36/train_sample/
        contour_pipeline_comparison.png
        contour_overlay.png
        data/summary.json
        README.md (written by hand)

Run from anywhere:
    python src/one_heliostat_demo/blocking_study/contour_gt_vs_pred_train_sample.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_here = pathlib.Path(__file__).resolve().parent          # blocking_study/
_src = _here.parents[1]                                   # src/
_ROOT = _src.parent                                       # master-thesis/
_ARTIST = _ROOT.parent / "ARTIST"
for _p in (str(_src), str(_here), str(_ARTIST)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import contextlib

from artist.util import indices  # noqa: E402
from artist.raytracing import blocking as _artist_blocking  # noqa: E402


@contextlib.contextmanager
def chunked_blocking_mask(chunk_points: int = 4000):
    """Chunk ``soft_ray_blocking_mask`` over the surface-point dimension.

    The stock implementation broadcasts to
    [heliostats, rays, points, primitives, 3] — at 100² points/facet and 1276
    blocking primitives that is tens of GB and gets the process OOM-killed.
    The computation is elementwise per (ray, point, primitive), so slicing the
    points dimension and concatenating is exact, not approximate. Same
    monkeypatch pattern as ``brute_blocking.exact_blocking``.
    """
    original = _artist_blocking.soft_ray_blocking_mask

    def _wrapped(ray_origins, ray_directions, *args, **kwargs):
        n_points = ray_origins.shape[1]
        outs = [
            original(ray_origins[:, s : s + chunk_points],
                     ray_directions[:, :, s : s + chunk_points], *args, **kwargs)
            for s in range(0, n_points, chunk_points)
        ]
        return torch.cat(outs, dim=2)  # blocked: [heliostats, rays, points]

    _artist_blocking.soft_ray_blocking_mask = _wrapped
    try:
        yield
    finally:
        _artist_blocking.soft_ray_blocking_mask = original

from artist_extensions.contour_loss import (  # noqa: E402
    ContourExtractor,
    WortbergContourLoss,
    build_contour_ground_truth,
)
from blocking_utils import forward_pass_blocking  # noqa: E402
from one_heliostat_demo.single_heliostat.train import _load_scenario  # noqa: E402

log = logging.getLogger("contour_gt_vs_pred_train")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HELIOSTAT_ID = "AY36"
SCENARIO_PATH = _ROOT / "scenarios" / "full_benchmark_ideal" / "ideal_1277_AY36_deflectometry.h5"
NEIGHBOURHOOD_SCENARIO = _ROOT / "scenarios" / "neighbourhoods_fullfield" / HELIOSTAT_ID / "scenario.h5"
BLOCKER_TARGET_NAME = "solar_tower_juelich_lower"
DATASET_DIR = _ROOT / "datasets" / "synthetic" / "fullfield_blocking_dataset" / "dataset"
SPLIT = "train"
EXPERIMENT_DIR = (
    _ROOT / "outputs" / "new_mapping_function" / "blocking_study" / "experiment_full_field_long"
)
STAGE1_CHECKPOINTS = [
    EXPERIMENT_DIR / HELIOSTAT_ID / "B1_blocking_on_long" / "stage1_checkpoint.pt",
    EXPERIMENT_DIR / HELIOSTAT_ID / "B0_blocking_off_long" / "stage1_checkpoint.pt",
]
OUT_DIR = (
    _ROOT / "outputs" / "new_mapping_function" / "blocking_study"
    / "contour_loss_analysis" / HELIOSTAT_ID / "train_sample"
)

SCAN_RAYS = 5                     # cheap blocked-fraction scan (15-hel scenario)
SCAN_SURFACE_POINTS = 25
RENDER_SURFACE_POINTS = 100       # dense render, per user request
RENDER_RAYS = 10

CONTOUR_BETA = 1e-4
CONTOUR_GAMMA = 0.3

cfg = SimpleNamespace(SURFACE_POINTS_PER_FACET=25)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_sample(sample_dir: pathlib.Path, device: torch.device):
    cal = json.load(open(sample_dir / "calibration_properties.json"))
    img = Image.open(sample_dir / "flux_image.png").convert("L")
    gt_flux = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    rays = torch.tensor([cal["incident_ray_direction"]], dtype=torch.float32, device=device)
    motors = torch.tensor([cal["motor_position"]], dtype=torch.float32, device=device)
    target_mask = torch.tensor([int(cal["target_area_index"])], dtype=torch.long, device=device)
    return gt_flux, rays, motors, target_mask, cal


def load_stage1_kinematics(scenario_ref, hg_ref, hel_idx_ref: int, device: torch.device):
    """Load the post-Stage-1 checkpoint and return per-parameter DEVIATIONS.

    The checkpoint stores ABSOLUTE parameter tensors of the training scenario
    (15-heliostat neighbourhood), saved after restoring the best Stage-1 epoch.
    Deviations = checkpoint row − nominal row of a fresh load of the SAME
    scenario, so they can be applied additively to the 1277-heliostat scenario.
    Returns (deviation dict incl. base_position_dev_m, checkpoint path used).
    """
    kinematic = hg_ref.kinematics
    ckpt_path = next(p for p in STAGE1_CHECKPOINTS if p.exists())
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("heliostat_id") not in (None, HELIOSTAT_ID):
        raise ValueError(f"Checkpoint {ckpt_path} is for {ckpt.get('heliostat_id')!r}")

    def _dev(ckpt_key, nominal_tensor):
        return (ckpt[ckpt_key][hel_idx_ref].float()
                - nominal_tensor.detach().float().cpu()[hel_idx_ref])

    dev = {
        "rotation_dev_rad": _dev("rotation", kinematic.rotation_deviation_parameters),
        "translation_dev_m": _dev("translation", kinematic.translation_deviation_parameters),
        "act_opt_dev": _dev("act_angle", kinematic.actuators.optimizable_parameters),
        "act_nonopt_dev": _dev("act_offset", kinematic.actuators.non_optimizable_parameters),
        "base_position_dev_m": ckpt["base_pos"][hel_idx_ref].float().cpu(),
    }
    log.info(f"Stage-1 checkpoint: {ckpt_path}")
    log.info(f"  rotation dev (mrad): {[round(1000*v, 3) for v in dev['rotation_dev_rad'].tolist()]}")
    return dev, ckpt_path


def apply_kinematic_deviations(hg, hel_idx: int, dev: dict, device: torch.device) -> torch.Tensor:
    kinematic = hg.kinematics
    with torch.no_grad():
        kinematic.rotation_deviation_parameters.data[hel_idx] += dev["rotation_dev_rad"].to(device)
        kinematic.translation_deviation_parameters.data[hel_idx] += dev["translation_dev_m"].to(device)
        kinematic.actuators.optimizable_parameters.data[hel_idx] += dev["act_opt_dev"].to(device)
        kinematic.actuators.non_optimizable_parameters.data[hel_idx] += dev["act_nonopt_dev"].to(device)
    base_pos_delta = torch.zeros(hg.number_of_heliostats, 3, device=device)
    base_pos_delta[hel_idx] = dev["base_position_dev_m"].to(device)
    return base_pos_delta


def mask_edge(img: np.ndarray, level: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """(cols, rows) of the sigmoid soft-mask `level` crossing (4-neighbour)."""
    above = img > level
    edge = np.zeros_like(above)
    edge[:-1] |= above[:-1] != above[1:]
    edge[1:] |= above[:-1] != above[1:]
    edge[:, :-1] |= above[:, :-1] != above[:, 1:]
    edge[:, 1:] |= above[:, :-1] != above[:, 1:]
    rows, cols = np.nonzero(edge)
    return cols, rows


def contour_pixels(contour_img: np.ndarray, threshold: float = 0.05):
    rows, cols = np.nonzero(contour_img > threshold)
    return cols, rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    device = torch.device("cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "data").mkdir(exist_ok=True)

    # 1. Load the 15-heliostat neighbourhood scenario (training scenario) for
    #    the checkpoint reference and the cheap blocked-fraction scan.
    scenario_nb, hg_nb, _d, hel_idx_nb = _load_scenario(
        HELIOSTAT_ID, cfg, device, scenario_path=NEIGHBOURHOOD_SCENARIO)
    blocker_index_nb = int(scenario_nb.solar_tower.target_name_to_index[BLOCKER_TARGET_NAME])

    # 2. Stage-1 kinematics as deviations (B1 preferred, B0 fallback).
    dev, ckpt_path = load_stage1_kinematics(scenario_nb, hg_nb, hel_idx_nb, device)

    # 3. Scan train samples aimed at juelich_lower for blocked fraction
    #    (nominal kinematics — blocking is a geometric property of the field,
    #    the small kinematic deviations barely move it).
    split_root = DATASET_DIR / SPLIT / HELIOSTAT_ID
    candidates = []
    for p in sorted(split_root.iterdir()):
        if not p.is_dir():
            continue
        cal = json.load(open(p / "calibration_properties.json"))
        if int(cal["target_area_index"]) == blocker_index_nb:
            candidates.append(p)
    log.info(f"Scanning {len(candidates)} train samples aimed at '{BLOCKER_TARGET_NAME}'")

    scenario_nb.set_number_of_rays(SCAN_RAYS)
    scan: list[tuple[str, float]] = []
    zero_bpd = torch.zeros(hg_nb.number_of_heliostats, 3, device=device)
    with torch.no_grad():
        for p in candidates:
            _gt, rays, motors, target_mask, _c = load_sample(p, device)
            _cents, _flux, blocked = forward_pass_blocking(
                scenario_nb, hg_nb, hel_idx_nb, rays, target_mask, device,
                motor_positions=motors, base_pos_delta=zero_bpd,
                target_index_override=blocker_index_nb,
            )
            scan.append((p.name, blocked[0]))
            log.info(f"  {p.name}: blocked {blocked[0]:.3f}")
    scan.sort(key=lambda t: t[1], reverse=True)
    sample_name, scan_blocked = scan[0]
    log.info(f"Chosen sample {sample_name} (scan blocked fraction {scan_blocked:.3f})")

    # 4. Dense render with the full 1277-heliostat scenario, stage-1 kinematics.
    # Free the neighbourhood scenario first — 100² points/facet over 1277
    # heliostats is ~1 GB of surface tensors, and other jobs may be running.
    del scenario_nb, hg_nb
    import gc; gc.collect()

    sample_dir = split_root / sample_name
    gt_flux, rays, motors, target_mask, cal = load_sample(sample_dir, device)

    cfg_dense = SimpleNamespace(SURFACE_POINTS_PER_FACET=RENDER_SURFACE_POINTS)
    scenario, hg, _dist, hel_idx = _load_scenario(
        HELIOSTAT_ID, cfg_dense, device, scenario_path=SCENARIO_PATH)
    blocker_index = int(scenario.solar_tower.target_name_to_index[BLOCKER_TARGET_NAME])
    base_pos_delta = apply_kinematic_deviations(hg, hel_idx, dev, device)
    scenario.set_number_of_rays(RENDER_RAYS)
    log.info(f"Dense render: {RENDER_SURFACE_POINTS}² points/facet, "
             f"{RENDER_RAYS} rays/point, blocking on")
    with torch.no_grad(), chunked_blocking_mask():
        _cents, pred_flux, pred_blocked = forward_pass_blocking(
            scenario, hg, hel_idx, rays, target_mask, device,
            motor_positions=motors, base_pos_delta=base_pos_delta,
            target_index_override=blocker_index,
        )
    pred_flux = pred_flux[0].float().cpu()
    log.info(f"Predicted flux {tuple(pred_flux.shape)}, rendered blocked {pred_blocked[0]:.3f}")

    resized = False
    if pred_flux.shape != gt_flux.shape:
        resized = True
        log.info(f"Resizing predicted {tuple(pred_flux.shape)} → GT {tuple(gt_flux.shape)} (bilinear)")
        pred_flux = F.interpolate(
            pred_flux[None, None], size=tuple(gt_flux.shape), mode="bilinear",
            align_corners=False,
        )[0, 0]

    # 5. Contour extraction on both.
    extractor = ContourExtractor()
    gt_steps = extractor.intermediate_steps(gt_flux)
    pred_steps = extractor.intermediate_steps(pred_flux)
    gt_dict, pred_dict = dict(gt_steps), dict(pred_steps)
    gt_contour = torch.from_numpy(gt_dict["Upper contour"])[None]
    pred_contour = torch.from_numpy(pred_dict["Upper contour"])[None]

    # 6. Loss terms (batch of 1).
    bitmap_resolution = torch.tensor(
        [pred_flux.shape[1], pred_flux.shape[0]], dtype=torch.long, device=device)
    gt_side = build_contour_ground_truth(
        gt_flux[None].to(device), extractor, bitmap_resolution, scenario.solar_tower,
        target_mask, device,
    )
    loss_fn = WortbergContourLoss(extractor, weight_coarse=CONTOUR_BETA, weight_gravity=CONTOUR_GAMMA)
    with torch.no_grad():
        total, comp = loss_fn(
            pred_flux[None].to(device), gt_side.contours, gt_side.distance_maps,
            gt_side.com_enu, target_mask, bitmap_resolution, scenario.solar_tower, device,
        )
    log.info(f"Loss terms: {comp} | weighted total {float(total.item()):.4f}")

    # Extra diagnostics for the critical evaluation.
    diag = {
        "gt_flux_minmax": [float(gt_flux.min()), float(gt_flux.max())],
        "pred_flux_minmax": [float(pred_flux.min()), float(pred_flux.max())],
        "gt_softmask_mass_frac": float((gt_dict["Soft mask"] > 0.5).mean()),
        "pred_softmask_mass_frac": float((pred_dict["Soft mask"] > 0.5).mean()),
        "gt_contour_mass": float(gt_contour.sum()),
        "pred_contour_mass": float(pred_contour.sum()),
        "gt_contour_pixels_gt_0.05": int((gt_contour.numpy() > 0.05).sum()),
        "pred_contour_pixels_gt_0.05": int((pred_contour.numpy() > 0.05).sum()),
        "gt_contour_max": float(gt_contour.max()),
        "pred_contour_max": float(pred_contour.max()),
    }
    log.info(f"Diagnostics: {json.dumps(diag, indent=1)}")

    # ------------------------------------------------------------------
    # Figure 1: pipeline comparison grid (shared scale per column).
    # ------------------------------------------------------------------
    col_names = ["Raw", "Denoised", "Soft mask", "Eroded", "Upper contour"]
    fig, axes = plt.subplots(2, len(col_names), figsize=(4.2 * len(col_names), 8.6))
    for c, name in enumerate(col_names):
        vmax = max(float(np.max(gt_dict[name])), float(np.max(pred_dict[name]))) or 1.0
        for r, (label, d) in enumerate((("Ground truth", gt_dict), ("Predicted", pred_dict))):
            ax = axes[r, c]
            ax.imshow(d[name], cmap="hot", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{name}\n(shared scale 0–{vmax:.2g})", fontsize=11)
            if c == 0:
                ax.set_ylabel(label, fontsize=12)
    fig.suptitle(
        f"{HELIOSTAT_ID} TRAIN sample {sample_name} — contour pipeline, GT vs predicted\n"
        f"stage-1 kinematics ({ckpt_path.parent.parent.name}), rendered blocked "
        f"{pred_blocked[0]:.1%}, dense render {RENDER_SURFACE_POINTS}²×{RENDER_RAYS}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "contour_pipeline_comparison.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: overlay + threshold diagnostic.
    # ------------------------------------------------------------------
    gt_c, pred_c = gt_contour[0].numpy(), pred_contour[0].numpy()
    gt_cols, gt_rows = contour_pixels(gt_c)
    pr_cols, pr_rows = contour_pixels(pred_c)
    gt_mc, gt_mr = mask_edge(gt_dict["Soft mask"], 0.5)
    pr_mc, pr_mr = mask_edge(pred_dict["Soft mask"], 0.5)

    annotation = (
        f"train sample {sample_name}  |  blocked: scan {scan_blocked:.1%}, "
        f"rendered {pred_blocked[0]:.1%}\n"
        f"coarse {comp['coarse']:.2f}   fine {comp['fine']:.4f}   "
        f"gravity {comp['gravity']:.4f} m   weighted total {float(total.item()):.4f}\n"
        f"kinematics: stage-1 checkpoint {ckpt_path.relative_to(_ROOT)}"
    )
    fig, axes = plt.subplots(1, 3, figsize=(22, 7.4))
    for ax, bg, title in (
        (axes[0], gt_flux.numpy(), "GT flux (background)"),
        (axes[1], pred_flux.numpy(), "Predicted flux (background)"),
    ):
        ax.imshow(bg, cmap="gray", vmin=0, vmax=1)
        ax.scatter(gt_cols, gt_rows, s=4, c="lime", label="GT upper contour")
        ax.scatter(pr_cols, pr_rows, s=4, c="red", label="Predicted upper contour")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="lower right", fontsize=9)
    # Panel 3: threshold diagnostic — soft-mask 0.5 outlines on GT background.
    axes[2].imshow(gt_flux.numpy(), cmap="gray", vmin=0, vmax=1)
    axes[2].scatter(gt_mc, gt_mr, s=2, c="lime", label="GT soft-mask 0.5 edge (τ level)")
    axes[2].scatter(pr_mc, pr_mr, s=2, c="red", label="Pred soft-mask 0.5 edge (τ level)")
    axes[2].scatter(gt_cols, gt_rows, s=6, c="cyan", marker="x", label="GT upper contour")
    axes[2].scatter(pr_cols, pr_rows, s=6, c="magenta", marker="+", label="Pred upper contour")
    axes[2].set_title("Threshold diagnostic (on GT background):\nwhere the sigmoid crosses 0.5", fontsize=12)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    axes[2].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        f"{HELIOSTAT_ID} TRAIN sample {sample_name} — upper contours & threshold behaviour\n{annotation}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT_DIR / "contour_overlay.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # summary.json
    # ------------------------------------------------------------------
    summary = {
        "heliostat_id": HELIOSTAT_ID,
        "split": SPLIT,
        "sample_id": sample_name,
        "blocked_fraction_scan_neighbourhood_scenario": scan_blocked,
        "blocked_fraction_rendered_full_scenario": pred_blocked[0],
        "scan_blocked_fractions_all_candidates": dict(scan),
        "target_area_index": int(target_mask.item()),
        "target_area_name": BLOCKER_TARGET_NAME,
        "blocker_target_name": BLOCKER_TARGET_NAME,
        "scenario_render": str(SCENARIO_PATH.relative_to(_ROOT)),
        "scenario_scan": str(NEIGHBOURHOOD_SCENARIO.relative_to(_ROOT)),
        "kinematics": {
            "stage": "stage1_only",
            "checkpoint": str(ckpt_path.relative_to(_ROOT)),
            "rotation_dev_rad": dev["rotation_dev_rad"].tolist(),
            "translation_dev_m": dev["translation_dev_m"].tolist(),
            "actuator_optimizable_dev": dev["act_opt_dev"].tolist(),
            "actuator_non_optimizable_dev": dev["act_nonopt_dev"].tolist(),
            "base_position_dev_m": dev["base_position_dev_m"].tolist(),
        },
        "render_settings": {
            "surface_points_per_facet": RENDER_SURFACE_POINTS,
            "rays_per_point": RENDER_RAYS,
        },
        "gt_image_size_hw": list(gt_flux.shape),
        "predicted_image_size_hw_after_resize": list(pred_flux.shape),
        "predicted_resized_to_gt": resized,
        "contour_loss_weights": {"beta_coarse": CONTOUR_BETA, "gamma_gravity": CONTOUR_GAMMA},
        "loss_terms": {
            "coarse_soft_distance_field": comp["coarse"],
            "fine_soft_dice": comp["fine"],
            "gravity_com_distance_m": comp["gravity"],
            "weighted_total": float(total.item()),
        },
        "diagnostics": diag,
        "incident_ray_direction": cal["incident_ray_direction"],
        "motor_position": cal["motor_position"],
    }
    with open(OUT_DIR / "data" / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
