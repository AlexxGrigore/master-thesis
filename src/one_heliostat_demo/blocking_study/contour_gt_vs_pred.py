"""Visualize how the upper-contour loss sees GT vs prediction on ONE blocked sample.

For one simulated sample of heliostat AY36 whose flux is partially blocked by
neighbours, this script renders the predicted flux WITH blocking (trained
kinematics applied), runs ``ContourExtractor.intermediate_steps()`` on both the
ground-truth and predicted flux images, computes the three contour-loss terms
(coarse / fine / gravity) and saves:

    outputs/new_mapping_function/blocking_study/contour_loss_analysis/AY36/
        contour_pipeline_comparison.png
        contour_overlay.png
        data/summary.json
        README.md   (written by hand after inspection)

Run from anywhere:
    python src/one_heliostat_demo/blocking_study/contour_gt_vs_pred.py
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

from artist.util import indices  # noqa: E402

from artist_extensions.contour_loss import (  # noqa: E402
    ContourExtractor,
    WortbergContourLoss,
    build_contour_ground_truth,
)
from blocking_utils import forward_pass_blocking  # noqa: E402
from one_heliostat_demo.single_heliostat.train import _load_scenario  # noqa: E402

log = logging.getLogger("contour_gt_vs_pred")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HELIOSTAT_ID = "AY36"
SCENARIO_PATH = _ROOT / "scenarios" / "full_benchmark_ideal" / "ideal_1277_AY36_deflectometry.h5"
BLOCKER_TARGET_NAME = "solar_tower_juelich_lower"
DATASET_DIR = _ROOT / "datasets" / "synthetic" / "fullfield_blocking_dataset" / "dataset"
SPLIT = "test"
EXPERIMENT_DIR = (
    _ROOT / "outputs" / "new_mapping_function" / "blocking_study" / "experiment_full_field_long"
)
BLOCKED_FRACTIONS_JSON = EXPERIMENT_DIR / HELIOSTAT_ID / "test_blocked_fractions.json"
KINEMATICS_CANDIDATES = [
    EXPERIMENT_DIR / HELIOSTAT_ID / "B1_blocking_on_long" / "kinematic_parameters.json",
    EXPERIMENT_DIR / HELIOSTAT_ID / "B0_blocking_off_long" / "kinematic_parameters.json",
    EXPERIMENT_DIR / HELIOSTAT_ID / "_backup_pre_stage1_logging_fix"
        / "B1_blocking_on_long" / "kinematic_parameters.json",
    EXPERIMENT_DIR / HELIOSTAT_ID / "_backup_pre_stage1_logging_fix"
        / "B0_blocking_off_long" / "kinematic_parameters.json",
]
OUT_DIR = (
    _ROOT / "outputs" / "new_mapping_function" / "blocking_study"
    / "contour_loss_analysis" / HELIOSTAT_ID
)
N_RAYS = 100  # rays per surface point — matches the GT dataset generation setting

# Contour-loss weights (config.py of single_heliostat training)
CONTOUR_BETA = 1e-4
CONTOUR_GAMMA = 0.3

cfg = SimpleNamespace(
    SCENARIO_PATH_TEMPLATE=str(SCENARIO_PATH),
    SURFACE_POINTS_PER_FACET=25,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pick_sample() -> tuple[str, float]:
    """Return (sample_dir_name, blocked_fraction) for the most-blocked sample."""
    data = json.load(open(BLOCKED_FRACTIONS_JSON))
    fracs = data["blocked_fractions"]
    i = int(np.argmax(fracs))
    return f"{i:04d}", float(fracs[i])


def load_sample(sample_dir: pathlib.Path, device: torch.device):
    """GT flux [H, W] float 0..1, sun direction [1, 4], motors [1, 2], target idx."""
    cal = json.load(open(sample_dir / "calibration_properties.json"))
    img = Image.open(sample_dir / "flux_image.png").convert("L")
    gt_flux = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    rays = torch.tensor([cal["incident_ray_direction"]], dtype=torch.float32, device=device)
    motors = torch.tensor([cal["motor_position"]], dtype=torch.float32, device=device)
    target_mask = torch.tensor([int(cal["target_area_index"])], dtype=torch.long, device=device)
    return gt_flux, rays, motors, target_mask, cal


def apply_kinematic_parameters(hg, hel_idx: int, params: dict, device: torch.device) -> torch.Tensor:
    """Additively apply trained deviation parameters to the studied row.

    Mirrors how train.py SAVED them: kinematic_parameters.json stores
    (trained − initial) deviations, so application = nominal + deviation.
    Returns the full-group base-position delta tensor for forward_pass_blocking.
    """
    kinematic = hg.kinematics
    with torch.no_grad():
        kinematic.rotation_deviation_parameters.data[hel_idx] += torch.tensor(
            params["rotation_dev_rad"], dtype=torch.float32, device=device)
        kinematic.translation_deviation_parameters.data[hel_idx] += torch.tensor(
            params["translation_dev_m"], dtype=torch.float32, device=device)
        kinematic.actuators.optimizable_parameters.data[
            hel_idx, indices.actuator_initial_angle, :] += torch.tensor(
            params["actuator_angle_dev_rad"], dtype=torch.float32, device=device)
        kinematic.actuators.optimizable_parameters.data[
            hel_idx, indices.actuator_initial_stroke_length, :] += torch.tensor(
            params["actuator_stroke_dev_m"], dtype=torch.float32, device=device)
        kinematic.actuators.non_optimizable_parameters.data[
            hel_idx, indices.actuator_offset, :] += torch.tensor(
            params["actuator_offset_dev_m"], dtype=torch.float32, device=device)
        kinematic.actuators.non_optimizable_parameters.data[
            hel_idx, indices.actuator_pivot_radius, :] += torch.tensor(
            params["pivot_radius_dev_m"], dtype=torch.float32, device=device)
    base_pos_delta = torch.zeros(hg.number_of_heliostats, 3, device=device)
    base_pos_delta[hel_idx] = torch.tensor(
        params["base_position_dev_m"], dtype=torch.float32, device=device)
    return base_pos_delta


def contour_pixels(contour_img: np.ndarray, threshold: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """(cols, rows) of pixels where the soft contour exceeds `threshold`."""
    rows, cols = np.nonzero(contour_img > threshold)
    return cols, rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    device = torch.device("cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "data").mkdir(exist_ok=True)

    # 1. Sample + GT
    sample_name, blocked_frac = pick_sample()
    sample_dir = DATASET_DIR / SPLIT / HELIOSTAT_ID / sample_name
    gt_flux, rays, motors, target_mask, cal = load_sample(sample_dir, device)
    log.info(f"Sample {sample_name}: blocked fraction {blocked_frac:.3f}, "
             f"target_area_index={int(target_mask.item())}, GT {tuple(gt_flux.shape)}")

    # 2. Kinematics
    kin_file = next(p for p in KINEMATICS_CANDIDATES if p.exists())
    kin_params = json.load(open(kin_file))
    log.info(f"Kinematics from: {kin_file}")

    # 3. Scenario + trained kinematics on AY36
    scenario, hg, _dist, hel_idx = _load_scenario(HELIOSTAT_ID, cfg, device)
    base_pos_delta = apply_kinematic_parameters(hg, hel_idx, kin_params, device)
    blocker_index = int(scenario.solar_tower.target_name_to_index[BLOCKER_TARGET_NAME])
    target_name_by_index = {v: k for k, v in scenario.solar_tower.target_name_to_index.items()}
    log.info(f"AY36 row {hel_idx}; blockers aim at '{BLOCKER_TARGET_NAME}' "
             f"(index {blocker_index}); sample target = "
             f"'{target_name_by_index[int(target_mask.item())]}'")

    # 4. Render predicted flux WITH blocking
    scenario.set_number_of_rays(N_RAYS)
    with torch.no_grad():
        _cents, pred_flux, pred_blocked = forward_pass_blocking(
            scenario, hg, hel_idx, rays, target_mask, device,
            motor_positions=motors,
            base_pos_delta=base_pos_delta,
            target_index_override=blocker_index,
        )
    pred_flux = pred_flux[0].float().cpu()
    log.info(f"Predicted flux {tuple(pred_flux.shape)}, "
             f"rendered blocked fraction {pred_blocked[0]:.3f}")

    # Resize predicted to GT size if the ray-tracer resolution differs
    resized = False
    if pred_flux.shape != gt_flux.shape:
        resized = True
        log.info(f"Resizing predicted {tuple(pred_flux.shape)} → GT {tuple(gt_flux.shape)} (bilinear)")
        pred_flux = F.interpolate(
            pred_flux[None, None], size=tuple(gt_flux.shape), mode="bilinear",
            align_corners=False,
        )[0, 0]

    # 5. Contour extraction on both
    extractor = ContourExtractor()
    gt_steps = extractor.intermediate_steps(gt_flux)
    pred_steps = extractor.intermediate_steps(pred_flux)
    gt_contour = torch.from_numpy(dict(gt_steps)["Upper contour"])[None]
    pred_contour = torch.from_numpy(dict(pred_steps)["Upper contour"])[None]

    # 6. Loss terms (batch of 1). GT side via the module's own builder.
    gt_batch = gt_flux[None].to(device)
    bitmap_resolution = torch.tensor(
        [pred_flux.shape[1], pred_flux.shape[0]], dtype=torch.long, device=device)  # (w, h)
    gt_side = build_contour_ground_truth(
        gt_batch, extractor, bitmap_resolution, scenario.solar_tower,
        target_mask, device,
    )
    loss_fn = WortbergContourLoss(
        extractor, weight_coarse=CONTOUR_BETA, weight_gravity=CONTOUR_GAMMA)
    pred_batch = pred_flux[None].to(device)
    with torch.no_grad():
        total, comp = loss_fn(
            pred_batch, gt_side.contours, gt_side.distance_maps, gt_side.com_enu,
            target_mask, bitmap_resolution, scenario.solar_tower, device,
        )
    log.info(f"Loss terms: {comp} | weighted total {float(total.item()):.4f}")

    # ------------------------------------------------------------------
    # Figure 1: pipeline comparison grid
    # ------------------------------------------------------------------
    col_names = ["Raw", "Denoised", "Soft mask", "Eroded", "Upper contour"]
    gt_dict, pred_dict = dict(gt_steps), dict(pred_steps)
    fig, axes = plt.subplots(2, len(col_names), figsize=(4.2 * len(col_names), 8.6))
    for r, (label, d) in enumerate((("Ground truth", gt_dict), ("Predicted", pred_dict))):
        for c, name in enumerate(col_names):
            ax = axes[r, c]
            img = d[name]
            vmax = float(np.max(img)) or 1.0
            ax.imshow(img, cmap="hot", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=12)
            if c == 0:
                ax.set_ylabel(label, fontsize=12)
    fig.suptitle(
        f"{HELIOSTAT_ID} sample {sample_name} — contour-extraction pipeline, GT vs predicted "
        f"(blocked fraction {blocked_frac:.1%})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "contour_pipeline_comparison.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: contour overlay on GT and predicted backgrounds
    # ------------------------------------------------------------------
    gt_c = gt_contour[0].numpy()
    pred_c = pred_contour[0].numpy()
    gt_cols, gt_rows = contour_pixels(gt_c)
    pr_cols, pr_rows = contour_pixels(pred_c)
    annotation = (
        f"blocked fraction (recorded) {blocked_frac:.1%}  |  rendered {pred_blocked[0]:.1%}\n"
        f"coarse {comp['coarse']:.2f}   fine {comp['fine']:.4f}   gravity {comp['gravity']:.4f} m\n"
        f"weighted total {float(total.item()):.4f}  (β={CONTOUR_BETA}, γ={CONTOUR_GAMMA})\n"
        f"kinematics: {kin_file.relative_to(_ROOT)}"
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
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
    fig.suptitle(
        f"{HELIOSTAT_ID} sample {sample_name} — upper contours, GT vs predicted\n{annotation}",
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
        "blocked_fraction_recorded": blocked_frac,
        "blocked_fraction_rendered": pred_blocked[0],
        "target_area_index": int(target_mask.item()),
        "target_area_name": target_name_by_index[int(target_mask.item())],
        "blocker_target_name": BLOCKER_TARGET_NAME,
        "blocker_target_index": blocker_index,
        "scenario": str(SCENARIO_PATH.relative_to(_ROOT)),
        "kinematics_file": str(kin_file.relative_to(_ROOT)),
        "n_rays_per_surface_point": N_RAYS,
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
        "incident_ray_direction": cal["incident_ray_direction"],
        "motor_position": cal["motor_position"],
    }
    with open(OUT_DIR / "data" / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
