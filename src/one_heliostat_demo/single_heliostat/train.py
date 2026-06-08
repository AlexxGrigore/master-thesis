"""
Two-stage kinematic reconstruction training for a single heliostat.

Direct port of one_heliostat_training_demo.ipynb (Cells 6-31).

Stage 1 — AlignmentLoss (motor-position MSE, no ray tracing, fast)
Stage 2 — FocalSpotLoss (ray-traced centroid MSE, mini-batched)

Saves to output_dir/:
    results.json               — pre/after-S1/after-S2 mrad metrics
    perturbations.json         — copy of GT perturbations
    convergence_history.csv    — epoch, stage, train_loss, val_loss, mrad_train_mean, ...
    kinematic_parameters.json  — final trained parameter values
    kinematic_history.json     — per-epoch parameter trajectory
    metrics_table.txt          — ASCII accuracy table
    plots/
        mrad_convergence.png
        loss_curves.png
        param_trajectories.png (only when GT perturbations are available)
        trail_morning.png
        trail_noon.png
        trail_afternoon.png
        test_flux/
            sample_NNNN.png
"""

import csv
import json
import logging
import pathlib
import sys
import time

import h5py
import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

_here     = pathlib.Path(__file__).resolve().parent   # single_heliostat/
_src      = _here.parent.parent                       # src/
_paint    = _src.parent.parent / "PAINT"              # Master Thesis/PAINT/
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_here))
if _paint.exists():
    sys.path.insert(0, str(_paint))

from artist.optim.loss import FocalSpotLoss
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario
from artist.util import constants as _const, get_device, indices, set_logger_config
from artist.util import setup_distributed_environment

from artist_extensions.loss_functions_ext import AlignmentLoss
from utils.synth_data import _forward_pass, SyntheticDatasetParser

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers: scenario / data loading
# ---------------------------------------------------------------------------

def _load_scenario(heliostat_id: str, cfg, device: torch.device):
    scenario_path = pathlib.Path(cfg.SCENARIO_PATH_TEMPLATE.format(heliostat_id=heliostat_id))
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario not found: {scenario_path}")

    with h5py.File(scenario_path, "r") as fh:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=fh,
            device=device,
            number_of_surface_points_per_facet=torch.tensor(
                [cfg.SURFACE_POINTS_PER_FACET, cfg.SURFACE_POINTS_PER_FACET]
            ),
        )

    hg = scenario.heliostat_field.heliostat_groups[0]
    hel_pos      = hg.positions[0, :3].float()
    target_areas = scenario.solar_tower.target_areas
    target_ctrs  = target_areas[indices.planar_target_areas].centers[:, :3].float()
    tower_ref    = target_ctrs.mean(dim=0)
    hel_dist_m   = torch.norm(hel_pos - tower_ref.to(device)).item()

    log.info(f"Scenario: {scenario_path}")
    log.info(f"Heliostat {heliostat_id}  |  dist-to-tower = {hel_dist_m:.1f} m")
    return scenario, hg, hel_dist_m


def _load_split(
    heliostat_id: str,
    dataset_dir: pathlib.Path,
    split: str,
    heliostat_group,
    scenario,
    device: torch.device,
):
    """Return (flux, centroids, rays, motor_pos, active_mask, target_mask) or None."""
    split_dir = dataset_dir / split / heliostat_id
    if not split_dir.exists():
        return None

    n_samples = sum(1 for d in split_dir.iterdir() if d.is_dir() and d.name.isdigit())
    if n_samples == 0:
        return None

    parser  = SyntheticDatasetParser(dataset_dir / split)
    mapping = [(heliostat_id, list(range(n_samples)), list(range(n_samples)))]
    return parser.parse_data_for_reconstruction(
        heliostat_data_mapping=mapping,
        heliostat_group=heliostat_group,
        scenario=scenario,
        device=device,
    )



def _load_split_real(
    heliostat_id: str,
    cfg,
    paint_split: str,
    hg,
    scenario,
    device: torch.device,
):
    """Load one PAINT benchmark split for a single heliostat using real flux images.

    paint_split is the PAINT convention name: "train", "validation", or "test".
    Returns the same 6-tuple as _load_split, or None if the heliostat is absent.
    """
    from utils.evaluation import build_heliostat_data_mapping
    from artist.io.paint_calibration_parser import PaintCalibrationDataParser

    full_mapping = build_heliostat_data_mapping(
        pathlib.Path(cfg.BENCHMARK_CSV),
        pathlib.Path(cfg.CALIBRATION_DIR),
        pathlib.Path(cfg.REAL_FLUX_DIR),
        paint_split,
    )
    hel_mapping = [(h, c, f) for h, c, f in full_mapping if h == heliostat_id]
    if not hel_mapping:
        log.warning(f"  {heliostat_id} not found in PAINT {paint_split} split")
        return None

    log.info(f"  PAINT {paint_split}: {len(hel_mapping[0][1])} samples for {heliostat_id}")
    parser = PaintCalibrationDataParser(
        centroid_extraction_method=getattr(cfg, "CENTROID_METHOD", "UTIS"),
    )
    return parser.parse_data_for_reconstruction(
        heliostat_data_mapping=hel_mapping,
        heliostat_group=hg,
        scenario=scenario,
        device=device,
    )


def _pool_and_split(
    heliostat_id: str,
    train_data, val_data, test_data,
    train_size: int,
    cfg,
    device: torch.device,
    swap_val_test: bool = True,
):
    """
    Aggregate train+val+test into one pool, filter for active pixels, then
    re-split with the PAINT DatasetSplitter (azimuth or balanced strategy).

    Returns
    -------
    train_tuple, val_tuple, test_tuple, pool_rays
        Each tuple is (flux, centroids, rays, motor_pos, active_mask, target_mask).
        pool_rays [N_pool, *] contains rays for every kept sample — used for the
        split-coverage polar plot in run().
    """
    try:
        import pandas as pd
        from paint.data.dataset_splits import DatasetSplitter
        import paint.util.paint_mappings as paint_mappings
    except ImportError as exc:
        raise ImportError(
            "PAINT library not found. Ensure Master Thesis/PAINT/ is on sys.path."
        ) from exc

    val_size   = getattr(cfg, "SPLITTER_VAL_SIZE", 50)
    split_type = getattr(cfg, "SPLITTER_TYPE", "balanced")

    # Aggregate all available splits (val may be None)
    parts = [d for d in (train_data, val_data, test_data) if d is not None]
    pool_flux        = torch.cat([p[0] for p in parts], dim=0)
    pool_centroids   = torch.cat([p[1] for p in parts], dim=0)
    pool_rays        = torch.cat([p[2] for p in parts], dim=0)
    pool_motor_pos   = torch.cat([p[3] for p in parts], dim=0)
    pool_target_mask = torch.cat([p[5] for p in parts], dim=0)   # skip active_mask (index 4)

    # Filter pool for active pixels (single pass, same threshold as per-split filter)
    min_pct = getattr(cfg, "MIN_ACTIVE_PIXEL_PERCENT", 2.0)
    n_raw   = pool_flux.shape[0]
    active_pct = torch.tensor(
        [float((pool_flux[i] > 0.01).sum()) / float(pool_flux[i].numel()) * 100.0
         for i in range(n_raw)],
        dtype=torch.float32,
    )
    keep             = active_pct >= min_pct
    pool_flux        = pool_flux[keep]
    pool_centroids   = pool_centroids[keep]
    pool_rays        = pool_rays[keep]
    pool_motor_pos   = pool_motor_pos[keep]
    pool_target_mask = pool_target_mask[keep]
    n_pool = int(keep.sum().item())
    log.info(f"  pool after filter: {n_pool}/{n_raw}  (rejected {n_raw - n_pool})")

    if n_pool < train_size + 2 * val_size:
        effective_train = n_pool - 2 * val_size
        if effective_train < 10:
            raise RuntimeError(
                f"Pool too small: capped train_size would be {effective_train} "
                f"(pool={n_pool}, 2×val_size={2*val_size}), minimum is 10."
            )
        log.warning(
            f"  Pool too small for train_size={train_size}: "
            f"only {n_pool} samples, need {train_size + 2 * val_size}. "
            f"Capping train_size to {effective_train}."
        )
        train_size = effective_train

    # Azimuth / elevation from incident_ray_direction
    # The ray points FROM the sun TO the heliostat (downward), so negate for sun direction.
    _sun       = -pool_rays.cpu().numpy()
    azimuths   = np.degrees(np.arctan2(_sun[:, 0], _sun[:, 1])) % 360.0
    elevations = np.degrees(np.arcsin(np.clip(_sun[:, 2], -1.0, 1.0)))

    heliostat_df = pd.DataFrame({
        paint_mappings.HELIOSTAT_ID: heliostat_id,
        paint_mappings.AZIMUTH:     azimuths,
        paint_mappings.ELEVATION:   elevations,
        paint_mappings.DATETIME:    pd.Timestamp("2020-06-15 12:00:00"),  # dummy tiebreaker
        paint_mappings.SPLIT_KEY:   "",
    }, index=range(n_pool))

    if split_type == "azimuth":
        split_df = DatasetSplitter._get_azimuth_splits(heliostat_df, train_size, val_size)
    elif split_type == "balanced":
        split_df = DatasetSplitter._get_balanced_splits(heliostat_df, train_size, val_size)
    else:
        raise ValueError(
            f"Unsupported SPLITTER_TYPE {split_type!r}. Use 'azimuth' or 'balanced'."
        )

    train_idx = split_df[split_df[paint_mappings.SPLIT_KEY] == paint_mappings.TRAIN_INDEX].index.tolist()
    val_idx   = split_df[split_df[paint_mappings.SPLIT_KEY] == paint_mappings.VALIDATION_INDEX].index.tolist()
    test_idx  = split_df[split_df[paint_mappings.SPLIT_KEY] == paint_mappings.TEST_INDEX].index.tolist()
    log.info(
        f"  DatasetSplitter ({split_type}): "
        f"train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}"
    )
    if swap_val_test:
        val_idx, test_idx = test_idx, val_idx
        log.info("  val↔test swapped (SWAP_VAL_TEST=True)")

    def _slice(idx):
        t = torch.tensor(idx, device=device)
        return (
            pool_flux[t],
            pool_centroids[t],
            pool_rays[t],
            pool_motor_pos[t],
            torch.tensor([len(idx)], device=device, dtype=torch.long),
            pool_target_mask[t],
        )

    return _slice(train_idx), _slice(val_idx), _slice(test_idx), pool_rays


def _load_perturbations(dataset_dir: pathlib.Path, heliostat_id: str, device: torch.device):
    """Load perturbations.json and return as tensor dict (or None if absent)."""
    pfile = dataset_dir / "perturbations.json"
    if not pfile.exists():
        return None
    with open(pfile) as fh:
        raw = json.load(fh)
    if heliostat_id not in raw:
        return None
    hp = raw[heliostat_id]
    return {
        "rotation":        torch.tensor([hp["rotation_rad"]],       dtype=torch.float32, device=device),
        "actuator_angle":  torch.tensor([hp["actuator_angle_rad"]], dtype=torch.float32, device=device),
        "actuator_stroke": torch.tensor([hp["actuator_stroke_m"]],  dtype=torch.float32, device=device),
        "actuator_offset": torch.tensor([hp["actuator_offset_m"]],  dtype=torch.float32, device=device),
        "translation":     torch.tensor([hp["translation_m"]],      dtype=torch.float32, device=device),
        "base_position":   torch.tensor([hp["base_position_m"]],    dtype=torch.float32, device=device),
    }


# ---------------------------------------------------------------------------
# Helpers: visualization utilities
# ---------------------------------------------------------------------------

def _bitmap_centroid(flux_img: torch.Tensor):
    """Return (col, row) pixel centroid, or (None, None) if flux is empty."""
    f    = flux_img.cpu().float().numpy()
    fsum = f.sum()
    if fsum < 1e-12:
        return None, None
    h, w = f.shape
    cols = np.arange(w, dtype=np.float32).reshape(1, -1)
    rows = np.arange(h, dtype=np.float32).reshape(-1, 1)
    return float((f * cols).sum() / fsum), float((f * rows).sum() / fsum)


def _to_norm(flux: torch.Tensor) -> np.ndarray:
    f    = flux.cpu().float()
    fmax = f.max()
    return (f / fmax).numpy() if fmax > 1e-12 else f.numpy()


# ---------------------------------------------------------------------------
# Helpers: evaluation
# ---------------------------------------------------------------------------

def _eval_test(scenario, hg, test_rays, test_active_mask, test_target_mask,
               test_centroids, hel_dist_m: float, cfg, device, label: str):
    """Forward-pass test set, return eval dict with flux, per-sample errs, label."""
    old_n_rays = scenario.light_sources.light_source_list[0].number_of_rays
    scenario.set_number_of_rays(cfg.DISPLAY_RAYS)

    kinematic = hg.kinematics
    bpd = kinematic._base_position_deviation.detach() if hasattr(kinematic, "_base_position_deviation") \
          else torch.zeros(1, 3, device=device)

    with torch.no_grad():
        pred_cents, pred_flux = _forward_pass(
            scenario, hg, test_rays, test_active_mask, test_target_mask, bpd, device,
        )

    scenario.set_number_of_rays(old_n_rays)

    errs_mrad = (
        torch.norm(pred_cents[:, :3] - test_centroids[:, :3], dim=1) / hel_dist_m * 1000
    ).cpu().numpy()
    errs_m = errs_mrad * hel_dist_m / 1000.0

    log.info(
        f"Eval [{label}]: mean={errs_mrad.mean():.4f} mrad  "
        f"median={float(np.median(errs_mrad)):.4f} mrad"
    )
    return {
        "label":     label,
        "flux":      pred_flux.cpu(),
        "errs_mrad": errs_mrad,
        "errs_m":    errs_m,
    }


# ---------------------------------------------------------------------------
# Helpers: optimizer setup (replicates notebook Cell 17)
# ---------------------------------------------------------------------------

def _setup_kinematic_for_training(kinematic, device: torch.device):
    """
    Enable gradients, register gradient hooks, initialise _base_position_deviation.
    Returns snapshots of initial values used for deviation-bound clamping.
    """
    for attr in ("_initial_actuator_angle", "_initial_actuator_offset",
                 "_initial_translation", "_base_position_deviation"):
        if hasattr(kinematic, attr):
            delattr(kinematic, attr)

    kinematic.translation_deviation_parameters.requires_grad_(True)
    kinematic.rotation_deviation_parameters.requires_grad_(True)
    kinematic.actuators.optimizable_parameters.requires_grad_(True)
    kinematic.actuators.non_optimizable_parameters.requires_grad_(True)

    def _freeze_stroke(grad):
        mask = torch.ones_like(grad)
        mask[:, indices.actuator_initial_stroke_length, :] = 0.0
        return grad * mask

    def _only_c_i(grad):
        mask = torch.zeros_like(grad)
        mask[:, indices.actuator_offset, :] = 1.0
        return grad * mask

    kinematic.actuators.optimizable_parameters.register_hook(_freeze_stroke)
    kinematic.actuators.non_optimizable_parameters.register_hook(_only_c_i)

    kinematic._base_position_deviation = torch.zeros(1, 3, device=device, requires_grad=True)

    init_translation = kinematic.translation_deviation_parameters.detach().clone()
    init_angle  = kinematic.actuators.optimizable_parameters[
        :, indices.actuator_initial_angle, :
    ].detach().clone()
    init_offset = kinematic.actuators.non_optimizable_parameters[
        :, indices.actuator_offset, :
    ].detach().clone()

    return init_angle, init_offset, init_translation


def _build_s1_optimizer(kinematic, cfg):
    opt = torch.optim.Adam(
        [
            {"params": kinematic.translation_deviation_parameters,     "lr": cfg.BASE_LR * 5.0},
            {"params": kinematic.rotation_deviation_parameters,        "lr": cfg.BASE_LR},
            {"params": kinematic.actuators.optimizable_parameters,     "lr": cfg.BASE_LR},
            {"params": kinematic.actuators.non_optimizable_parameters, "lr": cfg.BASE_LR},
            {"params": kinematic._base_position_deviation,             "lr": cfg.BASE_LR * 5.0},
        ],
        lr=cfg.BASE_LR,
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5,
        threshold=1e-4, cooldown=3, min_lr=1e-8,
    )
    return opt, sched


def _build_s2_optimizer(kinematic, cfg):
    kinematic._base_position_deviation = (
        kinematic._base_position_deviation.detach().requires_grad_(True)
    )
    opt = torch.optim.Adam(
        [
            {"params": kinematic.translation_deviation_parameters,     "lr": cfg.BASE_LR * 5.0},
            {"params": kinematic.rotation_deviation_parameters,        "lr": cfg.BASE_LR},
            {"params": kinematic.actuators.optimizable_parameters,     "lr": cfg.BASE_LR},
            {"params": kinematic.actuators.non_optimizable_parameters, "lr": cfg.BASE_LR},
            {"params": kinematic._base_position_deviation,             "lr": cfg.BASE_LR * 5.0},
        ],
        lr=cfg.BASE_LR,
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10,
        threshold=1e-4, cooldown=5, min_lr=1e-8,
    )
    return opt, sched


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_mrad_convergence(trail_checkpoints: list, stage1_epochs: int,
                            plots_dir: pathlib.Path, heliostat_id: str) -> None:
    epochs      = [c["epoch"]      for c in trail_checkpoints]
    mrad_train  = [c["mrad_mean"]  for c in trail_checkpoints]
    mrad_val    = [c["mrad_val_mean"] for c in trail_checkpoints]
    has_val     = not all(np.isnan(v) for v in mrad_val)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, mrad_train, lw=2.0, color="steelblue", label="train mrad (mean)")
    if has_val:
        ax.plot(epochs, mrad_val, lw=2.0, color="darkorange", ls="--", label="val mrad (mean)")
    ax.axvline(stage1_epochs, color="gray", ls=":", lw=1.5, zorder=0)
    ymax = max(ax.get_ylim()[1], 0.1)
    ax.text(stage1_epochs - 0.4, ymax * 0.97, "Stage 1\n(AlignmentLoss)",
            fontsize=7, color="gray", ha="right", va="top")
    ax.text(stage1_epochs + 0.4, ymax * 0.97, "Stage 2\n(FocalSpotLoss)",
            fontsize=7, color="gray", ha="left",  va="top")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Focal-spot error [mrad]")
    ax.set_title(f"{heliostat_id} — focal-spot error across training")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "mrad_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_loss_curves(s1_hist, s1_val, s2_hist, s2_val, plots_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    ep1 = range(1, len(s1_hist) + 1)
    ep2 = range(1, len(s2_hist) + 1)

    axes[0].plot(ep1, s1_hist, lw=1.5, color="steelblue",  label="train")
    if s1_val:
        axes[0].plot(ep1, s1_val, lw=1.5, color="darkorange", ls="--", label="val")
    axes[0].set(title="Stage 1 — AlignmentLoss [rad²]", xlabel="Epoch", ylabel="Loss [rad²]")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].plot(ep2, s2_hist, lw=1.5, color="steelblue",  label="train")
    if s2_val:
        axes[1].plot(ep2, s2_val, lw=1.5, color="darkorange", ls="--", label="val")
    axes[1].set(title="Stage 2 — FocalSpotLoss [m²]", xlabel="Epoch", ylabel="Loss [m²]")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(plots_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_param_trajectories(grad_history: list, param_history: list, pert_tensors: dict,
                              stage1_epochs: int, plots_dir: pathlib.Path) -> None:
    if not grad_history or not param_history:
        return

    groups = [
        ("translation",    "Translation deviation [m]",   [f"t{i}" for i in range(9)]),
        ("rotation",       "Rotation deviation [rad]",    [f"r{i}" for i in range(4)]),
        ("actuator_angle", "Actuator angle dev [rad]",    ["a0", "a1"]),
        ("actuator_offset","Actuator offset dev [m]",     ["c0", "c1"]),
        ("base_position",  "Base position dev [m]",       ["E", "N", "U"]),
    ]
    n_groups = len(groups)
    gt = {
        "translation":    pert_tensors["translation"][0].cpu().tolist(),
        "rotation":       pert_tensors["rotation"][0].cpu().tolist(),
        "actuator_angle": pert_tensors["actuator_angle"][0].cpu().tolist(),
        "actuator_offset":pert_tensors["actuator_offset"][0].cpu().tolist(),
        "base_position":  pert_tensors["base_position"][0].cpu().tolist(),
    }

    epochs_g = [d["epoch"] for d in grad_history]
    epochs_p = [d["epoch"] for d in param_history]
    colors   = ["steelblue", "darkorange", "green", "red", "purple"]

    fig = plt.figure(figsize=(14, 4 + n_groups * 2.6))
    gs  = mgridspec.GridSpec(1 + n_groups, 1, figure=fig, hspace=0.55)

    # Gradient norms (top panel)
    ax_g = fig.add_subplot(gs[0])
    for (key, _, _), col in zip(groups, colors):
        ax_g.semilogy(epochs_g, [d[key] for d in grad_history], lw=1.4, color=col, label=key)
    ax_g.axvline(stage1_epochs, color="gray", ls=":", lw=1.2)
    ax_g.set_xlabel("Epoch"); ax_g.set_ylabel("Grad norm (log scale)")
    ax_g.set_title("Gradient norms per parameter group"); ax_g.legend(fontsize=7)

    # Parameter trajectories (one panel per group)
    for gi, (key, ylabel, labels) in enumerate(groups):
        ax_p = fig.add_subplot(gs[gi + 1])
        param_vals = np.array([d[key] for d in param_history])  # [T, D]
        if param_vals.ndim == 1:
            param_vals = param_vals[:, None]
        gt_vals = gt[key] if isinstance(gt[key], list) else [gt[key]]
        n_dims  = param_vals.shape[1]
        for di in range(n_dims):
            lbl = labels[di] if di < len(labels) else f"d{di}"
            col = plt.cm.tab10(di / max(n_dims - 1, 1))
            ax_p.plot(epochs_p, param_vals[:, di], lw=1.2, color=col, label=lbl)
            if di < len(gt_vals):
                ax_p.axhline(gt_vals[di], color=col, ls="--", lw=0.9, alpha=0.7)
        ax_p.axvline(stage1_epochs, color="gray", ls=":", lw=1.2)
        ax_p.set_xlabel("Epoch"); ax_p.set_ylabel(ylabel)
        ax_p.legend(fontsize=6, ncol=min(n_dims, 5))
        ax_p.grid(alpha=0.2)

    fig.suptitle("Parameter trajectories (solid=trained, dashed=GT)", fontsize=11)
    fig.savefig(plots_dir / "param_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_centroid_trails(trail_checkpoints: list, train_flux: torch.Tensor,
                           train_rays: torch.Tensor, stage1_epochs: int,
                           n_train: int, plots_dir: pathlib.Path, heliostat_id: str) -> None:
    if not trail_checkpoints:
        return

    sun = -train_rays.cpu().numpy()
    compass_az = np.degrees(np.arctan2(sun[:, 0], sun[:, 1])) % 360.0

    bins = [
        (45,  135, "morning",   "Morning (East arc, 45–135°)"),
        (135, 225, "noon",      "Solar noon (South arc, 135–225°)"),
        (225, 315, "afternoon", "Afternoon (West arc, 225–315°)"),
    ]

    n_s1   = sum(1 for c in trail_checkpoints if not c["label"].startswith("S2"))
    n_s2   = sum(1 for c in trail_checkpoints if     c["label"].startswith("S2"))
    colors = []
    s1_i = s2_i = 0
    for snap in trail_checkpoints:
        if snap["label"].startswith("S2"):
            t = s2_i / max(n_s2 - 1, 1)
            colors.append(plt.cm.RdYlGn(t))
            s2_i += 1
        else:
            t = s1_i / max(n_s1 - 1, 1)
            colors.append(plt.cm.Blues(0.35 + 0.65 * t))
            s1_i += 1

    final_mrad = np.array(trail_checkpoints[-1]["errs_train_mrad"])

    for az_lo, az_hi, fname, title_str in bins:
        in_mask = (compass_az >= az_lo) & (compass_az < az_hi)
        idx_in  = np.where(in_mask)[0]
        if len(idx_in) == 0:
            continue

        el_deg = np.degrees(np.arcsin(np.clip(-sun[:, 2], -1.0, 1.0)))
        sorted_in = idx_in[np.argsort(el_deg[idx_in])]
        n_in      = len(sorted_in)
        n_cols    = 8
        n_rows    = (n_in + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(n_cols * 1.9 + 0.5, n_rows * 2.4), constrained_layout=True)
        fig.suptitle(
            f"{heliostat_id}  —  {title_str}  ({n_in} samples)\n"
            "Trail: Blues=Stage1, RdYlGn=Stage2 (red→green)",
            fontsize=8, fontweight="bold",
        )
        gs = mgridspec.GridSpec(n_rows, n_cols, figure=fig)

        for j, idx in enumerate(sorted_in):
            row, col = divmod(j, n_cols)
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(_to_norm(train_flux[idx]), cmap="inferno", origin="upper")

            for ck_i, ckpt in enumerate(trail_checkpoints):
                cx, cy = ckpt["centroids"][idx]
                if cx is not None:
                    ax.plot(cx, cy, "o", color=colors[ck_i], ms=2.5, alpha=0.75, zorder=5)

            gt_cx, gt_cy = _bitmap_centroid(train_flux[idx])
            if gt_cx is not None:
                ax.plot(gt_cx, gt_cy, "x", color="limegreen", ms=6, mew=1.5, zorder=8)

            ax.set_title(f"el={el_deg[idx]:.0f}°  {final_mrad[idx]:.1f}mrad", fontsize=5, pad=1)
            ax.axis("off")

        for j in range(n_in, n_rows * n_cols):
            row, col = divmod(j, n_cols)
            fig.add_subplot(gs[row, col]).axis("off")

        fig.savefig(plots_dir / f"trail_{fname}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_sun_positions_split(
    train_rays: torch.Tensor,
    val_rays: torch.Tensor | None,
    test_rays: torch.Tensor,
    plots_dir: pathlib.Path,
    heliostat_id: str,
) -> None:
    """Polar sun-path diagram coloured by dataset split (South=top, East=left)."""
    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(111, projection="polar")

    for rays, color, label in [
        (train_rays, "steelblue",  "train"),
        (val_rays,   "darkorange", "val"),
        (test_rays,  "green",      "test"),
    ]:
        if rays is None or rays.shape[0] == 0:
            continue
        r_np = rays.cpu().float().numpy()
        el   = np.degrees(np.arcsin(np.clip(-r_np[:, 2], -1.0, 1.0)))
        az   = np.degrees(np.arctan2(-r_np[:, 0], -r_np[:, 1])) % 360.0
        # South=top, East=left mapping onto standard matplotlib polar (0=right, CCW):
        #   theta = 270° - compass_az
        #   r     = zenith angle = 90° - elevation (0=zenith/centre, 90=horizon/edge)
        theta = np.radians(270.0 - az)
        r     = 90.0 - el
        ax.scatter(theta, r, c=color, s=15, alpha=0.75, label=f"{label} (n={len(r)})", zorder=3)

    # Cardinal direction labels: at polar angles 0°(right)=W, 90°(top)=S, 180°(left)=E, 270°(bottom)=N
    ax.set_thetagrids([0, 90, 180, 270], labels=["W", "S", "E", "N"], fontsize=10)

    # Radial ticks are zenith angle; annotate as elevation
    ax.set_rticks([30, 60, 90])
    ax.set_yticklabels(["el 60°", "el 30°", "horizon"], fontsize=8)
    ax.set_rlim(0, 90)

    ax.set_title(f"{heliostat_id} — sun positions by split", pad=15, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = plots_dir / "sun_positions_split.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Sun positions plot → {out}")


def _plot_test_flux(test_eval: dict, test_flux: torch.Tensor, test_rays: torch.Tensor,
                    hel_dist_m: float, plots_dir: pathlib.Path, heliostat_id: str) -> None:
    test_flux_dir = plots_dir / "test_flux"
    test_flux_dir.mkdir(exist_ok=True)

    el_np    = torch.asin(-test_rays[:, 2].clamp(-1, 1)).rad2deg().cpu().numpy()
    sort_idx = np.argsort(el_np)
    pred_flux = test_eval["flux"]
    errs_mrad = test_eval["errs_mrad"]
    errs_m    = test_eval["errs_m"]

    n_pairs_per_row = 4
    n_t   = len(test_flux)
    n_rows = (n_t + n_pairs_per_row - 1) // n_pairs_per_row

    wr = [1.0, 1.0, 0.45] * n_pairs_per_row
    fig = plt.figure(
        figsize=(n_pairs_per_row * (2 * 2.5 + 1.1), n_rows * 3.0),
        constrained_layout=True,
    )
    gs = mgridspec.GridSpec(n_rows, len(wr), figure=fig, width_ratios=wr)
    fig.suptitle(
        f"{heliostat_id}  —  {n_t} test samples  (After Stage 2)\n"
        "each pair: predicted (left) / GT (right)   "
        "green × = GT centroid  |  red + = predicted centroid",
        fontsize=8,
    )

    for j, idx in enumerate(sort_idx):
        row  = j // n_pairs_per_row
        pair = j %  n_pairs_per_row

        pf = pred_flux[idx]
        gf = test_flux[idx]
        em = errs_mrad[idx]
        el = el_np[idx]

        pcx, pcy = _bitmap_centroid(pf)
        gcx, gcy = _bitmap_centroid(gf)

        ax_p = fig.add_subplot(gs[row, pair * 3])
        ax_p.imshow(_to_norm(pf), cmap="gray", vmin=0, vmax=1)
        if gcx is not None:
            ax_p.plot(gcx, gcy, "x", color="green", ms=7, mew=1.8, zorder=10)
        if pcx is not None:
            ax_p.plot(pcx, pcy, "+", color="red",   ms=7, mew=1.8, zorder=9)
        ax_p.set_title(f"pred  el={el:.0f}°", fontsize=6, pad=2)
        ax_p.text(0.03, 0.97, f"{em:.4f} mrad",
                  transform=ax_p.transAxes, fontsize=5, va="top", ha="left", color="white",
                  bbox=dict(facecolor="black", alpha=0.55, pad=1, linewidth=0))
        ax_p.axis("off")

        ax_g = fig.add_subplot(gs[row, pair * 3 + 1])
        ax_g.imshow(_to_norm(gf), cmap="gray", vmin=0, vmax=1)
        if gcx is not None:
            ax_g.plot(gcx, gcy, "x", color="green", ms=7, mew=1.8, zorder=10)
        ax_g.set_title("GT", fontsize=6, pad=2)
        ax_g.axis("off")

        fig.add_subplot(gs[row, pair * 3 + 2]).axis("off")

    for j in range(n_t, n_rows * n_pairs_per_row):
        row  = j // n_pairs_per_row
        pair = j %  n_pairs_per_row
        for off in range(3):
            fig.add_subplot(gs[row, pair * 3 + off]).axis("off")

    fig.savefig(plots_dir / "test_flux" / "all_samples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Also save each sample individually.
    for j, idx in enumerate(sort_idx):
        pf = pred_flux[idx]
        gf = test_flux[idx]
        em = errs_mrad[idx]
        el = el_np[idx]
        pcx, pcy = _bitmap_centroid(pf)
        gcx, gcy = _bitmap_centroid(gf)

        fig2, axes2 = plt.subplots(1, 2, figsize=(6, 3))
        for ax, fl, cx, cy, ttl in [
            (axes2[0], pf, pcx, pcy, f"Predicted  el={el:.0f}°\n{em:.4f} mrad"),
            (axes2[1], gf, gcx, gcy, "GT"),
        ]:
            ax.imshow(_to_norm(fl), cmap="gray", vmin=0, vmax=1)
            if gcx is not None:
                ax.plot(gcx, gcy, "x", color="green", ms=8, mew=2, zorder=10)
            if cx is not None and ax is axes2[0]:
                ax.plot(cx, cy, "+", color="red", ms=8, mew=2, zorder=9)
            ax.set_title(ttl, fontsize=8)
            ax.axis("off")
        plt.tight_layout()
        fig2.savefig(test_flux_dir / f"sample_{idx:04d}.png", dpi=120, bbox_inches="tight")
        plt.close(fig2)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run(
    heliostat_id: str,
    dataset_dir: pathlib.Path | str,
    output_dir: pathlib.Path | str,
    cfg,
    device: torch.device,
    train_size: int | None = None,
    sampling_seed: int = 42,
) -> dict:
    """
    Run the two-stage training pipeline and save all outputs.

    Parameters
    ----------
    heliostat_id  : heliostat to reconstruct
    dataset_dir   : root of the generated dataset (contains train/val/test/{id}/)
    output_dir    : where to write results, plots, histories
    cfg           : config module (or SimpleNamespace) with all hyperparameters
    device        : torch device

    Returns
    -------
    dict with keys "pre", "after_s1", "after_s2" (each holding mrad metrics)
    """
    dataset_dir = pathlib.Path(dataset_dir)
    output_dir  = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    t_start = time.time()

    # ------------------------------------------------------------------ #
    # 1. Load scenario                                                     #
    # ------------------------------------------------------------------ #
    scenario, hg, hel_dist_m = _load_scenario(heliostat_id, cfg, device)
    kinematic = hg.kinematics

    # ------------------------------------------------------------------ #
    # 2. Load data                                                         #
    # ------------------------------------------------------------------ #
    _data_mode = getattr(cfg, "DATA_MODE", "synthetic")

    if _data_mode == "real":
        log.info("Data mode: real (PAINT benchmark)")
        train_data = _load_split_real(heliostat_id, cfg, "train",      hg, scenario, device)
        val_data   = _load_split_real(heliostat_id, cfg, "validation", hg, scenario, device)
        test_data  = _load_split_real(heliostat_id, cfg, "test",       hg, scenario, device)
        pert_tensors = None
    else:
        log.info(f"Data mode: {_data_mode} (synthetic dataset)")
        train_data = _load_split(heliostat_id, dataset_dir, "train", hg, scenario, device)
        val_data   = _load_split(heliostat_id, dataset_dir, "val",   hg, scenario, device)
        test_data  = _load_split(heliostat_id, dataset_dir, "test",  hg, scenario, device)
        pert_tensors = _load_perturbations(dataset_dir, heliostat_id, device)

    if train_data is None:
        raise RuntimeError(
            f"No training data found for {heliostat_id} "
            f"({'PAINT benchmark' if _data_mode == 'real' else dataset_dir})"
        )
    if test_data is None:
        raise RuntimeError(
            f"No test data found for {heliostat_id} "
            f"({'PAINT benchmark' if _data_mode == 'real' else dataset_dir})"
        )

    # ------------------------------------------------------------------ #
    # 3. Pool all splits, re-split with DatasetSplitter                    #
    # ------------------------------------------------------------------ #
    if train_size is None:
        train_size = getattr(cfg, "SPLITTER_TRAIN_SIZE", 100)

    swap = getattr(cfg, "SWAP_VAL_TEST", True)
    (train_data, val_data, test_data, full_train_rays) = _pool_and_split(
        heliostat_id, train_data, val_data, test_data,
        train_size, cfg, device, swap_val_test=swap,
    )
    (train_flux, train_centroids, train_rays,
     train_motor_pos, train_active_mask, train_target_mask) = train_data
    (val_flux, val_centroids, val_rays,
     val_motor_pos, val_active_mask, val_target_mask) = val_data
    (test_flux, test_centroids, test_rays,
     test_motor_pos, test_active_mask, test_target_mask) = test_data

    N_TRAIN = train_flux.shape[0]
    N_VAL   = val_flux.shape[0] if val_flux is not None else 0
    N_TEST  = test_flux.shape[0]
    log.info(f"Final counts: train={N_TRAIN}  val={N_VAL}  test={N_TEST}")

    # ------------------------------------------------------------------ #
    # 4. Pre-training evaluation                                           #
    # ------------------------------------------------------------------ #
    pre_eval = _eval_test(
        scenario, hg,
        test_rays, test_active_mask, test_target_mask, test_centroids,
        hel_dist_m, cfg, device, "Pre-training",
    )

    # ------------------------------------------------------------------ #
    # 5. Optimizer setup (replicates notebook Cell 17)                    #
    # ------------------------------------------------------------------ #
    init_angle, init_offset, init_translation = _setup_kinematic_for_training(kinematic, device)

    def _apply_bounds():
        kinematic.translation_deviation_parameters.data.clamp_(
            init_translation - cfg._BOUND_TRANSLATION_M,
            init_translation + cfg._BOUND_TRANSLATION_M,
        )
        kinematic.rotation_deviation_parameters.data.clamp_(
            -cfg._BOUND_ROTATION_RAD, cfg._BOUND_ROTATION_RAD,
        )
        kinematic.actuators.optimizable_parameters.data[
            :, indices.actuator_initial_angle, :
        ].clamp_(
            init_angle - cfg._BOUND_ACTUATOR_ANGLE_RAD,
            init_angle + cfg._BOUND_ACTUATOR_ANGLE_RAD,
        )
        kinematic.actuators.non_optimizable_parameters.data[
            :, indices.actuator_offset, :
        ].clamp_(
            init_offset - cfg._BOUND_ACTUATOR_OFFSET_M,
            init_offset + cfg._BOUND_ACTUATOR_OFFSET_M,
        )
        if hasattr(kinematic, "_base_position_deviation"):
            kinematic._base_position_deviation.data.clamp_(
                -cfg._BOUND_BASE_POSITION_M, cfg._BOUND_BASE_POSITION_M,
            )

    def _all_params():
        p = [
            kinematic.translation_deviation_parameters,
            kinematic.rotation_deviation_parameters,
            kinematic.actuators.optimizable_parameters,
            kinematic.actuators.non_optimizable_parameters,
        ]
        if hasattr(kinematic, "_base_position_deviation"):
            p.append(kinematic._base_position_deviation)
        return p

    def _current_base_pos() -> torch.Tensor:
        if hasattr(kinematic, "_base_position_deviation"):
            return kinematic._base_position_deviation.detach()
        return torch.zeros(1, 3, device=device)

    optimizer_s1, scheduler_s1 = _build_s1_optimizer(kinematic, cfg)

    # ------------------------------------------------------------------ #
    # 6. Trail capture state                                               #
    # ------------------------------------------------------------------ #
    trail_checkpoints: list[dict] = []
    grad_history:      list[dict] = []
    param_history:     list[dict] = []
    PLOT_EVERY = getattr(cfg, "PLOT_EVERY", 1)

    def capture_trails(label: str, epoch: int) -> None:
        with torch.no_grad():
            pred_cents, pred_flux = _forward_pass(
                scenario, hg,
                train_rays, train_active_mask, train_target_mask,
                _current_base_pos(), device,
            )
        errs = (
            torch.norm(pred_cents[:, :3] - train_centroids[:, :3], dim=1)
            / hel_dist_m * 1000
        ).cpu().numpy()

        mrad_val_mean = float("nan")
        if val_flux is not None:
            with torch.no_grad():
                val_cents, _ = _forward_pass(
                    scenario, hg,
                    val_rays, val_active_mask, val_target_mask,
                    _current_base_pos(), device,
                )
            mrad_val_mean = float(
                (torch.norm(val_cents[:, :3] - val_centroids[:, :3], dim=1)
                 / hel_dist_m * 1000).mean().item()
            )

        trail_checkpoints.append({
            "label":           label,
            "epoch":           epoch,
            "centroids":       [_bitmap_centroid(pred_flux[i]) for i in range(N_TRAIN)],
            "mrad_mean":       float(errs.mean()),
            "mrad_median":     float(np.median(errs)),
            "mrad_val_mean":   mrad_val_mean,
            "errs_train_mrad": errs.tolist(),
        })

    def capture_grad_and_params(abs_epoch: int) -> None:
        def _gnorm(p):
            return float(p.grad.norm().item()) if p is not None and p.grad is not None else 0.0

        grad_history.append({
            "epoch":          abs_epoch,
            "translation":    _gnorm(kinematic.translation_deviation_parameters),
            "rotation":       _gnorm(kinematic.rotation_deviation_parameters),
            "actuator_angle": _gnorm(kinematic.actuators.optimizable_parameters),
            "actuator_offset":_gnorm(kinematic.actuators.non_optimizable_parameters),
            "base_position":  _gnorm(kinematic._base_position_deviation)
                              if hasattr(kinematic, "_base_position_deviation") else 0.0,
        })

        angle_dev  = (
            kinematic.actuators.optimizable_parameters[:, indices.actuator_initial_angle, :]
            - init_angle
        ).detach().cpu().squeeze().tolist()
        offset_dev = (
            kinematic.actuators.non_optimizable_parameters[:, indices.actuator_offset, :]
            - init_offset
        ).detach().cpu().squeeze().tolist()

        param_history.append({
            "epoch":          abs_epoch,
            "translation":    kinematic.translation_deviation_parameters.detach().cpu().squeeze().tolist(),
            "rotation":       kinematic.rotation_deviation_parameters.detach().cpu().squeeze().tolist(),
            "actuator_angle": angle_dev  if isinstance(angle_dev,  list) else [angle_dev],
            "actuator_offset":offset_dev if isinstance(offset_dev, list) else [offset_dev],
            "base_position":  kinematic._base_position_deviation.detach().cpu().squeeze().tolist()
                              if hasattr(kinematic, "_base_position_deviation") else [0.0, 0.0, 0.0],
        })

    # ------------------------------------------------------------------ #
    # 7. Stage 1 — AlignmentLoss                                          #
    # ------------------------------------------------------------------ #
    capture_trails("pre", 0)

    alignment_loss_fn = AlignmentLoss()
    stage1_history:     list[float] = []
    stage1_val_history: list[float] = []
    best_s1_mrad   = float("inf")
    best_s1_params = None

    log.info(f"Stage 1: AlignmentLoss  |  {cfg.STAGE1_EPOCHS} epochs  |  {N_TRAIN} samples/epoch")
    t_s1 = time.time()

    for epoch in tqdm(range(1, cfg.STAGE1_EPOCHS + 1), desc="Stage 1"):
        optimizer_s1.zero_grad()

        hg.activate_heliostats(active_heliostats_mask=train_active_mask, device=device)
        _bpd = kinematic._base_position_deviation
        _rep = _bpd.repeat_interleave(train_active_mask, dim=0)
        _pad = torch.zeros(_rep.shape[0], 1, device=device)
        kinematic.active_heliostat_positions = (
            kinematic.active_heliostat_positions + torch.cat([_rep, _pad], dim=1)
        )

        hg.align_surfaces_with_incident_ray_directions(
            aim_points=scenario.solar_tower.get_centers_of_target_areas(
                train_target_mask, device
            ),
            incident_ray_directions=train_rays,
            active_heliostats_mask=train_active_mask,
            device=device,
        )

        lps  = alignment_loss_fn(
            predicted_motor_positions=kinematic.active_motor_positions,
            measured_motor_positions=train_motor_pos,
            actuators=kinematic.actuators,
            device=device,
        )
        loss = lps.mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(_all_params(), max_norm=1.0)
        optimizer_s1.step()
        _apply_bounds()
        capture_grad_and_params(epoch)
        stage1_history.append(loss.item())

        s1_val_loss = None
        if val_flux is not None:
            with torch.no_grad():
                hg.activate_heliostats(active_heliostats_mask=val_active_mask, device=device)
                _bpd_v = kinematic._base_position_deviation
                _rep_v = _bpd_v.repeat_interleave(val_active_mask, dim=0)
                _pad_v = torch.zeros(_rep_v.shape[0], 1, device=device)
                kinematic.active_heliostat_positions = (
                    kinematic.active_heliostat_positions + torch.cat([_rep_v, _pad_v], dim=1)
                )
                hg.align_surfaces_with_incident_ray_directions(
                    aim_points=scenario.solar_tower.get_centers_of_target_areas(
                        val_target_mask, device
                    ),
                    incident_ray_directions=val_rays,
                    active_heliostats_mask=val_active_mask,
                    device=device,
                )
                _val_lps = alignment_loss_fn(
                    predicted_motor_positions=kinematic.active_motor_positions,
                    measured_motor_positions=val_motor_pos,
                    actuators=kinematic.actuators,
                    device=device,
                )
            s1_val_loss = _val_lps.mean().item()
            stage1_val_history.append(s1_val_loss)

        scheduler_s1.step(s1_val_loss if s1_val_loss is not None else loss.item())

        if epoch % PLOT_EVERY == 0:
            capture_trails(f"S1/{epoch}", epoch)
            metric = (trail_checkpoints[-1]["mrad_val_mean"]
                      if val_flux is not None else trail_checkpoints[-1]["mrad_mean"])
            if metric < best_s1_mrad:
                best_s1_mrad = metric
                best_s1_params = {
                    "translation": kinematic.translation_deviation_parameters.clone().detach(),
                    "rotation":    kinematic.rotation_deviation_parameters.clone().detach(),
                    "act_angle":   kinematic.actuators.optimizable_parameters.clone().detach(),
                    "act_offset":  kinematic.actuators.non_optimizable_parameters.clone().detach(),
                    "base_pos":    kinematic._base_position_deviation.clone().detach(),
                }

    if best_s1_params is not None:
        kinematic.translation_deviation_parameters.data.copy_(best_s1_params["translation"])
        kinematic.rotation_deviation_parameters.data.copy_(best_s1_params["rotation"])
        kinematic.actuators.optimizable_parameters.data.copy_(best_s1_params["act_angle"])
        kinematic.actuators.non_optimizable_parameters.data.copy_(best_s1_params["act_offset"])
        kinematic._base_position_deviation = best_s1_params["base_pos"].clone().requires_grad_(True)
        log.info(f"Restored best Stage 1 params (mrad={best_s1_mrad:.4f})")

    t_s1_min = (time.time() - t_s1) / 60.0
    log.info(f"Stage 1 done in {t_s1_min:.1f} min. Final loss={stage1_history[-1]:.6f}")

    s1_eval = _eval_test(
        scenario, hg,
        test_rays, test_active_mask, test_target_mask, test_centroids,
        hel_dist_m, cfg, device, "After Stage 1",
    )

    # ------------------------------------------------------------------ #
    # 8. Stage 2 — FocalSpotLoss                                          #
    # ------------------------------------------------------------------ #
    optimizer_s2, scheduler_s2 = _build_s2_optimizer(kinematic, cfg)
    focal_spot_loss_fn = FocalSpotLoss(scenario=scenario)
    scenario.set_number_of_rays(cfg.TRAIN_RAYS)

    n_mb = (N_TRAIN + cfg.MINI_BATCH_SIZE - 1) // cfg.MINI_BATCH_SIZE
    stage2_history:     list[float] = []
    stage2_val_history: list[float] = []
    best_s2_loss   = float("inf")
    best_s2_params = None

    log.info(
        f"Stage 2: FocalSpotLoss  |  {cfg.STAGE2_EPOCHS} epochs  |  "
        f"{n_mb} mini-batches of ≤{cfg.MINI_BATCH_SIZE} samples"
    )
    t_s2 = time.time()

    for epoch in tqdm(range(1, cfg.STAGE2_EPOCHS + 1), desc="Stage 2"):
        optimizer_s2.zero_grad()
        loss_accum = None

        for mb in range(n_mb):
            s  = mb * cfg.MINI_BATCH_SIZE
            e  = min(s + cfg.MINI_BATCH_SIZE, N_TRAIN)
            mb_size = e - s

            mb_rays    = train_rays[s:e]
            mb_target  = train_target_mask[s:e]
            mb_gt      = train_centroids[s:e]
            mb_active  = torch.tensor([mb_size], device=device, dtype=torch.long)

            hg.activate_heliostats(active_heliostats_mask=mb_active, device=device)
            _bpd = kinematic._base_position_deviation
            _rep = _bpd.repeat_interleave(mb_active, dim=0)
            _pad = torch.zeros(_rep.shape[0], 1, device=device)
            kinematic.active_heliostat_positions = (
                kinematic.active_heliostat_positions + torch.cat([_rep, _pad], dim=1)
            )

            hg.align_surfaces_with_incident_ray_directions(
                aim_points=scenario.solar_tower.get_centers_of_target_areas(mb_target, device),
                incident_ray_directions=mb_rays,
                active_heliostats_mask=mb_active,
                device=device,
            )

            ray_tracer = HeliostatRayTracer(
                scenario=scenario,
                heliostat_group=hg,
                blocking_active=False,
                world_size=1, rank=0,
                batch_size=max(8, mb_size),
                random_seed=epoch * 1000 + mb,
            )
            flux, _, _, _ = ray_tracer.trace_rays(
                incident_ray_directions=mb_rays,
                active_heliostats_mask=mb_active,
                target_area_indices=mb_target,
                device=device,
            )
            sample_idx = ray_tracer.get_sampler_indices()

            lps = focal_spot_loss_fn(
                prediction=flux,
                ground_truth=mb_gt[sample_idx],
                target_area_indices=mb_target[sample_idx],
                reduction_dimensions=(indices.focal_spots,),
                device=device,
            )
            weight = mb_size / N_TRAIN
            (lps.mean() * weight).backward()
            mb_loss   = lps.detach().mean() * weight
            loss_accum = mb_loss if loss_accum is None else loss_accum + mb_loss

        torch.nn.utils.clip_grad_norm_(_all_params(), max_norm=1.0)
        optimizer_s2.step()
        _apply_bounds()
        capture_grad_and_params(cfg.STAGE1_EPOCHS + epoch)
        stage2_history.append(loss_accum.item())

        s2_val_loss = None
        if val_flux is not None:
            n_mb_v   = (N_VAL + cfg.MINI_BATCH_SIZE - 1) // cfg.MINI_BATCH_SIZE
            val_accum = 0.0
            with torch.no_grad():
                for mbv in range(n_mb_v):
                    sv  = mbv * cfg.MINI_BATCH_SIZE
                    ev  = min(sv + cfg.MINI_BATCH_SIZE, N_VAL)
                    msv = ev - sv
                    mbr_v = val_rays[sv:ev];    mbt_v = val_target_mask[sv:ev]
                    mbg_v = val_centroids[sv:ev]; mba_v = torch.tensor([msv], device=device, dtype=torch.long)

                    hg.activate_heliostats(active_heliostats_mask=mba_v, device=device)
                    _bpd_v = kinematic._base_position_deviation
                    _rep_v = _bpd_v.repeat_interleave(mba_v, dim=0)
                    _pad_v = torch.zeros(_rep_v.shape[0], 1, device=device)
                    kinematic.active_heliostat_positions = (
                        kinematic.active_heliostat_positions + torch.cat([_rep_v, _pad_v], dim=1)
                    )
                    hg.align_surfaces_with_incident_ray_directions(
                        aim_points=scenario.solar_tower.get_centers_of_target_areas(mbt_v, device),
                        incident_ray_directions=mbr_v,
                        active_heliostats_mask=mba_v,
                        device=device,
                    )
                    rt_v = HeliostatRayTracer(
                        scenario=scenario, heliostat_group=hg,
                        blocking_active=False, world_size=1, rank=0,
                        batch_size=max(8, msv), random_seed=42,
                    )
                    fl_v, _, _, _ = rt_v.trace_rays(
                        incident_ray_directions=mbr_v,
                        active_heliostats_mask=mba_v,
                        target_area_indices=mbt_v,
                        device=device,
                    )
                    sidx_v = rt_v.get_sampler_indices()
                    lps_v  = focal_spot_loss_fn(
                        prediction=fl_v,
                        ground_truth=mbg_v[sidx_v],
                        target_area_indices=mbt_v[sidx_v],
                        reduction_dimensions=(indices.focal_spots,),
                        device=device,
                    )
                    val_accum += lps_v.mean().item() * (msv / N_VAL)
            s2_val_loss = val_accum
            stage2_val_history.append(s2_val_loss)

        scheduler_s2.step(s2_val_loss if s2_val_loss is not None else loss_accum.item())

        monitor = s2_val_loss if s2_val_loss is not None else loss_accum.item()
        if monitor < best_s2_loss:
            best_s2_loss = monitor
            best_s2_params = {
                "translation": kinematic.translation_deviation_parameters.clone().detach(),
                "rotation":    kinematic.rotation_deviation_parameters.clone().detach(),
                "act_angle":   kinematic.actuators.optimizable_parameters.clone().detach(),
                "act_offset":  kinematic.actuators.non_optimizable_parameters.clone().detach(),
                "base_pos":    kinematic._base_position_deviation.clone().detach(),
            }

        if epoch % PLOT_EVERY == 0:
            capture_trails(f"S2/{epoch}", cfg.STAGE1_EPOCHS + epoch)

    if best_s2_params is not None:
        kinematic.translation_deviation_parameters.data.copy_(best_s2_params["translation"])
        kinematic.rotation_deviation_parameters.data.copy_(best_s2_params["rotation"])
        kinematic.actuators.optimizable_parameters.data.copy_(best_s2_params["act_angle"])
        kinematic.actuators.non_optimizable_parameters.data.copy_(best_s2_params["act_offset"])
        kinematic._base_position_deviation = best_s2_params["base_pos"].clone().requires_grad_(True)
        log.info(f"Restored best Stage 2 params (loss={best_s2_loss:.6f})")

    t_s2_min = (time.time() - t_s2) / 60.0
    log.info(f"Stage 2 done in {t_s2_min:.1f} min. Final loss={stage2_history[-1]:.6f}")

    s2_eval = _eval_test(
        scenario, hg,
        test_rays, test_active_mask, test_target_mask, test_centroids,
        hel_dist_m, cfg, device, "After Stage 2",
    )

    total_min = (time.time() - t_start) / 60.0

    # ------------------------------------------------------------------ #
    # 9. Save outputs                                                      #
    # ------------------------------------------------------------------ #

    def _mrad_stats(ev: dict) -> dict:
        return {
            "mrad_mean":   float(ev["errs_mrad"].mean()),
            "mrad_median": float(np.median(ev["errs_mrad"])),
            "m_mean":      float(ev["errs_m"].mean()),
            "m_median":    float(np.median(ev["errs_m"])),
        }

    results = {
        "heliostat_id":    heliostat_id,
        "hel_dist_m":      hel_dist_m,
        "n_train":         N_TRAIN,
        "n_val":           N_VAL,
        "n_test":          N_TEST,
        "total_time_min":  total_min,
        "pre_training":    _mrad_stats(pre_eval),
        "after_stage1":    _mrad_stats(s1_eval),
        "after_stage2":    _mrad_stats(s2_eval),
    }
    with open(output_dir / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    # Metrics ASCII table
    with open(output_dir / "metrics_table.txt", "w") as fh:
        fh.write(f"Heliostat: {heliostat_id}  |  test samples: {N_TEST}\n\n")
        fh.write(f"{'Stage':<22} {'Mean [mrad]':>12} {'Median [mrad]':>14} {'Mean [m]':>10} {'Median [m]':>10}\n")
        fh.write("-" * 70 + "\n")
        for ev in [pre_eval, s1_eval, s2_eval]:
            mn  = ev["errs_mrad"].mean()
            med = float(np.median(ev["errs_mrad"]))
            fh.write(
                f"{ev['label']:<22} {mn:12.4f} {med:14.4f}"
                f" {mn * hel_dist_m / 1000:10.6f} {med * hel_dist_m / 1000:10.6f}\n"
            )

    # Convergence CSV
    with open(output_dir / "convergence_history.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "epoch", "stage", "train_loss", "val_loss",
            "mrad_train_mean", "mrad_train_median", "mrad_val_mean",
        ])
        writer.writeheader()
        for ckpt in trail_checkpoints:
            ep  = ckpt["epoch"]
            if ep == 0:
                stage = "pre"
            elif ep <= cfg.STAGE1_EPOCHS:
                stage = "stage1"
            else:
                stage = "stage2"
            # Match loss to epoch
            if ep == 0:
                t_loss = v_loss = None
            elif stage == "stage1":
                idx = ep - 1
                t_loss = stage1_history[idx] if idx < len(stage1_history) else None
                v_loss = stage1_val_history[idx] if idx < len(stage1_val_history) else None
            else:
                idx = ep - cfg.STAGE1_EPOCHS - 1
                t_loss = stage2_history[idx] if idx < len(stage2_history) else None
                v_loss = stage2_val_history[idx] if idx < len(stage2_val_history) else None

            writer.writerow({
                "epoch":             ep,
                "stage":             stage,
                "train_loss":        t_loss,
                "val_loss":          v_loss,
                "mrad_train_mean":   ckpt["mrad_mean"],
                "mrad_train_median": ckpt["mrad_median"],
                "mrad_val_mean":     ckpt["mrad_val_mean"],
            })

    # Kinematic parameters (final)
    kin_params = {
        "rotation_dev_rad":       kinematic.rotation_deviation_parameters.detach().cpu().tolist(),
        "actuator_angle_dev_rad": (
            kinematic.actuators.optimizable_parameters[:, indices.actuator_initial_angle, :]
            - init_angle
        ).detach().cpu().tolist(),
        "actuator_offset_dev_m":  (
            kinematic.actuators.non_optimizable_parameters[:, indices.actuator_offset, :]
            - init_offset
        ).detach().cpu().tolist(),
        "translation_dev_m":      kinematic.translation_deviation_parameters.detach().cpu().tolist(),
        "base_position_dev_m":    kinematic._base_position_deviation.detach().cpu().tolist()
                                  if hasattr(kinematic, "_base_position_deviation") else [0.0, 0.0, 0.0],
    }
    with open(output_dir / "kinematic_parameters.json", "w") as fh:
        json.dump(kin_params, fh, indent=2)

    # Per-epoch kinematic history
    with open(output_dir / "kinematic_history.json", "w") as fh:
        json.dump(param_history, fh, indent=2)

    # Gradient history
    with open(output_dir / "gradient_history.json", "w") as fh:
        json.dump(grad_history, fh, indent=2)

    # Trail checkpoints (excludes per-flux data to keep file small)
    trail_json = [
        {k: v for k, v in ckpt.items() if k != "centroids"}
        for ckpt in trail_checkpoints
    ]
    with open(output_dir / "trail_checkpoints.json", "w") as fh:
        json.dump(trail_json, fh, indent=2)

    log.info(f"Outputs saved to {output_dir}")

    # ------------------------------------------------------------------ #
    # 10. Plots                                                            #
    # ------------------------------------------------------------------ #
    log.info("Generating plots...")

    _plot_mrad_convergence(trail_checkpoints, cfg.STAGE1_EPOCHS, plots_dir, heliostat_id)
    _plot_loss_curves(stage1_history, stage1_val_history, stage2_history, stage2_val_history, plots_dir)

    if pert_tensors is not None:
        _plot_param_trajectories(grad_history, param_history, pert_tensors, cfg.STAGE1_EPOCHS, plots_dir)

    _plot_centroid_trails(
        trail_checkpoints, train_flux, train_rays,
        cfg.STAGE1_EPOCHS, N_TRAIN, plots_dir, heliostat_id,
    )
    _plot_test_flux(s2_eval, test_flux, test_rays, hel_dist_m, plots_dir, heliostat_id)
    _plot_sun_positions_split(train_rays, val_rays, test_rays, plots_dir, heliostat_id)

    log.info(f"All plots saved to {plots_dir}")
    log.info(f"Total time: {total_min:.1f} min")

    return results
