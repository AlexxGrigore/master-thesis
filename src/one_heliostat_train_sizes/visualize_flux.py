"""
Visualize measured vs predicted flux for the one-heliostat train-size experiment.

Loads the trained kinematic parameters saved by main.py and reruns the ray tracer
on test samples to produce a measured/predicted/diff comparison grid.

Usage
-----
    python visualize_flux.py                           # AO34, train_size=20, 6 samples
    python visualize_flux.py --heliostat-id AW36 --train-size 20 --n-samples 9
    python visualize_flux.py --heliostat-id AO34 --train-size 50 --n-rays 50
"""
import argparse
import pathlib
import sys
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch

_here = pathlib.Path(__file__).resolve().parent
_src  = _here.parent
sys.path.insert(0, str(_src))

from artist.scenario.scenario import Scenario
from artist.util import set_logger_config
from artist.util.env import get_device
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
from artist.flux import get_center_of_mass
from artist.geometry import bitmap_coordinates_to_target_coordinates
from artist.util import indices as index_mapping

from utils.synth_data import SyntheticDatasetParser
from utils.evaluation import _gaussian_blur_batch
import config as cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_scenario(heliostat_id: str, device: torch.device, n_surface_pts: int) -> "Scenario":
    scenario_path = cfg.ONE_HELIOSTAT_SCENARIOS_DIR / heliostat_id / "scenario.h5"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario not found: {scenario_path}")
    with h5py.File(scenario_path, "r") as f:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=f,
            device=device,
            number_of_surface_points_per_facet=torch.tensor([n_surface_pts, n_surface_pts]),
        )
    return scenario


def _apply_kinematic_parameters(scenario, params_path: pathlib.Path, device: torch.device) -> None:
    """Overwrite in-place the scenario kinematics with the saved trained values."""
    with open(params_path) as fh:
        saved = json.load(fh)["group_0"]

    kinematic = scenario.heliostat_field.heliostat_groups[0].kinematics

    kinematic.translation_deviation_parameters.data.copy_(
        torch.tensor(saved["translation_deviation_parameters"], dtype=torch.float32, device=device)
    )
    kinematic.rotation_deviation_parameters.data.copy_(
        torch.tensor(saved["rotation_deviation_parameters"], dtype=torch.float32, device=device)
    )
    kinematic.actuators.optimizable_parameters.data.copy_(
        torch.tensor(saved["actuator_optimizable_parameters"], dtype=torch.float32, device=device)
    )
    kinematic.actuators.non_optimizable_parameters.data.copy_(
        torch.tensor(saved["actuator_nonoptimizable_parameters"], dtype=torch.float32, device=device)
    )
    kinematic._base_position_deviation = torch.tensor(
        saved["base_position_deviation_parameters"], dtype=torch.float32, device=device
    )


def _dummy_mapping(heliostat_id: str, n_samples: int) -> list:
    """SyntheticDatasetParser only needs (hid, len(cal), _) — file paths are ignored."""
    return [(heliostat_id, [None] * n_samples, [])]


@torch.no_grad()
def _run_inference(
    scenario,
    heliostat_id: str,
    n_samples: int,
    n_rays: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Returns (measured, predicted, fse_mrad_list) each shaped [n_samples, H, W].
    Images are peak-normalised and Gaussian-blurred (sigma=1) for display.
    """
    scenario.set_number_of_rays(n_rays)
    test_parser = SyntheticDatasetParser(cfg.SYNTH_DATA_DIR / "test")
    mapping     = _dummy_mapping(heliostat_id, n_samples)

    heliostat_group = scenario.heliostat_field.heliostat_groups[0]

    (
        measured_flux,
        focal_spots,
        incident_ray_directions,
        _,
        active_heliostats_mask,
        target_area_mask,
    ) = test_parser.parse_data_for_reconstruction(
        heliostat_data_mapping=mapping,
        heliostat_group=heliostat_group,
        scenario=scenario,
        device=device,
    )

    if active_heliostats_mask.sum() == 0:
        raise RuntimeError(f"No active heliostats found for {heliostat_id}.")

    heliostat_group.activate_heliostats(active_heliostats_mask=active_heliostats_mask, device=device)
    kinematic = heliostat_group.kinematics

    if hasattr(kinematic, "_base_position_deviation"):
        base_dev = kinematic._base_position_deviation.repeat_interleave(active_heliostats_mask, dim=0)
        pad = torch.zeros(base_dev.shape[0], 1, device=device)
        kinematic.active_heliostat_positions = (
            kinematic.active_heliostat_positions + torch.cat([base_dev, pad], dim=1)
        )

    heliostat_group.align_surfaces_with_incident_ray_directions(
        aim_points=scenario.solar_tower.get_centers_of_target_areas(target_area_mask, device=device),
        incident_ray_directions=incident_ray_directions,
        active_heliostats_mask=active_heliostats_mask,
        device=device,
    )

    bitmap_resolution = torch.tensor([256, 256], device=device)
    ray_tracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=heliostat_group,
        blocking_active=False,
        batch_size=min(int(active_heliostats_mask.sum().item()), 32),
        bitmap_resolution=bitmap_resolution,
    )
    predicted_sampler, _, _, _ = ray_tracer.trace_rays(
        incident_ray_directions=incident_ray_directions,
        active_heliostats_mask=active_heliostats_mask,
        target_area_indices=target_area_mask,
        device=device,
    )

    sample_indices = ray_tracer.get_sampler_indices()
    inv_perm       = torch.argsort(sample_indices)
    predicted_nat  = predicted_sampler[inv_perm]

    # FSE
    bitmap_coords    = get_center_of_mass(bitmaps=predicted_sampler, device=device)
    predicted_spots  = bitmap_coordinates_to_target_coordinates(
        bitmap_coordinates=bitmap_coords,
        bitmap_resolution=ray_tracer.bitmap_resolution,
        solar_tower=scenario.solar_tower,
        target_area_indices=target_area_mask[sample_indices],
        device=device,
    )
    fse_sampler = torch.norm(
        predicted_spots[:, :3] - focal_spots[sample_indices][:, :3], dim=1
    )
    fse_nat = fse_sampler[inv_perm]

    reference_target = scenario.solar_tower.target_areas[
        index_mapping.planar_target_areas
    ].centers[:, :3].mean(dim=0).to(device)
    active_idx = torch.where(active_heliostats_mask.bool())[0]
    distances  = torch.norm(
        heliostat_group.positions[active_idx, :3].to(device) - reference_target, dim=1
    )
    samples_per_hel = active_heliostats_mask[active_idx].long()

    # collect per-sample data
    measured_imgs  = []
    predicted_imgs = []
    fse_mrad_list  = []

    offset = 0
    for j in range(len(active_idx)):
        n = samples_per_hel[j].item()
        dist = distances[j].item()
        for s in range(n):
            meas_raw = measured_flux[offset + s].cpu()
            pred_raw = predicted_nat[offset + s].cpu()
            fse_val  = fse_nat[offset + s].item()
            fse_mrad = (fse_val / dist * 1000.0) if (dist > 0 and not math.isnan(fse_val)) else float("nan")

            pred_bl = _gaussian_blur_batch(pred_raw.unsqueeze(0), sigma=1.0).squeeze(0)
            pred_vis = (pred_bl / pred_bl.max().clamp(min=1e-12)).numpy()

            meas_bl = _gaussian_blur_batch(meas_raw.unsqueeze(0), sigma=1.0).squeeze(0)
            meas_vis = (meas_bl / meas_bl.max().clamp(min=1e-12)).numpy()

            measured_imgs.append(meas_vis)
            predicted_imgs.append(pred_vis)
            fse_mrad_list.append(fse_mrad)

        offset += n

    return np.array(measured_imgs), np.array(predicted_imgs), fse_mrad_list


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_flux_grid(
    measured: np.ndarray,
    predicted: np.ndarray,
    fse_mrad_list: list[float],
    heliostat_id: str,
    train_size: int,
    n_rays: int,
    output_path: pathlib.Path,
    n_cols: int = 10,
) -> None:
    """
    Compact grid for any number of samples.

    Layout: pairs of rows — top row = measured, bottom row = predicted.
    Each group of n_cols samples occupies 2 rows. FSE is shown as a column title.
    For small n (<= 6) falls back to the 3-column (meas / pred / diff) layout.
    """
    n = len(measured)

    if n <= 6:
        _plot_flux_grid_small(measured, predicted, fse_mrad_list,
                              heliostat_id, train_size, n_rays, output_path)
        return

    # --- compact grid for large n ---
    n_cols  = min(n_cols, n)
    n_batch = math.ceil(n / n_cols)   # number of row-pairs
    n_rows  = n_batch * 2             # 2 rows per batch: measured + predicted

    img_h = 1.5   # inches per image cell
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * img_h, n_rows * img_h + 0.8),
        gridspec_kw={"hspace": 0.05, "wspace": 0.02},
    )
    fig.patch.set_facecolor("white")

    for b in range(n_batch):
        meas_row  = b * 2
        pred_row  = b * 2 + 1
        for c in range(n_cols):
            idx = b * n_cols + c
            ax_m = axes[meas_row, c]
            ax_p = axes[pred_row, c]

            if idx < n:
                fse_str = f"{fse_mrad_list[idx]:.2f} mrad" if not math.isnan(fse_mrad_list[idx]) else "NaN"
                ax_m.imshow(measured[idx],  cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
                ax_p.imshow(predicted[idx], cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
                ax_m.set_title(f"#{idx}\n{fse_str}", fontsize=6.5, pad=2)
            else:
                ax_m.set_visible(False)
                ax_p.set_visible(False)
                continue

            ax_m.axis("off")
            ax_p.axis("off")

        # row labels on the left
        axes[meas_row, 0].set_ylabel("meas", fontsize=7, rotation=0, labelpad=22, va="center")
        axes[pred_row, 0].set_ylabel("pred", fontsize=7, rotation=0, labelpad=22, va="center")

    mean_fse = float(np.nanmean(fse_mrad_list))
    fig.suptitle(
        f"Flux comparison — {heliostat_id}  |  train_size={train_size}  |  "
        f"n_rays={n_rays}  |  mean FSE={mean_fse:.3f} mrad",
        fontsize=11, fontweight="bold",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


def _plot_flux_grid_small(
    measured: np.ndarray,
    predicted: np.ndarray,
    fse_mrad_list: list[float],
    heliostat_id: str,
    train_size: int,
    n_rays: int,
    output_path: pathlib.Path,
) -> None:
    """3-column layout (measured / predicted / |diff|) for small n."""
    n = len(measured)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    fig.patch.set_facecolor("white")
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        diff = np.abs(predicted[i] - measured[i])
        fse_str = f"{fse_mrad_list[i]:.3f} mrad" if not math.isnan(fse_mrad_list[i]) else "NaN"
        for col, (img, title, cmap, vmax) in enumerate([
            (measured[i],  "Measured",   "inferno", 1.0),
            (predicted[i], "Predicted",  "inferno", 1.0),
            (diff,         f"|diff|  FSE={fse_str}", "hot", None),
        ]):
            ax = axes[i, col]
            vmax_arg = vmax if vmax is not None else (diff.max() or 1e-6)
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax_arg, interpolation="nearest")
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axes[i, 0].set_ylabel(f"#{i}", fontsize=9, rotation=0, labelpad=30, va="center")

    fig.suptitle(
        f"Flux comparison — {heliostat_id}  |  train_size={train_size}  |  n_rays={n_rays}",
        fontsize=13, fontweight="bold", y=1.002,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize measured vs predicted flux.")
    parser.add_argument("--heliostat-id", default="AO34",
                        help="Heliostat ID (must exist in one_heliostat_scenarios/ and outputs/)")
    parser.add_argument("--train-size", type=int, default=20,
                        help="Training sample count (must match a train_size_N subdir in outputs/)")
    parser.add_argument("--n-samples", type=int, default=6,
                        help="Number of test samples to visualise (evenly spaced)")
    parser.add_argument("--n-rays", type=int, default=50,
                        help="Number of rays for inference (higher = cleaner image, slower)")
    parser.add_argument("--surface-pts", type=int, default=25,
                        help="Surface points per facet (NxN); must match training config")
    parser.add_argument("--output", type=pathlib.Path, default=None,
                        help="Output PNG path (default: outputs/one_hel_train_sizes/{hel}/{size}/flux_grid.png)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Heliostat: {args.heliostat_id}  train_size: {args.train_size}  "
          f"n_samples: {args.n_samples}  n_rays: {args.n_rays}")

    # Resolve output path
    output_path = args.output or (
        cfg.BASE_DIR / "outputs" / "one_hel_train_sizes" / args.heliostat_id
        / f"train_size_{args.train_size}" / "flux_grid.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check trained params exist
    params_path = (
        cfg.BASE_DIR / "outputs" / "one_hel_train_sizes" / args.heliostat_id
        / f"train_size_{args.train_size}" / "kinematic_parameters.json"
    )
    if not params_path.exists():
        print(f"ERROR: kinematic_parameters.json not found at {params_path}")
        sys.exit(1)

    # Count available test samples
    test_hel_dir = cfg.SYNTH_DATA_DIR / "test" / args.heliostat_id
    if not test_hel_dir.exists():
        print(f"ERROR: Test data not found: {test_hel_dir}")
        sys.exit(1)
    n_available = len(sorted(test_hel_dir.iterdir()))
    n_to_load   = min(args.n_samples, n_available)
    if n_to_load < args.n_samples:
        print(f"WARNING: Only {n_available} test samples available; using {n_to_load}.")

    print(f"Loading scenario …")
    scenario = _load_scenario(args.heliostat_id, device, args.surface_pts)

    print(f"Applying trained kinematic parameters from {params_path} …")
    _apply_kinematic_parameters(scenario, params_path, device)

    print(f"Running inference on {n_to_load} test samples ({args.n_rays} rays) …")
    measured, predicted, fse_mrad_list = _run_inference(
        scenario=scenario,
        heliostat_id=args.heliostat_id,
        n_samples=n_to_load,
        n_rays=args.n_rays,
        device=device,
    )

    # Subsample evenly if we got more samples than requested
    if len(measured) > args.n_samples:
        indices       = np.linspace(0, len(measured) - 1, args.n_samples, dtype=int)
        measured      = measured[indices]
        predicted     = predicted[indices]
        fse_mrad_list = [fse_mrad_list[i] for i in indices]

    for i, fse in enumerate(fse_mrad_list):
        fse_str = f"{fse:.4f}" if not math.isnan(fse) else "NaN"
        print(f"  sample {i:2d}: FSE = {fse_str} mrad")

    print(f"Plotting …")
    plot_flux_grid(
        measured=measured,
        predicted=predicted,
        fse_mrad_list=fse_mrad_list,
        heliostat_id=args.heliostat_id,
        train_size=args.train_size,
        n_rays=args.n_rays,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
