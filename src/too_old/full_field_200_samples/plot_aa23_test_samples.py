"""
Plot all test-set flux predictions for heliostat AA23 using trained kinematic parameters.

Usage:
    python plot_aa23_test_samples.py \
        --run-dir ../../outputs/local_runs/full_field_200_20260518_235543 \
        --heliostat AA23

Produces:
    <run-dir>/aa23_all_test_samples.png  — grid of (measured, predicted) pairs
    <run-dir>/aa23_test_samples/         — individual per-sample PNGs
"""
import argparse
import json
import pathlib
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_here = pathlib.Path(__file__).resolve().parent
_src  = _here.parent
sys.path.insert(0, str(_src))

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario
from artist.util import index_mapping
from artist.util.environment_setup import get_device
from artist.util.utils import get_center_of_mass, bitmap_coordinates_to_target_coordinates
from five_heliostats_synth.data import SyntheticDatasetParser
from utils.evaluation import _gaussian_blur_batch

import config as cfg


# ---------------------------------------------------------------------------
# Kinematic parameter loading
# ---------------------------------------------------------------------------

def load_kinematic_parameters(path: pathlib.Path, scenario, device: torch.device) -> None:
    """Apply saved kinematic parameters back onto the scenario in-place."""
    with open(path) as f:
        payload = json.load(f)

    g = payload["group_0"]
    kinematic = scenario.heliostat_field.heliostat_groups[0].kinematics

    def t(data):
        return torch.tensor(data, dtype=torch.float32, device=device)

    with torch.no_grad():
        kinematic.translation_deviation_parameters.data.copy_(t(g["translation_deviation_parameters"]))
        kinematic.rotation_deviation_parameters.data.copy_(t(g["rotation_deviation_parameters"]))
        kinematic.actuators.optimizable_parameters.data.copy_(t(g["actuator_optimizable_parameters"]))
        kinematic.actuators.non_optimizable_parameters.data.copy_(t(g["actuator_nonoptimizable_parameters"]))
        if hasattr(kinematic, "_base_position_deviation"):
            kinematic._base_position_deviation.data.copy_(t(g["base_position_deviation_parameters"]))

    print(f"Kinematic parameters loaded from {path}")


# ---------------------------------------------------------------------------
# Inference for a single heliostat — all test samples
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference_for_heliostat(
    scenario,
    heliostat_id: str,
    test_data_dir: pathlib.Path,
    n_samples: int,
    device: torch.device,
    bitmap_resolution: torch.Tensor,
) -> dict:
    """
    Run ray tracing for all test samples of one heliostat.

    Returns a dict with:
        measured   : list of numpy arrays [H, W]  (peak-normalised, blurred)
        predicted  : list of numpy arrays [H, W]  (peak-normalised, blurred)
        fse_mrad   : list of float
        pixel_loss : list of float
    """
    heliostat_group = scenario.heliostat_field.heliostat_groups[0]

    # Dummy mapping — SyntheticDatasetParser only uses len(cal) per heliostat
    mapping = [(heliostat_id, [None] * n_samples, [None] * n_samples)]

    test_parser = SyntheticDatasetParser(test_data_dir)

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

    print(f"  Loaded {measured_flux.shape[0]} test samples for {heliostat_id}")

    heliostat_group.activate_heliostats(
        active_heliostats_mask=active_heliostats_mask, device=device
    )
    heliostat_group.align_surfaces_with_incident_ray_directions(
        aim_points=scenario.solar_tower.get_centers_of_target_areas(
            target_area_mask, device=device
        ),
        incident_ray_directions=incident_ray_directions,
        active_heliostats_mask=active_heliostats_mask,
        device=device,
    )

    ray_tracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=heliostat_group,
        blocking_active=False,
        batch_size=min(heliostat_group.number_of_active_heliostats, 32),
        bitmap_resolution=bitmap_resolution.to(device),
    )
    predicted_sampler, _, _, _ = ray_tracer.trace_rays(
        incident_ray_directions=incident_ray_directions,
        active_heliostats_mask=active_heliostats_mask,
        target_area_indices=target_area_mask,
        device=device,
    )

    sample_indices = ray_tracer.get_sampler_indices()
    inv_perm = torch.argsort(sample_indices)

    # Reorder to natural (per-heliostat sequential) order
    predicted_natural = predicted_sampler[inv_perm]
    focal_spots_sampler = focal_spots[sample_indices]

    # Focal spot errors
    bitmap_coords = get_center_of_mass(bitmaps=predicted_sampler, device=device)
    predicted_spots = bitmap_coordinates_to_target_coordinates(
        bitmap_coordinates=bitmap_coords,
        bitmap_resolution=ray_tracer.bitmap_resolution,
        solar_tower=scenario.solar_tower,
        target_area_indices=target_area_mask[sample_indices],
        device=device,
    )
    fse_sampler = torch.norm(
        predicted_spots[:, :3] - focal_spots_sampler[:, :3], dim=1
    )  # metres
    fse_natural = fse_sampler[inv_perm]

    reference_target = scenario.solar_tower.target_areas[
        index_mapping.planar_target_areas
    ].centers[:, :3].mean(dim=0).to(device)
    active_idx = torch.where(active_heliostats_mask.bool())[0][0]
    distance_m = torch.norm(
        heliostat_group.positions[active_idx, :3].to(device) - reference_target
    ).item()

    results = {"measured": [], "predicted": [], "fse_mrad": [], "pixel_loss": []}

    for k in range(measured_flux.shape[0]):
        meas_raw = measured_flux[k].cpu()
        pred_raw = predicted_natural[k].cpu()

        pred_blurred = _gaussian_blur_batch(pred_raw.unsqueeze(0), sigma=1.0).squeeze(0)
        meas_blurred = _gaussian_blur_batch(meas_raw.unsqueeze(0), sigma=1.0).squeeze(0)

        pred_vis = (pred_blurred / pred_blurred.max().clamp(min=1e-12)).numpy()
        meas_vis = (meas_blurred / meas_blurred.max().clamp(min=1e-12)).numpy()

        pixel_loss = float(abs(torch.from_numpy(pred_vis) - torch.from_numpy(meas_vis)).sum())

        fse_val = fse_natural[k].item()
        fse_mrad = (fse_val / distance_m) * 1000.0 if not np.isnan(fse_val) else float("nan")

        results["measured"].append(meas_vis)
        results["predicted"].append(pred_vis)
        results["fse_mrad"].append(fse_mrad)
        results["pixel_loss"].append(pixel_loss)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_grid(results: dict, heliostat_id: str, output_path: pathlib.Path) -> None:
    """Save a grid of (measured | predicted) pairs for all test samples."""
    n = len(results["measured"])
    cols = 2  # measured + predicted per sample
    n_cols_pairs = 7  # pairs per row
    n_cols = n_cols_pairs * cols
    n_rows = (n + n_cols_pairs - 1) // n_cols_pairs

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.7))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for k in range(n):
        row = k // n_cols_pairs
        col_pair = k % n_cols_pairs
        ax_meas = axes[row, col_pair * 2]
        ax_pred = axes[row, col_pair * 2 + 1]

        ax_meas.imshow(results["measured"][k], cmap="hot", vmin=0, vmax=1)
        ax_pred.imshow(results["predicted"][k], cmap="hot", vmin=0, vmax=1)

        fse = results["fse_mrad"][k]
        fse_str = f"{fse:.1f}" if not np.isnan(fse) else "nan"

        ax_meas.set_title(f"#{k:02d} meas", fontsize=5)
        ax_pred.set_title(f"pred {fse_str}mr", fontsize=5)
        ax_meas.axis("off")
        ax_pred.axis("off")

    # Hide unused axes
    for k in range(n, n_rows * n_cols_pairs):
        row = k // n_cols_pairs
        col_pair = k % n_cols_pairs
        axes[row, col_pair * 2].axis("off")
        axes[row, col_pair * 2 + 1].axis("off")

    valid_fse = [v for v in results["fse_mrad"] if not np.isnan(v)]
    mean_fse = np.mean(valid_fse) if valid_fse else float("nan")
    mean_px  = np.mean(results["pixel_loss"])
    fig.suptitle(
        f"{heliostat_id} — {n} test samples  |  mean FSE={mean_fse:.2f} mrad  "
        f"mean pixel_loss={mean_px:.2f}\n"
        f"Left = measured, Right = predicted (Gaussian blur σ=1, peak-normalised)",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Grid saved → {output_path}")


def _plot_individual(results: dict, heliostat_id: str, output_dir: pathlib.Path) -> None:
    """Save one PNG per test sample showing measured and predicted side by side."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for k in range(len(results["measured"])):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
        ax1.imshow(results["measured"][k], cmap="hot", vmin=0, vmax=1)
        ax1.set_title("Measured")
        ax1.axis("off")
        ax2.imshow(results["predicted"][k], cmap="hot", vmin=0, vmax=1)
        fse = results["fse_mrad"][k]
        fse_str = f"{fse:.2f} mrad" if not np.isnan(fse) else "nan mrad"
        ax2.set_title(f"Predicted  FSE={fse_str}")
        ax2.axis("off")
        fig.suptitle(f"{heliostat_id} — sample {k:02d}  pixel_loss={results['pixel_loss'][k]:.2f}", fontsize=9)
        fig.tight_layout()
        path = output_dir / f"sample_{k:02d}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
    print(f"Individual PNGs saved → {output_dir}/  ({len(results['measured'])} files)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot all test predictions for one heliostat.")
    parser.add_argument("--run-dir", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[2] /
                                "outputs/local_runs/full_field_200_20260518_235543",
                        help="Path to the experiment output directory.")
    parser.add_argument("--heliostat", default="AA23", help="Heliostat ID to plot.")
    parser.add_argument("--n-samples", type=int, default=49,
                        help="Number of test samples to use (default: 49 for AA23).")
    parser.add_argument("--no-individual", action="store_true",
                        help="Skip saving individual per-sample PNGs.")
    args = parser.parse_args()

    run_dir = args.run_dir
    heliostat_id = args.heliostat
    n_samples = args.n_samples

    device = get_device()
    print(f"Device: {device}")

    # Load scenario
    print(f"Loading scenario from {cfg.SCENARIO_PATH} …")
    with h5py.File(cfg.SCENARIO_PATH, "r") as f:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=f,
            device=device,
            number_of_surface_points_per_facet=torch.tensor(
                [cfg.SURFACE_POINTS_PER_FACET, cfg.SURFACE_POINTS_PER_FACET]
            ),
        )

    # Apply trained kinematic parameters
    kin_path = run_dir / "kinematic_parameters.json"
    load_kinematic_parameters(kin_path, scenario, device)

    # Test data directory
    test_data_dir = (
        pathlib.Path(cfg.SCENARIO_PATH).parent / "synthetic_data" / "test"
    )
    print(f"Test data dir: {test_data_dir}")

    # Run inference
    print(f"Running inference for {heliostat_id} ({n_samples} samples) …")
    results = run_inference_for_heliostat(
        scenario=scenario,
        heliostat_id=heliostat_id,
        test_data_dir=test_data_dir,
        n_samples=n_samples,
        device=device,
        bitmap_resolution=torch.tensor([256, 256]),
    )

    valid_fse = [v for v in results["fse_mrad"] if not np.isnan(v)]
    print(f"  mean FSE     = {np.mean(valid_fse):.3f} mrad")
    print(f"  median FSE   = {np.median(valid_fse):.3f} mrad")
    print(f"  mean px_loss = {np.mean(results['pixel_loss']):.3f}")

    # Save grid
    grid_path = run_dir / f"{heliostat_id.lower()}_all_test_samples.png"
    _plot_grid(results, heliostat_id, grid_path)

    # Save individual PNGs
    if not args.no_individual:
        individual_dir = run_dir / f"{heliostat_id.lower()}_test_samples"
        _plot_individual(results, heliostat_id, individual_dir)


if __name__ == "__main__":
    main()
