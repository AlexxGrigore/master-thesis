"""
Perturbation Visualizer
=======================
Generates a side-by-side flux comparison for a single heliostat under a
user-defined kinematic perturbation.

Usage
-----
    python run.py            # use config.py in this directory
    python run.py --config /path/to/other_config.py

Outputs (written to config.OUTPUT_DIR)
---------------------------------------
    clean.png          — flux image with zero perturbation
    perturbed.png      — flux image with PERTURBATIONS from config.py
    comparison.png     — side-by-side plot with centroid markers & shift info
    results.json       — centroid positions and shift in metres
"""
import argparse
import importlib.util
import json
import logging
import pathlib
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle

from artist.geometry import bitmap_coordinates_to_target_coordinates
from artist.geometry.coordinates import (
    azimuth_elevation_to_enu,
    convert_3d_points_to_4d_format,
)
from artist.flux import get_center_of_mass
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario
from artist.util import constants as config_dictionary, set_logger_config
from artist.util import get_device, setup_distributed_environment

# Make src/ importable so utils.synth_data resolves correctly.
_here = pathlib.Path(__file__).resolve().parent
_src  = _here.parent       # master-thesis/src/
sys.path.insert(0, str(_src))

from utils.synth_data import apply_perturbations, reset_perturbations

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config(config_path: pathlib.Path):
    """Import a config.py file from an arbitrary path and return the module."""
    spec   = importlib.util.spec_from_file_location("cfg", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Incident-ray direction
# ---------------------------------------------------------------------------

def _build_incident_ray(azimuth_deg: float, elevation_deg: float, device: torch.device) -> torch.Tensor:
    """
    Convert a sun position given as (azimuth, elevation) to the ARTIST
    incident-ray direction convention: a 4-D homogeneous direction vector
    pointing FROM the sun TO the scene (w = 0).

    azimuth  : south-oriented, 0° = south, 90° = west.
    elevation: degrees above horizon.
    """
    az  = torch.tensor([azimuth_deg],   dtype=torch.float32, device=device)
    el  = torch.tensor([elevation_deg], dtype=torch.float32, device=device)
    enu = azimuth_elevation_to_enu(az, el, degree=True, device=device)          # [1, 3]
    sun_pos_4d = convert_3d_points_to_4d_format(enu, device=device)             # [1, 4], w=1
    origin_4d  = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)
    return origin_4d - sun_pos_4d                                                # [1, 4], w=0


# ---------------------------------------------------------------------------
# Single-heliostat forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def _forward_pass(
    scenario,
    heliostat_group,
    heliostat_idx: int,
    incident_ray: torch.Tensor,   # [1, 4]
    target_area_idx: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the ray-tracer for a single heliostat with a single sun direction.

    Returns
    -------
    centroid : torch.Tensor  [1, 4]  — focal-spot position in ENU+w coords
    flux     : torch.Tensor  [1, H, W] — normalised flux bitmap
    """
    n_heliostats = heliostat_group.kinematics.heliostat_positions.shape[0]

    # active_mask: 1 for the target heliostat, 0 for all others.
    active_mask = torch.zeros(n_heliostats, dtype=torch.int32, device=device)
    active_mask[heliostat_idx] = 1

    # target_mask: one entry per active sample (here just one sample).
    target_mask = torch.tensor([target_area_idx], dtype=torch.long, device=device)

    heliostat_group.activate_heliostats(active_heliostats_mask=active_mask, device=device)
    kinematic = heliostat_group.kinematics

    # Apply base-position deviation if set by apply_perturbations.
    if hasattr(kinematic, "_base_position_deviation"):
        base_dev = kinematic._base_position_deviation[heliostat_idx].unsqueeze(0)  # [1, 3]
        pad      = torch.zeros(1, 1, device=device)
        kinematic.active_heliostat_positions = (
            kinematic.active_heliostat_positions + torch.cat([base_dev, pad], dim=1)
        )

    aim_points = scenario.solar_tower.get_centers_of_target_areas(target_mask, device=device)
    heliostat_group.align_surfaces_with_incident_ray_directions(
        aim_points=aim_points,
        incident_ray_directions=incident_ray,
        active_heliostats_mask=active_mask,
        device=device,
    )

    ray_tracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=heliostat_group,
        blocking_active=False,
        world_size=1,
        rank=0,
        batch_size=1,
        random_seed=42,
    )
    flux_sampler, _, _, _ = ray_tracer.trace_rays(
        incident_ray_directions=incident_ray,
        active_heliostats_mask=active_mask,
        target_area_indices=target_mask,
        device=device,
    )

    bitmap_coords = get_center_of_mass(bitmaps=flux_sampler, device=device)
    centroid = bitmap_coordinates_to_target_coordinates(
        bitmap_coordinates=bitmap_coords,
        bitmap_resolution=ray_tracer.bitmap_resolution,
        solar_tower=scenario.solar_tower,
        target_area_indices=target_mask,
        device=device,
    )  # [1, 4]

    return centroid, flux_sampler   # flux_sampler: [1, H, W]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _flux_to_uint8(flux: torch.Tensor) -> np.ndarray:
    """Normalise a [H, W] flux tensor to uint8 for display."""
    arr  = flux.cpu().numpy()
    fmax = arr.max()
    if fmax > 1e-12:
        arr = (arr / fmax * 255).clip(0, 255).astype(np.uint8)
    else:
        arr = np.zeros_like(arr, dtype=np.uint8)
    return arr


def _bitmap_centroid_px(flux: torch.Tensor) -> tuple[float, float]:
    """Return the (col, row) centroid in pixel coordinates."""
    arr = flux.cpu().float().numpy()
    H, W = arr.shape
    total = arr.sum()
    if total < 1e-12:
        return W / 2.0, H / 2.0
    col_idx = np.arange(W, dtype=np.float32)
    row_idx = np.arange(H, dtype=np.float32)
    cx = (arr * col_idx[None, :]).sum() / total
    cy = (arr * row_idx[:, None]).sum() / total
    return float(cx), float(cy)


def _save_comparison(
    flux_clean:     torch.Tensor,  # [H, W]
    flux_perturbed: torch.Tensor,  # [H, W]
    centroid_clean:     torch.Tensor,  # [1, 4]  ENU+w
    centroid_perturbed: torch.Tensor,  # [1, 4]
    shift_m:   float,
    output_dir: pathlib.Path,
    heliostat_id: str,
    azimuth_deg:   float,
    elevation_deg: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    img_clean     = _flux_to_uint8(flux_clean)
    img_perturbed = _flux_to_uint8(flux_perturbed)
    cx_c, cy_c   = _bitmap_centroid_px(flux_clean)
    cx_p, cy_p   = _bitmap_centroid_px(flux_perturbed)

    enu_c = centroid_clean[0, :3].cpu().numpy()
    enu_p = centroid_perturbed[0, :3].cpu().numpy()

    # --- Individual PNGs ---
    from PIL import Image
    Image.fromarray(img_clean,     mode="L").save(output_dir / "clean.png")
    Image.fromarray(img_perturbed, mode="L").save(output_dir / "perturbed.png")

    # --- Side-by-side comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Heliostat {heliostat_id}  |  az={azimuth_deg:.1f}°  el={elevation_deg:.1f}°",
        fontsize=13, fontweight="bold",
    )

    for ax, img, cx, cy, label, color in [
        (axes[0], img_clean,     cx_c, cy_c, "Clean (no perturbation)", "lime"),
        (axes[1], img_perturbed, cx_p, cy_p, "Perturbed",               "red"),
    ]:
        ax.imshow(img, cmap="hot", origin="upper", interpolation="nearest")
        ax.add_patch(Circle((cx, cy), radius=max(2, img.shape[1] * 0.015),
                            edgecolor=color, facecolor="none", linewidth=1.5))
        ax.plot(cx, cy, "+", color=color, markersize=8, markeredgewidth=1.5)
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    # Annotate shift below the images.
    note = (
        f"Centroid shift: {shift_m * 100:.2f} cm  "
        f"(ΔE={enu_p[0]-enu_c[0]:+.3f} m,  "
        f"ΔN={enu_p[1]-enu_c[1]:+.3f} m,  "
        f"ΔU={enu_p[2]-enu_c[2]:+.3f} m)"
    )
    fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=10,
             style="italic", color="#333333")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Comparison saved → {output_dir / 'comparison.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize perturbation effect on flux image.")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=_here / "config.py",
        help="Path to a config.py file (default: config.py next to run.py).",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config.resolve())

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)

    if not cfg.SCENARIO_PATH.exists():
        sys.exit(f"Scenario not found: {cfg.SCENARIO_PATH}")

    device   = get_device()
    n_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(cfg.SCENARIO_PATH)

    with setup_distributed_environment(
        number_of_heliostat_groups=n_groups, device=device
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        with h5py.File(cfg.SCENARIO_PATH, "r") as f:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=f,
                device=device,
                number_of_surface_points_per_facet=torch.tensor(
                    [cfg.SURFACE_POINTS_PER_FACET, cfg.SURFACE_POINTS_PER_FACET]
                ),
            )
        scenario.set_number_of_rays(cfg.N_RAYS)

        heliostat_group = scenario.heliostat_field.heliostat_groups[0]
        heliostat_ids   = list(heliostat_group.names)

        if cfg.HELIOSTAT_ID not in heliostat_ids:
            sys.exit(
                f"Heliostat '{cfg.HELIOSTAT_ID}' not found in scenario.\n"
                f"Available IDs: {heliostat_ids}"
            )
        heliostat_idx = heliostat_ids.index(cfg.HELIOSTAT_ID)
        log.info(f"Target heliostat: {cfg.HELIOSTAT_ID}  (index {heliostat_idx})")

        incident_ray = _build_incident_ray(
            cfg.SUN_AZIMUTH_DEG, cfg.SUN_ELEVATION_DEG, device
        )
        log.info(
            f"Sun  az={cfg.SUN_AZIMUTH_DEG:.1f}°  el={cfg.SUN_ELEVATION_DEG:.1f}°  "
            f"→  incident ray {incident_ray[0].tolist()}"
        )

        # ------------------------------------------------------------------ #
        # 1. Clean forward pass (no perturbation)                             #
        # ------------------------------------------------------------------ #
        log.info("Running clean forward pass …")
        centroid_clean, flux_clean = _forward_pass(
            scenario, heliostat_group,
            heliostat_idx, incident_ray,
            cfg.TARGET_AREA_INDEX, device,
        )
        enu_clean = centroid_clean[0, :3].cpu()
        log.info(f"  Clean centroid ENU: {enu_clean.tolist()}")

        # ------------------------------------------------------------------ #
        # 2. Build perturbations tensor (non-zero only for target heliostat)  #
        # ------------------------------------------------------------------ #
        n_hel = len(heliostat_ids)
        p     = cfg.PERTURBATIONS

        def _sparse(values: list, cols: int) -> torch.Tensor:
            """[N_hel, cols] tensor, non-zero only at heliostat_idx."""
            t = torch.zeros(n_hel, cols, device=device)
            t[heliostat_idx] = torch.tensor(values, dtype=torch.float32, device=device)
            return t

        perturbations = {
            "rotation":        _sparse(p["rotation_rad"],        4),
            "actuator_angle":  _sparse(p["actuator_angle_rad"],  2),
            "actuator_stroke": _sparse(p["actuator_stroke_m"],   2),
            "actuator_offset": _sparse(p["actuator_offset_m"],   2),
            "translation":     _sparse(p["translation_m"],       9),
            "base_position":   _sparse(p["base_position_m"],     3),
        }

        log.info("Perturbations (target heliostat only):")
        for k, v in p.items():
            log.info(f"  {k}: {v}")

        # ------------------------------------------------------------------ #
        # 3. Perturbed forward pass                                           #
        # ------------------------------------------------------------------ #
        original = apply_perturbations(heliostat_group.kinematics, perturbations, device)
        log.info("Running perturbed forward pass …")

        centroid_perturbed, flux_perturbed = _forward_pass(
            scenario, heliostat_group,
            heliostat_idx, incident_ray,
            cfg.TARGET_AREA_INDEX, device,
        )
        reset_perturbations(heliostat_group.kinematics, original)
        log.info("Perturbations reset.")

        enu_perturbed = centroid_perturbed[0, :3].cpu()
        log.info(f"  Perturbed centroid ENU: {enu_perturbed.tolist()}")

        shift_m = float((enu_perturbed - enu_clean).norm())
        log.info(f"  Centroid shift: {shift_m * 100:.2f} cm")

        # ------------------------------------------------------------------ #
        # 4. Save results                                                     #
        # ------------------------------------------------------------------ #
        cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        results = {
            "heliostat_id":     cfg.HELIOSTAT_ID,
            "sun_azimuth_deg":  cfg.SUN_AZIMUTH_DEG,
            "sun_elevation_deg": cfg.SUN_ELEVATION_DEG,
            "target_area_index": cfg.TARGET_AREA_INDEX,
            "centroid_clean_enu":     enu_clean.tolist(),
            "centroid_perturbed_enu": enu_perturbed.tolist(),
            "centroid_shift_m":       shift_m,
            "perturbations":          p,
        }
        with open(cfg.OUTPUT_DIR / "results.json", "w") as fh:
            json.dump(results, fh, indent=2)
        log.info(f"Results saved → {cfg.OUTPUT_DIR / 'results.json'}")

        _save_comparison(
            flux_clean=flux_clean[0],
            flux_perturbed=flux_perturbed[0],
            centroid_clean=centroid_clean,
            centroid_perturbed=centroid_perturbed,
            shift_m=shift_m,
            output_dir=cfg.OUTPUT_DIR,
            heliostat_id=cfg.HELIOSTAT_ID,
            azimuth_deg=cfg.SUN_AZIMUTH_DEG,
            elevation_deg=cfg.SUN_ELEVATION_DEG,
        )

        log.info("Done.")


if __name__ == "__main__":
    main()
