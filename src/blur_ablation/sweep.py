"""
Core ray-tracing sweep for the Gaussian blur ablation study.

Provides two main functions:
  - trace_flux_for_mapping: trace rays for a set of heliostats, return per-heliostat flux.
  - run_blur_sweep: run all (n_rays × sigma) configs vs a high-quality reference.
"""

import logging

import torch
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario

from artist_extensions.kinematic_reconstructors import WortbergPixelReconstructor

log = logging.getLogger(__name__)

_gaussian_blur = WortbergPixelReconstructor._gaussian_blur
_peak_normalize = WortbergPixelReconstructor._peak_normalize


# ---------------------------------------------------------------------------
# Low-level: trace rays → per-heliostat flux dict
# ---------------------------------------------------------------------------

def trace_flux_for_mapping(
    scenario: Scenario,
    heliostat_data_mapping: list[tuple[str, list, list]],
    data_parser: PaintCalibrationDataParser,
    device: torch.device,
    bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
    ray_tracing_batch_size: int = 32,
) -> dict[str, torch.Tensor]:
    """Trace rays for each heliostat in the mapping and return per-heliostat flux.

    Returns
    -------
    dict[str, torch.Tensor]
        {heliostat_name: flux [N_meas, H, W]} on CPU, raw (un-normalised) intensity.
    """
    result: dict[str, torch.Tensor] = {}

    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        (
            _measured_flux,
            _focal_spots,
            incident_ray_directions,
            _,
            active_heliostats_mask,
            target_area_mask,
        ) = data_parser.parse_data_for_reconstruction(
            heliostat_data_mapping=heliostat_data_mapping,
            heliostat_group=heliostat_group,
            scenario=scenario,
            bitmap_resolution=bitmap_resolution,
            device=device,
        )

        if active_heliostats_mask.sum() == 0:
            continue

        heliostat_group.activate_heliostats(
            active_heliostats_mask=active_heliostats_mask,
            device=device,
        )

        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=scenario.target_areas.centers[target_area_mask],
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            device=device,
        )

        ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=heliostat_group,
            blocking_active=False,
            batch_size=min(heliostat_group.number_of_active_heliostats, ray_tracing_batch_size),
            bitmap_resolution=bitmap_resolution.to(device),
        )

        with torch.no_grad():
            predicted_flux = ray_tracer.trace_rays(
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_heliostats_mask,
                target_area_mask=target_area_mask,
                device=device,
            )  # [N_active_heliostats * N_meas, H, W]

        # Split the flat batch back into per-heliostat tensors.
        # active_heliostats_mask[i] == N_meas for active heliostats, 0 otherwise.
        active_indices = torch.where(active_heliostats_mask > 0)[0]
        n_active = len(active_indices)
        if n_active == 0:
            continue

        n_meas = int(active_heliostats_mask[active_indices[0]].item())
        # predicted_flux shape: [n_active * n_meas, H, W]
        H, W = predicted_flux.shape[1], predicted_flux.shape[2]
        per_heliostat_flux = predicted_flux.reshape(n_active, n_meas, H, W)

        for i, idx in enumerate(active_indices):
            name = heliostat_group.names[idx.item()]
            result[name] = per_heliostat_flux[i].cpu()  # [n_meas, H, W]

    return result


# ---------------------------------------------------------------------------
# High-level: run the full sweep
# ---------------------------------------------------------------------------

def run_blur_sweep(
    scenarios: dict[int, Scenario],
    selected_mapping: list[tuple[str, list, list]],
    data_parser: PaintCalibrationDataParser,
    rays_configs: list[int],
    sigma_configs: list[float],
    ref_rays: int,
    device: torch.device,
    bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
) -> list[dict]:
    """Run the full (surface_pts × n_rays × sigma) grid and compute MSE vs reference.

    For each surface_pts config the reference is computed at (surface_pts, ref_rays, no blur),
    so comparisons are always within the same surface discretisation.

    Parameters
    ----------
    scenarios : dict[int, Scenario]
        {surface_pts: scenario} — one loaded scenario per surface discretisation,
        e.g. {25: scenario_25, 50: scenario_50, 75: scenario_75, 100: scenario_100}.
    selected_mapping : list
        Filtered heliostat_data_mapping containing only the selected heliostats.
    data_parser : PaintCalibrationDataParser
        Parser configured with sample_limit (10 for train split).
    rays_configs : list[int]
        Number-of-rays values to sweep, e.g. [10, 20, 50].
    sigma_configs : list[float]
        Gaussian blur sigma values to sweep, e.g. [0, 1, 2, 3, 5, 7, 10].
    ref_rays : int
        Number of rays for the reference simulation (e.g. 200).
    device : torch.device

    Returns
    -------
    list[dict]
        One record per (surface_pts, heliostat, n_rays, sigma):
        {
          "surface_pts": int,          # surface discretisation (N in N×N)
          "heliostat_id": str,
          "n_rays": int,
          "sigma": float,
          "mse": float,                # mean MSE over all measurements
          "mse_std": float,            # std over measurements
        }
    """
    records = []

    for surface_pts, scenario in sorted(scenarios.items()):
        log.info(f"=== Surface config: {surface_pts}×{surface_pts} ===")

        # ------------------------------------------------------------------ #
        # Step 1: reference flux for this surface config (ref_rays, no blur)
        # ------------------------------------------------------------------ #
        log.info(f"  Computing reference flux ({surface_pts}×{surface_pts}, {ref_rays} rays) …")
        scenario.set_number_of_rays(ref_rays)

        ref_flux = trace_flux_for_mapping(
            scenario=scenario,
            heliostat_data_mapping=selected_mapping,
            data_parser=data_parser,
            device=device,
            bitmap_resolution=bitmap_resolution,
        )  # {name: [n_meas, H, W] on CPU}

        ref_norm = {
            name: _peak_normalize(flux.to(device)).cpu()
            for name, flux in ref_flux.items()
        }
        log.info(f"  Reference flux computed for {len(ref_norm)} heliostats.")

        # ------------------------------------------------------------------ #
        # Step 2: sweep (n_rays × sigma) for this surface config
        # ------------------------------------------------------------------ #
        total_configs = len(rays_configs) * len(sigma_configs)
        cfg_idx = 0

        for n_rays in rays_configs:
            log.info(f"  Setting n_rays = {n_rays} …")
            scenario.set_number_of_rays(n_rays)

            test_flux = trace_flux_for_mapping(
                scenario=scenario,
                heliostat_data_mapping=selected_mapping,
                data_parser=data_parser,
                device=device,
                bitmap_resolution=bitmap_resolution,
            )  # {name: [n_meas, H, W] on CPU}

            for sigma in sigma_configs:
                cfg_idx += 1
                log.info(f"  Config {cfg_idx}/{total_configs}: surface={surface_pts}×{surface_pts}, n_rays={n_rays}, sigma={sigma}")

                for name, flux_cpu in test_flux.items():
                    if name not in ref_norm:
                        continue

                    flux_dev = flux_cpu.to(device)

                    # Apply Gaussian blur then peak-normalise.
                    blurred = _gaussian_blur(flux_dev, sigma)       # [n_meas, H, W]
                    test_norm = _peak_normalize(blurred).cpu()      # [n_meas, H, W]

                    ref = ref_norm[name]  # [n_meas, H, W]

                    # MSE per measurement image, then average + std.
                    mse_per_meas = ((test_norm - ref) ** 2).mean(dim=(1, 2))  # [n_meas]
                    mse_mean = mse_per_meas.mean().item()
                    mse_std = mse_per_meas.std().item() if mse_per_meas.numel() > 1 else 0.0

                    records.append({
                        "surface_pts": surface_pts,
                        "heliostat_id": name,
                        "n_rays": n_rays,
                        "sigma": sigma,
                        "mse": mse_mean,
                        "mse_std": mse_std,
                    })

    log.info(f"Sweep complete. {len(records)} records collected.")
    return records
