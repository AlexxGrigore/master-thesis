"""
Core ray-tracing sweep for the Gaussian blur ablation study.

Provides two main functions:
  - trace_flux_for_mapping: trace rays for a set of heliostats, return per-heliostat flux.
  - run_blur_sweep: run all (n_rays × sigma) configs vs a high-quality reference.
"""

import logging
import math

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
    heliostat_chunk_size: int | None = None,
) -> dict[str, torch.Tensor]:
    """Trace rays for each heliostat in the mapping and return per-heliostat flux.

    Parameters
    ----------
    heliostat_chunk_size : int or None
        If set, process at most this many heliostats per forward pass to cap
        GPU memory. None (default) processes the whole active group at once.

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

        N_samples = int(active_heliostats_mask.max().item())
        active_group_positions = torch.where(active_heliostats_mask > 0)[0]
        N_active = len(active_group_positions)
        chunk_size = heliostat_chunk_size if heliostat_chunk_size is not None else N_active

        with torch.no_grad():
            for c_start in range(0, N_active, chunk_size):
                c_end = min(c_start + chunk_size, N_active)
                chunk_active_local = list(range(c_start, c_end))
                K = len(chunk_active_local)

                chunk_mask = torch.zeros_like(active_heliostats_mask)
                chunk_mask[active_group_positions[chunk_active_local]] = N_samples

                data_rows = torch.cat([
                    torch.arange(i * N_samples, (i + 1) * N_samples, device=device)
                    for i in chunk_active_local
                ])
                chunk_incident = incident_ray_directions[data_rows]
                chunk_target_mask = target_area_mask[data_rows]

                heliostat_group.activate_heliostats(
                    active_heliostats_mask=chunk_mask,
                    device=device,
                )
                heliostat_group.align_surfaces_with_incident_ray_directions(
                    aim_points=scenario.target_areas.centers[chunk_target_mask],
                    incident_ray_directions=chunk_incident,
                    active_heliostats_mask=chunk_mask,
                    device=device,
                )

                ray_tracer = HeliostatRayTracer(
                    scenario=scenario,
                    heliostat_group=heliostat_group,
                    blocking_active=False,
                    batch_size=min(K * N_samples, ray_tracing_batch_size),
                    bitmap_resolution=bitmap_resolution.to(device),
                )

                predicted_flux = ray_tracer.trace_rays(
                    incident_ray_directions=chunk_incident,
                    active_heliostats_mask=chunk_mask,
                    target_area_mask=chunk_target_mask,
                    device=device,
                )  # [K * N_samples, H, W]

                H, W = predicted_flux.shape[1], predicted_flux.shape[2]
                per_chunk_flux = predicted_flux.reshape(K, N_samples, H, W)

                for local_i, global_local in enumerate(chunk_active_local):
                    name = heliostat_group.names[active_group_positions[global_local].item()]
                    result[name] = per_chunk_flux[local_i].cpu()  # [N_samples, H, W]

    return result


# ---------------------------------------------------------------------------
# High-level: run the full sweep
# ---------------------------------------------------------------------------

def run_blur_sweep(
    scenario: Scenario,
    selected_mapping: list[tuple[str, list, list]],
    data_parser: PaintCalibrationDataParser,
    rays_configs: list[int],
    sigma_configs: list[float],
    ref_rays: int,
    device: torch.device,
    bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
    heliostat_chunk_size: int | None = None,
) -> list[dict]:
    """Run the full (n_rays × sigma) grid and compute MSE vs high-quality reference.

    Parameters
    ----------
    scenario : Scenario
        Scenario used for both the reference pass (ref_rays) and all test configs.
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
        One record per (heliostat, n_rays, sigma):
        {
          "heliostat_id": str,
          "n_rays": int,
          "sigma": float,
          "mse": float,               # mean MSE over all measurements
          "mse_std": float,           # std over measurements
        }
    """
    # ------------------------------------------------------------------ #
    # Step 1: compute reference flux (25×25, ref_rays, no blur)
    # ------------------------------------------------------------------ #
    log.info(f"Computing reference flux ({ref_rays} rays) …")
    scenario.set_number_of_rays(ref_rays)

    ref_flux = trace_flux_for_mapping(
        scenario=scenario,
        heliostat_data_mapping=selected_mapping,
        data_parser=data_parser,
        device=device,
        bitmap_resolution=bitmap_resolution,
        heliostat_chunk_size=heliostat_chunk_size,
    )  # {name: [n_meas, H, W] on CPU}

    # Peak-normalise reference images once (per image).
    ref_norm = {
        name: _peak_normalize(flux.to(device)).cpu()
        for name, flux in ref_flux.items()
    }
    log.info(f"Reference flux computed for {len(ref_norm)} heliostats.")

    # ------------------------------------------------------------------ #
    # Step 2: sweep (n_rays × sigma)
    # ------------------------------------------------------------------ #
    records = []
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
            heliostat_chunk_size=heliostat_chunk_size,
        )  # {name: [n_meas, H, W] on CPU}

        for sigma in sigma_configs:
            cfg_idx += 1
            log.info(f"  Config {cfg_idx}/{total_configs}: n_rays={n_rays}, sigma={sigma}")

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
                    "heliostat_id": name,
                    "n_rays": n_rays,
                    "sigma": sigma,
                    "mse": mse_mean,
                    "mse_std": mse_std,
                })

    log.info(f"Sweep complete. {len(records)} records collected.")
    return records
