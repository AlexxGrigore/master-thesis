import json
import logging
import math
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import index_mapping
from artist.util.utils import get_center_of_mass, bitmap_coordinates_to_target_coordinates

log = logging.getLogger(__name__)


def build_heliostat_data_mapping(
    benchmark_csv: pathlib.Path,
    calibration_properties_dir: pathlib.Path,
    flux_image_dir: pathlib.Path,
    split: str = "train",
    deflectometry_only: bool = False,
    deflectometry_available_json: pathlib.Path | None = None,
) -> list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]:
    """
    Build the heliostat_data_mapping from the benchmark CSV file.

    Parameters
    ----------
    benchmark_csv : pathlib.Path
        Path to the benchmark split CSV file.
    calibration_properties_dir : pathlib.Path
        Base directory containing calibration properties JSON files.
    flux_image_dir : pathlib.Path
        Base directory containing flux image PNG files.
    split : str
        Which split to use: "train", "validation", or "test".
    deflectometry_only : bool
        If True, only include heliostats that have deflectometry data available.
        Requires either a ``DeflectometryAvailable`` column in the CSV or a
        ``deflectometry_available_json`` mapping file. Defaults to False.
    deflectometry_available_json : pathlib.Path or None
        Path to a JSON file mapping heliostat IDs to booleans
        (``{"AA23": true, "AA24": false, ...}``). Used as a fallback when
        ``deflectometry_only=True`` and the CSV lacks a
        ``DeflectometryAvailable`` column.

    Returns
    -------
    list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]
        List of tuples (heliostat_name, calibration_paths, flux_paths).
    """
    df = pd.read_csv(benchmark_csv)
    df_split = df[df["Split"] == split]

    if deflectometry_only:
        deflectometry_col = "DeflectometryAvailable"
        if deflectometry_col in df_split.columns:
            df_split = df_split[df_split[deflectometry_col].astype(bool)]
        elif deflectometry_available_json is not None:
            with open(deflectometry_available_json) as f:
                availability: dict[str, bool] = json.load(f)
            df_split = df_split[df_split["HeliostatId"].map(availability).fillna(False)]
        else:
            raise ValueError(
                f"deflectometry_only=True but column '{deflectometry_col}' not found in "
                f"benchmark CSV and no deflectometry_available_json was provided. "
                f"Available columns: {list(df_split.columns)}"
            )
        log.info(f"Filtered to heliostats with deflectometry data. Remaining samples: {len(df_split)}")

    log.info(f"Building heliostat_data_mapping for split '{split}'")
    log.info(f"Total samples in split: {len(df_split)}")

    heliostat_groups = defaultdict(list)
    for _, row in df_split.iterrows():
        measurement_id = row["Id"]
        heliostat_id = row["HeliostatId"]
        heliostat_groups[heliostat_id].append(measurement_id)

    log.info(f"Number of unique heliostats: {len(heliostat_groups)}")

    heliostat_data_mapping = []
    for heliostat_id, measurement_ids in sorted(heliostat_groups.items()):
        calibration_paths = []
        flux_paths = []

        for mid in measurement_ids:
            cal_path = calibration_properties_dir / split / f"{mid}-calibration-properties.json"
            flux_path = flux_image_dir / split / f"{mid}-flux.png"

            if cal_path.exists() and flux_path.exists():
                calibration_paths.append(cal_path)
                flux_paths.append(flux_path)

        if calibration_paths:
            heliostat_data_mapping.append((heliostat_id, calibration_paths, flux_paths))

    log.info(f"Built mapping for {len(heliostat_data_mapping)} heliostats")

    return heliostat_data_mapping


def evaluate_flux_accuracy(
    scenario: Scenario,
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    data_parser: PaintCalibrationDataParser,
    device: torch.device,
    bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
    ray_tracing_batch_size: int = 32,
) -> dict:
    """
    Evaluate flux image prediction accuracy after kinematic reconstruction.

    Returns a dict with:
      - min/mean/median/max focal spot errors in mrad (centroid-based)
      - mean/median peak-normalised per-pixel L1 loss (image-based)
      - per-heliostat breakdown of both metrics
    """
    all_focal_spot_errors_m = []
    all_focal_spot_errors_mrad = []
    all_pixel_losses = []
    results_per_heliostat = {}
    nan_heliostat_ids = set()

    # Reference target center used for distance computation.
    reference_target = scenario.solar_tower.target_areas[
        index_mapping.planar_target_areas
    ].centers[:, :3].mean(dim=0).to(device)

    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        (
            measured_flux,
            focal_spots,
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
            batch_size=min(heliostat_group.number_of_active_heliostats, ray_tracing_batch_size),
            bitmap_resolution=bitmap_resolution.to(device),
        )

        predicted_flux, _, _, _ = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            target_area_indices=target_area_mask,
            device=device,
        )

        # Align ground-truth focal spots with sampler order.
        sample_indices = ray_tracer.get_sampler_indices()
        focal_spots = focal_spots[sample_indices]

        bitmap_coords = get_center_of_mass(bitmaps=predicted_flux, device=device)
        predicted_focal_spots = bitmap_coordinates_to_target_coordinates(
            bitmap_coordinates=bitmap_coords,
            bitmap_resolution=ray_tracer.bitmap_resolution,
            solar_tower=scenario.solar_tower,
            target_area_indices=target_area_mask[sample_indices],
            device=device,
        )

        focal_spot_error = torch.norm(predicted_focal_spots[:, :3] - focal_spots[:, :3], dim=1)
        all_focal_spot_errors_m.extend(focal_spot_error.cpu().tolist())

        # Pixel-wise L1 loss — blur both images (sigma=1) to smooth out ray-tracing
        # noise, then peak-normalise so the comparison is scale-invariant
        # (predicted flux is in physical units, measured is in [0,1]).
        # measured_flux is in natural order; align it to sampler order first.
        measured_flux_sampler = measured_flux[sample_indices]
        pred_blurred = _gaussian_blur_batch(predicted_flux,       sigma=1.0)
        meas_blurred = _gaussian_blur_batch(measured_flux_sampler, sigma=1.0)
        N = pred_blurred.shape[0]
        pred_peak = pred_blurred.view(N, -1).max(dim=1).values.clamp(min=1e-12)
        meas_peak = meas_blurred.view(N, -1).max(dim=1).values.clamp(min=1e-12)
        pred_pnorm = pred_blurred / pred_peak.view(N, 1, 1)
        meas_pnorm = meas_blurred / meas_peak.view(N, 1, 1)
        pixel_loss_sampler = (pred_pnorm - meas_pnorm).abs().sum(dim=(1, 2))  # [N], sampler order

        # Compute per-heliostat distances to the reference target for mrad conversion.
        active_indices = torch.where(active_heliostats_mask.bool())[0]
        active_positions = heliostat_group.positions[active_indices, :3].to(device)
        distances = torch.norm(active_positions - reference_target.unsqueeze(0), dim=1)

        # Build name -> distance lookup for per-heliostat results.
        name_to_distance = {
            heliostat_group.names[idx.item()]: dist.item()
            for idx, dist in zip(active_indices, distances)
        }

        # Number of samples per heliostat, read from the mask (each active entry holds
        # the replication count N_samples, so mask[active_indices] gives [N, N, ..., N]).
        samples_per_heliostat_tensor = active_heliostats_mask[active_indices].long()

        # Build a per-sample distance tensor in natural (incident_rays) order.
        distances_natural = distances.repeat_interleave(samples_per_heliostat_tensor)

        # Both focal_spot_error and pixel_loss_sampler are in sampler order; invert
        # back to natural order so per-heliostat slicing is well-defined.
        inv_perm = torch.argsort(sample_indices)
        focal_spot_error_natural = focal_spot_error[inv_perm]
        pixel_loss_natural = pixel_loss_sampler[inv_perm]

        focal_spot_error_mrad = (focal_spot_error_natural / distances_natural) * 1000.0
        all_focal_spot_errors_mrad.extend(focal_spot_error_mrad.cpu().tolist())
        all_pixel_losses.extend(
            pixel_loss_natural[torch.isfinite(pixel_loss_natural)].cpu().tolist()
        )

        # Track heliostats that produced NaN focal spot errors (zero flux on target).
        # Work in natural order so heliostat index arithmetic is well-defined.
        nan_natural_indices = torch.where(torch.isnan(focal_spot_error_natural))[0].tolist()
        offset = 0
        for j, idx in enumerate(active_indices):
            n = samples_per_heliostat_tensor[j].item()
            for k in nan_natural_indices:
                if offset <= k < offset + n:
                    nan_heliostat_ids.add(heliostat_group.names[idx.item()])
                    break
            offset += n

        # Per-heliostat mean focal spot error and pixel loss.
        offset = 0
        for j, idx in enumerate(active_indices):
            name = heliostat_group.names[idx.item()]
            n = samples_per_heliostat_tensor[j].item()
            fse_slice = focal_spot_error_natural[offset : offset + n]
            pix_slice = pixel_loss_natural[offset : offset + n]
            offset += n

            fse_valid = fse_slice[torch.isfinite(fse_slice)]
            pix_valid = pix_slice[torch.isfinite(pix_slice)]

            fse_mrad = None
            if len(fse_valid) > 0:
                dist_m = name_to_distance[name]
                fse_mrad = fse_valid.mean().item() / dist_m * 1000.0

            pixel_loss_mean = pix_valid.mean().item() if len(pix_valid) > 0 else None

            results_per_heliostat[name] = {
                "focal_spot_error_mrad": fse_mrad,
                "pixel_loss": pixel_loss_mean,
            }

    def _safe_mean(lst):
        valid = [x for x in lst if not math.isnan(x)]
        return sum(valid) / len(valid) if valid else float("inf")

    def _safe_median(lst):
        valid = [x for x in lst if not math.isnan(x)]
        return float(np.median(valid)) if valid else float("inf")

    num_nan_samples = sum(1 for x in all_focal_spot_errors_mrad if math.isnan(x))

    return {
        "mean_focal_spot_error_mrad":   _safe_mean(all_focal_spot_errors_mrad),
        "median_focal_spot_error_mrad": _safe_median(all_focal_spot_errors_mrad),
        "mean_focal_spot_error_m":      _safe_mean(all_focal_spot_errors_m),
        "all_errors_mrad":              all_focal_spot_errors_mrad,
        # Legacy aliases used by five_heliostats_synth and full_field_200_samples.
        "mean_mrad":                    _safe_mean(all_focal_spot_errors_mrad),
        "median_mrad":                  _safe_median(all_focal_spot_errors_mrad),
        "mean_m":                       _safe_mean(all_focal_spot_errors_m),
        "min_mrad":                     float(np.nanmin(all_focal_spot_errors_mrad)) if all_focal_spot_errors_mrad else float("inf"),
        "max_mrad":                     float(np.nanmax(all_focal_spot_errors_mrad)) if all_focal_spot_errors_mrad else float("inf"),
        "mean_pixel_loss":              _safe_mean(all_pixel_losses),
        "median_pixel_loss":            _safe_median(all_pixel_losses),
        "num_samples":                  len(all_focal_spot_errors_m),
        "num_nan_samples":              num_nan_samples,
        "nan_heliostat_ids":            sorted(nan_heliostat_ids),
        "per_heliostat":                results_per_heliostat,
    }


def _gaussian_blur_batch(flux: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply a separable Gaussian blur to a batch of 2-D flux images [N, H, W]."""
    if sigma <= 0:
        return flux
    kernel_size = int(4 * sigma + 0.5) * 2 + 1
    coords = torch.arange(kernel_size, device=flux.device, dtype=flux.dtype) - kernel_size // 2
    gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel = (gauss_1d[:, None] * gauss_1d[None, :]).view(1, 1, kernel_size, kernel_size)
    return F.conv2d(flux.unsqueeze(1), kernel, padding=kernel_size // 2).squeeze(1)


@torch.no_grad()
def compute_pixel_test_loss(
    scenario: Scenario,
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    data_parser: PaintCalibrationDataParser,
    device: torch.device,
    blur_sigma: float = 0.0,
    bitmap_resolution: torch.Tensor = torch.tensor([256, 256]),
    ray_tracing_batch_size: int = 32,
) -> float:
    """
    Compute the mean PixelLossL1 value on a data split using the trained scenario.
    """

    all_losses: list[float] = []

    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        (
            measured_flux,
            _focal_spots,
            incident_ray_directions,
            _motor_positions,
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
            batch_size=min(heliostat_group.number_of_active_heliostats, ray_tracing_batch_size),
            bitmap_resolution=bitmap_resolution.to(device),
        )
        predicted_flux, _, _, _ = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            target_area_indices=target_area_mask,
            device=device,
        )

        sample_indices = ray_tracer.get_sampler_indices()
        measured_flux = measured_flux[sample_indices]

        # Gaussian blur
        predicted_flux = _gaussian_blur_batch(predicted_flux, blur_sigma)

        # Peak-normalise
        N = predicted_flux.shape[0]
        pred_max = predicted_flux.view(N, -1).max(dim=1).values.clamp(min=1e-12)
        meas_max = measured_flux.view(N, -1).max(dim=1).values.clamp(min=1e-12)
        pred_norm = predicted_flux / pred_max.view(N, 1, 1)
        meas_norm = measured_flux / meas_max.view(N, 1, 1)

        # L1 summed over pixels per sample
        per_sample = (pred_norm - meas_norm).abs().sum(dim=(1, 2))
        finite_losses = per_sample[torch.isfinite(per_sample)]
        if len(finite_losses) > 0:
            all_losses.extend(finite_losses.cpu().tolist())

    if not all_losses:
        return float("nan")
    return float(sum(all_losses) / len(all_losses))
