import json
import logging
import math
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import index_mapping
from artist.util.utils import get_center_of_mass

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
    heliostat_chunk_size: int | None = None,
) -> dict:
    """
    Evaluate flux image prediction accuracy after kinematic reconstruction.

    Returns a dict with min/mean/median/max focal spot errors in mrad, the
    sample count, per-heliostat mrad errors, and the raw error list (needed
    for histogram plotting but not persisted to JSON).

    heliostat_chunk_size : int or None
        If set, process at most this many heliostats per forward pass to cap
        GPU memory usage. None (default) processes the whole group at once.
    """
    all_focal_spot_errors_m = []
    all_focal_spot_errors_mrad = []
    results_per_heliostat = {}
    nan_heliostat_ids = set()

    # Reference target center (mean over all target areas) used for distance computation.
    # All heliostats aim at roughly the same tower, so this is a good approximation.
    reference_target = scenario.target_areas.centers[:, :3].mean(dim=0).to(device)

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

        N_samples_per_heliostat = int(active_heliostats_mask.max().item())
        active_group_positions = torch.where(active_heliostats_mask > 0)[0]
        N_active_heliostats = len(active_group_positions)

        # Distances computed once for all active heliostats (no activation needed).
        active_positions = heliostat_group.positions[active_group_positions, :3].to(device)
        distances = torch.norm(active_positions - reference_target.unsqueeze(0), dim=1)
        name_to_distance = {
            heliostat_group.names[idx.item()]: dist.item()
            for idx, dist in zip(active_group_positions, distances)
        }

        chunk_size = heliostat_chunk_size if heliostat_chunk_size is not None else N_active_heliostats

        with torch.no_grad():
            for c_start in range(0, N_active_heliostats, chunk_size):
                c_end = min(c_start + chunk_size, N_active_heliostats)
                chunk_active_local = list(range(c_start, c_end))
                K = len(chunk_active_local)

                chunk_mask = torch.zeros_like(active_heliostats_mask)
                chunk_mask[active_group_positions[chunk_active_local]] = N_samples_per_heliostat

                data_rows = torch.cat([
                    torch.arange(i * N_samples_per_heliostat, (i + 1) * N_samples_per_heliostat, device=device)
                    for i in chunk_active_local
                ])

                chunk_incident = incident_ray_directions[data_rows]
                chunk_target_mask = target_area_mask[data_rows]
                chunk_focal_spots = focal_spots[data_rows]

                heliostat_group.activate_heliostats(
                    active_heliostats_mask=chunk_mask,
                    device=device,
                )

                kinematic = heliostat_group.kinematic
                if hasattr(kinematic, "_base_position_deviation"):
                    chunk_base_dev = kinematic._base_position_deviation[
                        active_group_positions[chunk_active_local]
                    ].repeat_interleave(N_samples_per_heliostat, dim=0)
                    pad = torch.zeros(chunk_base_dev.shape[0], 1, device=device)
                    kinematic.active_heliostat_positions = (
                        kinematic.active_heliostat_positions + torch.cat([chunk_base_dev, pad], dim=1)
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
                    batch_size=min(K * N_samples_per_heliostat, ray_tracing_batch_size),
                    bitmap_resolution=bitmap_resolution.to(device),
                )

                predicted_flux = ray_tracer.trace_rays(
                    incident_ray_directions=chunk_incident,
                    active_heliostats_mask=chunk_mask,
                    target_area_mask=chunk_target_mask,
                    device=device,
                )

                target_centers = scenario.target_areas.centers[chunk_target_mask]
                target_widths = scenario.target_areas.dimensions[chunk_target_mask][
                    :, index_mapping.target_area_width
                ]
                target_heights = scenario.target_areas.dimensions[chunk_target_mask][
                    :, index_mapping.target_area_height
                ]

                predicted_focal_spots = get_center_of_mass(
                    bitmaps=predicted_flux,
                    target_centers=target_centers,
                    target_widths=target_widths,
                    target_heights=target_heights,
                    device=device,
                )

                focal_spot_error = torch.norm(
                    predicted_focal_spots[:, :3] - chunk_focal_spots[:, :3], dim=1
                )
                all_focal_spot_errors_m.extend(focal_spot_error.cpu().tolist())

                # mrad conversion using per-chunk distances.
                chunk_distances = distances[chunk_active_local].repeat_interleave(N_samples_per_heliostat)
                focal_spot_error_mrad = (focal_spot_error / chunk_distances) * 1000.0
                all_focal_spot_errors_mrad.extend(focal_spot_error_mrad.cpu().tolist())

                # Track heliostats with NaN focal spot errors (zero flux on target).
                nan_sample_indices = torch.where(torch.isnan(focal_spot_error))[0].tolist()
                for sample_idx in nan_sample_indices:
                    heliostat_local_idx = sample_idx // N_samples_per_heliostat
                    global_idx = active_group_positions[chunk_active_local[heliostat_local_idx]].item()
                    nan_heliostat_ids.add(heliostat_group.names[global_idx])

                # Per-heliostat results: mean error across all samples for that heliostat.
                for local_i, global_local in enumerate(chunk_active_local):
                    name = heliostat_group.names[active_group_positions[global_local].item()]
                    sample_errors = focal_spot_error[
                        local_i * N_samples_per_heliostat : (local_i + 1) * N_samples_per_heliostat
                    ]
                    fse_m = sample_errors.nanmean().item()
                    dist_m = name_to_distance.get(name)
                    fse_mrad = (fse_m / dist_m * 1000.0) if dist_m else None
                    results_per_heliostat[name] = {"focal_spot_error_mrad": fse_mrad}

    def _safe_mean(lst):
        valid = [x for x in lst if not math.isnan(x)]
        return sum(valid) / len(valid) if valid else float("inf")

    def _safe_median(lst):
        valid = [x for x in lst if not math.isnan(x)]
        return float(np.median(valid)) if valid else float("inf")

    num_nan_samples = sum(1 for x in all_focal_spot_errors_mrad if math.isnan(x))

    return {
        "mean_focal_spot_error_mrad": _safe_mean(all_focal_spot_errors_mrad),
        "median_focal_spot_error_mrad": _safe_median(all_focal_spot_errors_mrad),
        "min_focal_spot_error_mrad": float(np.nanmin(all_focal_spot_errors_mrad)) if all_focal_spot_errors_mrad else float("inf"),
        "max_focal_spot_error_mrad": float(np.nanmax(all_focal_spot_errors_mrad)) if all_focal_spot_errors_mrad else float("inf"),
        "num_samples_evaluated": len(all_focal_spot_errors_m),
        "num_nan_samples": num_nan_samples,
        "nan_heliostat_ids": sorted(nan_heliostat_ids),
        "all_errors_mrad": all_focal_spot_errors_mrad,  # kept in memory for histogram; not saved to JSON
        "per_heliostat": results_per_heliostat,
    }
