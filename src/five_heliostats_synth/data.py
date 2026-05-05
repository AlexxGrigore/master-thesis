"""
Synthetic data generation and loading for the 5-heliostat experiment.

Workflow (generation)
---------------------
1. Sample random per-heliostat perturbations (seeded, once per run).
2. Load PAINT calibration data for a split (geometry only; real flux discarded).
3. Apply perturbations to kinematics in-place.
4. Ray-trace synthetic focal spot centroids with high ray count.
5. Reset perturbations.

Workflow (loading)
------------------
- SyntheticDatasetParser reads from synthetic_data/{split}/{heliostat_id}/{idx:04d}/:
    calibration_properties.json  — incident ray direction, centroid ENU, motor pos, target index
    flux_image.png               — ray-traced flux bitmap (uint8, normalised to [0,255])
"""
import json
import logging
import pathlib

import numpy as np
import torch
from PIL import Image
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.util import index_mapping
from artist.util.utils import get_center_of_mass, bitmap_coordinates_to_target_coordinates

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Perturbation sampling
# ---------------------------------------------------------------------------

def sample_perturbations(n_heliostats: int, ranges: dict, seed: int) -> dict:
    """
    Sample independent uniform random perturbations for each heliostat.

    Returns
    -------
    dict with keys:
        rotation          : Tensor [N, 4]  (rad)  — 4 joint tilts
        actuator_angle    : Tensor [N, 2]  (rad)  — a_i, optimized
        actuator_stroke   : Tensor [N, 2]  (m)    — b_i, frozen during training
        actuator_offset   : Tensor [N, 2]  (m)    — c_i, optimized
        translation       : Tensor [N, 9]  (m)    — joint + concentrator translations, optimized
        base_position     : Tensor [N, 3]  (m)    — (east, north, up), optimized

    Each value is drawn uniformly from [-bound, +bound] per element.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    def uniform(shape: tuple, bound: float) -> torch.Tensor:
        return (torch.rand(shape, generator=rng) * 2.0 - 1.0) * bound

    return {
        "rotation":        uniform((n_heliostats, 4), ranges["rotation_rad"]),
        "actuator_angle":  uniform((n_heliostats, 2), ranges["actuator_angle_rad"]),
        "actuator_stroke": uniform((n_heliostats, 2), ranges["actuator_stroke_m"]),
        "actuator_offset": uniform((n_heliostats, 2), ranges["actuator_offset_m"]),
        "translation":     uniform((n_heliostats, 9), ranges["translation_m"]),
        "base_position":   uniform((n_heliostats, 3), ranges["base_position_m"]),
    }


def perturbations_to_json(perturbations: dict, heliostat_ids: list) -> dict:
    """Convert perturbation tensors to a JSON-serialisable dict keyed by heliostat ID."""
    return {
        hid: {
            "rotation_rad":       perturbations["rotation"][i].tolist(),
            "actuator_angle_rad": perturbations["actuator_angle"][i].tolist(),
            "actuator_stroke_m":  perturbations["actuator_stroke"][i].tolist(),
            "actuator_offset_m":  perturbations["actuator_offset"][i].tolist(),
            "translation_m":      perturbations["translation"][i].tolist(),
            "base_position_m":    perturbations["base_position"][i].tolist(),
        }
        for i, hid in enumerate(heliostat_ids)
    }


# ---------------------------------------------------------------------------
# Apply / reset perturbations
# ---------------------------------------------------------------------------

def apply_perturbations(kinematic, perturbations: dict, device: torch.device) -> dict:
    """
    Add per-heliostat perturbation deltas to all kinematic parameters in-place.

    Covers all parameters optimized (or frozen) by WortbergKinematicReconstructor:
      - rotation_deviation_parameters       [N, 4]  ±5 mrad   (optimized)
      - actuator_initial_angle (a_i)        [N, 2]  ±5 mrad   (optimized)
      - actuator_initial_stroke_length (b_i)[N, 2]  ±5 mm     (frozen — not recovered)
      - actuator_offset (c_i)               [N, 2]  ±5 mm     (optimized)
      - translation_deviation_parameters    [N, 9]  ±50 mm    (optimized)
      - _base_position_deviation            [N, 3]  ±50 mm    (optimized, created if absent)

    Returns a snapshot of original values for reset_perturbations().
    """
    original = {}

    # rotation
    rot = perturbations["rotation"].to(device)
    original["rotation"] = kinematic.rotation_deviation_parameters.data.clone()
    kinematic.rotation_deviation_parameters.data += rot

    # a_i — actuator initial angle
    act = perturbations["actuator_angle"].to(device)
    original["actuator_angle"] = kinematic.actuators.optimizable_parameters.data[
        :, index_mapping.actuator_initial_angle, :
    ].clone()
    kinematic.actuators.optimizable_parameters.data[
        :, index_mapping.actuator_initial_angle, :
    ] += act

    # b_i — actuator initial stroke length (frozen, but still perturbed)
    stroke = perturbations["actuator_stroke"].to(device)
    original["actuator_stroke"] = kinematic.actuators.optimizable_parameters.data[
        :, index_mapping.actuator_initial_stroke_length, :
    ].clone()
    kinematic.actuators.optimizable_parameters.data[
        :, index_mapping.actuator_initial_stroke_length, :
    ] += stroke

    # c_i — actuator offset
    offset = perturbations["actuator_offset"].to(device)
    original["actuator_offset"] = kinematic.actuators.non_optimizable_parameters.data[
        :, index_mapping.actuator_offset, :
    ].clone()
    kinematic.actuators.non_optimizable_parameters.data[
        :, index_mapping.actuator_offset, :
    ] += offset

    # translation_deviation_parameters
    trans = perturbations["translation"].to(device)
    original["translation"] = kinematic.translation_deviation_parameters.data.clone()
    kinematic.translation_deviation_parameters.data += trans

    # base_position — stored as _base_position_deviation attribute (created if absent)
    base = perturbations["base_position"].to(device)
    if hasattr(kinematic, "_base_position_deviation"):
        original["base_position"] = kinematic._base_position_deviation.detach().clone()
        kinematic._base_position_deviation = (
            kinematic._base_position_deviation.detach() + base
        )
    else:
        original["base_position"] = None
        kinematic._base_position_deviation = base.clone().detach()

    return original


def reset_perturbations(kinematic, original: dict) -> None:
    """Restore kinematic parameters to their pre-perturbation values."""
    kinematic.rotation_deviation_parameters.data.copy_(original["rotation"])
    kinematic.actuators.optimizable_parameters.data[
        :, index_mapping.actuator_initial_angle, :
    ].copy_(original["actuator_angle"])
    kinematic.actuators.optimizable_parameters.data[
        :, index_mapping.actuator_initial_stroke_length, :
    ].copy_(original["actuator_stroke"])
    kinematic.actuators.non_optimizable_parameters.data[
        :, index_mapping.actuator_offset, :
    ].copy_(original["actuator_offset"])
    kinematic.translation_deviation_parameters.data.copy_(original["translation"])
    if original["base_position"] is not None:
        kinematic._base_position_deviation = original["base_position"].clone().detach()
    elif hasattr(kinematic, "_base_position_deviation"):
        del kinematic._base_position_deviation


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def _forward_pass(
    scenario,
    heliostat_group,
    incident_rays: torch.Tensor,
    active_mask: torch.Tensor,
    target_mask: torch.Tensor,
    base_pos_delta: torch.Tensor,   # [N_heliostats, 3]
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Activate heliostats, inject base-position offset, align surfaces, trace rays,
    and return (centroids, flux) both in natural (incident_rays) order.

    centroids : [N_instances, 4]
    flux      : [N_instances, H, W]  — physical intensity units
    """
    heliostat_group.activate_heliostats(active_heliostats_mask=active_mask, device=device)
    kinematic = heliostat_group.kinematics

    # Expand base_pos_delta from [N_heliostats, 3] to [N_active_instances, 3].
    # active_mask[i] = N_samples for active heliostats, 0 for inactive.
    repeated = base_pos_delta.to(device).repeat_interleave(active_mask, dim=0)
    pad = torch.zeros(repeated.shape[0], 1, device=device)
    kinematic.active_heliostat_positions = (
        kinematic.active_heliostat_positions + torch.cat([repeated, pad], dim=1)
    )

    heliostat_group.align_surfaces_with_incident_ray_directions(
        aim_points=scenario.solar_tower.get_centers_of_target_areas(
            target_mask, device=device
        ),
        incident_ray_directions=incident_rays,
        active_heliostats_mask=active_mask,
        device=device,
    )

    ray_tracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=heliostat_group,
        blocking_active=False,
        world_size=1,
        rank=0,
        batch_size=max(8, int(active_mask.sum().item())),
        random_seed=42,
    )
    flux_sampler, _, _, _ = ray_tracer.trace_rays(
        incident_ray_directions=incident_rays,
        active_heliostats_mask=active_mask,
        target_area_indices=target_mask,
        device=device,
    )
    sample_indices = ray_tracer.get_sampler_indices()

    bitmap_coords = get_center_of_mass(bitmaps=flux_sampler, device=device)
    centroids_sampler = bitmap_coordinates_to_target_coordinates(
        bitmap_coordinates=bitmap_coords,
        bitmap_resolution=ray_tracer.bitmap_resolution,
        solar_tower=scenario.solar_tower,
        target_area_indices=target_mask[sample_indices],
        device=device,
    )  # [N_sampled, 4] in sampler order

    # Undo sampler reordering — training loop re-applies sample_indices when computing loss.
    inverse_perm = torch.argsort(sample_indices)
    return centroids_sampler[inverse_perm], flux_sampler[inverse_perm]


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def _equalize_mapping(
    mapping: list,
    sample_limit: int | None = None,
) -> list:
    """
    Trim all heliostats in mapping to the same number of calibration samples.

    ARTIST's RestrictedDistributedSampler uses floor division to distribute
    samples across heliostats, so if any heliostat has fewer files than the
    rest (e.g. BC32 has 9 while others have 10), the sampler silently drops
    the excess samples and produces a length mismatch between flux bitmaps
    and target_area_indices.  Equalising to min(counts) avoids this.
    """
    trimmed = [
        (hid, cal[:sample_limit], flux[:sample_limit])
        for hid, cal, flux in mapping
    ] if sample_limit else mapping

    active = [(hid, cal, flux) for hid, cal, flux in trimmed if cal]
    if not active:
        return trimmed

    min_count = min(len(cal) for _, cal, _ in active)
    return [(hid, cal[:min_count], flux[:min_count]) for hid, cal, flux in trimmed]


# ---------------------------------------------------------------------------
# Public builder (perturbed training data — used outside file-based workflow)
# ---------------------------------------------------------------------------

def build_synthetic_parser(
    real_parser: PaintCalibrationDataParser,
    scenario,
    heliostat_group,
    mapping: list,
    perturbations: dict,
    n_rays: int,
    device: torch.device,
) -> "SyntheticCalibrationDataParser":
    """
    Generate synthetic focal spots for one data split and return a parser.

    Steps
    -----
    1. Load PAINT data to get incident ray directions and geometry.
    2. Temporarily apply per-heliostat perturbations.
    3. Ray-trace with n_rays rays (high count → clean, near-noiseless centroids).
    4. Reset perturbations.
    5. Return SyntheticCalibrationDataParser wrapping the real parser.
    """
    mapping = _equalize_mapping(mapping, real_parser.sample_limit)

    with torch.no_grad():
        _, _, incident_rays, _, active_mask, target_mask = (
            real_parser.parse_data_for_reconstruction(
                heliostat_data_mapping=mapping,
                heliostat_group=heliostat_group,
                scenario=scenario,
                device=device,
            )
        )

    if active_mask.sum() == 0:
        raise RuntimeError("No active heliostats found in mapping. Check heliostat IDs.")

    old_rays = scenario.light_sources.light_source_list[0].number_of_rays
    scenario.set_number_of_rays(n_rays)

    original = apply_perturbations(heliostat_group.kinematics, perturbations, device)

    synthetic_focal_spots, synthetic_flux = _forward_pass(
        scenario, heliostat_group, incident_rays, active_mask, target_mask,
        perturbations["base_position"], device,
    )

    reset_perturbations(heliostat_group.kinematics, original)
    scenario.set_number_of_rays(old_rays)

    n_active = int((active_mask > 0).sum().item())
    n_samples = int(active_mask[active_mask > 0][0].item())
    log.info(
        f"Generated {len(synthetic_focal_spots)} synthetic focal spots "
        f"({n_active} heliostats × {n_samples} samples, {n_rays} rays)."
    )

    return SyntheticCalibrationDataParser(real_parser, synthetic_focal_spots, synthetic_flux)


# ---------------------------------------------------------------------------
# In-memory parser (kept for non-file-based workflows)
# ---------------------------------------------------------------------------

class SyntheticCalibrationDataParser:
    """
    Drop-in replacement for PaintCalibrationDataParser backed by in-memory tensors.

    Used by build_synthetic_parser for quick perturbed-data experiments.
    For the main file-based workflow use SyntheticDatasetParser instead.
    """

    def __init__(
        self,
        real_parser: PaintCalibrationDataParser,
        synthetic_focal_spots: torch.Tensor,
        synthetic_flux: torch.Tensor | None = None,
    ) -> None:
        self._real_parser = real_parser
        self._synthetic_focal_spots = synthetic_focal_spots  # [N_instances, 4]
        self._synthetic_flux = synthetic_flux                # [N_instances, H, W] or None

    def parse_data_for_reconstruction(
        self,
        heliostat_data_mapping,
        heliostat_group,
        scenario,
        bitmap_resolution=None,
        device=None,
    ):
        kwargs = dict(
            heliostat_data_mapping=heliostat_data_mapping,
            heliostat_group=heliostat_group,
            scenario=scenario,
            device=device,
        )
        if bitmap_resolution is not None:
            kwargs["bitmap_resolution"] = bitmap_resolution

        measured_flux, _, incident_rays, motor_pos, active_mask, target_mask = (
            self._real_parser.parse_data_for_reconstruction(**kwargs)
        )

        flux_out = (
            self._synthetic_flux.to(device)
            if self._synthetic_flux is not None
            else measured_flux
        )

        return (
            flux_out,
            self._synthetic_focal_spots.to(device),
            incident_rays,
            motor_pos,
            active_mask,
            target_mask,
        )


# ---------------------------------------------------------------------------
# File-based parser
# ---------------------------------------------------------------------------

class SyntheticDatasetParser:
    """
    Reads synthetic calibration data from a folder structure generated by
    generate_dataset.py:

        data_dir/
            {heliostat_id}/
                {idx:04d}/
                    calibration_properties.json
                    flux_image.png

    The sample count per heliostat is controlled by the heliostat_data_mapping
    passed to parse_data_for_reconstruction (same contract as PaintCalibrationDataParser).
    The mapping's file paths are ignored — only the heliostat IDs and per-heliostat
    sample counts matter.
    """

    def __init__(self, data_dir: pathlib.Path | str) -> None:
        self._data_dir = pathlib.Path(data_dir)

    def parse_data_for_reconstruction(
        self,
        heliostat_data_mapping,
        heliostat_group,
        scenario,
        bitmap_resolution=None,
        device=None,
    ):
        # Derive n_samples per heliostat_id from the mapping.
        mapping_dict = {hid: len(cal) for hid, cal, _ in heliostat_data_mapping if cal}

        n_total = len(heliostat_group.names)
        active_mask = torch.zeros(n_total, dtype=torch.long)

        all_flux: list[torch.Tensor] = []
        all_centroids: list[list] = []
        all_incident_rays: list[list] = []
        all_motor_pos: list[list] = []
        all_target_indices: list[int] = []

        for i, hid in enumerate(heliostat_group.names):
            n = mapping_dict.get(hid, 0)
            if n == 0:
                continue
            active_mask[i] = n

            hel_dir = self._data_dir / hid
            for k in range(n):
                meas_dir = hel_dir / f"{k:04d}"
                with open(meas_dir / "calibration_properties.json") as fh:
                    cal = json.load(fh)

                all_incident_rays.append(cal["incident_ray_direction"])
                all_centroids.append(cal["focal_spot_enu"])
                mp = cal["motor_position"]
                all_motor_pos.append(mp if isinstance(mp, list) else [
                    mp["axis_1_motor_position"], mp["axis_2_motor_position"]
                ])
                all_target_indices.append(int(cal["target_area_index"]))

                img = Image.open(meas_dir / "flux_image.png").convert("L")
                flux_arr = np.array(img, dtype=np.float32) / 255.0
                all_flux.append(torch.from_numpy(flux_arr))

        if not all_flux:
            raise RuntimeError(
                f"No synthetic data found in {self._data_dir} for the given mapping. "
                f"Expected heliostat IDs: {list(mapping_dict.keys())}"
            )

        measured_flux   = torch.stack(all_flux).to(device)
        focal_spots     = torch.tensor(all_centroids,     dtype=torch.float32).to(device)
        incident_rays   = torch.tensor(all_incident_rays, dtype=torch.float32).to(device)
        motor_positions = torch.tensor(all_motor_pos,     dtype=torch.float32).to(device)
        target_mask     = torch.tensor(all_target_indices, dtype=torch.long).to(device)

        return measured_flux, focal_spots, incident_rays, motor_positions, active_mask.to(device), target_mask
