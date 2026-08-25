"""
Warm start for fine_error_learning.

Loads each per-heliostat scenario, writes the stage-1 checkpoint tensors into
its kinematic module (the loader precedent is
``src/one_heliostat_demo/single_heliostat/train.py:2437-2454``), and snapshots
the 24-D vector as the frozen base θ_KR.

Also measures the per-heliostat on-target fraction under θ_KR (plan
deliverable): the fraction of measurements whose predicted centroid is finite
and lands within the target bitmap extent. When this is low, the focal-spot
loss has no gradient for that heliostat (the beam misses the target).
"""
from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass

import h5py
import torch
from artist.scenario.scenario import Scenario
from artist.util import indices

from fine_error_learning import pipeline as fel_pipeline

log = logging.getLogger(__name__)


@dataclass
class WarmStartState:
    """Everything the training loop needs for one heliostat."""

    heliostat_id: str
    scenario: Scenario
    heliostat_group: object
    hel_idx: int                      # row of the heliostat within its group
    theta_kr: torch.Tensor            # [24] frozen base parameter vector
    act_nonopt_template: torch.Tensor  # [1, 7, 2] detached actuator constants
    heliostat_position: torch.Tensor  # [3] absolute ENU
    hel_dist_m: float                 # distance to the tower reference point [m]
    on_target_fraction: float = float("nan")
    on_target_mean_error_m: float = float("nan")


def load_warm_start_state(
    heliostat_id: str,
    cfg,
    device: torch.device,
) -> WarmStartState:
    """Load the scenario and stage-1 checkpoint of one heliostat.

    With ``cfg.WARM_START == "nominal"`` the checkpoint is skipped and θ_KR is
    the scenario's nominal kinematics (guaranteed on-target on synthetic data).
    """
    scenario_path = pathlib.Path(
        cfg.SCENARIO_PATH_TEMPLATE.format(heliostat_id=heliostat_id)
    )
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

    heliostat_group = scenario.heliostat_field.heliostat_groups[0]
    hel_idx = heliostat_group.names.index(heliostat_id)
    kinematic = heliostat_group.kinematics

    # Same loader semantics as the live pipeline (train.py:2447-2453): the
    # checkpoint holds ABSOLUTE tensor values that replace the kinematic state.
    warm_start_mode = getattr(cfg, "WARM_START", "stage1")
    if warm_start_mode == "stage1":
        ckpt_path = (
            pathlib.Path(cfg.STAGE1_CHECKPOINT_DIR) / heliostat_id / "stage1_checkpoint.pt"
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ckpt.get("heliostat_id") not in (None, heliostat_id):
            raise ValueError(
                f"Stage-1 checkpoint {ckpt_path} belongs to heliostat "
                f"{ckpt.get('heliostat_id')!r}, not {heliostat_id!r}"
            )
        kinematic.translation_deviation_parameters = ckpt["translation"].to(device).clone()
        kinematic.rotation_deviation_parameters = ckpt["rotation"].to(device).clone()
        kinematic.actuators.optimizable_parameters = ckpt["act_angle"].to(device).clone()
        kinematic.actuators.non_optimizable_parameters = ckpt["act_offset"].to(device).clone()
        base_position = ckpt["base_pos"].to(device).clone()
    elif warm_start_mode == "nominal":
        base_position = torch.zeros(
            kinematic.rotation_deviation_parameters.shape[0], 3, device=device
        )
    else:
        raise ValueError(f"Unknown WARM_START {warm_start_mode!r} ('stage1' | 'nominal')")

    kinematic._base_position_deviation = base_position

    # Snapshot the frozen 24-D base vector θ_KR.
    theta_kr = torch.cat(
        [
            kinematic.rotation_deviation_parameters[0],
            kinematic.translation_deviation_parameters[0],
            kinematic.actuators.optimizable_parameters[0, indices.actuator_initial_angle, :],
            kinematic.actuators.optimizable_parameters[0, indices.actuator_initial_stroke_length, :],
            kinematic.actuators.non_optimizable_parameters[0, indices.actuator_offset, :],
            kinematic.actuators.non_optimizable_parameters[0, indices.actuator_pivot_radius, :],
            kinematic._base_position_deviation[0],
        ]
    ).detach().float()
    act_nonopt_template = kinematic.actuators.non_optimizable_parameters.detach().clone()

    hel_pos = heliostat_group.positions[hel_idx, :3].float()
    target_areas = scenario.solar_tower.target_areas
    tower_ref = target_areas[indices.planar_target_areas].centers[:, :3].float().mean(dim=0)
    hel_dist_m = torch.norm(hel_pos - tower_ref.to(device)).item()

    log.info(
        f"Warm start [{warm_start_mode}]: {heliostat_id}  |  dist-to-tower = {hel_dist_m:.1f} m"
    )
    return WarmStartState(
        heliostat_id=heliostat_id,
        scenario=scenario,
        heliostat_group=heliostat_group,
        hel_idx=hel_idx,
        theta_kr=theta_kr,
        act_nonopt_template=act_nonopt_template,
        heliostat_position=hel_pos,
        hel_dist_m=hel_dist_m,
    )


@torch.no_grad()
def measure_on_target_fraction(
    state: WarmStartState,
    measurements,
    device: torch.device,
    max_measurements: int,
    n_rays: int,
) -> tuple[float, float]:
    """Fraction of measurements on target under θ_KR (Δθ = 0), plus mean error.

    "On target" = predicted centroid finite and within the planar target's
    bitmap extent (center ± plane dimensions / 2). Off-target measurements
    produce an empty flux bitmap and therefore zero focal-spot gradient.
    """
    n = min(measurements.n, max_measurements)
    state.scenario.set_number_of_rays(n_rays)
    flux, bitmap_resolution, sampler_indices = fel_pipeline.predict_flux(
        state=state,
        theta_final=state.theta_kr,
        incident_rays=measurements.incident_rays[:n],
        motor_positions=measurements.motor_positions[:n],
        target_indices=measurements.target_indices[:n],
        device=device,
        random_seed=0,
    )
    _, pred_coords = fel_pipeline.focal_spot_centroid_loss(
        predicted_flux=flux,
        focal_spots=measurements.focal_spots[:n][sampler_indices],
        target_indices=measurements.target_indices[:n][sampler_indices],
        bitmap_resolution=bitmap_resolution,
        scenario=state.scenario,
        device=device,
    )

    target_areas = state.scenario.solar_tower.target_areas[indices.planar_target_areas]
    tgt_idx = measurements.target_indices[:n][sampler_indices]
    centers = target_areas.centers[tgt_idx, :3].float()
    dims = target_areas.dimensions[tgt_idx].float()  # [N, 2] = (plane_e, plane_u)

    coords = pred_coords[:, :3]
    finite = torch.isfinite(coords).all(dim=-1)
    within_e = (coords[:, 0] - centers[:, 0]).abs() <= dims[:, 0] / 2
    within_u = (coords[:, 2] - centers[:, 2]).abs() <= dims[:, 1] / 2
    on_target = finite & within_e & within_u

    error_m = torch.norm(coords - measurements.focal_spots[:n][sampler_indices, :3], dim=1)
    fraction = on_target.float().mean().item()
    state.on_target_fraction = fraction
    state.on_target_mean_error_m = error_m.mean().item()
    log.info(
        f"  {state.heliostat_id}: on-target under θ_KR "
        f"{int(on_target.sum())}/{n} ({fraction:.0%}), "
        f"mean centroid error {state.on_target_mean_error_m:.3f} m"
    )
    return fraction, state.on_target_mean_error_m
