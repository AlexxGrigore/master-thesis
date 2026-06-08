"""Validation loss helpers (no gradient) for alignment-loss and focal-spot-loss stages."""

import torch
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
from artist.util import indices


@torch.no_grad()
def val_alignment_loss(
    v_rays: torch.Tensor,
    v_am: torch.Tensor,
    v_tm: torch.Tensor,
    v_mp: torch.Tensor,
    k,
    alignment_fn,
    *,
    scenario,
    heliostat_group,
    device: torch.device,
) -> float:
    """Compute AlignmentLoss on a validation batch (no gradient)."""
    heliostat_group.activate_heliostats(active_heliostats_mask=v_am, device=device)
    _bpd = k._base_position_deviation
    _rep = _bpd.repeat_interleave(v_am, dim=0)
    k.active_heliostat_positions = k.active_heliostat_positions + torch.cat(
        [_rep, torch.zeros(_rep.shape[0], 1, device=device)], dim=1
    )
    heliostat_group.align_surfaces_with_incident_ray_directions(
        aim_points=scenario.solar_tower.get_centers_of_target_areas(v_tm, device),
        incident_ray_directions=v_rays,
        active_heliostats_mask=v_am,
        device=device,
    )
    return alignment_fn(
        predicted_motor_positions=k.active_motor_positions,
        measured_motor_positions=v_mp,
        actuators=k.actuators,
        device=device,
    ).mean().item()


@torch.no_grad()
def val_focal_loss(
    v_rays: torch.Tensor,
    v_am: torch.Tensor,
    v_tm: torch.Tensor,
    v_cents: torch.Tensor,
    focal_fn,
    k,
    *,
    scenario,
    heliostat_group,
    device: torch.device,
) -> float:
    """Compute FocalSpotLoss on a validation batch (full batch, no gradient)."""
    heliostat_group.activate_heliostats(active_heliostats_mask=v_am, device=device)
    _bpd = k._base_position_deviation
    _rep = _bpd.repeat_interleave(v_am, dim=0)
    k.active_heliostat_positions = k.active_heliostat_positions + torch.cat(
        [_rep, torch.zeros(_rep.shape[0], 1, device=device)], dim=1
    )
    heliostat_group.align_surfaces_with_incident_ray_directions(
        aim_points=scenario.solar_tower.get_centers_of_target_areas(v_tm, device),
        incident_ray_directions=v_rays,
        active_heliostats_mask=v_am,
        device=device,
    )
    rt = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=heliostat_group,
        blocking_active=False,
        world_size=1,
        rank=0,
        batch_size=max(8, int(v_am.sum().item())),
        random_seed=0,
    )
    flux, _, _, _ = rt.trace_rays(
        incident_ray_directions=v_rays,
        active_heliostats_mask=v_am,
        target_area_indices=v_tm,
        device=device,
    )
    inv_perm = torch.argsort(rt.get_sampler_indices())
    return focal_fn(
        prediction=flux[inv_perm],
        ground_truth=v_cents,
        target_area_indices=v_tm,
        reduction_dimensions=(indices.focal_spots,),
        device=device,
    ).mean().item()
