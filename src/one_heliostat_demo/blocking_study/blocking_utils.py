"""Shared helpers for blocking-aware generation / training / evaluation.

Blocking is injected through the two public attributes of ``HeliostatRayTracer``
(``blocking_active`` and ``blocking_heliostat_surfaces_active``), exactly as
``gate.py`` established — ARTIST itself is never modified. Every blocked trace
goes through ``brute_blocking.exact_blocking`` because ARTIST's own LBVH filter
silently under-reports (see brute_blocking.py).

Structural constraint (verified in the gate study): ARTIST's blocking assumes
ONE active instance per heliostat row, so every function here works per sample.
"""

from __future__ import annotations

import logging
import sys
import pathlib

import numpy as np
import torch

_here = pathlib.Path(__file__).resolve().parent
_src = _here.parents[1]
for _p in (str(_src), str(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from artist.flux import get_center_of_mass  # noqa: E402
from artist.geometry import bitmap_coordinates_to_target_coordinates  # noqa: E402
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer  # noqa: E402

from brute_blocking import exact_blocking  # noqa: E402

log = logging.getLogger(__name__)


def apply_stage1_checkpoint(kinematic, checkpoint_path, heliostat_id: str, device: torch.device) -> None:
    """Overwrite a kinematic's parameters in place with a trained Stage-1 checkpoint.

    Checkpoint format (shared across `all63_stage1/` and `full_field_1277/
    stage1_only_ideal_surfaces/`): dict with keys `heliostat_id`, `translation`,
    `rotation`, `act_angle`, `act_offset`, `base_pos`. Used to compare a
    heliostat's IDEAL nominal kinematics against its TRAINED/recovered ones
    under otherwise identical conditions (see `project_ba72_blocking_
    centroid_inversion.md`: the trained-vs-ideal difference can by itself
    flip a blocking-direction finding).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    if ckpt.get("heliostat_id") not in (None, heliostat_id):
        raise ValueError(f"Checkpoint belongs to {ckpt.get('heliostat_id')!r}, not {heliostat_id!r}")
    kinematic.translation_deviation_parameters.data.copy_(ckpt["translation"].to(device))
    kinematic.rotation_deviation_parameters.data.copy_(ckpt["rotation"].to(device))
    kinematic.actuators.optimizable_parameters.data.copy_(ckpt["act_angle"].to(device))
    kinematic.actuators.non_optimizable_parameters.data.copy_(ckpt["act_offset"].to(device))
    kinematic._base_position_deviation = ckpt["base_pos"].clone().to(device)


def one_hot_mask(index: int, count: int, size: int, device: torch.device) -> torch.Tensor:
    """Active mask of length `size` holding `count` at `index`, zero elsewhere."""
    mask = torch.zeros(size, dtype=torch.long, device=device)
    mask[index] = count
    return mask


def rotation_about_axis(axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Rodrigues rotation matrix [3, 3] for unit ``axis`` and ``angle_rad``.

    Lives here (not just in validate_blocking_flux.py, which re-exports it)
    because forward_pass_blocking/aimed_neighbour_surfaces need it too, and
    validate_blocking_flux.py already imports FROM this module -- defining it
    there and importing back here would be circular.
    """
    k = axis / axis.norm()
    K = torch.tensor(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]]
    )
    eye = torch.eye(3)
    return (
        eye
        + float(np.sin(angle_rad)) * K
        + (1.0 - float(np.cos(angle_rad))) * (K @ K)
    )


def manually_rotated_surfaces(hg, row: int, rotation: torch.Tensor) -> torch.Tensor:
    """World surface points of ``row`` for an exact orientation (bypasses kinematics).

    Local surface points (x = mirror width, y = mirror height, z ~ 0, normal =
    local +z) are rigidly rotated by ``rotation`` and translated to the
    heliostat position, using the same homogeneous convention as
    ``align_surfaces_*`` (``world = local @ R4.T``). Legitimate for blocking
    surfaces (fixed environment geometry, never trained).
    """
    r4 = torch.eye(4)
    r4[:3, :3] = rotation.float()
    r4[:3, 3] = hg.positions[row, :3].float()
    return hg.surface_points[row].float().cpu() @ r4.T


def blocker_rotation_for_sun(incident_ray_direction: torch.Tensor, tilt: float) -> torch.Tensor:
    """Rigid rotation for a blocker mirror at ``tilt`` (0 = vertical/max block,
    1 = horizontal/stow/no block), facing the given sample's own sun azimuth --
    the fixed-tilt convention used by generate_occlusion_dataset.py so that a
    blocking-aware training/eval forward pass can reproduce EXACTLY the same
    occlusion geometry that generated a given dataset, instead of the default
    "every neighbour aimed normally at the target" (which models the field's
    natural blocking, not a synthetic controlled sweep).
    """
    d3 = incident_ray_direction[:3].detach().cpu()
    sun_h = -d3[:2]
    n_vertical = torch.tensor([sun_h[0], sun_h[1], 0.0]) / torch.norm(sun_h)
    n_horizontal = torch.tensor([0.0, 0.0, 1.0])
    up = torch.tensor([0.0, 0.0, 1.0])
    x_w = torch.linalg.cross(up, n_vertical)
    x_w = x_w / x_w.norm()
    r_vertical = torch.stack([x_w, up, n_vertical], dim=1)
    tilt_axis = torch.linalg.cross(n_vertical, n_horizontal)
    tilt_axis = tilt_axis / tilt_axis.norm()
    return rotation_about_axis(tilt_axis, float(tilt) * np.pi / 2.0) @ r_vertical


def aimed_neighbour_surfaces(
    heliostat_group,
    scenario,
    incident_ray_direction: torch.Tensor,
    target_area_index: int,
    device: torch.device,
    target_index_override: int | None = None,
    fixed_tilt_rows: list[int] | None = None,
    fixed_tilt: float | None = None,
) -> torch.Tensor:
    """World-coordinate surface points of every heliostat in the group, aimed.

    Every group member (studied heliostat included — its plane is excluded from
    blocking later by the ray-owner self-exclusion) is aimed at the centre of
    the given target area under the given sun direction. Computed under no_grad:
    the neighbours are fixed environment, not trainable. Follows gate.py's
    ``blocker_surfaces`` (aimed branch), keyed by target index directly.

    ``target_index_override`` (Experiment F, full-field blocking): when given,
    the group is aimed at THIS target index instead of ``target_area_index``
    (e.g. every blocker fixed on ``solar_tower_juelich_lower`` regardless of
    where the studied heliostat aims). ``None`` (default) reproduces the
    Experiment-S convention exactly.

    ``fixed_tilt_rows`` + ``fixed_tilt`` (controlled-occlusion datasets, e.g.
    generate_occlusion_dataset.py): after aiming everyone normally, these
    specific rows are overridden to the fixed vertical(0)->horizontal(1) tilt
    via ``manually_rotated_surfaces`` / ``blocker_rotation_for_sun``, instead
    of "aimed normally" -- so a blocking-aware training/eval forward pass can
    reproduce EXACTLY the occlusion geometry a controlled dataset was
    generated with, not the field's natural blocking. Both args or neither.

    Returns ``[number_of_heliostats, number_of_surface_points, 4]``.
    """
    n_hel = heliostat_group.number_of_heliostats
    aim_index = target_area_index if target_index_override is None else target_index_override
    aim_point = scenario.solar_tower.get_centers_of_target_areas(
        target_area_indices=torch.tensor([aim_index], device=device),
        device=device,
    )[0]
    mask = torch.ones(n_hel, dtype=torch.long, device=device)
    with torch.no_grad():
        heliostat_group.activate_heliostats(active_heliostats_mask=mask, device=device)
        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=aim_point.expand(n_hel, -1),
            incident_ray_directions=incident_ray_direction.expand(n_hel, -1),
            active_heliostats_mask=mask,
            device=device,
        )
        surfaces = heliostat_group.active_surface_points.detach().clone()
    if fixed_tilt_rows:
        rotation = blocker_rotation_for_sun(incident_ray_direction, fixed_tilt)
        for row in fixed_tilt_rows:
            surfaces[row] = manually_rotated_surfaces(heliostat_group, row, rotation).to(device)
    return surfaces


def forward_pass_blocking(
    scenario,
    heliostat_group,
    heliostat_index: int,
    incident_rays: torch.Tensor,
    target_mask: torch.Tensor,
    device: torch.device,
    motor_positions: torch.Tensor | None = None,
    aim_points: torch.Tensor | None = None,
    base_pos_delta: torch.Tensor | None = None,
    random_seed: int = 0,
    target_index_override: int | None = None,
    fixed_tilt_rows: list[int] | None = None,
    fixed_tilt: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Per-sample forward pass with blocking ON for one heliostat of a group.

    Orientation source (exactly one must be given):
      * ``motor_positions`` [N, 2] — orient from recorded motors m_c (centre-free
        forward map; training Stage 2 / evaluation / trails).
      * ``aim_points`` [N, 4] — aim at these points (dataset generation).

    ``base_pos_delta`` is the full-group ``[n_hel, 3]`` base-position deviation
    tensor; the studied row is injected per sample (matching _forward_pass).

    ``target_index_override`` (Experiment F): aim the passive blockers at this
    fixed target instead of each sample's own target. Default None = Experiment-S
    convention (blockers co-aimed with the studied heliostat).

    Gradients flow through the studied heliostat's trace when the caller is not
    under no_grad; the neighbour surfaces are always constants.

    Returns
    -------
    centroids : [N, 4] ENU target-plane coordinates of the flux COM
    flux      : [N, H, W]
    blocked   : list of per-sample blocked ray fractions (0..1)
    """
    if (motor_positions is None) == (aim_points is None):
        raise ValueError("Give exactly one of motor_positions / aim_points.")

    n_hel = heliostat_group.number_of_heliostats
    kinematic = heliostat_group.kinematics
    fluxes: list[torch.Tensor] = []
    blocked: list[float] = []
    bitmap_resolution = None

    for i in range(incident_rays.shape[0]):
        sun = incident_rays[i : i + 1]
        tgt = target_mask[i : i + 1]

        surfaces = aimed_neighbour_surfaces(
            heliostat_group, scenario, incident_rays[i], int(tgt.item()), device,
            target_index_override=target_index_override,
            fixed_tilt_rows=fixed_tilt_rows, fixed_tilt=fixed_tilt,
        )

        mask = one_hot_mask(heliostat_index, 1, n_hel, device)
        heliostat_group.activate_heliostats(active_heliostats_mask=mask, device=device)
        if base_pos_delta is not None:
            pad = torch.zeros(1, 1, device=device)
            kinematic.active_heliostat_positions = (
                kinematic.active_heliostat_positions
                + torch.cat([base_pos_delta[heliostat_index : heliostat_index + 1, :3].to(device), pad], dim=1)
            )
        if motor_positions is not None:
            heliostat_group.align_surfaces_with_motor_positions(
                motor_positions=motor_positions[i : i + 1],
                active_heliostats_mask=mask,
                device=device,
            )
        else:
            heliostat_group.align_surfaces_with_incident_ray_directions(
                aim_points=aim_points[i : i + 1],
                incident_ray_directions=sun,
                active_heliostats_mask=mask,
                device=device,
            )

        ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=heliostat_group,
            blocking_active=False,
            world_size=1,
            rank=0,
            batch_size=1,
            random_seed=random_seed,
        )
        ray_tracer.blocking_active = True
        ray_tracer.blocking_heliostat_surfaces_active = surfaces
        with exact_blocking():
            flux, _, _, blocking_factor = ray_tracer.trace_rays(
                incident_ray_directions=sun,
                active_heliostats_mask=mask,
                target_area_indices=tgt,
                device=device,
            )
        bitmap_resolution = ray_tracer.bitmap_resolution
        fluxes.append(flux[0])
        blocked.append(float(1.0 - blocking_factor.item()))

    flux = torch.stack(fluxes)
    bitmap_coords = get_center_of_mass(bitmaps=flux, device=device)
    centroids = bitmap_coordinates_to_target_coordinates(
        bitmap_coordinates=bitmap_coords,
        bitmap_resolution=bitmap_resolution,
        solar_tower=scenario.solar_tower,
        target_area_indices=target_mask,
        device=device,
    )
    return centroids, flux, blocked
