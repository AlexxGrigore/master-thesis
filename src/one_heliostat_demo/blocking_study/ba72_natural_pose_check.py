"""Does BA72 still reverse under a REALISTIC (naturally-aimed) neighbour pose?

`ba72_flux_diff_diagnostic.py` forced AZ70/AZ71 to a synthetic fully-VERTICAL
pose (tilt=0), which is how the controlled occlusion-sweep dataset was built,
but is not a pose any real neighbouring heliostat would ever take (a real
neighbour is either stowed/horizontal or aimed at a target). BE25's real-data
check (`be25_blocking_direction_check.py`) used naturally-aimed neighbours
and found the INTUITIVE direction (blocked centroid moves up), even at 25%
blocked, opposite to BA72's forced-vertical result. This isolates which of
the two differences (BA72's own oblique geometry vs. the artificial vertical
blocker pose) is responsible, by re-running BA72 with AZ70/AZ71 naturally
aimed at the same target instead of forced vertical.
"""

from __future__ import annotations

import logging
import pathlib
import sys

import h5py
import torch

_here = pathlib.Path(__file__).resolve().parent
_src = _here.parents[1]
for _p in (str(_src), str(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from artist.flux import get_center_of_mass  # noqa: E402
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer  # noqa: E402
from artist.scenario.scenario import Scenario  # noqa: E402
from artist.util import get_device, set_logger_config  # noqa: E402

from blocking_utils import aimed_neighbour_surfaces, one_hot_mask  # noqa: E402
from brute_blocking import exact_blocking  # noqa: E402

log = logging.getLogger(__name__)

_ROOT = _here.parents[2]
HELIOSTAT_ID = "BA72"
BLOCKER_NAMES = ["AZ70", "AZ71"]
TARGET_NAME = "solar_tower_juelich_lower"
SCENARIO_PATH = _ROOT / "scenarios" / "neighbourhoods" / HELIOSTAT_ID / "scenario.h5"
STAGE1_CHECKPOINT = (
    _ROOT / "outputs" / "new_mapping_function" / "ba72_occlusion" / "stage1_only"
    / "stage1_checkpoint.pt"
)
SUN_INCIDENT_RAY_DIRECTION = [0.037787847220897675, 0.4960194528102875, -0.8674888014793396, 0.0]


def load_stage1_kinematics(kinematic, device: torch.device) -> None:
    ckpt = torch.load(STAGE1_CHECKPOINT, map_location=device)
    kinematic.translation_deviation_parameters.data.copy_(ckpt["translation"].to(device))
    kinematic.rotation_deviation_parameters.data.copy_(ckpt["rotation"].to(device))
    kinematic.actuators.optimizable_parameters.data.copy_(ckpt["act_angle"].to(device))
    kinematic.actuators.non_optimizable_parameters.data.copy_(ckpt["act_offset"].to(device))
    kinematic._base_position_deviation = ckpt["base_pos"].clone().to(device)


def render(hg, scenario, sun, tgt, aim_point, mask, blocker_rows, device, blocking_on: bool):
    if blocking_on:
        surfaces = aimed_neighbour_surfaces(hg, scenario, sun[0], int(tgt.item()), device)
    with torch.no_grad():
        hg.activate_heliostats(active_heliostats_mask=mask, device=device)
        hg.align_surfaces_with_incident_ray_directions(
            aim_points=aim_point, incident_ray_directions=sun,
            active_heliostats_mask=mask, device=device,
        )
        ray_tracer = HeliostatRayTracer(
            scenario=scenario, heliostat_group=hg, blocking_active=False,
            world_size=1, rank=0, batch_size=1, random_seed=7,
        )
        if blocking_on:
            ray_tracer.blocking_active = True
            ray_tracer.blocking_heliostat_surfaces_active = surfaces
            with exact_blocking():
                flux, _, _, bf = ray_tracer.trace_rays(
                    incident_ray_directions=sun, active_heliostats_mask=mask,
                    target_area_indices=tgt, device=device,
                )
        else:
            flux, _, _, bf = ray_tracer.trace_rays(
                incident_ray_directions=sun, active_heliostats_mask=mask,
                target_area_indices=tgt, device=device,
            )
    return flux[0].detach().cpu(), float(1.0 - bf.item())


def com_u(flux_2d: torch.Tensor) -> float:
    bc = get_center_of_mass(bitmaps=flux_2d.unsqueeze(0))
    return float(bc[0, 1].item())


def main() -> None:
    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("artist", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    device = get_device()
    with h5py.File(SCENARIO_PATH) as fh:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=fh, device=device,
            number_of_surface_points_per_facet=torch.tensor([100, 100]),
        )
    scenario.set_number_of_rays(10)
    hg = scenario.heliostat_field.heliostat_groups[0]
    kinematic = hg.kinematics
    hel_idx = hg.names.index(HELIOSTAT_ID)
    n_hel = hg.number_of_heliostats
    blocker_rows = [hg.names.index(b) for b in BLOCKER_NAMES]
    load_stage1_kinematics(kinematic, device)

    sun = torch.tensor(SUN_INCIDENT_RAY_DIRECTION, dtype=torch.float, device=device).view(1, 4)
    target_index = scenario.solar_tower.target_name_to_index[TARGET_NAME]
    tgt = torch.tensor([target_index], device=device)
    aim_point = scenario.solar_tower.get_centers_of_target_areas(target_area_indices=tgt, device=device)
    mask = one_hot_mask(hel_idx, 1, n_hel, device)

    flux_none, bf_none = render(hg, scenario, sun, tgt, aim_point, mask, blocker_rows, device, blocking_on=False)
    flux_nat, bf_nat = render(hg, scenario, sun, tgt, aim_point, mask, blocker_rows, device, blocking_on=True)

    print(f"blocked fraction: no-block=0.0000 (by construction)  naturally-aimed AZ70/AZ71={bf_nat:.4f}")
    u_none = com_u(flux_none)
    u_nat = com_u(flux_nat)
    print(f"centroid bitmap-u: no-block={u_none:.3f}px  naturally-aimed-block={u_nat:.3f}px "
          f"(u grows DOWNWARD; blocked centroid is "
          f"{'LOWER' if u_nat > u_none else 'HIGHER'} on target than unblocked "
          f"=> {'matches BA72 forced-vertical result (reversed)' if u_nat > u_none else 'matches BE25 (intuitive)'})")

    diff = flux_none - flux_nat
    removed_total = float(diff.clamp(min=0).sum())
    added_total = float((-diff).clamp(min=0).sum())
    diff_pos = diff.clamp(min=0)
    u_removed = com_u(diff_pos) if removed_total > 0 else float("nan")
    print(f"flux removed = {removed_total:.1f}, flux 'added' (redistribution artefact) = {added_total:.1f}")
    print(f"center-of-mass (bitmap-u) of REMOVED flux = {u_removed:.3f}px vs. no-block centroid "
          f"u={u_none:.3f}px => removed flux sits "
          f"{'BELOW' if u_removed > u_none else 'ABOVE'} the centroid (u grows downward)")


if __name__ == "__main__":
    main()
