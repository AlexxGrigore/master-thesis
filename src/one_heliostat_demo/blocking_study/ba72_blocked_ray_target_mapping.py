"""Where do BA72's BLOCKED mirror rays actually land on target?

`plot_centroid_vs_occlusion.py` found that as AZ70/AZ71 tilt from horizontal
(no block) to vertical (max block), BA72's focal-spot centroid moves DOWN, not
up. The naive intuition ("blockers occlude the mirror's bottom -> that flux is
missing -> the remaining flux, now more concentrated near the top, pulls the
centroid up") predicts the opposite sign. This script tests that intuition
directly, using the exact same scenario/kinematics/sun direction, by tracing
UNBLOCKED rays from every BA72 surface point to the target and asking: which
mirror points are the ones AZ70/AZ71 (at tilt=0, vertical) actually block, and
where on the target would THEIR rays have landed?

Two diagnostics:
  1. Mirror surface (local width/height), coloured by blocked / unblocked.
  2. Target bitmap positions of each mirror point's ray, coloured the same way
     -- i.e. the "would-be" contribution of the blocked points, overlaid on
     the full unblocked spot.

Usage
-----
    python ba72_blocked_ray_target_mapping.py
"""

from __future__ import annotations

import logging
import pathlib
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_here = pathlib.Path(__file__).resolve().parent
_src = _here.parents[1]
for _p in (str(_src), str(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer  # noqa: E402
from artist.scenario.scenario import Scenario  # noqa: E402
from artist.util import get_device, set_logger_config  # noqa: E402

from blocking_utils import aimed_neighbour_surfaces, one_hot_mask  # noqa: E402
from brute_blocking import (  # noqa: E402
    capture_blocking_mask,
    capture_target_intersections,
    exact_blocking,
)

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
OUT_DIR = _ROOT / "outputs" / "new_mapping_function" / "ba72_occlusion" / "plots"
SUN_INCIDENT_RAY_DIRECTION = [0.037787847220897675, 0.4960194528102875, -0.8674888014793396, 0.0]


def load_stage1_kinematics(kinematic, device: torch.device) -> None:
    ckpt = torch.load(STAGE1_CHECKPOINT, map_location=device)
    kinematic.translation_deviation_parameters.data.copy_(ckpt["translation"].to(device))
    kinematic.rotation_deviation_parameters.data.copy_(ckpt["rotation"].to(device))
    kinematic.actuators.optimizable_parameters.data.copy_(ckpt["act_angle"].to(device))
    kinematic.actuators.non_optimizable_parameters.data.copy_(ckpt["act_offset"].to(device))
    kinematic._base_position_deviation = ckpt["base_pos"].clone().to(device)


def main() -> None:
    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("artist", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    device = get_device()
    surface_points = 100
    with h5py.File(SCENARIO_PATH) as fh:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=fh, device=device,
            number_of_surface_points_per_facet=torch.tensor([surface_points, surface_points]),
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

    local_xy = hg.surface_points[hel_idx, :, :2].detach().cpu().numpy()

    mask = one_hot_mask(hel_idx, 1, n_hel, device)
    surfaces = aimed_neighbour_surfaces(
        hg, scenario, sun[0], target_index, device,
        fixed_tilt_rows=blocker_rows, fixed_tilt=0.0,  # vertical, max block
    )
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
        ray_tracer.blocking_active = True
        ray_tracer.blocking_heliostat_surfaces_active = surfaces
        with exact_blocking(), capture_blocking_mask() as blk, capture_target_intersections() as tgt_int:
            flux, _, _, bf = ray_tracer.trace_rays(
                incident_ray_directions=sun, active_heliostats_mask=mask,
                target_area_indices=tgt, device=device,
            )
    print(f"blocked fraction (rays) = {1.0 - bf.item():.4f}")

    # blk["blocked"]: [1, n_rays, n_points], ~1 = blocked. Average over rays -> per-point blocked frac.
    per_point_blocked = blk["blocked"][0].mean(dim=0).cpu().numpy()
    blocked_mask = per_point_blocked > 0.5
    print(f"mirror points blocked (>50% of their rays): {blocked_mask.mean() * 100:.1f} %")

    # tgt_int["bitmap_e"/"bitmap_u"]: [1, n_rays, n_points] -- average over rays per point.
    e = tgt_int["bitmap_e"][0].mean(dim=0).cpu().numpy()
    u = tgt_int["bitmap_u"][0].mean(dim=0).cpu().numpy()
    bitmap_res = ray_tracer.bitmap_resolution
    height_u = float(bitmap_res[1] if bitmap_res.numel() > 1 else bitmap_res)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4))

    ax1.scatter(local_xy[~blocked_mask, 0], local_xy[~blocked_mask, 1], c="0.75", s=4, linewidths=0,
                label="unblocked mirror points")
    ax1.scatter(local_xy[blocked_mask, 0], local_xy[blocked_mask, 1], c="crimson", s=4, linewidths=0,
                label="blocked mirror points")
    ax1.set_title(f"{HELIOSTAT_ID} mirror surface\n(which points does AZ70/AZ71 vertical shadow?)")
    ax1.set_xlabel("mirror width [m]"); ax1.set_ylabel("mirror height [m]")
    ax1.set_aspect("equal")
    ax1.legend(loc="upper right", fontsize=8)

    ax2.scatter(e[~blocked_mask], u[~blocked_mask], c="0.75", s=4, linewidths=0,
                label="rays from unblocked points")
    ax2.scatter(e[blocked_mask], u[blocked_mask], c="crimson", s=4, linewidths=0,
                label="rays from blocked points\n(the flux REMOVED by blocking)")
    ax2.axhline(height_u / 2.0, c="k", lw=0.8, ls="--", alpha=0.6, label="target vertical centre")
    ax2.invert_yaxis()  # bitmap u grows downward, matches the flux renders (origin='upper')
    ax2.set_title("Where those same rays land on target\n(before blocking removes the red ones)")
    ax2.set_xlabel("bitmap e [px]"); ax2.set_ylabel("bitmap u [px] (down -->)")
    ax2.set_aspect("equal")
    ax2.legend(loc="upper right", fontsize=7)

    mean_u_blocked = u[blocked_mask].mean() if blocked_mask.any() else float("nan")
    mean_u_all = u.mean()
    fig.suptitle(
        f"{HELIOSTAT_ID}: mean target-u of blocked rays = {mean_u_blocked:.1f} px vs. "
        f"all rays = {mean_u_all:.1f} px  (bitmap u grows DOWNWARD)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT_DIR / "ba72_blocked_ray_target_mapping.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"mean target-u: blocked-rays={mean_u_blocked:.2f}px  all-rays={mean_u_all:.2f}px "
          f"(bitmap_height={height_u:.0f}px)")
    print("If mean_u(blocked) < mean_u(all): the removed rays sit ABOVE centre on target "
          "(u grows downward) -> removing them pulls the remaining centroid DOWN. "
          "If mean_u(blocked) > mean_u(all): removed rays sit below centre -> centroid should move UP.")


if __name__ == "__main__":
    main()
