"""Does BA72's mirror form an inverted (top/bottom-flipped) image on target?

Colours every BA72 surface point by its LOCAL MIRROR HEIGHT (bottom=blue,
top=red), traces all of them (unblocked, no sunshape jitter beyond ARTIST's
default) and plots where each point's ray lands on the target bitmap in the
same colour. If the mirror's bottom (blue) rays land at the target's TOP and
the mirror's top (red) rays land at the target's BOTTOM, the mirror forms a
vertically-inverted image -- exactly like any converging mirror/lens (pinhole
camera analogy) -- which would explain why AZ70/AZ71 shadowing BA72's
physical bottom (confirmed by `ba72_blocked_ray_target_mapping.py`) removes
flux from the target's TOP (confirmed by `ba72_flux_diff_diagnostic.py`),
pulling the remaining centroid DOWN.
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

from blocking_utils import one_hot_mask  # noqa: E402
from brute_blocking import capture_target_intersections  # noqa: E402

log = logging.getLogger(__name__)

_ROOT = _here.parents[2]
HELIOSTAT_ID = "BA72"
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
    with h5py.File(SCENARIO_PATH) as fh:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=fh, device=device,
            number_of_surface_points_per_facet=torch.tensor([100, 100]),
        )
    scenario.set_number_of_rays(3)
    hg = scenario.heliostat_field.heliostat_groups[0]
    kinematic = hg.kinematics
    hel_idx = hg.names.index(HELIOSTAT_ID)
    n_hel = hg.number_of_heliostats
    load_stage1_kinematics(kinematic, device)

    sun = torch.tensor(SUN_INCIDENT_RAY_DIRECTION, dtype=torch.float, device=device).view(1, 4)
    target_index = scenario.solar_tower.target_name_to_index[TARGET_NAME]
    tgt = torch.tensor([target_index], device=device)
    aim_point = scenario.solar_tower.get_centers_of_target_areas(target_area_indices=tgt, device=device)
    mask = one_hot_mask(hel_idx, 1, n_hel, device)

    local_xy = hg.surface_points[hel_idx, :, :2].detach().cpu().numpy()

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
        with capture_target_intersections() as captured:
            ray_tracer.trace_rays(
                incident_ray_directions=sun, active_heliostats_mask=mask,
                target_area_indices=tgt, device=device,
            )

    e = captured["bitmap_e"][0].mean(dim=0).cpu().numpy()
    u = captured["bitmap_u"][0].mean(dim=0).cpu().numpy()

    mirror_height = local_xy[:, 1]  # local y = mirror "up/down" axis
    bottom_mask = mirror_height < 0.0  # "red" is top (positive), "blue"/other colour is bottom (negative)
    top_mask = ~bottom_mask
    corr = np.corrcoef(mirror_height, u)[0, 1]
    vmin, vmax = float(mirror_height.min()), float(mirror_height.max())

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    ax = axes[0, 0]
    sc = ax.scatter(local_xy[:, 0], local_xy[:, 1], c=mirror_height, cmap="coolwarm", s=4, linewidths=0,
                     vmin=vmin, vmax=vmax)
    ax.set_title(f"{HELIOSTAT_ID} mirror surface\n(colour = local height; blue=bottom, red=top)")
    ax.set_xlabel("mirror width [m]"); ax.set_ylabel("mirror height [m]")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="mirror height [m]")

    def _target_panel(ax, mask, title):
        sc = ax.scatter(e[mask], u[mask], c=mirror_height[mask], cmap="coolwarm", s=4, linewidths=0,
                         vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("bitmap e [px]"); ax.set_ylabel("bitmap u [px] (down -->)")
        ax.set_aspect("equal")
        ax.set_xlim(e.min() - 5, e.max() + 5)
        ax.set_ylim(u.max() + 5, u.min() - 5)  # inverted, matches invert_yaxis
        return sc

    _target_panel(axes[0, 1], bottom_mask, "Rays from BOTTOM-half mirror points only\n(blue, mirror height < 0)")
    _target_panel(axes[1, 0], top_mask, "Rays from TOP-half mirror points only\n(red, mirror height > 0)")
    sc_merged = _target_panel(axes[1, 1], np.ones_like(bottom_mask), "Merged: rays from ALL mirror points\n(same colour scale)")
    fig.colorbar(sc_merged, ax=axes[1, 1], fraction=0.046, pad=0.04, label="mirror height [m] (source point)")

    fig.suptitle(
        f"{HELIOSTAT_ID}: mirror-height vs. target-u correlation = {corr:.3f} "
        f"({'INVERTED image (mirror bottom -> target top)' if corr < 0 else 'upright image (mirror bottom -> target bottom)'})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "ba72_mirror_inversion_check.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"correlation(mirror_height, bitmap_u) = {corr:.4f}")


if __name__ == "__main__":
    main()
