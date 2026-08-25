"""Direct flux-difference diagnostic for the BA72 "centroid moves the wrong way" puzzle.

`plot_centroid_vs_occlusion.py`: as AZ70/AZ71 tilt vertical->horizontal (max
block -> no block), BA72's centroid moves from up_m=-0.18 (max block) to
up_m=0 (no block), i.e. blocking pushes the centroid DOWN. A per-mirror-point
ray-landing proxy (`ba72_blocked_ray_target_mapping.py`) said the opposite:
rays from blocked mirror points land BELOW target centre on average, so
removing them should push the remaining centroid UP. Since that proxy
coarsens over per-point ray sunshape jitter and a 50%-blocked threshold, it
may not represent the actual (continuous, intensity-weighted) removed flux.

This script renders the identical scene at max-block and no-block, takes the
literal flux DIFFERENCE image, and computes the difference image's own
center of mass -- i.e. exactly where the removed flux physically sits on the
target, with no proxy or thresholding.
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

from artist.flux import get_center_of_mass  # noqa: E402
from artist.geometry import bitmap_coordinates_to_target_coordinates  # noqa: E402
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
OUT_DIR = _ROOT / "outputs" / "new_mapping_function" / "ba72_occlusion" / "plots"
SUN_INCIDENT_RAY_DIRECTION = [0.037787847220897675, 0.4960194528102875, -0.8674888014793396, 0.0]


def load_stage1_kinematics(kinematic, device: torch.device) -> None:
    ckpt = torch.load(STAGE1_CHECKPOINT, map_location=device)
    kinematic.translation_deviation_parameters.data.copy_(ckpt["translation"].to(device))
    kinematic.rotation_deviation_parameters.data.copy_(ckpt["rotation"].to(device))
    kinematic.actuators.optimizable_parameters.data.copy_(ckpt["act_angle"].to(device))
    kinematic.actuators.non_optimizable_parameters.data.copy_(ckpt["act_offset"].to(device))
    kinematic._base_position_deviation = ckpt["base_pos"].clone().to(device)


def render(hg, scenario, sun, tgt, aim_point, mask, blocker_rows, tilt, device):
    surfaces = aimed_neighbour_surfaces(
        hg, scenario, sun[0], int(tgt.item()), device,
        fixed_tilt_rows=blocker_rows, fixed_tilt=tilt,
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
        with exact_blocking():
            flux, _, _, bf = ray_tracer.trace_rays(
                incident_ray_directions=sun, active_heliostats_mask=mask,
                target_area_indices=tgt, device=device,
            )
    return flux[0].detach().cpu(), float(1.0 - bf.item()), ray_tracer.bitmap_resolution


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

    flux_max, bf_max, res = render(hg, scenario, sun, tgt, aim_point, mask, blocker_rows, 0.0, device)
    flux_none, bf_none, _ = render(hg, scenario, sun, tgt, aim_point, mask, blocker_rows, 1.0, device)

    print(f"blocked fraction: max-block={bf_max:.4f}  no-block={bf_none:.4f}")
    print(f"total flux: max-block={flux_max.sum():.1f}  no-block={flux_none.sum():.1f}")

    u_max = com_u(flux_max)
    u_none = com_u(flux_none)
    print(f"centroid bitmap-u: max-block={u_max:.3f}px  no-block={u_none:.3f}px  "
          f"(u grows DOWNWARD; max-block u {'>' if u_max > u_none else '<'} no-block u "
          f"=> max-block centroid is {'LOWER' if u_max > u_none else 'HIGHER'} on target)")

    diff = flux_none - flux_max  # positive where blocking removed flux
    removed_total = float(diff.clamp(min=0).sum())
    added_total = float((-diff).clamp(min=0).sum())
    print(f"flux removed by blocking (positive part of diff) = {removed_total:.1f}")
    print(f"flux INCREASED by blocking (negative part of diff) = {added_total:.1f}  "
          "(nonzero = blocking redistributes some intensity, not pure removal)")

    diff_pos = diff.clamp(min=0)
    u_removed = com_u(diff_pos) if removed_total > 0 else float("nan")
    print(f"center-of-mass (bitmap-u) of the REMOVED flux = {u_removed:.3f}px "
          f"vs. no-block centroid u={u_none:.3f}px "
          f"=> removed flux sits {'BELOW' if u_removed > u_none else 'ABOVE'} the centroid "
          "(u grows downward)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmax = float(max(flux_max.max(), flux_none.max()))
    for ax, f, title in zip(axes[:2], [flux_none, flux_max], ["no block (horizontal)", "max block (vertical)"]):
        im = ax.imshow(f.numpy(), cmap="inferno", origin="upper", vmin=0, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    dmax = float(diff.abs().max())
    im3 = axes[2].imshow(diff.numpy(), cmap="RdBu_r", origin="upper", vmin=-dmax, vmax=dmax)
    axes[2].set_title("diff = no_block - max_block\n(red = flux REMOVED by blocking)")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_xlabel("bitmap e [px]"); ax.set_ylabel("bitmap u [px] (down -->)")
    fig.suptitle(f"{HELIOSTAT_ID}: literal flux difference, max-block vs no-block")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT_DIR / "ba72_flux_diff_diagnostic.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
