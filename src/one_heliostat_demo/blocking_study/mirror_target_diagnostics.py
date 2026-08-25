"""Two follow-up diagnostics for the BH58 mirror-shadow-map result.

1. Mirror-to-target MAPPING check: is the target image a direct copy of the
   mirror surface, or is it flipped/rotated (as any converging mirror would
   produce, like a pinhole camera or camera-lens image)? Colors the mirror's
   surface points with a synthetic "color wheel" (hue = angle, brightness =
   radius, easy to eyeball for flips/rotations) and renders where each point's
   ray actually lands on the target bitmap, in the same colors.

2. PER-BLOCKER breakdown: the combined mirror shadow map
   (`mirror_shadow_map.py`) shows one continuous wedge for all 4 blockers
   together. This traces each of the 4 blocker heliostats INDIVIDUALLY (one
   primitive at a time) to show which specific heliostat is responsible for
   which part of the mirror's shadow, and whether their silhouettes are
   actually distinguishable or genuinely fused.

Usage
-----
    python mirror_target_diagnostics.py --heliostat-id BH58 --sun-sample 0011
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_here = pathlib.Path(__file__).resolve().parent
_src = _here.parents[1]
_sh = _src / "one_heliostat_demo" / "single_heliostat"
for _p in (str(_src), str(_sh), str(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer  # noqa: E402
from artist.util import set_logger_config  # noqa: E402

from blocking_utils import one_hot_mask  # noqa: E402
from brute_blocking import (  # noqa: E402
    capture_blocking_mask,
    capture_target_intersections,
    exact_blocking,
)
from mirror_shadow_map import trace_capture  # noqa: E402
from validate_blocking_flux import (  # noqa: E402
    RANDOM_SEED,
    configure,
    identify_blockers,
    load_context,
    load_train_sun_positions,
    manually_rotated_surfaces,
    rotation_about_axis,
)

log = logging.getLogger(__name__)


def wheel_colors(local_xy: np.ndarray) -> np.ndarray:
    """HSV colour wheel keyed on each point's angle/radius around the mirror centre."""
    x, y = local_xy[:, 0], local_xy[:, 1]
    r = np.hypot(x, y)
    r = r / r.max()
    theta = (np.arctan2(y, x) + np.pi) / (2 * np.pi)  # 0..1
    hsv = np.stack([theta, np.full_like(theta, 0.85), 0.35 + 0.65 * r], axis=-1)
    return mcolors.hsv_to_rgb(hsv)


def mapping_check(scenario, hg, hel_idx, target_index, aim_center, device, local_xy, sun, out_path,
                   heliostat_id):
    """Part 1: render the mirror (colour wheel) next to where those rays land on target."""
    n_hel = hg.number_of_heliostats
    mask = one_hot_mask(hel_idx, 1, n_hel, device)
    with torch.no_grad():
        hg.activate_heliostats(active_heliostats_mask=mask, device=device)
        hg.align_surfaces_with_incident_ray_directions(
            aim_points=aim_center.view(1, 4), incident_ray_directions=sun.view(1, 4),
            active_heliostats_mask=mask, device=device,
        )
        ray_tracer = HeliostatRayTracer(
            scenario=scenario, heliostat_group=hg, blocking_active=False,
            world_size=1, rank=0, batch_size=1, random_seed=RANDOM_SEED,
        )
        with capture_target_intersections() as captured:
            ray_tracer.trace_rays(
                incident_ray_directions=sun.view(1, 4), active_heliostats_mask=mask,
                target_area_indices=torch.tensor([target_index], device=device), device=device,
            )
    # [1, n_rays, n_points] -> average over the (near-identical, point-source-like few) rays per point
    e = captured["bitmap_e"][0].mean(dim=0).cpu().numpy()
    u = captured["bitmap_u"][0].mean(dim=0).cpu().numpy()

    colors = wheel_colors(local_xy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.scatter(local_xy[:, 0], local_xy[:, 1], c=colors, s=3, marker="s", linewidths=0)
    ax1.set_title(f"{heliostat_id} mirror surface (colour = synthetic marker)")
    ax1.set_xlabel("mirror width [m]"); ax1.set_ylabel("mirror height [m]")
    ax1.set_aspect("equal")
    ax2.scatter(e, u, c=colors, s=3, marker="s", linewidths=0)
    ax2.set_title("Where those same rays land on the target bitmap")
    ax2.set_xlabel("bitmap e [px]"); ax2.set_ylabel("bitmap u [px]")
    ax2.invert_yaxis()  # bitmap u grows downward, like the flux renders
    ax2.set_aspect("equal")
    fig.suptitle("Mirror -> target mapping check: is the target image flipped relative to the mirror?")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    log.info(f"wrote {out_path}")


def per_blocker_breakdown(scenario, hg, hel_idx, target_index, aim_center, sun, blocker_rows,
                            blocker_names, rotation, device, local_xy, out_path, heliostat_id):
    """Part 2: trace each blocker alone; show which one is responsible for which mirror region."""
    n_points = hg.surface_points.shape[1]
    per_blocker = np.zeros((len(blocker_rows), n_points))
    for i, row in enumerate(blocker_rows):
        single = manually_rotated_surfaces(hg, row, rotation).unsqueeze(0)
        _, _, per_point = trace_capture(scenario, hg, hel_idx, sun, target_index, aim_center, single, device)
        per_blocker[i] = per_point.numpy()
        log.info(f"{blocker_names[i]}: {(per_blocker[i] > 0.5).mean() * 100:.1f} % of mirror points blocked alone")

    # sanity check vs. the joint (all 4 at once) trace
    joint_surfaces = torch.stack(
        [manually_rotated_surfaces(hg, row, rotation) for row in blocker_rows], dim=0
    )
    _, joint_frac, joint_per_point = trace_capture(
        scenario, hg, hel_idx, sun, target_index, aim_center, joint_surfaces, device
    )
    union = (per_blocker > 0.5).any(axis=0)
    joint_bin = (joint_per_point.numpy() > 0.5)
    agreement = (union == joint_bin).mean()
    log.info(f"joint blocked={joint_frac * 100:.2f} %  union-of-singles vs joint agreement={agreement * 100:.2f} %")

    assignment = np.full(n_points, -1, dtype=int)  # -1 = none blocked
    max_frac = per_blocker.max(axis=0)
    argmax = per_blocker.argmax(axis=0)
    assignment[max_frac > 0.5] = argmax[max_frac > 0.5]
    n_multi = int(((per_blocker > 0.5).sum(axis=0) > 1).sum())

    cmap = plt.get_cmap("tab10")
    colors = np.array([[0.85, 0.85, 0.85, 1.0]] * n_points)
    for i in range(len(blocker_rows)):
        colors[assignment == i] = cmap(i)

    fig, ax = plt.subplots(figsize=(7, 5.6))
    ax.scatter(local_xy[:, 0], local_xy[:, 1], c=colors, s=4, marker="s", linewidths=0)
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=cmap(i), markersize=10,
                           label=f"{blocker_names[i]} ({(assignment == i).sum()} pts)")
               for i in range(len(blocker_rows))]
    handles.append(plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=(0.85, 0.85, 0.85, 1.0),
                                markersize=10, label="unblocked"))
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    ax.set_xlabel("mirror width [m]"); ax.set_ylabel("mirror height [m]")
    ax.set_aspect("equal")
    ax.set_title(
        f"{heliostat_id} mirror, vertical pose: which blocker shadows which region\n"
        f"{n_multi} pts blocked by >1 heliostat  |  union-vs-joint agreement {agreement * 100:.1f} %"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    log.info(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heliostat-id", default="BH58")
    parser.add_argument("--sun-sample", default="0011")
    parser.add_argument("--surface-points", type=int, default=60)
    parser.add_argument("--rays", type=int, default=3)
    parser.add_argument("--point-source-std-mrad", type=float, default=0.001)
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("artist", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    configure(args.heliostat_id, out_name=f"{args.heliostat_id}_point_source")
    from validate_blocking_flux import PLOT_DIR  # noqa: E402

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    point_source_covariance = (args.point_source_std_mrad * 1e-3) ** 2
    scenario, hg, hel_idx, target_index, aim_center, hel_dist_m = load_context(
        device, surface_points_per_facet=args.surface_points, rays_per_surface_point=args.rays,
        point_source_covariance=point_source_covariance,
    )
    local_xy = hg.surface_points[hel_idx, :, :2].detach().cpu().numpy()

    samples = load_train_sun_positions()
    sun_record = next(s for s in samples if s["sample_id"] == args.sun_sample)
    sun = torch.tensor(sun_record["incident_ray_direction"] + [0.0], dtype=torch.float, device=device)
    d3 = sun[:3].detach().cpu()

    mapping_check(scenario, hg, hel_idx, target_index, aim_center, device, local_xy, sun,
                  PLOT_DIR / "mirror_target_mapping_check.png", args.heliostat_id)

    blocker_rows = identify_blockers(scenario, hg, hel_idx, sun, target_index, aim_center, device)
    blocker_names = [str(hg.names[i]) for i in blocker_rows]

    sun_h = -d3[:2]
    n_vertical = torch.tensor([sun_h[0], sun_h[1], 0.0]) / torch.norm(sun_h)
    up = torch.tensor([0.0, 0.0, 1.0])
    x_w = torch.linalg.cross(up, n_vertical)
    x_w = x_w / x_w.norm()
    r_vertical = torch.stack([x_w, up, n_vertical], dim=1)  # tilt=0 (fully vertical) rotation

    per_blocker_breakdown(
        scenario, hg, hel_idx, target_index, aim_center, sun, blocker_rows, blocker_names,
        r_vertical, device, local_xy, PLOT_DIR / "mirror_per_blocker_breakdown.png",
        args.heliostat_id,
    )


if __name__ == "__main__":
    main()
