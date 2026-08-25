"""Generic per-heliostat mirror-to-target mapping check.

Colours every studied heliostat's surface point by local mirror height (blue
= bottom, red = top), traces all of them UNBLOCKED under a real TRAIN sun
sample (same generic pipeline as `geometry_schematic.py` /
`blocked_vs_unblocked_grid.py` / `occlusion_sweep.py`), and plots where each
point's ray lands on the target bitmap -- split into bottom-half-only,
top-half-only, and merged panels so the two populations aren't overplotted on
each other. Same layout as `ba72_mirror_inversion_check.py`.

Usage
-----
    python mirror_inversion_check.py --heliostat-id BE25 --sun-sample 0027
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

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
from artist.util import set_logger_config  # noqa: E402

from blocking_utils import apply_stage1_checkpoint, one_hot_mask  # noqa: E402
from brute_blocking import capture_target_intersections  # noqa: E402
from validate_blocking_flux import configure, load_context, load_train_sun_positions  # noqa: E402

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heliostat-id", required=True)
    parser.add_argument("--sun-sample", required=True)
    parser.add_argument("--surface-points", type=int, default=100)
    parser.add_argument("--rays", type=int, default=3)
    parser.add_argument("--checkpoint", default=None,
                         help="Stage-1 checkpoint .pt to apply instead of ideal nominal kinematics.")
    args = parser.parse_args()
    heliostat_id = args.heliostat_id

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("artist", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    out_name = f"{heliostat_id}_trained" if args.checkpoint else heliostat_id
    configure(heliostat_id, out_name=out_name)
    from validate_blocking_flux import PLOT_DIR  # noqa: E402
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    scenario, hg, hel_idx, target_index, aim_point, hel_dist_m = load_context(
        device, surface_points_per_facet=args.surface_points, rays_per_surface_point=args.rays,
    )
    n_hel = hg.number_of_heliostats
    if args.checkpoint:
        apply_stage1_checkpoint(hg.kinematics, args.checkpoint, heliostat_id, device)
        log.info(f"Applied trained Stage-1 checkpoint: {args.checkpoint}")

    samples = load_train_sun_positions()
    sun_record = next(s for s in samples if s["sample_id"] == args.sun_sample)
    sun = torch.tensor(sun_record["incident_ray_direction"] + [0.0], dtype=torch.float, device=device).view(1, 4)
    tgt = torch.tensor([target_index], device=device)

    local_xy = hg.surface_points[hel_idx, :, :2].detach().cpu().numpy()

    mask = one_hot_mask(hel_idx, 1, n_hel, device)
    with torch.no_grad():
        hg.activate_heliostats(active_heliostats_mask=mask, device=device)
        hg.align_surfaces_with_incident_ray_directions(
            aim_points=aim_point.view(1, 4), incident_ray_directions=sun,
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

    mirror_height = local_xy[:, 1]
    bottom_mask = mirror_height < 0.0
    top_mask = ~bottom_mask
    corr = np.corrcoef(mirror_height, u)[0, 1]
    vmin, vmax = float(mirror_height.min()), float(mirror_height.max())

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    ax = axes[0, 0]
    sc = ax.scatter(local_xy[:, 0], local_xy[:, 1], c=mirror_height, cmap="coolwarm", s=4, linewidths=0,
                     vmin=vmin, vmax=vmax)
    ax.set_title(f"{heliostat_id} mirror surface\n(colour = local height; blue=bottom, red=top)")
    ax.set_xlabel("mirror width [m]"); ax.set_ylabel("mirror height [m]")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="mirror height [m]")

    def _target_panel(ax, mask_, title):
        sc = ax.scatter(e[mask_], u[mask_], c=mirror_height[mask_], cmap="coolwarm", s=4, linewidths=0,
                         vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("bitmap e [px]"); ax.set_ylabel("bitmap u [px] (down -->)")
        ax.set_aspect("equal")
        ax.set_xlim(e.min() - 5, e.max() + 5)
        ax.set_ylim(u.max() + 5, u.min() - 5)
        return sc

    _target_panel(axes[0, 1], bottom_mask, "Rays from BOTTOM-half mirror points only\n(blue, mirror height < 0)")
    _target_panel(axes[1, 0], top_mask, "Rays from TOP-half mirror points only\n(red, mirror height > 0)")
    sc_merged = _target_panel(axes[1, 1], np.ones_like(bottom_mask), "Merged: rays from ALL mirror points\n(same colour scale)")
    fig.colorbar(sc_merged, ax=axes[1, 1], fraction=0.046, pad=0.04, label="mirror height [m] (source point)")

    fig.suptitle(
        f"{heliostat_id} sample {args.sun_sample}: mirror-height vs. target-u correlation = {corr:.3f} "
        f"({'INVERTED image (mirror bottom -> target top)' if corr < 0 else 'upright image (mirror bottom -> target bottom)'})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = PLOT_DIR / "mirror_inversion_check.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"correlation(mirror_height, bitmap_u) = {corr:.4f}")


if __name__ == "__main__":
    main()
