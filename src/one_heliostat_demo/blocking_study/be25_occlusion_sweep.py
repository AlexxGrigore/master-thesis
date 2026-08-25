"""BE25 version of `plot_centroid_vs_occlusion.py`, for direct comparison with BA72.

Unlike the BA72 script (which uses a synthetic forward-aim with a fixed
Stage-1 checkpoint and a hand-picked sun direction), this uses the SAME
generic real-data pipeline as `geometry_schematic.py` /
`blocked_vs_unblocked_grid.py`: the full 1277-heliostat field scenario, a real
TRAIN sun sample, and `identify_blockers`'s automatic geometric cone test for
BE25's real neighbours (BC26/BD25/BD26/BE26 at sample 0027, 25.9% blocked --
matches the field-documented near-max exposure).

BE25 itself is aimed at the target's centre (synthetic forward-aim, same
convention as the BA72 script -- there is no trained kinematic model needed
for this geometric sweep). The identified blockers are swept continuously
from VERTICAL (tilt=0, max blocking) to HORIZONTAL (tilt=1, no blocking).

Four outputs, matching the BA72 script's naming:
  1. `centroid_trail_over_flux.png`
  2. `centroid_trajectory_meters.png`
  3. `focal_spot_occlusion_sweep.gif`
  4. `focal_spot_occlusion_sweep_with_centroid.gif`

Usage
-----
    python be25_occlusion_sweep.py
    python be25_occlusion_sweep.py --sun-sample 0027 --n-steps 20 --fps 2
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

_here = pathlib.Path(__file__).resolve().parent
_src = _here.parents[1]
for _p in (str(_src), str(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from artist.flux import get_center_of_mass  # noqa: E402
from artist.geometry import bitmap_coordinates_to_target_coordinates  # noqa: E402
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer  # noqa: E402
from artist.util import set_logger_config  # noqa: E402

from blocking_utils import manually_rotated_surfaces, one_hot_mask, rotation_about_axis  # noqa: E402
from brute_blocking import exact_blocking  # noqa: E402
from validate_blocking_flux import configure, identify_blockers, load_context, load_train_sun_positions  # noqa: E402

log = logging.getLogger(__name__)

HELIOSTAT_ID = "BE25"


def render_frame(flux: torch.Tensor, path: pathlib.Path, title_lines: list[str], vmax: float,
                  centroid_px: tuple[float, float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=110)
    im = ax.imshow(flux.numpy(), cmap="inferno", origin="upper", vmin=0.0, vmax=vmax)
    if centroid_px is not None:
        ax.scatter(*centroid_px, marker="+", s=140, c="deepskyblue", linewidths=2.2, zorder=6,
                   label="centroid")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax.set_title("\n".join(title_lines), fontsize=10)
    ax.set_xlabel("bitmap e [px]", fontsize=8); ax.set_ylabel("bitmap u [px]", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="flux [a.u.]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sun-sample", default="0027")
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--surface-points", type=int, default=100)
    parser.add_argument("--rays", type=int, default=10)
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("artist", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    configure(HELIOSTAT_ID)
    from validate_blocking_flux import PLOT_DIR, GIF_DIR  # noqa: E402
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    GIF_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    scenario, hg, hel_idx, target_index, aim_point, hel_dist_m = load_context(
        device, surface_points_per_facet=args.surface_points, rays_per_surface_point=args.rays,
    )
    n_hel = hg.number_of_heliostats

    samples = load_train_sun_positions()
    sun_record = next(s for s in samples if s["sample_id"] == args.sun_sample)
    sun = torch.tensor(sun_record["incident_ray_direction"] + [0.0], dtype=torch.float, device=device).view(1, 4)
    tgt = torch.tensor([target_index], device=device)

    blocker_rows = identify_blockers(scenario, hg, hel_idx, sun[0], target_index, aim_point, device)
    blocker_names = [str(hg.names[r]) for r in blocker_rows]
    log.info(f"blockers: {blocker_names}")

    d3 = sun[0, :3].detach().cpu()
    sun_h = -d3[:2]
    n_vertical = torch.tensor([sun_h[0], sun_h[1], 0.0]) / torch.norm(sun_h)
    n_horizontal = torch.tensor([0.0, 0.0, 1.0])
    up = torch.tensor([0.0, 0.0, 1.0])
    x_w = torch.linalg.cross(up, n_vertical)
    x_w = x_w / x_w.norm()
    r_vertical = torch.stack([x_w, up, n_vertical], dim=1)
    tilt_axis = torch.linalg.cross(n_vertical, n_horizontal)
    tilt_axis = tilt_axis / tilt_axis.norm()

    frames_dir = PLOT_DIR / "frames_occlusion_sweep"
    frames_dir.mkdir(parents=True, exist_ok=True)

    tilts = np.linspace(0.0, 1.0, args.n_steps)
    records = []
    fluxes = []
    mask = one_hot_mask(hel_idx, 1, n_hel, device)
    for tilt in tilts:
        rotation = rotation_about_axis(tilt_axis, float(tilt) * np.pi / 2.0) @ r_vertical
        surfaces = torch.stack(
            [manually_rotated_surfaces(hg, row, rotation).to(device) for row in blocker_rows], dim=0
        )
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
            ray_tracer.blocking_active = True
            ray_tracer.blocking_heliostat_surfaces_active = surfaces
            with exact_blocking():
                flux, _, _, bf = ray_tracer.trace_rays(
                    incident_ray_directions=sun, active_heliostats_mask=mask,
                    target_area_indices=tgt, device=device,
                )
        bitmap_res = ray_tracer.bitmap_resolution
        bc = get_center_of_mass(bitmaps=flux, device=device)
        cent = bitmap_coordinates_to_target_coordinates(
            bitmap_coordinates=bc, bitmap_resolution=bitmap_res,
            solar_tower=scenario.solar_tower, target_area_indices=tgt, device=device,
        )[0]
        fluxes.append(flux[0].detach().cpu())
        records.append({
            "tilt": float(tilt),
            "blocked_fraction": float(1.0 - bf.item()),
            "centroid_enu": cent.cpu().tolist(),
            "centroid_px": (float(bc[0, 0].item()), float(bc[0, 1].item())),
            "total_flux": float(flux.sum()),
        })
        print(f"tilt={tilt:.3f}  blocked={records[-1]['blocked_fraction']*100:5.2f}%  "
              f"flux={records[-1]['total_flux']:.0f}  centroid_px={records[-1]['centroid_px']}")

    unblocked_enu = np.array(records[-1]["centroid_enu"][:3])
    for r in records:
        r["shift_mrad"] = float(
            np.linalg.norm(np.array(r["centroid_enu"][:3]) - unblocked_enu) / hel_dist_m * 1000.0
        )

    # ---------------------------------------------------------------- GIF
    shared_vmax = max(float(f.max()) for f in fluxes)
    frames_dir_c = PLOT_DIR / "frames_occlusion_sweep_with_centroid"
    frames_dir_c.mkdir(parents=True, exist_ok=True)
    for step, (flux, r) in enumerate(zip(fluxes, records)):
        title_lines = [
            f"{HELIOSTAT_ID} focal spot (sample {args.sun_sample}) | step {step + 1}/{args.n_steps}",
            f"tilt {r['tilt'] * 100:.0f} %   blocked {r['blocked_fraction'] * 100:.2f} %   "
            f"flux {r['total_flux']:.0f}",
        ]
        render_frame(flux, frames_dir / f"frame_{step:03d}.png", title_lines, vmax=shared_vmax)
        render_frame(flux, frames_dir_c / f"frame_{step:03d}.png", title_lines, vmax=shared_vmax,
                     centroid_px=r["centroid_px"])
    pngs = sorted(frames_dir.glob("frame_*.png"))
    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in pngs]
    gif_path = GIF_DIR / "focal_spot_occlusion_sweep.gif"
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=int(1000.0 / args.fps), loop=0)
    print(f"wrote {gif_path} ({len(pngs)} frames @ {args.fps} fps)")

    pngs_c = sorted(frames_dir_c.glob("frame_*.png"))
    images_c = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in pngs_c]
    gif_path_c = GIF_DIR / "focal_spot_occlusion_sweep_with_centroid.gif"
    images_c[0].save(gif_path_c, save_all=True, append_images=images_c[1:], duration=int(1000.0 / args.fps), loop=0)
    print(f"wrote {gif_path_c} ({len(pngs_c)} frames @ {args.fps} fps)")

    # ------------------------------------------------------------ plot 1
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

    px = np.array([r["centroid_px"] for r in records])
    unblocked_flux = fluxes[-1].numpy()

    def _plot_endpoints(ax, marker_size=60, lw=0.8):
        ax.scatter(*px[0], marker="X", s=marker_size, c="red", edgecolor="k", linewidths=lw,
                   zorder=6, label=f"vertical (max block, {records[0]['blocked_fraction']*100:.0f}%)")
        ax.scatter(*px[-1], marker="o", s=marker_size, c="deepskyblue", edgecolor="k", linewidths=lw,
                   zorder=6, label=f"horizontal (no block, {records[-1]['blocked_fraction']*100:.1f}%)")

    fig1, ax1 = plt.subplots(figsize=(7.6, 6.4))
    im = ax1.imshow(unblocked_flux, cmap="inferno", origin="upper", vmin=0.0, vmax=float(unblocked_flux.max()))
    fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="flux [a.u.] (unblocked)")
    _plot_endpoints(ax1)
    ax1.set_xlabel("bitmap e [px]"); ax1.set_ylabel("bitmap u [px]")
    ax1.set_title(f"{HELIOSTAT_ID}: focal spot (unblocked) with the two occlusion endpoints\n"
                  f"(real neighbours {', '.join(blocker_names)}, sun sample {args.sun_sample})")
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)

    pad_px = 8
    x0, x1 = min(px[0, 0], px[-1, 0]) - pad_px, max(px[0, 0], px[-1, 0]) + pad_px
    y0, y1 = min(px[0, 1], px[-1, 1]) - pad_px, max(px[0, 1], px[-1, 1]) + pad_px
    axins = zoomed_inset_axes(ax1, zoom=unblocked_flux.shape[0] / (x1 - x0) * 0.16, loc="lower left",
                               bbox_to_anchor=(0.02, 0.02), bbox_transform=ax1.transAxes, borderpad=0)
    axins.imshow(unblocked_flux, cmap="inferno", origin="upper", vmin=0.0, vmax=float(unblocked_flux.max()))
    _plot_endpoints(axins, marker_size=110, lw=1.0)
    axins.set_xlim(x0, x1); axins.set_ylim(y1, y0)
    axins.set_xticks([]); axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor("white"); spine.set_linewidth(1.5)
    mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="white", lw=1.0, zorder=3)

    fig1.tight_layout()
    out1 = PLOT_DIR / "centroid_trail_over_flux.png"
    fig1.savefig(out1, dpi=140)
    plt.close(fig1)
    print(f"wrote {out1}")

    # ------------------------------------------------------------ plot 2
    enu = np.array([r["centroid_enu"][:3] for r in records])
    ref_east, ref_up = enu[-1, 0], enu[-1, 2]
    east_m, up_m = enu[:, 0] - ref_east, enu[:, 2] - ref_up

    fig2, ax2 = plt.subplots(figsize=(9.5, 6.4))
    ax2.plot(east_m, up_m, "-", c="gray", lw=1.2, alpha=0.7, zorder=4)
    sc2 = ax2.scatter(east_m, up_m, c=tilts, cmap="viridis", s=70, zorder=5, edgecolor="k", linewidths=0.5)
    ax2.scatter(east_m[0], up_m[0], marker="X", s=80, c="red", edgecolor="k", linewidths=0.8,
                zorder=6, label=f"vertical (max block, {records[0]['blocked_fraction']*100:.0f}%)")
    ax2.scatter(east_m[-1], up_m[-1], marker="o", s=80, c="deepskyblue", edgecolor="k", linewidths=0.8,
                zorder=6, label=f"horizontal (no block, {records[-1]['blocked_fraction']*100:.1f}%) = origin")
    fig2.colorbar(sc2, ax=ax2, label="tilt (0=vertical/max block, 1=horizontal/none)")
    ax2.axhline(0, c="0.85", lw=0.8, zorder=1); ax2.axvline(0, c="0.85", lw=0.8, zorder=1)
    ax2.set_xlabel("east offset from unblocked centroid [m]")
    ax2.set_ylabel("up offset from unblocked centroid [m]")
    ax2.set_title(f"{HELIOSTAT_ID}: centroid trajectory relative to the unblocked position\n"
                  f"as blockers tilt vertical -> horizontal (real neighbours, sample {args.sun_sample})")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    out2 = PLOT_DIR / "centroid_trajectory_meters.png"
    fig2.savefig(out2, dpi=140)
    plt.close(fig2)
    print(f"wrote {out2}")

    (PLOT_DIR / "centroid_vs_occlusion.json").write_text(json.dumps({
        "sun_sample": args.sun_sample,
        "blockers": blocker_names,
        "records": records,
        "hel_dist_m": hel_dist_m,
    }, indent=2))


if __name__ == "__main__":
    main()
