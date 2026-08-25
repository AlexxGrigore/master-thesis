"""Mirror-surface shadow map for the point-source blocking pose sweep.

`validate_blocking_flux.py --point-source-sun` renders the flux formed AT THE
TARGET; that image is small (the lit spot is a ~40x40 px patch of a 256x256
bitmap) because the target is far away and the mirror image is a projection
of a ~2x3 m mirror onto that plane. This script instead shades the STUDIED
HELIOSTAT'S OWN MIRROR SURFACE by, for each of its surface points, what
fraction of the rays leaving that point were blocked by the neighbouring
heliostats -- i.e. the actual shadow the blockers cast on the mirror, at full
mirror resolution (100x100 points/facet x 4 facets = 40 000 points), which is
both bigger on screen and free of the target-plane projection/binning.

Per-point blocked fractions come from ARTIST's own internal soft blocking
mask (`blocking.soft_ray_blocking_mask`), captured via
`brute_blocking.capture_blocking_mask()` -- not otherwise exposed by
`HeliostatRayTracer.trace_rays`, which only returns the scalar
`blocking_factor`.

Usage
-----
    python mirror_shadow_map.py --heliostat-id BH58 --sun-sample 0011 \\
        --surface-points 100 --rays 10 --pose-steps 15
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

_here = pathlib.Path(__file__).resolve().parent
_src = _here.parents[1]
_sh = _src / "one_heliostat_demo" / "single_heliostat"
for _p in (str(_src), str(_sh), str(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer  # noqa: E402
from artist.util import set_logger_config  # noqa: E402

from blocking_utils import one_hot_mask  # noqa: E402
from brute_blocking import capture_blocking_mask, exact_blocking  # noqa: E402
from validate_blocking_flux import (  # noqa: E402
    RANDOM_SEED,
    TARGET_NAME,
    configure,
    identify_blockers,
    load_context,
    load_train_sun_positions,
    manually_rotated_surfaces,
    render_frame,
    rotation_about_axis,
)

log = logging.getLogger(__name__)

FRAME_DURATION_S = 0.15


def trace_capture(
    scenario, hg, hel_idx, incident_ray_direction, target_index, aim_point, surfaces, device,
):
    """One blocking trace that also captures the per-surface-point blocked fraction.

    Returns (flux [H,W] cpu, blocked_ray_fraction: float, per_point_blocked: [n_points] cpu).
    """
    n_hel = hg.number_of_heliostats
    sun = incident_ray_direction.view(1, 4)
    mask = one_hot_mask(hel_idx, 1, n_hel, device)
    with torch.no_grad():
        hg.activate_heliostats(active_heliostats_mask=mask, device=device)
        hg.align_surfaces_with_incident_ray_directions(
            aim_points=aim_point.view(1, 4),
            incident_ray_directions=sun,
            active_heliostats_mask=mask,
            device=device,
        )
        ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=hg,
            blocking_active=False,
            world_size=1,
            rank=0,
            batch_size=1,
            random_seed=RANDOM_SEED,
        )
        ray_tracer.blocking_active = True
        ray_tracer.blocking_heliostat_surfaces_active = surfaces.to(device)
        with exact_blocking(), capture_blocking_mask() as captured:
            flux, _, _, blocking_factor = ray_tracer.trace_rays(
                incident_ray_directions=sun,
                active_heliostats_mask=mask,
                target_area_indices=torch.tensor([target_index], device=device),
                device=device,
            )
    blocked = captured.get("blocked")
    n_points = hg.surface_points.shape[1]
    per_point = (
        blocked[0].mean(dim=0).cpu() if blocked is not None else torch.zeros(n_points)
    )
    return flux[0].detach().cpu(), float(1.0 - blocking_factor.item()), per_point


def render_mirror_frame(
    local_xy: np.ndarray, per_point_blocked: torch.Tensor, path: pathlib.Path,
    title_lines: list[str],
) -> None:
    """Scatter the mirror's own surface points, shaded by blocked fraction."""
    frac = per_point_blocked.numpy()
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=110)
    sc = ax.scatter(
        local_xy[:, 0], local_xy[:, 1], c=frac, cmap="RdYlGn_r", vmin=0.0, vmax=1.0,
        s=3, marker="s", linewidths=0,
    )
    ax.set_title("\n".join(title_lines), fontsize=9)
    ax.set_xlabel("mirror width [m]", fontsize=8)
    ax.set_ylabel("mirror height [m]", fontsize=8)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="blocked fraction")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def assemble_gif(frames_dir: pathlib.Path, gif_path: pathlib.Path, duration_s: float) -> int:
    pngs = sorted(frames_dir.glob("frame_*.png"))
    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in pngs]
    images[0].save(
        gif_path, save_all=True, append_images=images[1:],
        duration=int(duration_s * 1000), loop=0,
    )
    return len(pngs)


def make_grid(frames_dir: pathlib.Path, plot_path: pathlib.Path, caption: str, n: int = 8) -> None:
    pngs = sorted(frames_dir.glob("frame_*.png"))
    picks = sorted(set(np.linspace(0, len(pngs) - 1, min(n, len(pngs))).astype(int).tolist()))
    ncols = 4
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, idx in zip(axes, picks):
        ax.imshow(plt.imread(pngs[idx]))
        ax.set_title(f"frame {idx:03d}", fontsize=11)
        ax.axis("off")
    for ax in axes[len(picks):]:
        ax.axis("off")
    fig.suptitle(caption, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(plot_path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heliostat-id", default="BH58")
    parser.add_argument("--sun-sample", default="0011")
    parser.add_argument("--surface-points", type=int, default=100)
    parser.add_argument("--rays", type=int, default=10)
    parser.add_argument("--pose-steps", type=int, default=15)
    parser.add_argument("--point-source-std-mrad", type=float, default=0.001)
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("artist", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    out_name = f"{args.heliostat_id}_point_source"
    configure(args.heliostat_id, out_name=out_name)
    from validate_blocking_flux import OUT_DIR, GIF_DIR, PLOT_DIR, DATA_DIR  # noqa: E402  (module globals, set by configure())

    frames_flux = DATA_DIR / "frames_pose_sweep"
    frames_mirror = DATA_DIR / "frames_mirror_shadow"
    for d in (GIF_DIR, PLOT_DIR, DATA_DIR, frames_flux, frames_mirror):
        d.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cpu")
    point_source_covariance = (args.point_source_std_mrad * 1e-3) ** 2
    scenario, hg, hel_idx, target_index, aim_center, hel_dist_m = load_context(
        device, surface_points_per_facet=args.surface_points,
        rays_per_surface_point=args.rays,
        point_source_covariance=point_source_covariance,
    )
    local_xy = hg.surface_points[hel_idx, :, :2].detach().cpu().numpy()

    samples = load_train_sun_positions()
    sun_record = next(s for s in samples if s["sample_id"] == args.sun_sample)
    sun = torch.tensor(sun_record["incident_ray_direction"] + [0.0], dtype=torch.float, device=device)
    d3 = sun[:3].detach().cpu()

    blocker_rows = identify_blockers(scenario, hg, hel_idx, sun, target_index, aim_center, device)
    blocker_names = [str(hg.names[i]) for i in blocker_rows]
    log.info(f"blockers: {blocker_names}")

    sun_h = -d3[:2]
    n_vertical = torch.tensor([sun_h[0], sun_h[1], 0.0]) / torch.norm(sun_h)
    n_horizontal = torch.tensor([0.0, 0.0, 1.0])
    up = torch.tensor([0.0, 0.0, 1.0])
    z_w = n_vertical
    x_w = torch.linalg.cross(up, z_w)
    x_w = x_w / x_w.norm()
    r_vertical = torch.stack([x_w, up, z_w], dim=1)
    tilt_axis = torch.linalg.cross(n_vertical, n_horizontal)
    tilt_axis = tilt_axis / tilt_axis.norm()

    # Blocking primitives passed to the ray tracer: ONLY the identified blocker
    # rows, not the full 1277-heliostat field. `identify_blockers`'s geometric
    # cone test already proved no other heliostat can intersect the ray
    # bundle, so testing against the whole field is wasted primitives -- and
    # at rays_per_surface_point > 1 it's not just wasted, it's the difference
    # between a few-MB intermediate tensor and one sized
    # [rays x n_points x n_field_primitives] that OOMs the machine (this is
    # exactly what happened at 10 rays x 40 000 points x ~1276 primitives).
    # Self-exclusion is moot here since BH58's own row is never included.

    records = []
    fluxes = []
    per_point_all = []
    for step in range(args.pose_steps):
        t0 = time.time()
        tilt = step / (args.pose_steps - 1) if args.pose_steps > 1 else 1.0
        angle = tilt * (np.pi / 2.0)
        rotation = rotation_about_axis(tilt_axis, angle) @ r_vertical
        surfaces = torch.stack(
            [manually_rotated_surfaces(hg, row, rotation) for row in blocker_rows], dim=0
        )
        flux, blocked_frac, per_point = trace_capture(
            scenario, hg, hel_idx, sun, target_index, aim_center, surfaces, device
        )
        total_flux = float(flux.sum())
        fluxes.append(flux)
        per_point_all.append(per_point)
        records.append({
            "frame": step, "tilt_fraction": tilt, "blocked_fraction": blocked_frac,
            "total_flux": total_flux,
            "mirror_frac_points_blocked_gt50pct": float((per_point > 0.5).float().mean()),
        })
        log.info(
            f"step {step + 1}/{args.pose_steps} tilt={tilt:.2f} blocked={blocked_frac:.4f} "
            f"flux={total_flux:.0f} mirror_frac>50%={records[-1]['mirror_frac_points_blocked_gt50pct']:.3f} "
            f"({time.time() - t0:.1f}s)"
        )

    shared_vmax = max(float(f.max()) for f in fluxes)
    for step, (flux, per_point, rec) in enumerate(zip(fluxes, per_point_all, records)):
        tilt = rec["tilt_fraction"]
        render_frame(
            flux, frames_flux / f"frame_{step:03d}.png",
            [
                f"{args.heliostat_id} target flux | step {step + 1}/{args.pose_steps}",
                f"tilt {tilt * 100:.0f} %   blocked {rec['blocked_fraction'] * 100:.2f} %   "
                f"flux {rec['total_flux']:.0f}",
            ],
            vmax=shared_vmax,
        )
        render_mirror_frame(
            local_xy, per_point, frames_mirror / f"frame_{step:03d}.png",
            [
                f"{args.heliostat_id} mirror surface | step {step + 1}/{args.pose_steps}",
                f"tilt {tilt * 100:.0f} %   blocked {rec['blocked_fraction'] * 100:.2f} %",
            ],
        )

    n_flux = assemble_gif(frames_flux, GIF_DIR / "pose_sweep.gif", 0.01)
    n_mirror = assemble_gif(frames_mirror, GIF_DIR / "mirror_shadow.gif", FRAME_DURATION_S)
    make_grid(frames_flux, PLOT_DIR / "pose_sweep_grid.png",
              f"{args.heliostat_id} point-source pose sweep -- target flux ({args.surface_points}x{args.surface_points} pts/facet, {args.rays} rays/pt)")
    make_grid(frames_mirror, PLOT_DIR / "mirror_shadow_grid.png",
              f"{args.heliostat_id} point-source pose sweep -- mirror shadow map ({args.surface_points}x{args.surface_points} pts/facet, {args.rays} rays/pt)")

    # ---- 2-row comparison figure at 5 representative tilts: mirror (top) vs target flux (bottom)
    picks = sorted(set(np.linspace(0, args.pose_steps - 1, 5).astype(int).tolist()))
    fig, axes = plt.subplots(2, len(picks), figsize=(4.2 * len(picks), 8.6))
    for col, step in enumerate(picks):
        img_m = plt.imread(frames_mirror / f"frame_{step:03d}.png")
        img_f = plt.imread(frames_flux / f"frame_{step:03d}.png")
        axes[0, col].imshow(img_m); axes[0, col].axis("off")
        axes[1, col].imshow(img_f); axes[1, col].axis("off")
    fig.suptitle(
        f"{args.heliostat_id}: where the blocking actually happens (mirror) vs. what it looks like downstream (target),\n"
        f"point-source sun, same blockers/tilts, {args.surface_points}x{args.surface_points} pts/facet x {args.rays} rays/pt",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.015, 0, 1, 0.93))
    row0_y = np.mean([axes[0, 0].get_position().y0, axes[0, 0].get_position().y1])
    row1_y = np.mean([axes[1, 0].get_position().y0, axes[1, 0].get_position().y1])
    fig.text(0.005, row0_y, "MIRROR SURFACE\n(shadow map, source)", fontsize=13, fontweight="bold",
              rotation=90, va="center", ha="center")
    fig.text(0.005, row1_y, "TARGET FLUX\n(image, far field)", fontsize=13, fontweight="bold",
              rotation=90, va="center", ha="center")
    fig.savefig(PLOT_DIR / "mirror_vs_target_comparison.png", dpi=130)
    plt.close(fig)

    summary_path = DATA_DIR / "mirror_shadow_run_summary.json"
    summary_path.write_text(json.dumps({
        "config": {
            "heliostat_id": args.heliostat_id, "sun_sample": args.sun_sample,
            "surface_points_per_facet": args.surface_points, "rays_per_surface_point": args.rays,
            "pose_steps": args.pose_steps, "point_source_std_mrad": args.point_source_std_mrad,
            "blocker_names": blocker_names, "n_mirror_points": int(local_xy.shape[0]),
        },
        "records": records,
    }, indent=1))
    log.info(f"wrote {summary_path}")
    log.info(f"gifs: {n_flux} flux frames, {n_mirror} mirror frames -> {GIF_DIR}")
    log.info(f"plots -> {PLOT_DIR}")


if __name__ == "__main__":
    main()
