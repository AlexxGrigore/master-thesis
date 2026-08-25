"""Blocked-vs-unblocked pose-sweep summary grid for one heliostat.

Columns: --n-steps blocker tilts, evenly spaced from VERTICAL (max blocking)
to HORIZONTAL/stow (no blocking) -- same convention as the pose sweeps
elsewhere in this study, just fewer, labelled steps instead of a full
animation.

Rows: target flux under the PRIMARY sun model (top -- real ARTIST sunshape by
default, or point-source if --point-source-sun is passed); optionally target
flux under the OTHER sun model too (--extra-flux-row, default "point_source"
when the primary is real, so you get the blurred oval next to the sharp
parallelogram it's the blur of); and the studied heliostat's own mirror
surface shaded by blocked fraction (bottom, where the blocking happens --
always rendered under the PRIMARY sun model). Each flux row shares one
intensity scale across its own columns so the loss is visible rather than
auto-scaled away; the mirror row shares the fixed 0..1 blocked-fraction
scale.

Usage
-----
    python blocked_vs_unblocked_grid.py --heliostat-id BA72 --sun-sample 0011
    python blocked_vs_unblocked_grid.py --heliostat-id BA72 --n-steps 2 --extra-flux-row none
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
_sh = _src / "one_heliostat_demo" / "single_heliostat"
for _p in (str(_src), str(_sh), str(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from artist.scene.sun import Sun  # noqa: E402
from artist.util import set_logger_config  # noqa: E402

from mirror_shadow_map import trace_capture  # noqa: E402
from validate_blocking_flux import (  # noqa: E402
    configure,
    identify_blockers,
    load_context,
    load_train_sun_positions,
    manually_rotated_surfaces,
    render_frame,
    rotation_about_axis,
)

log = logging.getLogger(__name__)

REAL_SUN_COVARIANCE = 4.3681e-06  # ARTIST default, ~2.1 mrad std (artist/scene/sun.py)


def make_sun(number_of_rays: int, covariance: float, device: torch.device) -> Sun:
    return Sun(
        number_of_rays=number_of_rays,
        distribution_parameters={"distribution_type": "normal", "mean": 0.0, "covariance": covariance},
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heliostat-id", default="BA72")
    parser.add_argument("--sun-sample", default="0011")
    parser.add_argument("--surface-points", type=int, default=100)
    parser.add_argument("--rays", type=int, default=10)
    parser.add_argument("--point-source-sun", action="store_true",
                        help="idealized geometric-optics limit (see validate_blocking_flux.py). "
                             "Default OFF: use the scenario's normal/default ARTIST sun "
                             "(real sunshape, ~2.1 mrad std).")
    parser.add_argument("--point-source-std-mrad", type=float, default=0.001)
    parser.add_argument("--extra-flux-row", choices=["point_source", "real_sun", "none"], default=None,
                        help="add a second flux row under the OTHER sun model, for direct comparison. "
                             "Default: 'point_source' if the primary is real_sun, else 'none' "
                             "(avoids defaulting to a redundant real_sun row when primary is already "
                             "point_source).")
    parser.add_argument("--n-steps", type=int, default=5,
                        help="number of tilt columns, evenly spaced 0 (vertical/max blocking) to "
                             "1 (horizontal/stow/no blocking). Default 5: blocked, 3 intermediate, "
                             "unblocked.")
    args = parser.parse_args()
    if args.extra_flux_row is None:
        args.extra_flux_row = "none" if args.point_source_sun else "point_source"

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("artist", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    out_name = f"{args.heliostat_id}_point_source" if args.point_source_sun else args.heliostat_id
    configure(args.heliostat_id, out_name=out_name)
    from validate_blocking_flux import PLOT_DIR  # noqa: E402
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    point_source_covariance = (
        (args.point_source_std_mrad * 1e-3) ** 2 if args.point_source_sun else None
    )
    scenario, hg, hel_idx, target_index, aim_center, hel_dist_m = load_context(
        device, surface_points_per_facet=args.surface_points, rays_per_surface_point=args.rays,
        point_source_covariance=point_source_covariance,
    )
    local_xy = hg.surface_points[hel_idx, :, :2].detach().cpu().numpy()

    samples = load_train_sun_positions()
    sun_record = next(s for s in samples if s["sample_id"] == args.sun_sample)
    sun = torch.tensor(sun_record["incident_ray_direction"] + [0.0], dtype=torch.float, device=device)
    d3 = sun[:3].detach().cpu()

    blocker_rows = identify_blockers(scenario, hg, hel_idx, sun, target_index, aim_center, device)
    blocker_names = [str(hg.names[i]) for i in blocker_rows]
    print(f"blockers: {blocker_names}")

    sun_h = -d3[:2]
    n_vertical = torch.tensor([sun_h[0], sun_h[1], 0.0]) / torch.norm(sun_h)
    n_horizontal = torch.tensor([0.0, 0.0, 1.0])
    up = torch.tensor([0.0, 0.0, 1.0])
    x_w = torch.linalg.cross(up, n_vertical)
    x_w = x_w / x_w.norm()
    r_vertical = torch.stack([x_w, up, n_vertical], dim=1)
    tilt_axis = torch.linalg.cross(n_vertical, n_horizontal)
    tilt_axis = tilt_axis / tilt_axis.norm()
    tilts = np.linspace(0.0, 1.0, args.n_steps)
    surfaces_by_tilt = {}
    for tilt in tilts:
        rotation = rotation_about_axis(tilt_axis, float(tilt) * np.pi / 2.0) @ r_vertical
        surfaces_by_tilt[tilt] = torch.stack(
            [manually_rotated_surfaces(hg, row, rotation) for row in blocker_rows], dim=0
        )

    def trace_all_tilts(sun_desc: str) -> dict:
        out = {}
        for tilt, surfaces in surfaces_by_tilt.items():
            flux, frac, per_point = trace_capture(
                scenario, hg, hel_idx, sun, target_index, aim_center, surfaces, device
            )
            out[tilt] = {"flux": flux, "frac": frac, "per_point": per_point}
            print(f"[{sun_desc}] tilt={tilt * 100:.0f} %: blocked_fraction={frac * 100:.2f} %  "
                  f"total_flux={float(flux.sum()):.0f}")
        return out

    def sun_tag(is_point_source: bool) -> str:
        return "point source" if is_point_source else "real sunshape"

    primary_desc = f"point-source sun ({args.point_source_std_mrad} mrad std)" if args.point_source_sun \
        else "normal ARTIST sun (real sunshape, ~2.1 mrad std)"
    primary_tag = sun_tag(args.point_source_sun)
    results = trace_all_tilts(primary_desc)

    extra_results = None
    extra_tag = None
    if args.extra_flux_row != "none":
        extra_covariance = (
            (args.point_source_std_mrad * 1e-3) ** 2 if args.extra_flux_row == "point_source"
            else REAL_SUN_COVARIANCE
        )
        extra_desc = f"point-source sun ({args.point_source_std_mrad} mrad std)" \
            if args.extra_flux_row == "point_source" else "normal ARTIST sun (real sunshape, ~2.1 mrad std)"
        extra_tag = sun_tag(args.extra_flux_row == "point_source")
        scenario.light_sources.light_source_list[0] = make_sun(args.rays, extra_covariance, device)
        extra_results = trace_all_tilts(extra_desc)

    n_rows = 3 if extra_results is not None else 2
    fig, axes = plt.subplots(n_rows, args.n_steps, figsize=(4.2 * args.n_steps, 4.6 * n_rows), squeeze=False)

    def tilt_label(tilt: float) -> str:
        if tilt == 0.0:
            return "blocked (vertical)"
        if tilt == 1.0:
            return "unblocked (horizontal)"
        return f"tilt {tilt * 100:.0f} %"

    def plot_flux_row(row: int, res: dict) -> None:
        shared_vmax = max(float(res[t]["flux"].max()) for t in tilts)
        for col, tilt in enumerate(tilts):
            r = res[tilt]
            im = axes[row, col].imshow(
                r["flux"].numpy(), cmap="inferno", origin="upper", vmin=0.0, vmax=shared_vmax
            )
            axes[row, col].set_title(
                f"{tilt_label(tilt)}\nblocked {r['frac'] * 100:.2f} %   flux {float(r['flux'].sum()):.0f}",
                fontsize=9.5,
            )
            axes[row, col].set_xlabel("bitmap e [px]"); axes[row, col].set_ylabel("bitmap u [px]")
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04, label="flux [a.u.]")

    plot_flux_row(0, results)
    row_descs = [f"target flux ({primary_tag})"]
    mirror_row = 1
    if extra_results is not None:
        plot_flux_row(1, extra_results)
        row_descs.append(f"target flux ({extra_tag})")
        mirror_row = 2
    row_descs.append(f"mirror surface ({primary_tag})")

    for col, tilt in enumerate(tilts):
        frac_arr = results[tilt]["per_point"].numpy()
        sc = axes[mirror_row, col].scatter(
            local_xy[:, 0], local_xy[:, 1], c=frac_arr, cmap="RdYlGn_r", vmin=0.0, vmax=1.0,
            s=3, marker="s", linewidths=0,
        )
        axes[mirror_row, col].set_title(tilt_label(tilt), fontsize=9.5)
        axes[mirror_row, col].set_xlabel("mirror width [m]"); axes[mirror_row, col].set_ylabel("mirror height [m]")
        axes[mirror_row, col].set_aspect("equal")
        fig.colorbar(sc, ax=axes[mirror_row, col], fraction=0.046, pad=0.04, label="blocked fraction")

    fig.suptitle(
        f"{args.heliostat_id}, sun sample {args.sun_sample}: blockers {', '.join(blocker_names)}\n"
        f"{args.n_steps} steps, vertical/max-blocking -> horizontal/stow -- "
        f"{args.surface_points}x{args.surface_points} pts/facet x {args.rays} rays/pt",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.045, 0, 1, 1 - 0.5 / n_rows))
    for row, desc in enumerate(row_descs):
        y = np.mean([axes[row, 0].get_position().y0, axes[row, 0].get_position().y1])
        fig.text(0.005, y, desc, fontsize=11, fontweight="bold", rotation=90, va="center", ha="center")
    out = PLOT_DIR / "blocked_vs_unblocked_grid.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
