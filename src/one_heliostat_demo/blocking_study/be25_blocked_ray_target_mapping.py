"""BE25 version of `ba72_blocked_ray_target_mapping.py`, for direct comparison.

Same layout: left panel shows which BE25 mirror points its real neighbours
(BB26/BC26/BD25/BD26/BE26, aimed at the target -- the "lower" hypothesis from
`gate.py`, already the documented worst-case real exposure) actually shadow;
right panel shows where those same points' rays would land on the target
(traced unblocked), red = the flux blocking removes, grey = unaffected.

Uses BE25's real PAINT calibration sample (default: sample 9, the
near-field-max exposure case, same as the other BE25 diagnostics in this
study) and its existing neighbourhood scenario/blockers -- no new scenario
needed, `scenarios/neighbourhoods/BE25/` already contains the real occluders.

Usage
-----
    python be25_blocked_ray_target_mapping.py
    python be25_blocked_ray_target_mapping.py --sample 0
"""

from __future__ import annotations

import argparse
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
from artist.util import set_logger_config  # noqa: E402

from brute_blocking import capture_blocking_mask, capture_target_intersections, exact_blocking  # noqa: E402
from gate import BENCHMARK, blocker_surfaces, one_hot_mask  # noqa: E402

log = logging.getLogger(__name__)

_ROOT = _here.parents[2]
HELIOSTAT_ID = "BE25"
HYPOTHESIS = "lower"
OUT_DIR = _ROOT / "outputs" / "new_mapping_function" / "blocking_study" / HELIOSTAT_ID / "direction_check_plots"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=9)
    parser.add_argument("--surface-points", type=int, default=100)
    parser.add_argument("--rays", type=int, default=10)
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("artist", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    torch.manual_seed(7)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    from artist.io.paint_calibration_parser import PaintCalibrationDataParser
    from utils.evaluation import build_heliostat_data_mapping

    paint_dir = _ROOT / "datasets" / "paint"
    scenario_path = _ROOT / "scenarios" / "neighbourhoods" / HELIOSTAT_ID / "scenario_ideal.h5"
    with h5py.File(scenario_path) as fh:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=fh, device=device,
            number_of_surface_points_per_facet=torch.tensor([args.surface_points, args.surface_points]),
        )
    scenario.set_number_of_rays(args.rays)
    hg = scenario.heliostat_field.heliostat_groups[0]
    hel_idx = hg.names.index(HELIOSTAT_ID)
    n_hel = hg.number_of_heliostats
    log.info(f"{HELIOSTAT_ID}: row {hel_idx}, group {hg.names}")

    mapping = build_heliostat_data_mapping(
        paint_dir / "splits" / f"{BENCHMARK}.csv",
        paint_dir / BENCHMARK / "calibration_properties",
        paint_dir / BENCHMARK / "flux_image",
        "test",
    )
    mapping = [entry for entry in mapping if entry[0] == HELIOSTAT_ID]
    _, centroids, rays, motor_positions, _, target_mask = PaintCalibrationDataParser(
        centroid_extraction_method="UTIS"
    ).parse_data_for_reconstruction(
        heliostat_data_mapping=mapping, heliostat_group=hg, scenario=scenario, device=device,
    )
    s = args.sample
    sun = rays[s]
    tgt = target_mask[s]
    print(f"sample {s}/{rays.shape[0]}, target index {int(tgt.item())}")

    local_xy = hg.surface_points[hel_idx, :, :2].detach().cpu().numpy()

    surf_lower = blocker_surfaces(hg, HYPOTHESIS, scenario, sun, device)
    mask = one_hot_mask(hel_idx, 1, n_hel, device)
    hg.activate_heliostats(active_heliostats_mask=mask, device=device)
    hg.align_surfaces_with_incident_ray_directions(
        aim_points=centroids[s].unsqueeze(0), incident_ray_directions=sun.unsqueeze(0),
        active_heliostats_mask=mask, device=device,
    )
    ray_tracer = HeliostatRayTracer(
        scenario=scenario, heliostat_group=hg, blocking_active=False,
        world_size=1, rank=0, batch_size=1, random_seed=7,
    )
    ray_tracer.blocking_active = True
    ray_tracer.blocking_heliostat_surfaces_active = surf_lower
    with exact_blocking(), capture_blocking_mask() as blk, capture_target_intersections() as tgt_int:
        flux, _, on_target, bf = ray_tracer.trace_rays(
            incident_ray_directions=sun.unsqueeze(0), active_heliostats_mask=mask,
            target_area_indices=tgt.unsqueeze(0), device=device,
        )
    print(f"on_target={on_target.item():.3f} blocked fraction (rays) = {1.0 - bf.item():.4f}")

    per_point_blocked = blk["blocked"][0].mean(dim=0).cpu().numpy()
    blocked_mask = per_point_blocked > 0.5
    print(f"mirror points blocked (>50% of their rays): {blocked_mask.mean() * 100:.1f} %")

    e = tgt_int["bitmap_e"][0].mean(dim=0).cpu().numpy()
    u = tgt_int["bitmap_u"][0].mean(dim=0).cpu().numpy()
    bitmap_res = ray_tracer.bitmap_resolution
    height_u = float(bitmap_res[1] if bitmap_res.numel() > 1 else bitmap_res)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4))

    ax1.scatter(local_xy[~blocked_mask, 0], local_xy[~blocked_mask, 1], c="0.75", s=4, linewidths=0,
                label="unblocked mirror points")
    ax1.scatter(local_xy[blocked_mask, 0], local_xy[blocked_mask, 1], c="crimson", s=4, linewidths=0,
                label="blocked mirror points")
    ax1.set_title(f"{HELIOSTAT_ID} mirror surface\n(which points do BB26/BC26/BD25/BD26/BE26 shadow?)")
    ax1.set_xlabel("mirror width [m]"); ax1.set_ylabel("mirror height [m]")
    ax1.set_aspect("equal")
    ax1.legend(loc="upper right", fontsize=8)

    ax2.scatter(e[~blocked_mask], u[~blocked_mask], c="0.75", s=4, linewidths=0,
                label="rays from unblocked points")
    ax2.scatter(e[blocked_mask], u[blocked_mask], c="crimson", s=4, linewidths=0,
                label="rays from blocked points\n(the flux REMOVED by blocking)")
    ax2.axhline(height_u / 2.0, c="k", lw=0.8, ls="--", alpha=0.6, label="target vertical centre")
    ax2.invert_yaxis()
    ax2.set_title("Where those same rays land on target\n(before blocking removes the red ones)")
    ax2.set_xlabel("bitmap e [px]"); ax2.set_ylabel("bitmap u [px] (down -->)")
    ax2.set_aspect("equal")
    ax2.legend(loc="upper right", fontsize=7)

    mean_u_blocked = u[blocked_mask].mean() if blocked_mask.any() else float("nan")
    mean_u_all = u.mean()
    fig.suptitle(
        f"{HELIOSTAT_ID} sample {s}: mean target-u of blocked rays = {mean_u_blocked:.1f} px vs. "
        f"all rays = {mean_u_all:.1f} px  (bitmap u grows DOWNWARD)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT_DIR / f"be25_blocked_ray_target_mapping_sample{s}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"mean target-u: blocked-rays={mean_u_blocked:.2f}px  all-rays={mean_u_all:.2f}px "
          f"(bitmap_height={height_u:.0f}px)")


if __name__ == "__main__":
    main()
