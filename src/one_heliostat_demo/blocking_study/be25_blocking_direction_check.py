"""Sanity check on a SECOND heliostat: does BE25 show the same "wrong-way" centroid shift as BA72?

`ba72_flux_diff_diagnostic.py` found that for BA72, blocking removes flux from
ABOVE the spot's centroid, so the centroid moves DOWN even though the
blockers shadow the mirror's physical bottom. Before trusting that as a real
optical effect (oblique/canted mirror -> target mapping is rotated, not a
flat-mirror flip), repeat the same three checks on BE25 -- the heliostat with
the best-documented real blocking exposure in this study (`GATE_RESULTS.md`:
10% median blocked, up to 25%, real PAINT data, "lower" hypothesis).

Reuses `gate.py`'s own real-data loading and blocker-pose logic directly (not
re-derived) so this is an apples-to-apples extension of the already-published
gate result, not a new toy scenario.

Usage
-----
    python be25_blocking_direction_check.py
    python be25_blocking_direction_check.py --sample 3
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

from artist.flux import get_center_of_mass  # noqa: E402
from artist.scenario.scenario import Scenario  # noqa: E402
from artist.util import set_logger_config  # noqa: E402

from brute_blocking import capture_blocking_mask, capture_target_intersections, exact_blocking  # noqa: E402
from gate import BENCHMARK, blocker_surfaces, one_hot_mask, trace_one_sample  # noqa: E402

log = logging.getLogger(__name__)

_ROOT = _here.parents[2]
HELIOSTAT_ID = "BE25"
HYPOTHESIS = "lower"
OUT_DIR = _ROOT / "outputs" / "new_mapping_function" / "blocking_study" / HELIOSTAT_ID
PLOT_DIR = OUT_DIR / "direction_check_plots"


def com_u(flux_2d: torch.Tensor) -> float:
    bc = get_center_of_mass(bitmaps=flux_2d.unsqueeze(0))
    return float(bc[0, 1].item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--surface-points", type=int, default=60)
    parser.add_argument("--rays", type=int, default=10)
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("artist", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    torch.manual_seed(7)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    from artist.io.paint_calibration_parser import PaintCalibrationDataParser
    from utils.evaluation import build_heliostat_data_mapping

    paint_dir = _ROOT / "datasets" / "paint"
    scenario_root = _ROOT / "scenarios" / "neighbourhoods"
    scenario_path = scenario_root / HELIOSTAT_ID / "scenario_ideal.h5"
    with h5py.File(scenario_path) as fh:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=fh, device=device,
            number_of_surface_points_per_facet=torch.tensor([args.surface_points, args.surface_points]),
        )
    scenario.set_number_of_rays(args.rays)
    hg = scenario.heliostat_field.heliostat_groups[0]
    hel_idx = hg.names.index(HELIOSTAT_ID)
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

    # ---- pose 1: stow (no block) ----
    surf_stow = blocker_surfaces(hg, "stow", scenario, sun, device)
    flux_stow, _, on_target_stow, bf_stow = trace_one_sample(
        scenario, hg, hel_idx, motor_positions[s], sun, tgt, surf_stow, device, centroid=centroids[s],
    )
    print(f"stow: on_target={on_target_stow.item():.3f} blocked={1.0 - bf_stow.item():.4f}")

    # ---- pose 2: lower (aimed, real exposure) ----
    surf_lower = blocker_surfaces(hg, HYPOTHESIS, scenario, sun, device)
    mask = one_hot_mask(hel_idx, 1, hg.number_of_heliostats, device)
    hg.activate_heliostats(active_heliostats_mask=mask, device=device)
    hg.align_surfaces_with_incident_ray_directions(
        aim_points=centroids[s].unsqueeze(0), incident_ray_directions=sun.unsqueeze(0),
        active_heliostats_mask=mask, device=device,
    )
    ray_tracer_kwargs = dict(scenario=scenario, heliostat_group=hg, blocking_active=False,
                              world_size=1, rank=0, batch_size=1, random_seed=7)
    from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
    ray_tracer = HeliostatRayTracer(**ray_tracer_kwargs)
    ray_tracer.blocking_active = True
    ray_tracer.blocking_heliostat_surfaces_active = surf_lower
    with exact_blocking(), capture_blocking_mask() as blk, capture_target_intersections() as tgt_int:
        flux_lower, _, on_target_lower, bf_lower = ray_tracer.trace_rays(
            incident_ray_directions=sun.unsqueeze(0), active_heliostats_mask=mask,
            target_area_indices=tgt.unsqueeze(0), device=device,
        )
    print(f"lower: on_target={on_target_lower.item():.3f} blocked={1.0 - bf_lower.item():.4f}")

    flux_stow2d = flux_stow[0].detach().cpu()
    flux_lower2d = flux_lower[0].detach().cpu()
    u_stow = com_u(flux_stow2d)
    u_lower = com_u(flux_lower2d)
    print(f"centroid bitmap-u: stow(no block)={u_stow:.3f}px  lower(blocked)={u_lower:.3f}px  "
          f"(u grows DOWNWARD; blocked centroid is "
          f"{'LOWER' if u_lower > u_stow else 'HIGHER'} on target than unblocked)")

    diff = flux_stow2d - flux_lower2d
    removed_total = float(diff.clamp(min=0).sum())
    diff_pos = diff.clamp(min=0)
    u_removed = com_u(diff_pos) if removed_total > 0 else float("nan")
    print(f"center-of-mass (bitmap-u) of REMOVED flux = {u_removed:.3f}px vs. unblocked centroid "
          f"u={u_stow:.3f}px => removed flux sits "
          f"{'BELOW' if u_removed > u_stow else 'ABOVE'} the unblocked centroid (u grows downward)")

    # mirror shadow region
    per_point_blocked = blk["blocked"][0].mean(dim=0).cpu().numpy()
    blocked_mask = per_point_blocked > 0.5
    print(f"mirror points blocked (>50% of rays): {blocked_mask.mean() * 100:.1f} %")

    e = tgt_int["bitmap_e"][0].mean(dim=0).cpu().numpy()
    u = tgt_int["bitmap_u"][0].mean(dim=0).cpu().numpy()
    mirror_height = local_xy[:, 1]
    corr = np.corrcoef(mirror_height, u)[0, 1]
    print(f"correlation(mirror_height, bitmap_u) [unblocked ray landings] = {corr:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5))
    ax = axes[0, 0]
    ax.scatter(local_xy[~blocked_mask, 0], local_xy[~blocked_mask, 1], c="0.75", s=4, linewidths=0)
    ax.scatter(local_xy[blocked_mask, 0], local_xy[blocked_mask, 1], c="crimson", s=4, linewidths=0)
    ax.set_title(f"{HELIOSTAT_ID} mirror: which points does '{HYPOTHESIS}' shadow?")
    ax.set_xlabel("mirror width [m]"); ax.set_ylabel("mirror height [m]"); ax.set_aspect("equal")

    ax = axes[0, 1]
    sc = ax.scatter(local_xy[:, 0], local_xy[:, 1], c=mirror_height, cmap="coolwarm", s=4, linewidths=0)
    ax.set_title("mirror surface, colour = local height")
    ax.set_xlabel("mirror width [m]"); ax.set_ylabel("mirror height [m]"); ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    sc = ax.scatter(e, u, c=mirror_height, cmap="coolwarm", s=4, linewidths=0)
    ax.invert_yaxis()
    ax.set_title(f"where unblocked rays land on target\ncorr(mirror_height, u) = {corr:.3f}")
    ax.set_xlabel("bitmap e [px]"); ax.set_ylabel("bitmap u [px] (down -->)"); ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    dmax = float(diff.abs().max()) or 1.0
    im = ax.imshow(diff.numpy(), cmap="RdBu_r", origin="upper", vmin=-dmax, vmax=dmax)
    ax.set_title(f"flux diff = stow - lower\n(red = REMOVED by blocking)")
    ax.set_xlabel("bitmap e [px]"); ax.set_ylabel("bitmap u [px] (down -->)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{HELIOSTAT_ID} sample {s}: centroid u stow={u_stow:.1f} lower={u_lower:.1f}px | "
        f"removed-flux u={u_removed:.1f}px"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = PLOT_DIR / f"be25_direction_check_sample{s}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
