"""BE25 version of `ba72_mirror_inversion_check.py`, for direct comparison.

Same layout: colour every BE25 surface point by local mirror height (blue =
bottom, red = top), trace all of them UNBLOCKED, and plot where each point's
ray lands on the target bitmap -- split into bottom-half-only, top-half-only,
and merged panels so the two populations aren't overplotted on each other.

Uses BE25's real PAINT calibration sample (same sample as
`be25_blocking_direction_check.py --sample 9`, the near-field-max exposure
case) and its real "lower" neighbour geometry, via `gate.py`'s own loaders.

Usage
-----
    python be25_mirror_inversion_check.py
    python be25_mirror_inversion_check.py --sample 0
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

from brute_blocking import capture_target_intersections  # noqa: E402
from gate import BENCHMARK, one_hot_mask  # noqa: E402

log = logging.getLogger(__name__)

_ROOT = _here.parents[2]
HELIOSTAT_ID = "BE25"
OUT_DIR = _ROOT / "outputs" / "new_mapping_function" / "blocking_study" / HELIOSTAT_ID / "direction_check_plots"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=9)
    parser.add_argument("--surface-points", type=int, default=60)
    parser.add_argument("--rays", type=int, default=3)
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

    mask = one_hot_mask(hel_idx, 1, hg.number_of_heliostats, device)
    hg.activate_heliostats(active_heliostats_mask=mask, device=device)
    hg.align_surfaces_with_incident_ray_directions(
        aim_points=centroids[s].unsqueeze(0), incident_ray_directions=sun.unsqueeze(0),
        active_heliostats_mask=mask, device=device,
    )
    ray_tracer = HeliostatRayTracer(
        scenario=scenario, heliostat_group=hg, blocking_active=False,
        world_size=1, rank=0, batch_size=1, random_seed=7,
    )
    with capture_target_intersections() as captured:
        ray_tracer.trace_rays(
            incident_ray_directions=sun.unsqueeze(0), active_heliostats_mask=mask,
            target_area_indices=tgt.unsqueeze(0), device=device,
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
    ax.set_title(f"{HELIOSTAT_ID} mirror surface\n(colour = local height; blue=bottom, red=top)")
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
        f"{HELIOSTAT_ID} sample {s}: mirror-height vs. target-u correlation = {corr:.3f} "
        f"({'INVERTED image (mirror bottom -> target top)' if corr < 0 else 'upright image (mirror bottom -> target bottom)'})"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / f"be25_mirror_inversion_check_sample{s}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"correlation(mirror_height, bitmap_u) = {corr:.4f}")


if __name__ == "__main__":
    main()
