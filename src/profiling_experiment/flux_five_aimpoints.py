"""Presentation visual: 5 heliostats each aimed at a different point of the receiver.

Aims 5 heliostats at the four corners + centre of a tower target, ray-traces them, and
saves the combined flux image — a quincunx of five flux spots on the receiver. Pure
forward ray tracing on an existing scenario; no data needed (the sun direction is
fabricated — alignment makes each beam land on its assigned aim point regardless).

    python profiling_experiment/flux_five_aimpoints.py
    python profiling_experiment/flux_five_aimpoints.py --surface-points 50 --rays-scale  # prettier
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import h5py
import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import paths  # noqa: E402

from artist.raytracing import HeliostatRayTracer  # noqa: E402
from artist.scenario import Scenario  # noqa: E402
from artist.util import indices  # noqa: E402
from artist.util.env import get_device  # noqa: E402


# Target planes are vertical and span East (x) × Up (z); dims = [width_e, height_u].
_E = torch.tensor([1.0, 0.0, 0.0, 0.0])  # East
_U = torch.tensor([0.0, 0.0, 1.0, 0.0])  # Up


def _corner_and_centre_aimpoints(center, dims, inset, device) -> torch.Tensor:
    """5 aim points: the 4 corners + centre, inset from the edges."""
    w, h = float(dims[0]) * 0.5 * inset, float(dims[1]) * 0.5 * inset
    e, u = _E.to(device), _U.to(device)
    offsets = [(-w, +h), (+w, +h), (-w, -h), (+w, -h), (0.0, 0.0)]
    return torch.stack([center + de * e + du * u for de, du in offsets])


def _random_aimpoints(center, dims, n, inset, seed, device) -> torch.Tensor:
    """``n`` aim points scattered uniformly at random over the receiver (inset)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    de = (torch.rand(n, generator=g) - 0.5) * float(dims[0]) * inset
    du = (torch.rand(n, generator=g) - 0.5) * float(dims[1]) * inset
    e, u = _E.to(device), _U.to(device)
    return center.unsqueeze(0) + de.to(device)[:, None] * e + du.to(device)[:, None] * u


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="5-heliostat corners+centre flux image.")
    p.add_argument("--scenario", type=pathlib.Path, default=None)
    p.add_argument("--surface-points", type=int, default=50,
                   help="Surface points per facet side (50 = smoother flux).")
    p.add_argument("--target", type=int, default=1, help="Planar target index to aim at.")
    p.add_argument("--heliostats", type=int, default=63)
    p.add_argument("--aim", choices=["random", "corners"], default="random",
                   help="random: aim points scattered over the receiver; corners: 4 corners + centre.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inset", type=float, default=0.85, help="Fraction of receiver used (0-1).")
    p.add_argument("--out", type=pathlib.Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    scenario_path = args.scenario or (
        paths.REPO / "scenarios" / "full_63_heli_kin_reconstruct" / "scenario.h5"
    )
    out = args.out or (paths.output_dir() / "flux_five_aimpoints.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    device = get_device()

    with h5py.File(scenario_path, "r") as fh:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=fh, device=device,
            number_of_surface_points_per_facet=torch.tensor(
                [args.surface_points, args.surface_points]),
        )
    hg = scenario.heliostat_field.heliostat_groups[0]
    planar = scenario.solar_tower.target_areas[indices.planar_target_areas]
    t = args.target
    center, dims = planar.centers[t], planar.dimensions[t]

    n = min(args.heliostats, hg.number_of_heliostats)
    if args.aim == "corners":
        aim_points = _corner_and_centre_aimpoints(center, dims, args.inset, device)[:n]
    else:
        aim_points = _random_aimpoints(center, dims, n, args.inset, args.seed, device)

    # Spread the chosen heliostats across the field for variety in the spots.
    chosen = torch.linspace(0, hg.number_of_heliostats - 1, n).round().long()
    mask = torch.zeros(hg.number_of_heliostats, dtype=torch.int64, device=device)
    mask[chosen] = 1
    target_indices = torch.full((n,), t, dtype=torch.int64, device=device)
    # Fabricated sun direction — must point DOWNWARD (negative Up) to be physical;
    # alignment then makes each beam hit its assigned aim point.
    rays = torch.tensor([0.1, 0.1, -1.0, 0.0], device=device)
    rays = (rays / rays[:3].norm()).repeat(n, 1)

    hg.activate_heliostats(active_heliostats_mask=mask, device=device)
    hg.align_surfaces_with_incident_ray_directions(
        aim_points=aim_points, incident_ray_directions=rays,
        active_heliostats_mask=mask, device=device,
    )
    ray_tracer = HeliostatRayTracer(
        scenario=scenario, heliostat_group=hg, blocking_active=False, batch_size=n)
    with torch.no_grad():
        flux, *_ = ray_tracer.trace_rays(
            incident_ray_directions=rays, active_heliostats_mask=mask,
            target_area_indices=target_indices, device=device)

    combined = flux.sum(dim=0).cpu().numpy()  # [H, W] — all 5 spots on one receiver

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w_m, h_m = float(dims[0]), float(dims[1])
    fig, ax = plt.subplots(figsize=(7, 7 * h_m / w_m))
    ax.imshow(combined, cmap="inferno", origin="lower",
              extent=[-w_m / 2, w_m / 2, -h_m / 2, h_m / 2], aspect="equal")
    _desc = "corners + centre" if args.aim == "corners" else "random points"
    ax.set_title(f"{n} heliostats aimed at {_desc} on the receiver", fontsize=12)
    ax.set_xlabel("East (m)"); ax.set_ylabel("Up (m)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"flux total energy={combined.sum():.3g}  saved -> {out}")


if __name__ == "__main__":
    main()
