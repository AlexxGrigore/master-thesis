"""Composite figure: ray-traced flux at 25/50/75/100 surface points + the VRAM each costs.

Top: a bar chart of peak ray-tracer VRAM per surface resolution.
Bottom: the ray-traced flux image at that resolution, under its bar.

VRAM is measured on the GPU (peak allocated during a forward, no-grad trace). On a machine
without CUDA you can pass ``--vram-gb`` to supply known/extrapolated values just to preview
the layout; on DAIC, omit it and the script measures for real.

    # DAIC (real VRAM):
    apptainer exec --nv --bind /tudelft.net:/tudelft.net <sif> \
        python profiling_experiment/flux_vram_vs_resolution.py --daic
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

_E = torch.tensor([1.0, 0.0, 0.0, 0.0])
_U = torch.tensor([0.0, 0.0, 1.0, 0.0])


def _random_aimpoints(center, dims, n, inset, seed, device):
    if n == 1:  # single heliostat -> clean centred spot for the resolution demo
        return center.unsqueeze(0).to(device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    de = (torch.rand(n, generator=g) - 0.5) * float(dims[0]) * inset
    du = (torch.rand(n, generator=g) - 0.5) * float(dims[1]) * inset
    e, u = _E.to(device), _U.to(device)
    return center.unsqueeze(0) + de.to(device)[:, None] * e + du.to(device)[:, None] * u


def _per_heliostat_slope_gb(res: int) -> float:
    """Marginal forward ray-tracer VRAM per heliostat (GB) at a given surface resolution.

    Fit to the measured 25×25 and 50×50 heliostat-count sweeps (slope ≈ linear in the
    number of surface points = res²). Used to extrapolate to many heliostats — scaling a
    single heliostat's VRAM directly would be wrong, as that value is dominated by the
    fixed CUDA/scenario overhead which does NOT grow with the field.
    """
    return 6.31e-6 * (res * res) + 8.7e-4


def render_flux_and_vram(scenario_path, res, n_hel, inset, seed, target, device):
    """Render the summed flux image at the given surface resolution; return (flux, vram_gb)."""
    with h5py.File(scenario_path, "r") as fh:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=fh, device=device,
            number_of_surface_points_per_facet=torch.tensor([res, res]))
    hg = scenario.heliostat_field.heliostat_groups[0]
    planar = scenario.solar_tower.target_areas[indices.planar_target_areas]
    center, dims = planar.centers[target], planar.dimensions[target]
    n = min(n_hel, hg.number_of_heliostats)

    aim = _random_aimpoints(center, dims, n, inset, seed, device)
    chosen = torch.linspace(0, hg.number_of_heliostats - 1, n).round().long()
    mask = torch.zeros(hg.number_of_heliostats, dtype=torch.int64, device=device)
    mask[chosen] = 1
    tidx = torch.full((n,), target, dtype=torch.int64, device=device)
    rays = torch.tensor([0.1, 0.1, -1.0, 0.0], device=device)
    rays = (rays / rays[:3].norm()).repeat(n, 1)

    hg.activate_heliostats(active_heliostats_mask=mask, device=device)
    hg.align_surfaces_with_incident_ray_directions(
        aim_points=aim, incident_ray_directions=rays,
        active_heliostats_mask=mask, device=device)
    ray_tracer = HeliostatRayTracer(
        scenario=scenario, heliostat_group=hg, blocking_active=False, batch_size=n)

    vram = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    with torch.no_grad():
        flux, *_ = ray_tracer.trace_rays(
            incident_ray_directions=rays, active_heliostats_mask=mask,
            target_area_indices=tidx, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize(); vram = torch.cuda.max_memory_allocated() / 2**30

    combined = flux.sum(dim=0).detach().cpu().numpy()
    extent = [-float(dims[0]) / 2, float(dims[0]) / 2, -float(dims[1]) / 2, float(dims[1]) / 2]
    del scenario, hg, ray_tracer, flux
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return combined, extent, vram


def _parse_args():
    p = argparse.ArgumentParser(description="Flux images + ray-tracer VRAM vs surface resolution.")
    p.add_argument("--resolutions", type=int, nargs="+", default=[10, 25, 50, 75, 100])
    p.add_argument("--scenario", type=pathlib.Path, default=None)
    p.add_argument("--heliostats", type=int, default=1)
    p.add_argument("--target", type=int, default=1)
    p.add_argument("--inset", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--daic", action="store_true")
    p.add_argument("--vram-gb", type=float, nargs="+", default=None,
                   help="Override VRAM values (preview on a CPU machine). One per resolution.")
    p.add_argument("--mid-heliostats", type=int, default=63,
                   help="Heliostat count for the middle (orange) bar.")
    p.add_argument("--extrapolate", type=int, default=1000,
                   help="Heliostat count to extrapolate the last (red) bar to.")
    p.add_argument("--overhead-gb", type=float, default=0.056,
                   help="Fixed VRAM overhead (intercept) held constant when extrapolating.")
    p.add_argument("--gamma", type=float, default=0.5,
                   help="Display gamma for the flux images (<1 brightens dim flux).")
    p.add_argument("--out", type=pathlib.Path, default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    scenario_path = args.scenario or (
        paths.REPO / "scenarios" / "full_63_heli_kin_reconstruct" / "scenario.h5")
    out = args.out or (paths.output_dir() / "flux_vram_vs_resolution.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    device = get_device()

    fluxes, extents, vrams = [], [], []
    for i, res in enumerate(args.resolutions):
        flux, extent, vram = render_flux_and_vram(
            scenario_path, res, args.heliostats, args.inset, args.seed, args.target, device)
        if vram is None and args.vram_gb:
            vram = args.vram_gb[i]
        fluxes.append(flux); extents.append(extent); vrams.append(vram)
        print(f"res {res}x{res}: vram={vram}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from matplotlib.gridspec import GridSpec

    k = len(args.resolutions)
    fig = plt.figure(figsize=(3.4 * k, 6.6))
    gs = GridSpec(2, k, height_ratios=[1.05, 1.15], hspace=0.30, wspace=0.12)

    # --- top: grouped VRAM bars — measured (blue) and extrapolated to N (red) ---
    axb = fig.add_subplot(gs[0, :])
    x = np.arange(k)
    width = 0.27
    n_render, n_mid, n_extrap = args.heliostats, args.mid_heliostats, args.extrapolate
    blue = [v if v is not None else 0 for v in vrams]
    # 63- and 1000-heliostat bars: extrapolate via the measured per-heliostat slope
    # (NOT by scaling the single-heliostat value, which is overhead-dominated).
    orange = [_per_heliostat_slope_gb(r) * n_mid for r in args.resolutions]
    red = [_per_heliostat_slope_gb(r) * n_extrap for r in args.resolutions]

    hel_lbl = f"{n_render} heliostat" + ("s" if n_render != 1 else "")
    axb.bar(x - width, blue, width, color="#2c6fbb", label=f"{hel_lbl} (measured)")
    axb.bar(x, orange, width, color="#e08214", label=f"{n_mid} heliostats")
    axb.bar(x + width, red, width, color="#c0392b", label=f"{n_extrap} heliostats (extrapolated)")

    def _fmt(v):
        return (f"{v:.2f}" if v < 1 else f"{v:.1f}" if v < 10 else f"{v:.0f}")
    for xi, b, o, r in zip(x, blue, orange, red):
        axb.annotate(_fmt(b), (xi - width, b), textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=8, color="#2c6fbb", fontweight="bold")
        axb.annotate(_fmt(o), (xi, o), textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=8, color="#a85a00", fontweight="bold")
        axb.annotate(f"{_fmt(r)} GB", (xi + width, r), textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=8, color="#c0392b", fontweight="bold")
    axb.axhline(44, ls="--", color="gray", lw=1.3, label="single A40 = 44 GB")
    axb.set_yscale("log")
    axb.set_xticks(x); axb.set_xticklabels([f"{r}×{r}" for r in args.resolutions])
    axb.set_ylabel("Peak ray-tracer VRAM (GB, log)")
    axb.set_title("Ray-tracer VRAM vs surface resolution (forward pass)", pad=10)
    axb.set_xlim(-0.5, k - 0.5)
    # Generous top headroom so the top bar labels don't clip and the legend floats
    # above the short bars instead of covering them.
    top = max(max(red), 44) * 8
    axb.set_ylim(min(b for b in blue if b) * 0.4, top)
    axb.legend(fontsize=9, loc="upper left", ncol=2, framealpha=0.95,
               borderaxespad=0.4)
    axb.grid(axis="y", alpha=0.3, which="both")

    # --- bottom: flux image under each bar ---
    # Normalise each image to its own max: total intensity grows with the number of
    # surface points (more points -> more rays), so a shared scale would just make the
    # high-res images look brighter. Per-image scaling shows what actually differs:
    # discretisation noise (25x25 is grainy, 100x100 is smooth).
    for i, (flux, extent, res) in enumerate(zip(fluxes, extents, args.resolutions)):
        axi = fig.add_subplot(gs[1, i])
        # Robust per-image scaling: clip noise spikes at the 99.5th percentile and apply
        # gamma<1 so the dim flux (esp. at sparse 25x25) is visible, not crushed to black.
        vmax = float(np.percentile(flux, 99.5)) or float(flux.max()) or 1.0
        axi.imshow(flux, cmap="inferno", origin="lower", extent=extent, aspect="equal",
                   norm=PowerNorm(gamma=args.gamma, vmin=0.0, vmax=vmax))
        axi.set_title(f"{res}×{res} points/facet", fontsize=10)
        axi.set_xticks([]); axi.set_yticks([])

    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
