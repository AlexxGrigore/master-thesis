"""Centroid error (mrad) vs surface resolution — blur vs no blur.

The focal-spot loss compares CENTROIDS, not pixels. This is the centroid analog of the
pixel-MSE blur sweep: for a single heliostat's flux at 10/25/50/75 points/facet, it measures
how far the centroid sits from the sharp 100×100 centroid (in mrad, the same unit the
focal-spot loss reports), unblurred and at the best blur σ*.

Because a symmetric Gaussian blur preserves a center-of-mass, blur is expected to barely move
the centroid — so the two lines should nearly overlap, unlike the pixel-MSE case.

Reuses render_unit_flux / blur_unit from blur_resolution_sweep.py. Local / CPU, seconds.
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
from blur_resolution_sweep import blur_unit, render_unit_flux  # noqa: E402

from artist.scenario import Scenario  # noqa: E402
from artist.util import indices  # noqa: E402
from artist.util.env import get_device  # noqa: E402


def get_geometry(scenario_path, target, device):
    """Return ((width_m, height_m) of the target, heliostat→target distance m)."""
    with h5py.File(scenario_path, "r") as fh:
        sc = Scenario.load_scenario_from_hdf5(
            scenario_file=fh, device=device,
            number_of_surface_points_per_facet=torch.tensor([25, 25]))
    hg = sc.heliostat_field.heliostat_groups[0]
    planar = sc.solar_tower.target_areas[indices.planar_target_areas]
    center = planar.centers[target][:3].float()
    dims = planar.dimensions[target]
    hel_pos = hg.positions[0, :3].float()
    dist = float(torch.norm(hel_pos - center.to(hel_pos.device)))
    return (float(dims[0]), float(dims[1])), dist


def centroid_metres(flux: np.ndarray, w: float, h: float) -> np.ndarray:
    """Flux center-of-mass as an (east, up) position in metres on the receiver."""
    H, W = flux.shape
    f = flux.astype(np.float64)
    s = f.sum()
    com_col = (f.sum(axis=0) @ np.arange(W)) / s
    com_row = (f.sum(axis=1) @ np.arange(H)) / s
    e = (com_col + 0.5) / W * w - w / 2.0
    u = (com_row + 0.5) / H * h - h / 2.0
    return np.array([e, u])


def _parse_args():
    p = argparse.ArgumentParser(description="Centroid error (mrad) vs resolution, blur sweep.")
    p.add_argument("--resolutions", type=int, nargs="+", default=[10, 25, 50, 75])
    p.add_argument("--reference", type=int, default=100)
    p.add_argument("--sigma-max", type=float, default=10.0)
    p.add_argument("--sigma-step", type=float, default=0.25)
    p.add_argument("--scenario", type=pathlib.Path, default=None)
    p.add_argument("--target", type=int, default=1)
    return p.parse_args()


def main():
    args = _parse_args()
    scenario_path = args.scenario or (
        paths.REPO / "scenarios" / "full_63_heli_kin_reconstruct" / "scenario.h5")
    outdir = paths.output_dir(); outdir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    (w, h), dist = get_geometry(scenario_path, args.target, device)
    print(f"target {w:.2f}×{h:.2f} m  |  heliostat distance {dist:.1f} m")

    all_res = sorted(set(args.resolutions + [args.reference]))
    flux = {r: render_unit_flux(scenario_path, r, args.target, device) for r in all_res}
    ref_c = centroid_metres(flux[args.reference], w, h)

    def err_mrad(img):
        return float(np.linalg.norm(centroid_metres(img, w, h) - ref_c) / dist * 1000.0)

    sigmas = np.arange(0.0, args.sigma_max + 1e-9, args.sigma_step)
    res = {}
    for r in args.resolutions:
        errs = np.array([err_mrad(blur_unit(flux[r], s)) for s in sigmas])
        i = int(np.argmin(errs))
        res[r] = dict(unblurred=float(errs[0]), sigma_star=float(sigmas[i]),
                      best=float(errs[i]))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = args.resolutions
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(rs, [res[r]["unblurred"] for r in rs], "s--", color="#8d99ae", lw=2,
            label="unblurred (σ=0)")
    ax.plot(rs, [res[r]["best"] for r in rs], "o-", color="#E76F51", lw=2,
            label="best blur (σ*)")
    for r in rs:
        ax.annotate(f"σ*={res[r]['sigma_star']:.2f}", (r, res[r]["best"]),
                    textcoords="offset points", xytext=(0, -14), ha="center",
                    fontsize=8, color="#c0392b")
    ax.set_xlabel("surface points/facet (N×N)")
    ax.set_ylabel("centroid error to sharp 100×100 (mrad)")
    ax.set_title("Centroid error: blur vs no blur\n(focal-spot loss ground truth)")
    ax.set_xticks(rs); ax.set_xticklabels([f"{r}×{r}" for r in rs])
    ax.legend(); ax.grid(alpha=0.3)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = outdir / "centroid_error_mrad.png"
    fig.savefig(out, dpi=150)

    lines = ["# Centroid error (mrad) vs surface resolution\n",
             f"Single heliostat, distance {dist:.0f} m, vs the sharp 100×100 centroid.\n",
             "| Resolution | unblurred (mrad) | σ* (px) | best blur (mrad) |",
             "|---|---|---|---|"]
    print("\ncentroid error (mrad):")
    for r in rs:
        s = res[r]
        lines.append(f"| {r}×{r} | {s['unblurred']:.3f} | {s['sigma_star']:.2f} | {s['best']:.3f} |")
        print(f"  {r}×{r}: unblurred={s['unblurred']:.3f}  σ*={s['sigma_star']:.2f}  best={s['best']:.3f} mrad")
    (outdir / "centroid_error_mrad.md").write_text("\n".join(lines) + "\n")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
