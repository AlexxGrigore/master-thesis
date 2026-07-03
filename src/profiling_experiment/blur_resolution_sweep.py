"""Does blurring let a low surface resolution match the 100×100 flux? Per-σ sweep.

Renders a single heliostat's flux at 10/25/50/75/100 points/facet (identical geometry and
ray-tracer seed, so only the surface sampling changes), normalises each to unit total
energy, and treats the sharp 100×100 as ground truth. For each lower resolution it sweeps a
Gaussian blur σ, blurring ONLY the low-res image, renormalising, and measuring pixelwise MSE
to the sharp 100×100. The minimum of each curve is σ* — the best blur for that resolution.

Outputs:
  * blur_mse_vs_sigma.png      — MSE vs σ, one curve per resolution, σ* marked
  * blur_sigma_and_mse.png     — σ* per resolution, and best-MSE (σ*) vs unblurred MSE
  * blur_image_grid.png        — unblurred vs blurred-at-σ* images, with 100×100 reference
  * blur_resolution_sweep.md   — σ* table + takeaway

Fully local / CPU: only 5 renders; the σ sweep just re-blurs the stored images.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import h5py
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import paths  # noqa: E402

from artist.raytracing import HeliostatRayTracer  # noqa: E402
from artist.scenario import Scenario  # noqa: E402
from artist.util import indices  # noqa: E402
from artist.util.env import get_device  # noqa: E402


def render_unit_flux(scenario_path, res, target, device) -> np.ndarray:
    """Single heliostat aimed at the target centre; flux normalised to unit total energy."""
    with h5py.File(scenario_path, "r") as fh:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=fh, device=device,
            number_of_surface_points_per_facet=torch.tensor([res, res]))
    hg = scenario.heliostat_field.heliostat_groups[0]
    center = scenario.solar_tower.target_areas[indices.planar_target_areas].centers[target]

    mask = torch.zeros(hg.number_of_heliostats, dtype=torch.int64, device=device)
    mask[0] = 1
    aim = center.unsqueeze(0).to(device)
    tidx = torch.tensor([target], dtype=torch.int64, device=device)
    rays = torch.tensor([0.1, 0.1, -1.0, 0.0], device=device)
    rays = (rays / rays[:3].norm()).repeat(1, 1)

    hg.activate_heliostats(active_heliostats_mask=mask, device=device)
    hg.align_surfaces_with_incident_ray_directions(
        aim_points=aim, incident_ray_directions=rays, active_heliostats_mask=mask, device=device)
    rt = HeliostatRayTracer(scenario=scenario, heliostat_group=hg, blocking_active=False, batch_size=1)
    with torch.no_grad():
        flux, *_ = rt.trace_rays(incident_ray_directions=rays, active_heliostats_mask=mask,
                                 target_area_indices=tidx, device=device)
    img = flux[0].detach().cpu().numpy().astype(np.float64)
    return img / img.sum()


def blur_unit(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur then renormalise to unit total energy."""
    b = img if sigma <= 0 else gaussian_filter(img, sigma=sigma, mode="constant")
    s = b.sum()
    return b / s if s > 0 else b


def _mse(a, b) -> float:
    return float(np.mean((a - b) ** 2))


def _parse_args():
    p = argparse.ArgumentParser(description="Blur-σ sweep vs surface resolution.")
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

    all_res = sorted(set(args.resolutions + [args.reference]))
    print(f"rendering resolutions {all_res} ...")
    flux = {r: render_unit_flux(scenario_path, r, args.target, device) for r in all_res}
    ref = flux[args.reference]

    sigmas = np.arange(0.0, args.sigma_max + 1e-9, args.sigma_step)
    sweep = {}  # res -> dict(sigmas, mses, sigma_star, mse_star, mse_unblurred)
    for r in args.resolutions:
        mses = np.array([_mse(blur_unit(flux[r], s), ref) for s in sigmas])
        i = int(np.argmin(mses))
        sweep[r] = dict(sigmas=sigmas, mses=mses, sigma_star=float(sigmas[i]),
                        mse_star=float(mses[i]), mse_unblurred=float(mses[0]))
    # "Blur penalty" floor: blurring the reference itself vs the sharp reference.
    floor = np.array([_mse(blur_unit(ref, s), ref) for s in sigmas])

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    SCALE = 1e9  # MSE values are tiny under unit-energy normalisation
    palette = ["#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#9b5de5"]
    cmap = {r: palette[i % len(palette)] for i, r in enumerate(args.resolutions)}

    # ---- Figure 1: MSE vs sigma, per resolution ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for r in args.resolutions:
        s = sweep[r]
        ax.plot(sigmas, s["mses"] * SCALE, "-", color=cmap[r], lw=2, label=f"{r}×{r}")
        ax.scatter([s["sigma_star"]], [s["mse_star"] * SCALE], color=cmap[r], s=60,
                   zorder=5, edgecolor="white")
        ax.annotate(f"σ*={s['sigma_star']:.2f}", (s["sigma_star"], s["mse_star"] * SCALE),
                    textcoords="offset points", xytext=(6, 6), fontsize=8, color=cmap[r])
    ax.plot(sigmas, floor * SCALE, "--", color="#8d99ae", lw=1.3,
            label=f"{args.reference}×{args.reference} blurred (blur penalty)")
    ax.set_xlabel("Gaussian blur σ (pixels)")
    ax.set_ylabel("MSE to sharp 100×100  (×10⁻⁹)")
    ax.set_title("Blurring low-resolution flux toward the sharp 100×100 reference")
    ax.legend(title="surface points/facet"); ax.grid(alpha=0.3)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(outdir / "blur_mse_vs_sigma.png", dpi=150); plt.close(fig)

    # ---- Figure 2: sigma* and best-MSE vs resolution ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    rs = args.resolutions
    a1.plot(rs, [sweep[r]["sigma_star"] for r in rs], "o-", color="#264653", lw=2)
    a1.set_xlabel("surface points/facet (N×N)"); a1.set_ylabel("optimal blur σ* (px)")
    a1.set_title("How much blur each resolution needs")
    a1.set_xticks(rs); a1.set_xticklabels([f"{r}×{r}" for r in rs]); a1.grid(alpha=0.3)

    a2.plot(rs, [sweep[r]["mse_unblurred"] * SCALE for r in rs], "s--", color="#8d99ae",
            lw=2, label="unblurred (σ=0)")
    a2.plot(rs, [sweep[r]["mse_star"] * SCALE for r in rs], "o-", color="#E76F51",
            lw=2, label="best blur (σ*)")
    a2.set_xlabel("surface points/facet (N×N)"); a2.set_ylabel("MSE to sharp 100×100 (×10⁻⁹)")
    a2.set_title("Best achievable error: blur vs no blur")
    a2.set_xticks(rs); a2.set_xticklabels([f"{r}×{r}" for r in rs])
    a2.legend(); a2.grid(alpha=0.3)
    for a in (a1, a2):
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(outdir / "blur_sigma_and_mse.png", dpi=150); plt.close(fig)

    # ---- Figure 3: image grid (unblurred vs blurred-at-σ*) ----
    cols = args.resolutions + [args.reference]
    fig, axes = plt.subplots(2, len(cols), figsize=(2.6 * len(cols), 5.4))
    for j, r in enumerate(cols):
        unb = flux[r]
        sig = sweep[r]["sigma_star"] if r in sweep else 0.0
        bl = blur_unit(flux[r], sig)
        for row, img, tag in ((0, unb, "unblurred"), (1, bl, f"σ*={sig:.2f}" if r in sweep else "reference")):
            ax = axes[row, j]
            vmax = float(np.percentile(img, 99.5)) or float(img.max()) or 1.0
            ax.imshow(img, cmap="inferno", origin="lower",
                      norm=PowerNorm(gamma=0.5, vmin=0, vmax=vmax))
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0: ax.set_title(f"{r}×{r}", fontsize=11)
            if j == 0: ax.set_ylabel({0: "unblurred", 1: "blurred (σ*)"}[row], fontsize=11)
            if r in sweep: ax.set_xlabel(tag, fontsize=8)
    fig.suptitle("Single-heliostat flux: unblurred (top) vs blurred at σ* (bottom)", fontsize=12)
    fig.tight_layout(); fig.savefig(outdir / "blur_image_grid.png", dpi=150); plt.close(fig)

    # ---- markdown summary ----
    lines = ["# Blur vs surface resolution (σ sweep)\n",
             "Single-heliostat flux, unit-energy normalised, MSE to the **sharp 100×100**.",
             "Blur applied only to the low-resolution image (Gaussian, renormalised).\n",
             "| Resolution | unblurred MSE (×10⁻⁹) | σ* (px) | best MSE (×10⁻⁹) | error removed |",
             "|---|---|---|---|---|"]
    for r in args.resolutions:
        s = sweep[r]
        drop = (1 - s["mse_star"] / s["mse_unblurred"]) * 100 if s["mse_unblurred"] else 0
        lines.append(f"| {r}×{r} | {s['mse_unblurred']*SCALE:.2f} | {s['sigma_star']:.2f} | "
                     f"{s['mse_star']*SCALE:.2f} | {drop:.0f}% |")
    (outdir / "blur_resolution_sweep.md").write_text("\n".join(lines) + "\n")

    print("\nσ* and MSE per resolution:")
    for r in args.resolutions:
        s = sweep[r]
        print(f"  {r}×{r}: σ*={s['sigma_star']:.2f}  unblurred={s['mse_unblurred']*SCALE:.2f}  "
              f"best={s['mse_star']*SCALE:.2f} (×1e-9)")
    print(f"\nsaved figures + md -> {outdir}")


if __name__ == "__main__":
    main()
