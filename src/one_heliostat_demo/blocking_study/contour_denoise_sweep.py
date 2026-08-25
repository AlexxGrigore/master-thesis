"""Stage 1 continuation: sweep q (smoothing_rounds) / sigma_gauss for
occlusion invariance AND noise robustness. tau/eta held at the Stage-1
winner (0.70, 70.0) from contour_tau_eta_occlusion_invariance.py.

Kernel size is DERIVED from sigma (k = smallest odd >= 4*sigma + 1, the same
relation the original sigma=3 -> k=13 stability fix already used) rather than
swept independently -- an untied (sigma, k) grid produces mostly degenerate
combinations (kernel too small to represent the blur, or vastly oversized
for a tiny sigma).

Two metrics, both real flux, same 60-image/10-heliostat set as the tau/eta
sweep:
  - occlusion-invariance drift (identical protocol to the tau/eta sweep)
  - noise-robustness drift: inject small Gaussian pixel noise into the RAW
    flux (real flux is noisier than the simulated data the thesis's
    defaults were tuned on -- this axis matters more here), 3 repeats per
    image per noise level, measure contour COM drift vs the noise-free
    baseline.

Usage
-----
    python contour_denoise_sweep.py
"""

from __future__ import annotations

import csv
import json
import logging
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

_here = pathlib.Path(__file__).resolve().parent          # blocking_study/
_src = _here.parents[1]                                   # src/
for _p in (str(_src), str(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from artist_extensions.contour_loss import ContourExtractor  # noqa: E402

log = logging.getLogger(__name__)

_ROOT = _here.parents[2]  # master-thesis/
BENCHMARK_NAME = "benchmark_split-balanced_train-50_validation-20"
PAINT_DIR = _ROOT / "datasets" / "paint"
OUT_DIR = _ROOT / "outputs" / "new_mapping_function" / "contour_hp_sweep" / "stage1_denoise"

HELIOSTATS = ["AH33", "AB50", "AP47", "AO29", "AN27", "AI56", "AG50", "BE29", "AF53", "BG41"]
N_SAMPLES_PER_HELIOSTAT = 6

TAU_WINNER = 0.70
ETA_WINNER = 70.0  # eta was shown irrelevant; keep the project default

Q_GRID = [1, 2, 3, 4]
SIGMA_GRID = [1.0, 2.0, 3.0, 4.0, 5.0]
KAPPA_LEVELS = [0.2, 0.4, 0.6]
NOISE_SIGMAS = [0.02, 0.05]
NOISE_REPEATS = 3
EMPTY_CONTOUR_MASS_EPS = 1e-3
RNG_SEED = 0


def ksize_for_sigma(sigma: float) -> int:
    k = int(np.ceil(4 * sigma + 1))
    return k if k % 2 == 1 else k + 1


def load_flux_images() -> list[torch.Tensor]:
    csv_path = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
    by_hel: dict[str, list[str]] = {}
    with open(csv_path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row[1] in HELIOSTATS and row[2] == "train":
                by_hel.setdefault(row[1], []).append(row[0])
    images = []
    for hid in HELIOSTATS:
        ids = sorted(by_hel.get(hid, []))[:N_SAMPLES_PER_HELIOSTAT]
        for sid in ids:
            p = PAINT_DIR / BENCHMARK_NAME / "flux_image" / "train" / f"{sid}-flux.png"
            if not p.exists():
                continue
            arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
            images.append(torch.from_numpy(arr))
    return images


def occlusion_augment(flux: torch.Tensor, kappa: float) -> torch.Tensor:
    if kappa <= 0:
        return flux.clone()
    H, _W = flux.shape
    row_sums = flux.sum(dim=1)
    total = row_sums.sum()
    if total <= 0:
        return flux.clone()
    target = kappa * total
    cum_from_bottom = torch.cumsum(row_sums.flip(0), dim=0)
    idx = torch.searchsorted(cum_from_bottom, target)
    k = min(int(idx.item()) + 1, H)
    out = flux.clone()
    if k > 0:
        out[H - k :, :] = 0.0
    return out


def com_px(contour: torch.Tensor) -> tuple[float, float] | None:
    mass = contour.sum()
    if mass <= EMPTY_CONTOUR_MASS_EPS:
        return None
    H, W = contour.shape
    rows = torch.arange(H, dtype=contour.dtype)
    cols = torch.arange(W, dtype=contour.dtype)
    r = float((contour.sum(dim=1) * rows).sum() / mass)
    c = float((contour.sum(dim=0) * cols).sum() / mass)
    return (c, r)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(RNG_SEED)

    images = load_flux_images()
    log.info(f"Loaded {len(images)} real flux images. tau={TAU_WINNER}, eta={ETA_WINNER} (fixed)")

    results = []
    n_grid = len(Q_GRID) * len(SIGMA_GRID)
    gi = 0
    for q in Q_GRID:
        for sigma in SIGMA_GRID:
            gi += 1
            ksize = ksize_for_sigma(sigma)
            extractor = ContourExtractor(
                tau=TAU_WINNER, eta=ETA_WINNER, smoothing_rounds=q,
                gaussian_sigma=sigma, gaussian_kernel_size=ksize,
            )

            # --- occlusion-invariance drift (same protocol as tau/eta sweep) ---
            com_drifts = []
            n_empty = 0
            n_checks = 0
            baselines = []
            with torch.no_grad():
                for flux in images:
                    c0 = extractor(flux.unsqueeze(0))[0]
                    com0 = com_px(c0)
                    n_checks += 1
                    baselines.append((flux, com0))
                    if com0 is None:
                        n_empty += 1
                        continue
                    for kappa in KAPPA_LEVELS:
                        occ = occlusion_augment(flux, kappa)
                        ck = extractor(occ.unsqueeze(0))[0]
                        n_checks += 1
                        comk = com_px(ck)
                        if comk is None:
                            n_empty += 1
                            continue
                        com_drifts.append(float(np.hypot(comk[0] - com0[0], comk[1] - com0[1])))
                mean_occ_drift = float(np.mean(com_drifts)) if com_drifts else float("nan")

                # --- noise-robustness drift ---
                noise_drifts = {ns: [] for ns in NOISE_SIGMAS}
                n_noise_empty = 0
                n_noise_checks = 0
                for flux, com0 in baselines:
                    if com0 is None:
                        continue
                    for ns in NOISE_SIGMAS:
                        for _rep in range(NOISE_REPEATS):
                            noisy = (flux + torch.randn_like(flux) * ns).clamp(0.0, 1.0)
                            cn = extractor(noisy.unsqueeze(0))[0]
                            n_noise_checks += 1
                            comn = com_px(cn)
                            if comn is None:
                                n_noise_empty += 1
                                continue
                            noise_drifts[ns].append(float(np.hypot(comn[0] - com0[0], comn[1] - com0[1])))
                mean_noise_drift = {ns: (float(np.mean(v)) if v else float("nan")) for ns, v in noise_drifts.items()}

            results.append({
                "q": q, "sigma": sigma, "ksize": ksize,
                "mean_occlusion_drift_px": mean_occ_drift,
                "n_empty_occlusion": n_empty, "n_checks_occlusion": n_checks,
                "mean_noise_drift_px": mean_noise_drift,
                "n_empty_noise": n_noise_empty, "n_checks_noise": n_noise_checks,
            })
            log.info(f"  [{gi}/{n_grid}] q={q} sigma={sigma} (k={ksize})  "
                     f"occ_drift={mean_occ_drift:.2f}px  "
                     f"noise_drift={mean_noise_drift}  "
                     f"empty(occ/noise)={n_empty}/{n_noise_empty}")

    with open(OUT_DIR / "sweep_results.json", "w") as fh:
        json.dump({
            "heliostats": HELIOSTATS, "n_images": len(images),
            "tau": TAU_WINNER, "eta": ETA_WINNER,
            "q_grid": Q_GRID, "sigma_grid": SIGMA_GRID, "kappa_levels": KAPPA_LEVELS,
            "noise_sigmas": NOISE_SIGMAS, "noise_repeats": NOISE_REPEATS,
            "results": results,
        }, fh, indent=2)

    valid = [r for r in results if r["n_empty_occlusion"] == 0 and r["n_empty_noise"] == 0]
    log.info(f"n_grid_points={len(results)}  n_valid={len(valid)}")
    pool = valid or results

    def combined_score(r):
        noise_mean = float(np.mean(list(r["mean_noise_drift_px"].values())))
        return r["mean_occlusion_drift_px"] + noise_mean

    best = min(pool, key=combined_score)
    log.info(f"BEST (occlusion+noise drift): q={best['q']} sigma={best['sigma']} (k={best['ksize']})  "
             f"occ_drift={best['mean_occlusion_drift_px']:.3f}px  noise_drift={best['mean_noise_drift_px']}")

    # ------------------------------------------------------------------
    # Figure: two heatmaps (occlusion drift, mean noise drift).
    # ------------------------------------------------------------------
    qs, sigmas = Q_GRID, SIGMA_GRID
    occ_grid = np.full((len(qs), len(sigmas)), np.nan)
    noise_grid = np.full((len(qs), len(sigmas)), np.nan)
    for r in results:
        i = qs.index(r["q"])
        j = sigmas.index(r["sigma"])
        occ_grid[i, j] = r["mean_occlusion_drift_px"]
        noise_grid[i, j] = float(np.mean(list(r["mean_noise_drift_px"].values())))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, grid, title in (
        (axes[0], occ_grid, "Mean occlusion-invariance drift [px]"),
        (axes[1], noise_grid, "Mean noise-robustness drift [px]\n(avg over noise sigma=0.02, 0.05)"),
    ):
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                         extent=[sigmas[0] - 0.5, sigmas[-1] + 0.5, qs[0] - 0.5, qs[-1] + 0.5])
        ax.set_xlabel("gaussian sigma"); ax.set_ylabel("q (smoothing rounds)")
        ax.set_yticks(qs)
        ax.set_title(title, fontsize=10.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].scatter([best["sigma"]], [best["q"]], marker="*", s=260, c="red", edgecolor="white", linewidths=1.2, zorder=5)
    axes[0].scatter([3.0], [2], marker="o", s=140, facecolor="none", edgecolor="white", linewidths=2, zorder=5)
    fig.suptitle(f"Denoise (q, sigma) sweep, tau={TAU_WINNER} eta={ETA_WINNER} fixed -- star=winner, circle=current default",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT_DIR / "denoise_heatmaps.png", dpi=150)
    plt.close(fig)

    log.info(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
