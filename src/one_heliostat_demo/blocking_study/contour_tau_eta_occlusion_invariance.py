"""Stage 1 of the contour-loss hyperparameter plan: sweep tau/eta for
occlusion invariance, no training needed, pure image processing.

A good (tau, eta) should extract a contour whose position barely moves when
the same real flux image has flux artificially removed from its lower region
(the mechanism Wortberg's contour loss is built to be robust to, confirmed
earlier this session via ray tracing: L1/L2 of the premise test). A bad
(tau, eta) lets the extracted contour drift as occlusion severity grows,
which defeats the entire point of using it.

Synthetic occlusion augmentation (occlusion_augment): removes real flux from
the BOTTOM of the image, sweeping a horizontal cutoff line upward until
kappa fraction of the image's total flux has been zeroed. This is a direct,
literal implementation of "eats flux from the lower part of the spot" -- no
ray tracing needed, pure image manipulation, hence "cheap to run" per the
plan.

Data: real PAINT flux images (train split), 10 heliostats x ~6 samples,
chosen to match the field ablation study's heliostat list for consistency.
Real images carry real camera noise, the actual regime this pipeline has to
work in.

Metric: mean contour-COM displacement (pixels) and mean 1-DICE drift between
the kappa=0 contour and each kappa>0 contour, averaged over all samples and
kappa in {0.2, 0.4, 0.6}. Any (tau, eta) that ever produces an EMPTY contour
(zero mass) at any kappa is excluded regardless of its drift score.

Usage
-----
    python contour_tau_eta_occlusion_invariance.py
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
OUT_DIR = _ROOT / "outputs" / "new_mapping_function" / "contour_hp_sweep" / "stage1_tau_eta"

HELIOSTATS = ["AH33", "AB50", "AP47", "AO29", "AN27", "AI56", "AG50", "BE29", "AF53", "BG41"]
N_SAMPLES_PER_HELIOSTAT = 6

TAU_GRID = np.round(np.arange(0.40, 0.76, 0.05), 2).tolist()
ETA_GRID = [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
KAPPA_LEVELS = [0.2, 0.4, 0.6]

# Denoise settings held at the current project defaults (already the
# stability-patched values, sigma=3/k=13/q=2) -- this sweep is tau/eta only.
GAUSS_SIGMA = 3.0
GAUSS_KSIZE = 13
SMOOTHING_ROUNDS = 2

EMPTY_CONTOUR_MASS_EPS = 1e-3


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
    """Zero rows from the bottom until ~kappa fraction of total flux is removed."""
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


def soft_dice(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
    inter = (a * b).sum()
    denom = a.sum() + b.sum()
    return float((2.0 * inter + eps) / (denom + eps))


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

    images = load_flux_images()
    log.info(f"Loaded {len(images)} real flux images across {len(HELIOSTATS)} heliostats")

    results = []
    n_grid = len(TAU_GRID) * len(ETA_GRID)
    gi = 0
    for tau in TAU_GRID:
        for eta in ETA_GRID:
            gi += 1
            extractor = ContourExtractor(
                tau=float(tau), eta=float(eta), smoothing_rounds=SMOOTHING_ROUNDS,
                gaussian_sigma=GAUSS_SIGMA, gaussian_kernel_size=GAUSS_KSIZE,
            )
            com_drifts, dice_drifts = [], []
            n_empty = 0
            n_total_checks = 0
            with torch.no_grad():
                for flux in images:
                    c0 = extractor(flux.unsqueeze(0))[0]
                    com0 = com_px(c0)
                    n_total_checks += 1
                    if com0 is None:
                        n_empty += 1
                        continue
                    for kappa in KAPPA_LEVELS:
                        occ = occlusion_augment(flux, kappa)
                        ck = extractor(occ.unsqueeze(0))[0]
                        n_total_checks += 1
                        comk = com_px(ck)
                        if comk is None:
                            n_empty += 1
                            continue
                        d = float(np.hypot(comk[0] - com0[0], comk[1] - com0[1]))
                        com_drifts.append(d)
                        dice_drifts.append(1.0 - soft_dice(ck, c0))
            mean_com_drift = float(np.mean(com_drifts)) if com_drifts else float("nan")
            mean_dice_drift = float(np.mean(dice_drifts)) if dice_drifts else float("nan")
            results.append({
                "tau": float(tau), "eta": float(eta),
                "mean_com_drift_px": mean_com_drift,
                "mean_dice_drift": mean_dice_drift,
                "n_empty": n_empty, "n_total_checks": n_total_checks,
                "empty_frac": n_empty / n_total_checks if n_total_checks else 1.0,
            })
            if gi % 8 == 0 or gi == n_grid:
                log.info(f"  [{gi}/{n_grid}] tau={tau:.2f} eta={eta:.0f}  "
                         f"com_drift={mean_com_drift:.2f}px  dice_drift={mean_dice_drift:.3f}  "
                         f"empty={n_empty}/{n_total_checks}")

    with open(OUT_DIR / "sweep_results.json", "w") as fh:
        json.dump({
            "heliostats": HELIOSTATS, "n_images": len(images),
            "tau_grid": TAU_GRID, "eta_grid": ETA_GRID, "kappa_levels": KAPPA_LEVELS,
            "gauss_sigma": GAUSS_SIGMA, "gauss_ksize": GAUSS_KSIZE, "smoothing_rounds": SMOOTHING_ROUNDS,
            "results": results,
        }, fh, indent=2)

    valid = [r for r in results if r["n_empty"] == 0]
    log.info(f"n_grid_points={len(results)}  n_valid(no empty contours)={len(valid)}")
    if valid:
        best = min(valid, key=lambda r: r["mean_com_drift_px"])
        log.info(f"BEST (min COM drift, zero empty contours): tau={best['tau']} eta={best['eta']}  "
                 f"com_drift={best['mean_com_drift_px']:.3f}px  dice_drift={best['mean_dice_drift']:.4f}")
    else:
        best = min(results, key=lambda r: r["empty_frac"])
        log.info(f"NO valid (empty-free) grid point. Least-bad: tau={best['tau']} eta={best['eta']}  "
                 f"empty_frac={best['empty_frac']:.3f}")

    # ------------------------------------------------------------------
    # Figure 1: heatmaps (COM drift, DICE drift, empty-contour fraction).
    # ------------------------------------------------------------------
    taus, etas = TAU_GRID, ETA_GRID
    com_grid = np.full((len(etas), len(taus)), np.nan)
    dice_grid = np.full((len(etas), len(taus)), np.nan)
    empty_grid = np.zeros((len(etas), len(taus)))
    for r in results:
        i = etas.index(r["eta"])
        j = taus.index(r["tau"])
        com_grid[i, j] = r["mean_com_drift_px"]
        dice_grid[i, j] = r["mean_dice_drift"]
        empty_grid[i, j] = r["empty_frac"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for ax, grid, title, cmap in (
        (axes[0], com_grid, "Mean COM drift [px]\n(lower = more occlusion-invariant)", "viridis"),
        (axes[1], dice_grid, "Mean 1-DICE drift\n(lower = more occlusion-invariant)", "viridis"),
        (axes[2], empty_grid, "Empty-contour fraction\n(must be 0)", "Reds"),
    ):
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap,
                         extent=[taus[0] - 0.025, taus[-1] + 0.025, etas[0] - 5, etas[-1] + 5])
        ax.set_xlabel("tau"); ax.set_ylabel("eta")
        ax.set_title(title, fontsize=10.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax_star = axes[0]
    ax_star.scatter([best["tau"]], [best["eta"]], marker="*", s=260, c="red", edgecolor="white", linewidths=1.2, zorder=5)
    fig.suptitle(f"Contour tau/eta occlusion-invariance sweep -- {len(images)} real flux images, "
                 f"{len(HELIOSTATS)} heliostats, kappa={KAPPA_LEVELS}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "tau_eta_heatmaps.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: drift-vs-kappa sanity curve for the winning (tau, eta),
    # vs the CURRENT project default (0.58, 70), computed fresh at full
    # per-sample granularity.
    # ------------------------------------------------------------------
    def per_kappa_curve(tau, eta):
        extractor = ContourExtractor(
            tau=float(tau), eta=float(eta), smoothing_rounds=SMOOTHING_ROUNDS,
            gaussian_sigma=GAUSS_SIGMA, gaussian_kernel_size=GAUSS_KSIZE,
        )
        curve = {0.0: []}
        for k in KAPPA_LEVELS:
            curve[k] = []
        with torch.no_grad():
            for flux in images:
                c0 = extractor(flux.unsqueeze(0))[0]
                com0 = com_px(c0)
                if com0 is None:
                    continue
                for kappa in KAPPA_LEVELS:
                    occ = occlusion_augment(flux, kappa)
                    ck = extractor(occ.unsqueeze(0))[0]
                    comk = com_px(ck)
                    if comk is None:
                        continue
                    d = float(np.hypot(comk[0] - com0[0], comk[1] - com0[1]))
                    curve[kappa].append(d)
        xs = [0.0] + KAPPA_LEVELS
        ys = [0.0] + [float(np.mean(curve[k])) if curve[k] else float("nan") for k in KAPPA_LEVELS]
        return xs, ys

    xs_best, ys_best = per_kappa_curve(best["tau"], best["eta"])
    xs_def, ys_def = per_kappa_curve(0.58, 70.0)

    fig2, ax2 = plt.subplots(figsize=(7, 5.2))
    ax2.plot(xs_best, ys_best, "o-", color="#1f78b4",
             label=f"winner: tau={best['tau']}, eta={best['eta']:.0f}")
    ax2.plot(xs_def, ys_def, "o--", color="#888888", label="current default: tau=0.58, eta=70")
    ax2.set_xlabel("kappa (fraction of flux removed from the bottom)")
    ax2.set_ylabel("mean contour COM drift [px] vs kappa=0")
    ax2.set_title(f"Occlusion-invariance sanity curve\n({len(images)} real flux images, {len(HELIOSTATS)} heliostats)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "drift_vs_kappa_sanity_curve.png", dpi=150)
    plt.close(fig2)

    log.info(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
