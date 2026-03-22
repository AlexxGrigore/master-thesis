"""
Standalone script to regenerate fig1_heatmap_rays_vs_distance.png
from existing JSON outputs, with a fixed colorbar layout.

Usage:
    python blur_ablation/regenerate_fig1.py \
        --results_dir ../../outputs/NewRuns/blur_ablation_20260317_000052
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


DISTANCE_BANDS = [
    ("near",  0,    100),
    ("mid",   100,  175),
    ("far",   175,  float("inf")),
]


def _assign_band(distance_m: float) -> str:
    for label, lo, hi in DISTANCE_BANDS:
        if lo <= distance_m < hi:
            return label
    return "far"


def _band_label(band: str) -> str:
    return {"near": "Near (<100 m)", "mid": "Mid (100–175 m)", "far": "Far (>175 m)"}[band]


def plot_heatmap_fixed(
    records: list[dict],
    heliostat_distances: dict[str, float],
    optimal_sigma: float,
    output_path: pathlib.Path,
) -> None:
    df = pd.DataFrame(records)
    df["distance_m"] = df["heliostat_id"].map(heliostat_distances)
    df["band"] = df["distance_m"].map(_assign_band)

    rays = sorted(df["n_rays"].unique())
    bands = ["near", "mid", "far"]

    def _build_grid(sigma_val: float) -> np.ndarray:
        grid = np.full((len(bands), len(rays)), np.nan)
        sub = df[df["sigma"] == sigma_val]
        for bi, band in enumerate(bands):
            band_df = sub[sub["band"] == band]
            for ri, n_rays in enumerate(rays):
                cell = band_df[band_df["n_rays"] == n_rays]["mse"]
                if not cell.empty:
                    grid[bi, ri] = cell.mean()
        return grid

    grid_off = _build_grid(0.0)
    grid_on  = _build_grid(optimal_sigma)

    vmin = np.nanmin([grid_off, grid_on])
    vmax = np.nanmax([grid_off, grid_on])

    # Use gridspec: two heatmap columns + one narrow colorbar column.
    fig = plt.figure(figsize=(11, 4))
    fig.suptitle(
        f"Simulation MSE vs #rays and distance\n"
        f"(left: no blur, right: Gaussian blur σ={optimal_sigma})",
        fontsize=12,
    )
    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[1, 1, 0.06],
        wspace=0.35,
        left=0.08, right=0.94, top=0.82, bottom=0.12,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    cax = fig.add_subplot(gs[0, 2])

    for ax, grid, title in [
        (ax0, grid_off, "Blur = OFF (σ=0)"),
        (ax1, grid_on,  f"Blur = ON  (σ={optimal_sigma})"),
    ]:
        im = ax.imshow(
            grid,
            aspect="auto",
            origin="upper",
            cmap="viridis_r",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xticks(range(len(rays)))
        ax.set_xticklabels([str(r) for r in rays])
        ax.set_xlabel("Number of rays")
        ax.set_yticks(range(len(bands)))
        ax.set_yticklabels([_band_label(b) for b in bands])
        ax.set_title(title)
        for bi in range(len(bands)):
            for ri in range(len(rays)):
                val = grid[bi, ri]
                if not np.isnan(val):
                    ax.text(
                        ri, bi, f"{val:.4f}",
                        ha="center", va="center",
                        fontsize=8,
                        color="white" if val > (vmin + vmax) / 2 else "black",
                    )

    # Hide y-tick labels on right panel (shared axis).
    plt.setp(ax1.get_yticklabels(), visible=False)

    fig.colorbar(im, cax=cax, label="MSE (peak-normalised)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent.parent
                / "outputs" / "NewRuns" / "blur_ablation_20260317_000052",
    )
    args = parser.parse_args()

    results_dir = args.results_dir

    with open(results_dir / "sweep_results.json") as f:
        records = json.load(f)

    with open(results_dir / "selected_heliostats.json") as f:
        selected = json.load(f)

    with open(results_dir / "optimal_sigma.json") as f:
        optimal_sigma = json.load(f)["optimal_sigma"]

    heliostat_distances = {h["name"]: h["distance_m"] for h in selected}

    print(f"Records: {len(records)}, optimal sigma: {optimal_sigma}")

    plot_heatmap_fixed(
        records=records,
        heliostat_distances=heliostat_distances,
        optimal_sigma=optimal_sigma,
        output_path=results_dir / "fig1_heatmap_rays_vs_distance.png",
    )


if __name__ == "__main__":
    main()
