"""
Plotting routines for the Gaussian blur ablation study.

Figure 1 (main): Two-panel heatmap — #rays vs distance, color = MSE.
                 Left panel: blur=off (sigma=0), right panel: blur=on (optimal sigma).
                 Directly addresses the teacher's requested graph.

Figure 2 (supplementary): Line plots — #rays vs MSE per distance band.
                           Two sub-panels: blur=off vs blur=on, with ±std shading.

Figure 3 (supplementary): Sigma sweep — sigma vs MSE, one line per distance band.
                           Shows which sigma is optimal and whether it varies with distance.
"""

import pathlib

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# Distance band boundaries in metres (horizontal distance from tower).
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
    mapping = {"near": "Near (<100 m)", "mid": "Mid (100–175 m)", "far": "Far (>175 m)"}
    return mapping.get(band, band)


def plot_heatmap(
    records: list[dict],
    heliostat_distances: dict[str, float],
    output_path: pathlib.Path,
    optimal_sigma: float | None = None,
) -> None:
    """Figure 1 (main): Two-panel heatmap — #rays × distance_band, color = MSE.

    Parameters
    ----------
    records : list[dict]
        Output of sweep.run_blur_sweep().
    heliostat_distances : dict[str, float]
        {heliostat_id: distance_from_tower_m}.
    output_path : pathlib.Path
        Where to save the figure (PNG).
    optimal_sigma : float, optional
        The sigma value to use for the "blur=ON" panel.
        If None, automatically chosen as the sigma minimising mean MSE across all heliostats.
    """
    import pandas as pd

    df = pd.DataFrame(records)
    df["distance_m"] = df["heliostat_id"].map(heliostat_distances)
    df["band"] = df["distance_m"].map(_assign_band)

    rays = sorted(df["n_rays"].unique())
    sigmas = sorted(df["sigma"].unique())
    bands = ["near", "mid", "far"]

    # Determine optimal sigma for blur=ON panel.
    if optimal_sigma is None:
        blur_df = df[df["sigma"] > 0]
        if not blur_df.empty:
            sigma_mse = blur_df.groupby("sigma")["mse"].mean()
            optimal_sigma = float(sigma_mse.idxmin())
        else:
            optimal_sigma = 0.0

    def _build_grid(sigma_val: float) -> np.ndarray:
        """Returns (n_bands × n_rays) MSE matrix, averaged over heliostats in each band."""
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
    grid_on = _build_grid(optimal_sigma)

    # Shared colorscale.
    vmin = np.nanmin([grid_off, grid_on])
    vmax = np.nanmax([grid_off, grid_on])

    fig = plt.figure(figsize=(11, 4))
    fig.suptitle(
        f"Simulation MSE vs #rays and distance\n"
        f"(left: no blur, right: Gaussian blur σ={optimal_sigma})",
        fontsize=12,
    )
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[1, 1, 0.06],
        wspace=0.35,
        left=0.08, right=0.94, top=0.82, bottom=0.12,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax  = fig.add_subplot(gs[0, 2])
    axes[1].sharey(axes[0])

    for ax, grid, title in [
        (axes[0], grid_off, "Blur = OFF (σ=0)"),
        (axes[1], grid_on,  f"Blur = ON  (σ={optimal_sigma})"),
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
        # Annotate cells with MSE values.
        for bi in range(len(bands)):
            for ri in range(len(rays)):
                val = grid[bi, ri]
                if not np.isnan(val):
                    ax.text(ri, bi, f"{val:.4f}", ha="center", va="center",
                            fontsize=8, color="white" if val > (vmin + vmax) / 2 else "black")

    plt.setp(axes[1].get_yticklabels(), visible=False)
    fig.colorbar(im, cax=cax, label="MSE (peak-normalised)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_line_plots(
    records: list[dict],
    heliostat_distances: dict[str, float],
    output_path: pathlib.Path,
    optimal_sigma: float | None = None,
) -> None:
    """Figure 2 (supplementary): Line plots — #rays vs MSE per distance band."""
    import pandas as pd

    df = pd.DataFrame(records)
    df["distance_m"] = df["heliostat_id"].map(heliostat_distances)
    df["band"] = df["distance_m"].map(_assign_band)

    rays = sorted(df["n_rays"].unique())
    bands = ["near", "mid", "far"]
    colors = {"near": "#2196F3", "mid": "#FF9800", "far": "#4CAF50"}

    if optimal_sigma is None:
        blur_df = df[df["sigma"] > 0]
        if not blur_df.empty:
            optimal_sigma = float(blur_df.groupby("sigma")["mse"].mean().idxmin())
        else:
            optimal_sigma = 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("MSE vs number of rays — blur off vs blur on", fontsize=12)

    for ax, sigma_val, title in [
        (axes[0], 0.0,          "Blur = OFF (σ=0)"),
        (axes[1], optimal_sigma, f"Blur = ON  (σ={optimal_sigma})"),
    ]:
        sub = df[df["sigma"] == sigma_val]
        for band in bands:
            band_df = sub[sub["band"] == band]
            if band_df.empty:
                continue
            means = band_df.groupby("n_rays")["mse"].mean().reindex(rays)
            stds  = band_df.groupby("n_rays")["mse"].std().reindex(rays).fillna(0)
            ax.plot(rays, means.values, marker="o", label=_band_label(band), color=colors[band])
            ax.fill_between(rays,
                            (means - stds).values,
                            (means + stds).values,
                            alpha=0.2, color=colors[band])
        ax.set_xticks(rays)
        ax.set_xlabel("Number of rays")
        ax.set_ylabel("MSE (peak-normalised)")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_field_heatmap(
    selected_heliostats: list[dict],
    output_path: pathlib.Path,
    heliostats_per_cell: int = 2,
) -> None:
    """Figure 4: Grid heatmap — rows=distance bands, cols=quadrants, value=# selected heliostats."""
    import pandas as pd

    bands = ["near", "mid", "far"]
    band_labels = {"near": "Near\n(<100 m)", "mid": "Mid\n(100–175 m)", "far": "Far\n(>175 m)"}
    quadrants = ["N", "E", "S", "W"]

    # Count selected per cell and collect names.
    counts = pd.DataFrame(0, index=bands, columns=quadrants)
    names: dict[tuple, list[str]] = {(b, q): [] for b in bands for q in quadrants}
    for h in selected_heliostats:
        counts.loc[h["band"], h["quadrant"]] += 1
        names[(h["band"], h["quadrant"])].append(h["name"])

    grid = counts.values.astype(float)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=heliostats_per_cell, aspect="auto")

    ax.set_xticks(range(len(quadrants)))
    ax.set_xticklabels(quadrants)
    ax.set_yticks(range(len(bands)))
    ax.set_yticklabels([band_labels[b] for b in bands])
    ax.set_xlabel("Quadrant")
    ax.set_ylabel("Distance band")
    ax.set_title(f"Heliostat selection grid\n(max {heliostats_per_cell} per cell)")

    for bi, band in enumerate(bands):
        for qi, q in enumerate(quadrants):
            count = int(grid[bi, qi])
            cell_names = names[(band, q)]
            label = f"{count}/{heliostats_per_cell}"
            if cell_names:
                label += "\n" + ", ".join(cell_names)
            text_color = "white" if count == heliostats_per_cell else "black"
            ax.text(qi, bi, label, ha="center", va="center", fontsize=7.5, color=text_color)

    fig.colorbar(im, ax=ax, label="# selected heliostats", ticks=range(heliostats_per_cell + 1))
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_field_coordinates(
    selected_heliostats: list[dict],
    all_heliostat_positions: dict[str, tuple[float, float]],
    output_path: pathlib.Path,
    deflectometry_names: set[str] | None = None,
) -> None:
    """Figure 5: Birds-eye field map — all heliostats, deflectometry subset, and selected sample.

    Parameters
    ----------
    selected_heliostats : list[dict]
        Output of the stratified selection — each dict has 'name', 'band', etc.
    all_heliostat_positions : dict[str, tuple[float, float]]
        {name: (east_m, north_m)} for every heliostat in the scenario.
    deflectometry_names : set[str], optional
        Names of heliostats that have deflectometry calibration data.
        If None, this layer is omitted.
    """
    selected_names = {h["name"] for h in selected_heliostats}

    # Split positions into three layers.
    bg_e, bg_n = [], []
    defl_e, defl_n = [], []
    for name, (e, n) in all_heliostat_positions.items():
        if name in selected_names:
            continue
        if deflectometry_names and name in deflectometry_names:
            defl_e.append(e)
            defl_n.append(n)
        else:
            bg_e.append(e)
            bg_n.append(n)

    sel_e = [all_heliostat_positions[h["name"]][0] for h in selected_heliostats if h["name"] in all_heliostat_positions]
    sel_n = [all_heliostat_positions[h["name"]][1] for h in selected_heliostats if h["name"] in all_heliostat_positions]

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(bg_e, bg_n, s=6, color="#cccccc", zorder=1, label="All heliostats")
    if deflectometry_names:
        ax.scatter(defl_e, defl_n, s=10, color="#90CAF9", zorder=2, label="Deflectometry heliostats")
    ax.scatter(sel_e, sel_n, s=60, color="#F44336", zorder=4, label="Selected sample",
               edgecolors="white", linewidths=0.6)

    # Tower at origin.
    ax.scatter([0], [0], s=120, marker="*", color="black", zorder=5, label="Tower")

    ax.set_aspect("equal")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Heliostat field — stratified sample selection")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sigma_sweep(
    records: list[dict],
    heliostat_distances: dict[str, float],
    output_path: pathlib.Path,
    fixed_n_rays: int = 10,
) -> None:
    """Figure 3 (supplementary): sigma vs MSE, one line per distance band (at fixed n_rays)."""
    import pandas as pd

    df = pd.DataFrame(records)
    df = df[df["n_rays"] == fixed_n_rays]
    df["distance_m"] = df["heliostat_id"].map(heliostat_distances)
    df["band"] = df["distance_m"].map(_assign_band)

    bands = ["near", "mid", "far"]
    colors = {"near": "#2196F3", "mid": "#FF9800", "far": "#4CAF50"}
    sigmas = sorted(df["sigma"].unique())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(f"Effect of blur sigma on MSE (n_rays={fixed_n_rays})", fontsize=12)

    for band in bands:
        band_df = df[df["band"] == band]
        if band_df.empty:
            continue
        means = band_df.groupby("sigma")["mse"].mean().reindex(sigmas)
        stds  = band_df.groupby("sigma")["mse"].std().reindex(sigmas).fillna(0)
        ax.plot(sigmas, means.values, marker="o", label=_band_label(band), color=colors[band])
        ax.fill_between(sigmas,
                        (means - stds).values,
                        (means + stds).values,
                        alpha=0.2, color=colors[band])

    # Mark sigma=0 (no blur baseline).
    if 0.0 in sigmas:
        for band in bands:
            band_df = df[(df["band"] == band) & (df["sigma"] == 0.0)]
            if not band_df.empty:
                baseline = band_df["mse"].mean()
                ax.axhline(baseline, linestyle="--", alpha=0.4, color=colors[band])

    ax.set_xlabel("Gaussian blur sigma (pixels)")
    ax.set_ylabel("MSE (peak-normalised)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
