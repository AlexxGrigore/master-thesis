import pathlib
import re

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario

# ---------------------------------------------------------------------------
# Shared style defaults
# ---------------------------------------------------------------------------
FONT_TITLE = 15
FONT_LABEL = 13
FONT_TICK = 11
FONT_LEGEND = 11
GRID_KW = dict(which="both", linestyle="--", linewidth=0.5, alpha=0.4, color="grey")


def _style_ax(ax, xlabel: str, ylabel: str, title: str) -> None:
    """Apply consistent axis styling."""
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_training_curves(log_file: pathlib.Path, output_dir: pathlib.Path) -> None:
    """
    Parse training.log and save a loss + learning-rate curve plot.

    Looks for log lines of the form (written every log_step epochs):
        Rank: 0, Epoch: {epoch}, Loss: {loss}, LR: {lr}
    Each heliostat group produces one such sequence, separated by
    'Kinematic reconstructed.' markers.
    """
    line_pattern = re.compile(
        r"Rank:\s*0\s*,\s*Epoch:\s*(\d+),\s*Loss:\s*([\d.eE+\-]+),\s*LR:\s*([\d.eE+\-]+)"
    )

    if not log_file.exists():
        print(f"Log file not found at {log_file}; skipping training curve plot.")
        return

    groups_data: list[dict] = []
    current: dict = {"epochs": [], "losses": [], "lrs": []}

    with open(log_file) as fh:
        for line in fh:
            m = line_pattern.search(line)
            if m:
                epoch, loss, lr = int(m.group(1)), float(m.group(2)), float(m.group(3))
                current["epochs"].append(epoch)
                current["losses"].append(loss)
                current["lrs"].append(lr)
            elif "Kinematic reconstructed." in line and current["epochs"]:
                groups_data.append(current)
                current = {"epochs": [], "losses": [], "lrs": []}

    if current["epochs"]:
        groups_data.append(current)

    if not groups_data:
        print("No training curve data found in log file; skipping plot.")
        return

    n_groups = len(groups_data)
    print(f"Plotting training curves for {n_groups} heliostat group(s).")

    fig, (ax_loss, ax_lr) = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    fig.patch.set_facecolor("white")

    individual = n_groups <= 10
    cmap = plt.cm.tab10 if n_groups <= 10 else plt.cm.viridis

    for i, gd in enumerate(groups_data):
        color = cmap(i / max(n_groups - 1, 1))
        alpha = 0.85 if individual else 0.12
        label = f"Group {i}" if individual else "_nolegend_"
        ax_loss.plot(gd["epochs"], gd["losses"], color=color, alpha=alpha, linewidth=1.2, label=label)
        ax_lr.plot(gd["epochs"], gd["lrs"], color=color, alpha=alpha, linewidth=1.2, label=label)

    if not individual:
        max_common = min(len(gd["epochs"]) for gd in groups_data)
        if max_common > 0:
            common_epochs = groups_data[0]["epochs"][:max_common]
            all_losses = np.array([gd["losses"][:max_common] for gd in groups_data])
            all_lrs = np.array([gd["lrs"][:max_common] for gd in groups_data])
            mean_losses = all_losses.mean(axis=0)
            mean_lrs = all_lrs.mean(axis=0)
            p10_losses, p90_losses = np.percentile(all_losses, 10, axis=0), np.percentile(all_losses, 90, axis=0)
            ax_loss.fill_between(common_epochs, p10_losses, p90_losses, alpha=0.15, color="steelblue", label="10–90th pct.")
            ax_loss.plot(common_epochs, mean_losses, color="black", linewidth=2.2, label=f"Mean ({n_groups} groups)", zorder=5)
            ax_lr.plot(common_epochs, mean_lrs, color="black", linewidth=2.2, label=f"Mean ({n_groups} groups)", zorder=5)

    ax_loss.set_yscale("log")
    ax_loss.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax_loss.yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
    ax_loss.grid(**GRID_KW)
    ax_loss.legend(fontsize=FONT_LEGEND, ncol=min(n_groups, 5), framealpha=0.8)
    _style_ax(ax_loss, "Epoch", "Loss", "Training Loss (log scale)")

    ax_lr.set_yscale("log")
    ax_lr.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax_lr.yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
    ax_lr.grid(**GRID_KW)
    if individual:
        ax_lr.legend(fontsize=FONT_LEGEND, ncol=min(n_groups, 5), framealpha=0.8)
    _style_ax(ax_lr, "Epoch", "Learning Rate", "Learning Rate Schedule")

    fig.suptitle("Kinematic Reconstruction Training", fontsize=FONT_TITLE + 1, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = output_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves to {out_path}")


def plot_tracking_error_histogram(
    errors_mrad: list[float],
    output_path: pathlib.Path,
    title: str = "Tracking Error Distribution",
) -> None:
    """
    Plot a histogram of tracking errors in mrad.

    Parameters
    ----------
    errors_mrad : list[float]
        All per-sample tracking errors in milliradians.
    output_path : pathlib.Path
        Where to save the PNG.
    title : str
        Plot title.
    """
    errors = np.array(errors_mrad)
    mean_val = float(np.mean(errors))
    median_val = float(np.median(errors))
    std_val = float(np.std(errors))
    n = len(errors)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")

    counts, bin_edges, patches = ax.hist(
        errors, bins=30, edgecolor="white", linewidth=0.5, alpha=0.85, color="steelblue"
    )

    ax.axvline(mean_val, color="crimson", linestyle="--", linewidth=2.0, label=f"Mean:   {mean_val:.2f} mrad")
    ax.axvline(median_val, color="darkorange", linestyle="-.", linewidth=2.0, label=f"Median: {median_val:.2f} mrad")
    ax.axvspan(mean_val - std_val, mean_val + std_val, alpha=0.10, color="crimson", label=f"±1 std: {std_val:.2f} mrad")

    # Stats annotation box
    stats_text = (
        f"$n$ = {n}\n"
        f"mean = {mean_val:.2f} mrad\n"
        f"median = {median_val:.2f} mrad\n"
        f"std = {std_val:.2f} mrad\n"
        f"min = {errors.min():.2f} mrad\n"
        f"max = {errors.max():.2f} mrad"
    )
    ax.text(
        0.97, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=FONT_TICK,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="grey", alpha=0.85),
    )

    ax.legend(fontsize=FONT_LEGEND, framealpha=0.85)
    ax.grid(axis="y", **GRID_KW)
    _style_ax(ax, "Tracking Error (mrad)", "Absolute Frequency", title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved tracking error histogram to {output_path}")


def visualize_flux_comparison(
    scenario: Scenario,
    heliostat_data_mapping: list[tuple[str, list[pathlib.Path], list[pathlib.Path]]],
    data_parser: PaintCalibrationDataParser,
    device: torch.device,
    output_dir: pathlib.Path = None,
    num_samples: int = 5,
    save_figures: bool = False,
):
    """
    Visualize comparison between predicted and measured flux images.
    """
    if save_figures and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    samples_visualized = 0

    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        if samples_visualized >= num_samples:
            break

        (
            measured_flux,
            _,
            incident_ray_directions,
            _,
            active_heliostats_mask,
            target_area_mask,
        ) = data_parser.parse_data_for_reconstruction(
            heliostat_data_mapping=heliostat_data_mapping,
            heliostat_group=heliostat_group,
            scenario=scenario,
            bitmap_resolution=torch.tensor([256, 256]),
            device=device,
        )

        if active_heliostats_mask.sum() == 0:
            continue

        heliostat_group.activate_heliostats(
            active_heliostats_mask=active_heliostats_mask,
            device=device,
        )
        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=scenario.target_areas.centers[target_area_mask],
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            device=device,
        )

        ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=heliostat_group,
            blocking_active=False,
            batch_size=min(heliostat_group.number_of_active_heliostats, 32),
            bitmap_resolution=torch.tensor([256, 256], device=device),
        )

        predicted_flux = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            target_area_mask=target_area_mask,
            device=device,
        )

        for i in range(min(len(predicted_flux), num_samples - samples_visualized)):
            pred_raw = predicted_flux[i].cpu().detach().numpy()
            meas = measured_flux[i].cpu().detach().numpy()

            # Normalize predicted flux to [0, 1] by matching total energy to the measured image.
            # The ray tracer returns physical intensity units (with 1/r² attenuation) that are
            # orders of magnitude smaller than the [0, 1]-normalised measured flux, so plotting
            # both on the same colour scale would make the prediction appear completely black.
            pred_sum = pred_raw.sum()
            meas_sum = meas.sum()
            if pred_sum > 0 and meas_sum > 0:
                pred = pred_raw * (meas_sum / pred_sum)
            else:
                pred = pred_raw  # fall back to raw values if one image is empty

            diff = pred - meas

            # Shared colour scale for predicted / measured
            vmax_flux = max(pred.max(), meas.max())
            pixel_mse = float(np.mean(diff ** 2))
            max_abs_diff = float(np.abs(diff).max())

            fig, axes = plt.subplots(1, 3, figsize=(17, 5))
            fig.patch.set_facecolor("white")
            fig.suptitle(
                f"Flux Comparison — sample {samples_visualized + 1}   "
                f"(pixel MSE: {pixel_mse:.5f})",
                fontsize=FONT_TITLE, fontweight="bold",
            )

            im0 = axes[0].imshow(pred, cmap="inferno", vmin=0, vmax=vmax_flux)
            axes[0].set_title("Predicted Flux (energy-normalised)", fontsize=FONT_LABEL, fontweight="bold")
            axes[0].axis("off")
            cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
            cb0.ax.tick_params(labelsize=FONT_TICK)

            im1 = axes[1].imshow(meas, cmap="inferno", vmin=0, vmax=vmax_flux)
            axes[1].set_title("Measured Flux", fontsize=FONT_LABEL, fontweight="bold")
            axes[1].axis("off")
            cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
            cb1.ax.tick_params(labelsize=FONT_TICK)

            im2 = axes[2].imshow(
                diff, cmap="coolwarm",
                vmin=-max_abs_diff, vmax=max_abs_diff,
            )
            axes[2].set_title(
                f"Residual (Pred − Meas)\nmax |diff|: {max_abs_diff:.4f}",
                fontsize=FONT_LABEL, fontweight="bold",
            )
            axes[2].axis("off")
            cb2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.03)
            cb2.ax.tick_params(labelsize=FONT_TICK)

            plt.tight_layout()

            if save_figures and output_dir is not None:
                plt.savefig(
                    output_dir / f"flux_comparison_{samples_visualized}.png",
                    dpi=150, bbox_inches="tight",
                )

            plt.show()

            samples_visualized += 1
            if samples_visualized >= num_samples:
                break

    print(f"Visualized {samples_visualized} flux comparison images")
