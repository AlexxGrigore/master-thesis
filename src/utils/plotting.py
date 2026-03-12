import pathlib
import re

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter
from scipy.ndimage import gaussian_filter

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
        Rank: 0, Epoch: {epoch}, Eval Loss: {eval_loss}
    Each heliostat group produces one such sequence, separated by
    'Kinematic reconstructed.' markers.
    """
    train_pattern = re.compile(
        r"Rank:\s*0\s*,\s*Epoch:\s*(\d+),\s*Loss:\s*([\d.eE+\-]+),\s*LR:\s*([\d.eE+\-]+)"
    )
    eval_pattern = re.compile(
        r"Rank:\s*0\s*,\s*Epoch:\s*(\d+),\s*Eval Loss:\s*([\d.eE+\-]+)"
    )

    if not log_file.exists():
        print(f"Log file not found at {log_file}; skipping training curve plot.")
        return

    groups_data: list[dict] = []
    current: dict = {"epochs": [], "losses": [], "lrs": [], "eval_epochs": [], "eval_losses": []}

    with open(log_file) as fh:
        for line in fh:
            m = train_pattern.search(line)
            if m:
                epoch, loss, lr = int(m.group(1)), float(m.group(2)), float(m.group(3))
                current["epochs"].append(epoch)
                current["losses"].append(loss)
                current["lrs"].append(lr)
                continue
            m = eval_pattern.search(line)
            if m:
                current["eval_epochs"].append(int(m.group(1)))
                current["eval_losses"].append(float(m.group(2)))
                continue
            if "Kinematic reconstructed." in line and current["epochs"]:
                groups_data.append(current)
                current = {"epochs": [], "losses": [], "lrs": [], "eval_epochs": [], "eval_losses": []}

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
        if individual and n_groups == 1:
            train_color, val_color = "tab:blue", "tab:orange"
            train_label, val_label = "Training Loss", "Validation Loss"
        elif individual:
            train_color = val_color = cmap(i / max(n_groups - 1, 1))
            train_label, val_label = f"Group {i} (train)", f"Group {i} (val)"
        else:
            train_color = val_color = cmap(i / max(n_groups - 1, 1))
            train_label = val_label = "_nolegend_"
        alpha = 0.85 if individual else 0.12
        ax_loss.plot(gd["epochs"], gd["losses"], color=train_color, alpha=alpha, linewidth=1.2, label=train_label)
        if gd["eval_epochs"]:
            ax_loss.plot(
                gd["eval_epochs"], gd["eval_losses"],
                color=val_color, alpha=alpha, linewidth=1.2, linestyle="--",
                label=val_label if individual else "_nolegend_",
            )
        ax_lr.plot(gd["epochs"], gd["lrs"], color=train_color, alpha=alpha, linewidth=1.2, label=train_label)

    if not individual:
        max_common = min(len(gd["epochs"]) for gd in groups_data)
        if max_common > 0:
            common_epochs = groups_data[0]["epochs"][:max_common]
            all_losses = np.array([gd["losses"][:max_common] for gd in groups_data])
            all_lrs = np.array([gd["lrs"][:max_common] for gd in groups_data])
            mean_losses = all_losses.mean(axis=0)
            mean_lrs = all_lrs.mean(axis=0)
            p10_losses, p90_losses = np.percentile(all_losses, 10, axis=0), np.percentile(all_losses, 90, axis=0)
            ax_loss.fill_between(common_epochs, p10_losses, p90_losses, alpha=0.15, color="tab:blue", label="10–90th pct.")
            ax_loss.plot(common_epochs, mean_losses, color="tab:blue", linewidth=2.2, label=f"Mean Training Loss ({n_groups} groups)", zorder=5)
            ax_lr.plot(common_epochs, mean_lrs, color="tab:blue", linewidth=2.2, label=f"Mean ({n_groups} groups)", zorder=5)
            # Mean eval loss across groups (only groups that have eval data)
            groups_with_eval = [gd for gd in groups_data if gd["eval_epochs"]]
            if groups_with_eval:
                max_eval_common = min(len(gd["eval_epochs"]) for gd in groups_with_eval)
                if max_eval_common > 0:
                    common_eval_epochs = groups_with_eval[0]["eval_epochs"][:max_eval_common]
                    all_eval_losses = np.array([gd["eval_losses"][:max_eval_common] for gd in groups_with_eval])
                    ax_loss.plot(
                        common_eval_epochs, all_eval_losses.mean(axis=0),
                        color="tab:orange", linewidth=2.2, linestyle="--",
                        label=f"Mean Validation Loss ({len(groups_with_eval)} groups)", zorder=5,
                    )

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

            # Peak-normalize both images to [0, 1] for a fair visual comparison.
            pred_max = pred_raw.max()
            meas_max = meas.max()
            pred = pred_raw / pred_max if pred_max > 0 else pred_raw
            meas_norm = meas / meas_max if meas_max > 0 else meas

            diff = pred - meas_norm

            # Gaussian-blurred versions (sigma matches WortbergPixelReconstructor.BLUR_SIGMA).
            pred_blur = gaussian_filter(pred, sigma=5)
            pred_blur_max = pred_blur.max()
            pred_blur = pred_blur / pred_blur_max if pred_blur_max > 0 else pred_blur
            diff_blur = pred_blur - meas_norm

            # Both images are in [0, 1], so shared colour scale is always 1.
            vmax_flux = 1.0
            pixel_mse = float(np.mean(diff ** 2))
            max_abs_diff = float(np.abs(diff).max())
            pixel_mse_blur = float(np.mean(diff_blur ** 2))
            max_abs_diff_blur = float(np.abs(diff_blur).max())

            fig, axes = plt.subplots(2, 3, figsize=(17, 10))
            fig.patch.set_facecolor("white")
            fig.suptitle(
                f"Flux Comparison — sample {samples_visualized + 1}",
                fontsize=FONT_TITLE, fontweight="bold",
            )

            # --- Row 0: raw (peak-normalised) ---
            im00 = axes[0, 0].imshow(pred, cmap="inferno", vmin=0, vmax=vmax_flux)
            axes[0, 0].set_title("Predicted Flux (peak-normalised)", fontsize=FONT_LABEL, fontweight="bold")
            axes[0, 0].axis("off")
            plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.03).ax.tick_params(labelsize=FONT_TICK)

            im01 = axes[0, 1].imshow(meas_norm, cmap="inferno", vmin=0, vmax=vmax_flux)
            axes[0, 1].set_title("Measured Flux", fontsize=FONT_LABEL, fontweight="bold")
            axes[0, 1].axis("off")
            plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.03).ax.tick_params(labelsize=FONT_TICK)

            im02 = axes[0, 2].imshow(diff, cmap="coolwarm", vmin=-max_abs_diff, vmax=max_abs_diff)
            axes[0, 2].set_title(
                f"Residual (Pred − Meas)\npixel MSE: {pixel_mse:.5f}  max |diff|: {max_abs_diff:.4f}",
                fontsize=FONT_LABEL, fontweight="bold",
            )
            axes[0, 2].axis("off")
            plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.03).ax.tick_params(labelsize=FONT_TICK)

            # --- Row 1: Gaussian-blurred predicted ---
            im10 = axes[1, 0].imshow(pred_blur, cmap="inferno", vmin=0, vmax=vmax_flux)
            axes[1, 0].set_title("Predicted Flux (Gaussian blur σ=3)", fontsize=FONT_LABEL, fontweight="bold")
            axes[1, 0].axis("off")
            plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.03).ax.tick_params(labelsize=FONT_TICK)

            im11 = axes[1, 1].imshow(meas_norm, cmap="inferno", vmin=0, vmax=vmax_flux)
            axes[1, 1].set_title("Measured Flux", fontsize=FONT_LABEL, fontweight="bold")
            axes[1, 1].axis("off")
            plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.03).ax.tick_params(labelsize=FONT_TICK)

            im12 = axes[1, 2].imshow(diff_blur, cmap="coolwarm", vmin=-max_abs_diff_blur, vmax=max_abs_diff_blur)
            axes[1, 2].set_title(
                f"Residual (Blurred − Meas)\npixel MSE: {pixel_mse_blur:.5f}  max |diff|: {max_abs_diff_blur:.4f}",
                fontsize=FONT_LABEL, fontweight="bold",
            )
            axes[1, 2].axis("off")
            plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.03).ax.tick_params(labelsize=FONT_TICK)

            plt.tight_layout()

            if save_figures and output_dir is not None:
                plt.savefig(
                    output_dir / f"flux_comparison_{samples_visualized}.png",
                    dpi=150, bbox_inches="tight",
                )

            plt.close()

            samples_visualized += 1
            if samples_visualized >= num_samples:
                break

    print(f"Visualized {samples_visualized} flux comparison images")
