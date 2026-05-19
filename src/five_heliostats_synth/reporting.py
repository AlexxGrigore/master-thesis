"""
Plots and summary outputs for the 5-heliostat synthetic perturbation experiment.

All plot functions read from pre-saved JSON files so that plots can be regenerated
and customized without re-running training.

Output files produced
---------------------
Per (subset, train_size) run:
  convergence.png          — train/val loss + 3 horizontal reference lines (all in metres)
  recovery_rotation.png    — bar chart: true vs recovered rotation per heliostat
  recovery_actuator_angle.png
  recovery_base_position.png
  kinematics_{hid}.png     — kinematic parameter evolution during training (one per heliostat)
  summary.json             — human-readable summary

Per subset (collected across train sizes):
  ablation_summary.json    — comparison across training-data-size runs
  ablation_comparison.png  — bar chart: mrad per stage per train size
  combined_convergence.png — all train-size convergence curves on one axes

Top-level 2D ablation:
  ablation_2d_summary.json     — full 4×3 table
  heatmap_post_train_mrad.png  — 4 rows × 3 cols heatmap, colour = post-train mrad
  ablation_trainsize.png       — 4 lines (one/subset), x=train count, y=post-train mrad
  recovery_by_subset.png       — 3 panels: mean abs error per param group per subset
"""
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Per-stage convergence plots (full_field_200_samples experiment)
# ---------------------------------------------------------------------------

def plot_stage_convergence(
    history: list,
    output_dir: pathlib.Path,
    stage_name: str,
    loss_label: str,
    filename: str,
) -> None:
    """
    Plot train/val loss for a single training stage.

    Parameters
    ----------
    history    : list of epoch dicts from convergence_history_stage{1,2}.json
    output_dir : destination directory
    stage_name : human-readable name shown in the title (e.g. "Stage 1 — AlignmentLoss")
    loss_label : y-axis label (e.g. "AlignmentLoss (rad²)" or "FocalSpotLoss (m)")
    filename   : output filename without directory (e.g. "convergence_stage1.png")
    """
    if not history:
        return

    epochs       = [e["epoch"] for e in history]
    train_losses = [e["loss"]  for e in history]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("white")
    ax.plot(epochs, train_losses, color="steelblue", linewidth=1.5, label="Train loss")
    if any("eval_loss" in e for e in history):
        eval_losses = [e.get("eval_loss", float("nan")) for e in history]
        ax.plot(epochs, eval_losses, color="darkorange", linewidth=1.5,
                linestyle="--", label="Val loss")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(f"{loss_label} — log scale", fontsize=11)
    ax.set_title(stage_name, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-heliostat accuracy table (full_field_200_samples experiment)
# ---------------------------------------------------------------------------

def plot_per_heliostat_accuracy_table(
    results: dict,
    heliostat_ids: list,
    output_dir: pathlib.Path,
) -> None:
    """
    Save a compact table image with per-heliostat accuracy at stage 1 and stage 2 end.

    Columns: heliostat | post-stage1 (mrad) | post-stage2 (mrad) | delta (mrad)
    Rows are sorted by heliostat ID.
    Also writes ``per_heliostat_accuracy.json`` for downstream use.
    """
    stage1_ph = results.get("post_stage1",  {}).get("per_heliostat", {})
    stage2_ph = results.get("post_training", {}).get("per_heliostat", {})

    hids = sorted(set(list(stage1_ph.keys()) + list(stage2_ph.keys())))
    if not hids and heliostat_ids:
        hids = sorted(heliostat_ids)

    rows = []
    for hid in hids:
        s1 = stage1_ph.get(hid, {}).get("focal_spot_error_mrad")
        s2 = stage2_ph.get(hid, {}).get("focal_spot_error_mrad")
        delta = (s2 - s1) if (s1 is not None and s2 is not None) else None
        rows.append({
            "heliostat":          hid,
            "post_stage1_mrad":   round(s1, 4)     if s1    is not None else None,
            "post_stage2_mrad":   round(s2, 4)     if s2    is not None else None,
            "delta_mrad":         round(delta, 4)  if delta is not None else None,
        })

    with open(output_dir / "per_heliostat_accuracy.json", "w") as f:
        json.dump(rows, f, indent=2)

    if not rows:
        return

    col_headers = ["Heliostat", "Stage 1 (mrad)", "Stage 2 (mrad)", "Δ (mrad)"]
    cell_data = [
        [
            r["heliostat"],
            f"{r['post_stage1_mrad']:.3f}" if r["post_stage1_mrad"] is not None else "—",
            f"{r['post_stage2_mrad']:.3f}" if r["post_stage2_mrad"] is not None else "—",
            f"{r['delta_mrad']:+.3f}"      if r["delta_mrad"]       is not None else "—",
        ]
        for r in rows
    ]

    n_rows = len(cell_data)
    row_h_in    = 0.30
    header_h_in = 0.50
    fig_h = max(3.0, header_h_in + n_rows * row_h_in + 0.5)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    header_h_frac = header_h_in / fig_h
    row_h_frac    = row_h_in    / fig_h

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    for r in range(1, n_rows + 1):
        for c in range(len(col_headers)):
            tbl[r, c].set_height(row_h_frac)
    for c in range(len(col_headers)):
        tbl[0, c].set_height(header_h_frac)
        tbl[0, c].set_facecolor("#3a3a3a")
        tbl[0, c].set_text_props(color="white", fontweight="bold")

    for r in range(1, n_rows + 1):
        delta_str = cell_data[r - 1][3]
        try:
            delta_val = float(delta_str.replace("+", ""))
        except ValueError:
            delta_val = 0.0
        bg = "#d4edda" if delta_val < 0 else ("#fce8e8" if delta_val > 0 else "#f5f5f5")
        for c in range(len(col_headers)):
            tbl[r, c].set_facecolor(bg if c == 3 else ("#f5f5f5" if r % 2 == 0 else "white"))

    ax.set_title(
        "Per-heliostat accuracy: end of stage 1 vs. stage 2",
        fontsize=10, fontweight="bold", pad=6,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "per_heliostat_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_heliostat_accuracy_histogram(
    rows: list[dict],
    output_dir: pathlib.Path,
) -> None:
    """
    Plot a histogram of per-heliostat focal-spot accuracy after stage 2.

    Parameters
    ----------
    rows : list[dict]
        Each entry must have ``post_stage2_mrad`` (float or None).
    output_dir : pathlib.Path
        Directory where ``per_heliostat_accuracy_histogram.png`` is saved.
    """
    values = [r["post_stage2_mrad"] for r in rows if r.get("post_stage2_mrad") is not None]
    if not values:
        print("WARNING: No post_stage2_mrad values — skipping histogram.")
        return

    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    mean_val   = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val    = float(np.std(arr))
    n          = len(arr)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    ax.hist(arr, bins=30, color="steelblue", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axvline(mean_val,   color="crimson",    linestyle="--", linewidth=2.0,
               label=f"Mean:   {mean_val:.2f} mrad")
    ax.axvline(median_val, color="darkorange", linestyle="-.", linewidth=2.0,
               label=f"Median: {median_val:.2f} mrad")
    ax.axvspan(mean_val - std_val, mean_val + std_val,
               alpha=0.10, color="crimson", label=f"±1 std: {std_val:.2f} mrad")

    stats_text = (
        f"$n$ = {n}\n"
        f"mean = {mean_val:.2f} mrad\n"
        f"median = {median_val:.2f} mrad\n"
        f"std = {std_val:.2f} mrad\n"
        f"min = {arr.min():.2f} mrad\n"
        f"max = {arr.max():.2f} mrad"
    )
    ax.text(
        0.97, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="grey", alpha=0.85),
    )

    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(axis="y", color="grey", alpha=0.3, linewidth=0.6)
    ax.set_xlabel("Focal-spot error after stage 2 (mrad)", fontsize=10)
    ax.set_ylabel("Number of heliostats", fontsize=10)
    ax.set_title("Per-heliostat accuracy distribution (post stage 2)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / "per_heliostat_accuracy_histogram.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-heliostat accuracy histogram to {out_path}")


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------

def plot_convergence(
    history: list,
    output_dir: pathlib.Path,
    pre_perturbation_m: float | None = None,
    post_perturbation_m: float | None = None,
    post_training_m: float | None = None,
    pre_perturbation_mrad: float | None = None,
    post_perturbation_mrad: float | None = None,
    post_training_mrad: float | None = None,
) -> None:
    """
    Plot training/validation loss over epochs.

    Horizontal reference lines are drawn at the FocalSpotLoss (metres) values
    for the three evaluation checkpoints. mrad values are shown in the legend
    labels for readability.

    Pass ``*_m`` (metres) for the y-axis position and ``*_mrad`` for the label.
    If only ``*_mrad`` is supplied the lines are omitted (units mismatch).
    """
    if not history:
        return

    epochs       = [e["epoch"] for e in history]
    train_losses = [e["loss"]  for e in history]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")
    ax.plot(epochs, train_losses, color="steelblue", linewidth=1.5, label="Train loss")
    if any("eval_loss" in e for e in history):
        eval_losses = [e.get("eval_loss", float("nan")) for e in history]
        ax.plot(epochs, eval_losses, color="darkorange", linewidth=1.5,
                linestyle="--", label="Val loss")

    ref_lines = [
        (pre_perturbation_m,  pre_perturbation_mrad,  "limegreen",   "Pre-perturb  (clean→clean)"),
        (post_perturbation_m, post_perturbation_mrad, "firebrick",   "Post-perturb (perturbed→clean)"),
        (post_training_m,     post_training_mrad,     "mediumpurple","Post-train   (trained→clean)"),
    ]
    for val_m, val_mrad, color, label in ref_lines:
        if val_m is not None:
            mrad_str = f"  {val_mrad:.2f} mrad" if val_mrad is not None else ""
            ax.axhline(val_m, color=color, linewidth=1.2, linestyle=":",
                       alpha=0.85, label=f"{label}  ({val_m:.5f} m{mrad_str})")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("FocalSpotLoss (m) — log scale", fontsize=11)
    ax.set_title("5-Heliostat Synth→Synth — Convergence", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Parameter recovery bar charts
# ---------------------------------------------------------------------------

def plot_param_recovery(recovery: dict, output_dir: pathlib.Path) -> None:
    """
    One bar chart per parameter group showing applied perturbation vs residual deviation.

    For each group: blue bars = |applied perturbation|, orange bars = |abs_residual|
    (deviation from clean state after training).  Orange → 0 means perfect recovery.
    Frozen group (actuator_stroke) shows full perturbation in both bars.
    """
    heliostat_ids = list(recovery.keys())

    param_specs = [
        # (group_key, perturb_key, residual_key, ylabel,                            scale,  filename_suffix)
        ("rotation",        "perturbation_rad", "abs_residual_rad", "Rotation (mrad)",              1000.0, "rotation"),
        ("actuator_angle",  "perturbation_rad", "abs_residual_rad", "Actuator angle aᵢ (mrad)",     1000.0, "actuator_angle"),
        ("actuator_stroke", "perturbation_m",   "abs_residual_m",   "Actuator stroke bᵢ (mm) [frozen]", 1000.0, "actuator_stroke"),
        ("actuator_offset", "perturbation_m",   "abs_residual_m",   "Actuator offset cᵢ (mm)",      1000.0, "actuator_offset"),
        ("translation",     "perturbation_m",   "abs_residual_m",   "Translation (mm)",             1000.0, "translation"),
        ("base_position",   "perturbation_m",   "abs_residual_m",   "Base position (mm)",           1000.0, "base_position"),
    ]

    for group_key, perturb_key, residual_key, ylabel, scale, suffix in param_specs:
        labels, perturb_vals, residual_vals = [], [], []

        for hid in heliostat_ids:
            p = recovery[hid].get(group_key, {})
            for j, (t, r) in enumerate(zip(p.get(perturb_key, []), p.get(residual_key, []))):
                labels.append(f"{hid}[{j}]")
                perturb_vals.append(abs(t) * scale)
                residual_vals.append(r * scale)

        if not labels:
            continue

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 4))
        fig.patch.set_facecolor("white")
        ax.bar(x - width / 2, perturb_vals,  width, label="Applied perturbation", color="steelblue",  alpha=0.85)
        ax.bar(x + width / 2, residual_vals, width, label="Residual (deviation from clean)", color="darkorange", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Parameter recovery — {group_key}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.85)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(output_dir / f"recovery_{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Kinematic parameter evolution
# ---------------------------------------------------------------------------

def plot_kinematic_evolution(
    kinematic_history: list,
    perturbations_json: dict,
    heliostat_ids: list,
    output_dir: pathlib.Path,
) -> None:
    """
    One figure per heliostat showing how kinematic parameters evolved during training.

    Each figure has three subplots:
      Top    : rotation (4 params, mrad)  — ±5 mrad scale
      Middle : actuator_angle_deviation (2 params, mrad) + actuator_offset_deviation (2 params, mm)
               Note: actuator_angle is in mrad, actuator_offset is in mm (different units).
               Actuator_offset is not perturbed in this experiment so its true value is 0.
      Bottom : base_position (3 params, mm) — ±50 mm scale

    Dashed horizontal lines mark the true perturbation value for each parameter.
    """
    if not kinematic_history:
        return

    epochs = [e["epoch"] for e in kinematic_history]

    for hid in heliostat_ids:
        rots    = []   # [epoch, 4]
        acts    = []   # [epoch, 2]
        offsets = []   # [epoch, 2]
        bases   = []   # [epoch, 3]

        for e in kinematic_history:
            h = e.get("heliostats", {}).get(hid, {})
            rots.append(h.get("rotation_rad")               or [0.0] * 4)
            acts.append(h.get("actuator_angle_deviation_rad") or [0.0] * 2)
            offsets.append(h.get("actuator_offset_deviation_m") or [0.0] * 2)
            bases.append(h.get("base_position_m")           or [0.0] * 3)

        rots    = np.array(rots)    * 1000.0   # rad  → mrad
        acts    = np.array(acts)    * 1000.0   # rad  → mrad
        offsets = np.array(offsets) * 1000.0   # m    → mm
        bases   = np.array(bases)   * 1000.0   # m    → mm

        true_rot  = [v * 1000.0 for v in perturbations_json[hid]["rotation_rad"]]
        true_act  = [v * 1000.0 for v in perturbations_json[hid]["actuator_angle_rad"]]
        true_base = [v * 1000.0 for v in perturbations_json[hid]["base_position_m"]]
        # actuator_offset not perturbed → true = 0
        true_offset = [0.0, 0.0]

        rot_colors    = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        act_colors    = ["#9467bd", "#8c564b"]
        offset_colors = ["#17becf", "#bcbd22"]
        base_colors   = ["#e377c2", "#7f7f7f", "#bcbd22"]

        fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.patch.set_facecolor("white")
        fig.suptitle(f"Kinematic parameter evolution — {hid}", fontsize=13, fontweight="bold")

        # Top: rotation in mrad.
        for j in range(rots.shape[1]):
            ax_top.plot(epochs, rots[:, j], color=rot_colors[j], linewidth=1.2,
                        label=f"rot[{j}]")
            ax_top.axhline(true_rot[j], color=rot_colors[j], linewidth=0.9,
                           linestyle="--", alpha=0.6)
        ax_top.set_ylabel("Deviation (mrad)", fontsize=10)
        ax_top.set_title("Rotation  |  dashed = true perturbation", fontsize=9)
        ax_top.legend(fontsize=8, ncol=4, framealpha=0.85)
        ax_top.grid(alpha=0.3, linestyle="--", linewidth=0.7)
        ax_top.axhline(0, color="black", linewidth=0.5, linestyle="-")

        # Middle: actuator_angle (mrad) + actuator_offset (mm).
        for j in range(acts.shape[1]):
            ax_mid.plot(epochs, acts[:, j], color=act_colors[j], linewidth=1.2,
                        label=f"act_angle[{j}] (mrad)")
            ax_mid.axhline(true_act[j], color=act_colors[j], linewidth=0.9,
                           linestyle="--", alpha=0.6)
        for j in range(offsets.shape[1]):
            ax_mid.plot(epochs, offsets[:, j], color=offset_colors[j], linewidth=1.2,
                        linestyle="-.", label=f"act_offset[{j}] (mm)")
            ax_mid.axhline(true_offset[j], color=offset_colors[j], linewidth=0.9,
                           linestyle="--", alpha=0.6)
        ax_mid.set_ylabel("mrad / mm", fontsize=10)
        ax_mid.set_title(
            "Actuator angle deviation (mrad, solid) + actuator offset deviation (mm, dash-dot)"
            "  |  dashed = true", fontsize=9
        )
        ax_mid.legend(fontsize=8, ncol=2, framealpha=0.85)
        ax_mid.grid(alpha=0.3, linestyle="--", linewidth=0.7)
        ax_mid.axhline(0, color="black", linewidth=0.5, linestyle="-")

        # Bottom: base_position in mm.
        base_labels = ["east", "north", "up"]
        for j in range(bases.shape[1]):
            ax_bot.plot(epochs, bases[:, j], color=base_colors[j], linewidth=1.2,
                        label=f"base_{base_labels[j]}")
            ax_bot.axhline(true_base[j], color=base_colors[j], linewidth=0.9,
                           linestyle="--", alpha=0.6)
        ax_bot.set_xlabel("Epoch", fontsize=10)
        ax_bot.set_ylabel("Deviation (mm)", fontsize=10)
        ax_bot.set_title("Base position  |  dashed = true perturbation", fontsize=9)
        ax_bot.legend(fontsize=8, ncol=3, framealpha=0.85)
        ax_bot.grid(alpha=0.3, linestyle="--", linewidth=0.7)
        ax_bot.axhline(0, color="black", linewidth=0.5, linestyle="-")

        fig.tight_layout()
        fig.savefig(output_dir / f"kinematics_{hid}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(results: dict, perturbations_json: dict, output_dir: pathlib.Path) -> None:
    with open(output_dir / "summary.json", "w") as f:
        json.dump({"results": results, "perturbations": perturbations_json}, f, indent=2)

    pre  = results.get("pre_perturbation", {})
    post = results.get("post_perturbation", {})
    st1  = results.get("post_stage1", {})
    trn  = results.get("post_training", {})

    W = 70
    print(f"\n{'=' * W}")
    print(f"  {'Checkpoint':<36} {'Mean (mrad)':>11}  {'Median (mrad)':>13}  n")
    print(f"  {'-' * 36} {'-' * 11}  {'-' * 13}  -")
    print(f"  {'Pre-perturbation  (clean→clean)':<36} {pre.get('mean_mrad', float('nan')):>11.3f}  {pre.get('median_mrad', float('nan')):>13.3f}  {pre.get('num_samples', '?')}")
    print(f"  {'Post-perturbation (perturbed→clean)':<36} {post.get('mean_mrad', float('nan')):>11.3f}  {post.get('median_mrad', float('nan')):>13.3f}  {post.get('num_samples', '?')}")
    if st1:
        print(f"  {'Post-stage1 (alignment, best-val)':<36} {st1.get('mean_mrad', float('nan')):>11.3f}  {st1.get('median_mrad', float('nan')):>13.3f}  {st1.get('num_samples', '?')}")
    print(f"  {'Post-stage2 / post-training':<36} {trn.get('mean_mrad', float('nan')):>11.3f}  {trn.get('median_mrad', float('nan')):>13.3f}  {trn.get('num_samples', '?')}")
    print(f"  Min / Max (post-training)          : {trn.get('min_mrad', float('nan')):.3f} / {trn.get('max_mrad', float('nan')):.3f} mrad")
    print(f"  Training time                      : {results.get('train_time_min', 0):.1f} min")
    if trn.get('num_nan_samples'):
        print(f"  NaN samples (post-training)        : {trn['num_nan_samples']}  ids={trn.get('nan_heliostat_ids', [])}")
    print(f"{'=' * W}\n")


# ---------------------------------------------------------------------------
# Combined convergence (all ablation runs on one plot)
# ---------------------------------------------------------------------------

def plot_combined_convergence(
    all_histories: dict,
    ablation_results: dict,
    output_dir: pathlib.Path,
) -> None:
    """
    Plot train + val loss curves for every ablation run on a single log-scale axes.

    Convention
    ----------
    - Solid line  = train loss
    - Dashed line = val loss
    - One colour per training-size run (tab10 palette)
    - Pre/post-perturbation shown as horizontal reference lines
    - Post-training final values shown as right-margin text annotations
      (not legend entries, since they sit at near-zero and clutter the legend)
    """
    # Use the first N colours from matplotlib's default tab10 cycle.
    tab10 = plt.cm.tab10.colors
    run_keys = list(all_histories.keys())
    colors = [tab10[i % 10] for i in range(len(run_keys))]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    for color, run_key in zip(colors, run_keys):
        history = all_histories[run_key]
        if not history:
            continue
        n = run_key.replace("train_", "")
        epochs       = [e["epoch"] for e in history]
        train_losses = [e["loss"]  for e in history]
        ax.plot(epochs, train_losses, color=color, linewidth=1.6,
                linestyle="-", label=f"N={n} train")
        if any("eval_loss" in e for e in history):
            eval_losses = [e.get("eval_loss", float("nan")) for e in history]
            ax.plot(epochs, eval_losses, color=color, linewidth=1.4,
                    linestyle="--", alpha=0.75, label=f"N={n} val")

    # Shared reference lines (pre/post perturbation) — identical across runs.
    first_res = next(iter(ablation_results.values()))
    pre_m     = first_res.get("pre_perturbation",  {}).get("mean_m")
    post_m    = first_res.get("post_perturbation", {}).get("mean_m")
    pre_mrad  = first_res.get("pre_perturbation",  {}).get("mean_mrad")
    post_mrad = first_res.get("post_perturbation", {}).get("mean_mrad")

    if pre_m is not None:
        mrad_str = f" = {pre_mrad:.2f} mrad" if pre_mrad is not None else ""
        ax.axhline(pre_m,  color="#2ca02c", linewidth=1.2, linestyle=":",
                   alpha=0.9, label=f"Pre-perturbation  ({pre_m:.2e} m{mrad_str})")
    if post_m is not None:
        mrad_str = f" = {post_mrad:.1f} mrad" if post_mrad is not None else ""
        ax.axhline(post_m, color="#d62728", linewidth=1.2, linestyle=":",
                   alpha=0.9, label=f"Post-perturbation ({post_m:.2e} m{mrad_str})")

    # Per-run post-training horizontal lines (same colour as the run, dash-dot).
    for color, run_key in zip(colors, run_keys):
        res = ablation_results.get(run_key, {})
        pt_m    = res.get("post_training", {}).get("mean_m")
        pt_mrad = res.get("post_training", {}).get("mean_mrad")
        n = run_key.replace("train_", "")
        if pt_m is not None:
            mrad_str = f" = {pt_mrad:.3f} mrad" if pt_mrad is not None else ""
            ax.axhline(pt_m, color=color, linewidth=1.0, linestyle="-.",
                       alpha=0.7, label=f"N={n} post-train ({pt_m:.2e} m{mrad_str})")

    # Post-training final values: text annotations stacked on the right margin.
    # Sort by value so labels don't collide, then offset them vertically if needed.
    x_max = max(
        (e["epoch"] for h in all_histories.values() for e in h),
        default=300,
    )
    pt_annotations = []
    for color, run_key in zip(colors, run_keys):
        res = ablation_results.get(run_key, {})
        pt_mrad = res.get("post_training", {}).get("mean_mrad")
        pt_m    = res.get("post_training", {}).get("mean_m")
        n = run_key.replace("train_", "")
        if pt_m is not None:
            pt_annotations.append((pt_m, pt_mrad, n, color))

    # Sort ascending by m value then spread labels so they don't overlap.
    pt_annotations.sort(key=lambda t: t[0])
    x_ann = x_max * 1.02
    prev_y = None
    min_sep = 0.15  # minimum log-space separation between labels
    for pt_m, pt_mrad, n, color in pt_annotations:
        import math
        y_log = math.log10(pt_m) if pt_m > 0 else -6
        if prev_y is not None and abs(y_log - prev_y) < min_sep:
            y_log = prev_y + min_sep
        prev_y = y_log
        y_display = 10 ** y_log
        mrad_str = f" ({pt_mrad:.3f} mrad)" if pt_mrad is not None else ""
        ax.annotate(
            f"N={n}: {pt_m:.2e} m{mrad_str}",
            xy=(x_max, pt_m),
            xytext=(x_ann, y_display),
            color=color, fontsize=7.5, va="center",
            annotation_clip=False,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("FocalSpotLoss — log scale (m)", fontsize=11)
    ax.set_title("5-Heliostat Synth — Convergence by training data size",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7, which="both")
    fig.tight_layout()
    fig.savefig(output_dir / "combined_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ablation comparison
# ---------------------------------------------------------------------------

def write_ablation_summary(ablation_results: dict, output_dir: pathlib.Path) -> None:
    """Save ablation summary across training-data-size runs."""
    summary = {}
    for run_key, res in ablation_results.items():
        summary[run_key] = {
            "pre_perturbation_mrad":  res.get("pre_perturbation", {}).get("mean_mrad"),
            "post_perturbation_mrad": res.get("post_perturbation", {}).get("mean_mrad"),
            "post_training_mrad":     res.get("post_training", {}).get("mean_mrad"),
            "post_training_median_mrad": res.get("post_training", {}).get("median_mrad"),
            "train_time_min":         res.get("train_time_min"),
        }
    with open(output_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  {'Run':<12} {'Pre':>10} {'Post-pert':>12} {'Post-train':>12} {'Time':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*12} {'-'*8}")
    for run_key, s in summary.items():
        print(
            f"  {run_key:<12} "
            f"{s['pre_perturbation_mrad']:>9.3f}m "
            f"{s['post_perturbation_mrad']:>11.3f}m "
            f"{s['post_training_mrad']:>11.3f}m "
            f"{s['train_time_min']:>7.1f}'"
        )
    print(f"{'=' * 70}\n")


def plot_ablation_comparison(ablation_results: dict, output_dir: pathlib.Path) -> None:
    """
    Render a summary table comparing all evaluation checkpoints across ablation runs.

    Replaces the bar chart (which is unreadable when pre-perturbation ~0 mrad sits
    beside post-perturbation ~8 mrad).  Saves as ``ablation_comparison.png``.
    """
    run_keys = list(ablation_results.keys())

    col_headers = [
        "Run", "Pre-perturb\n(mrad)", "Post-perturb\n(mrad)",
        "Post-train mean\n(mrad)", "Post-train median\n(mrad)", "Train time\n(min)"
    ]
    rows = []
    for rk in run_keys:
        res = ablation_results[rk]
        rows.append([
            rk,
            f"{res.get('pre_perturbation',  {}).get('mean_mrad', float('nan')):.4f}",
            f"{res.get('post_perturbation', {}).get('mean_mrad', float('nan')):.3f}",
            f"{res.get('post_training',     {}).get('mean_mrad', float('nan')):.4f}",
            f"{res.get('post_training',     {}).get('median_mrad', float('nan')):.4f}",
            f"{res.get('train_time_min', float('nan')):.1f}",
        ])

    n_rows = len(rows)
    n_cols = len(col_headers)

    # Row height and header height in inches; figure height derived from them.
    row_h_in   = 0.45
    header_h_in = 0.70   # tall enough for two-line header text
    fig_h = max(2.5, header_h_in + n_rows * row_h_in + 0.8)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    # Convert inch heights to axes-fraction heights.
    header_h_frac = header_h_in / fig_h
    row_h_frac    = row_h_in    / fig_h

    tbl = ax.table(
        cellText=rows,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    # Set every data row to a uniform height.
    for r in range(1, n_rows + 1):
        for c in range(n_cols):
            tbl[r, c].set_height(row_h_frac)

    # Header row: taller to accommodate two-line labels.
    for c in range(n_cols):
        tbl[0, c].set_height(header_h_frac)

    # Style header row.
    for c in range(n_cols):
        cell = tbl[0, c]
        cell.set_facecolor("#3a3a3a")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10,
                            va="center")

    # Alternating row shading.
    for r in range(1, n_rows + 1):
        bg = "#f5f5f5" if r % 2 == 0 else "white"
        for c in range(n_cols):
            tbl[r, c].set_facecolor(bg)

    ax.set_title("Training data ablation — evaluation summary",
                 fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2D ablation: subset × train_size
# ---------------------------------------------------------------------------

def write_ablation_2d_summary(
    results_2d: dict,
    subset_labels: dict,
    output_dir: pathlib.Path,
) -> None:
    """
    Save a full 2D summary table (JSON + printed table).

    Parameters
    ----------
    results_2d     : {subset_name: {train_key: results_dict}}
    subset_labels  : {subset_name: human_readable_label}
    output_dir     : destination directory
    """
    summary = {}
    for sname, train_results in results_2d.items():
        summary[sname] = {
            "label": subset_labels.get(sname, sname),
            "runs": {},
        }
        for train_key, res in train_results.items():
            summary[sname]["runs"][train_key] = {
                "pre_perturbation_mrad":     res.get("pre_perturbation",  {}).get("mean_mrad"),
                "post_perturbation_mrad":    res.get("post_perturbation", {}).get("mean_mrad"),
                "post_training_mrad":        res.get("post_training",     {}).get("mean_mrad"),
                "post_training_median_mrad": res.get("post_training",     {}).get("median_mrad"),
                "train_time_min":            res.get("train_time_min"),
            }

    with open(output_dir / "ablation_2d_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print table.
    train_keys = sorted({k for v in results_2d.values() for k in v})
    col_w = 11
    header = f"  {'Subset':<30}" + "".join(f"  {k:>{col_w}}" for k in train_keys)
    print(f"\n{'=' * len(header)}")
    print("  Post-training mean focal-spot error (mrad)")
    print(header)
    print(f"  {'-' * 30}" + "".join(f"  {'-' * col_w}" for _ in train_keys))
    for sname, data in summary.items():
        row = f"  {data['label']:<30}"
        for k in train_keys:
            val = data["runs"].get(k, {}).get("post_training_mrad")
            row += f"  {val:>{col_w}.3f}" if val is not None else f"  {'N/A':>{col_w}}"
        print(row)
    print(f"{'=' * len(header)}\n")


def plot_heatmap_post_train(
    results_2d: dict,
    subset_labels: dict,
    train_counts: list,
    output_dir: pathlib.Path,
) -> None:
    """
    4-row × N-col heatmap.  Rows = parameter subsets, columns = training sizes.
    Cell colour and annotation = post-training mean mrad.
    """
    subset_names = [s for s, _ in subset_labels.items() if s in results_2d]
    # Preserve insertion order from PARAM_SUBSETS (passed as dict keeps order in Python 3.7+)
    row_labels = [subset_labels[s] for s in subset_names]
    col_labels = [f"train_{n}" for n in train_counts]

    data = np.full((len(subset_names), len(train_counts)), np.nan)
    for r, sname in enumerate(subset_names):
        for c, n_train in enumerate(train_counts):
            val = (
                results_2d.get(sname, {})
                .get(f"train_{n_train}", {})
                .get("post_training", {})
                .get("mean_mrad")
            )
            if val is not None:
                data[r, c] = val

    fig, ax = plt.subplots(figsize=(max(5, len(train_counts) * 2.2), max(4, len(subset_names) * 1.3)))
    fig.patch.set_facecolor("white")
    vmax = np.nanmax(data) if not np.all(np.isnan(data)) else 1.0
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Post-training mean error (mrad)", shrink=0.8)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title("Post-training focal-spot error — parameter subset × training size",
                 fontsize=11, fontweight="bold")

    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            if not np.isnan(data[r, c]):
                ax.text(c, r, f"{data[r, c]:.3f}", ha="center", va="center",
                        fontsize=9, color="black", fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_dir / "heatmap_post_train_mrad.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_trainsize(
    results_2d: dict,
    subset_labels: dict,
    train_counts: list,
    output_dir: pathlib.Path,
) -> None:
    """
    Line plot: x = training data size, y = post-training mean mrad.
    One line per parameter subset.
    """
    subset_names = [s for s, _ in subset_labels.items() if s in results_2d]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")

    for idx, sname in enumerate(subset_names):
        vals = []
        for n in train_counts:
            val = (
                results_2d[sname]
                .get(f"train_{n}", {})
                .get("post_training", {})
                .get("mean_mrad")
            )
            vals.append(val if val is not None else float("nan"))
        ax.plot(train_counts, vals, marker="o", linewidth=1.8,
                color=colors[idx % len(colors)], label=subset_labels[sname])

    ax.set_xlabel("Training samples per heliostat", fontsize=11)
    ax.set_ylabel("Post-training mean error (mrad)", fontsize=11)
    ax.set_title("Effect of training data size per parameter subset", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xticks(train_counts)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_trainsize.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_recovery_by_subset(
    results_2d: dict,
    subset_labels: dict,
    perturbations_json: dict,
    output_dir: pathlib.Path,
    train_key: str = "train_50",
) -> None:
    """
    Three-panel bar chart: one panel per perturbed parameter group.

    Each panel shows 4 bars (one per subset) with mean absolute recovery error.
    A horizontal dashed line marks the mean absolute perturbation magnitude
    (= error if nothing is recovered).

    Uses the largest training-size run (``train_key``) for the comparison.
    Subsets that do not optimise a parameter group will show approximately
    the full perturbation magnitude as their error.
    """
    subset_names = [s for s, _ in subset_labels.items() if s in results_2d]

    # If the requested train_key is missing, fall back to any available key.
    for sname in subset_names:
        if train_key not in results_2d[sname]:
            available = list(results_2d[sname].keys())
            if available:
                train_key = sorted(available)[-1]
            break

    param_specs = [
        # (group_key, perturb_json_key, residual_key, ylabel,                 scale,  panel_title)
        ("rotation",       "rotation_rad",       "abs_residual_rad", "Mean residual (mrad)",  1000.0, "Rotation (4 params)"),
        ("actuator_angle", "actuator_angle_rad", "abs_residual_rad", "Mean residual (mrad)",  1000.0, "Actuator angle aᵢ (2 params)"),
        ("base_position",  "base_position_m",    "abs_residual_m",   "Mean residual (mm)",    1000.0, "Base position (3 params)"),
    ]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Parameter recovery by subset  (training size: {train_key.replace('_', ' ')})",
        fontsize=12, fontweight="bold",
    )

    for ax, (group_key, perturb_key, abs_err_key, ylabel, scale, panel_title) in zip(axes, param_specs):
        # Mean |true| perturbation across all heliostats and param indices.
        all_true = [abs(v) * scale
                    for hid in perturbations_json
                    for v in perturbations_json[hid][perturb_key]]
        true_mag = float(np.mean(all_true)) if all_true else 0.0

        bar_vals = []
        for sname in subset_names:
            recovery = results_2d[sname].get(train_key, {}).get("param_recovery")
            if recovery is None:
                bar_vals.append(float("nan"))
            else:
                errors = [e * scale
                          for hdata in recovery.values()
                          for e in hdata.get(group_key, {}).get(abs_err_key, [])]
                bar_vals.append(float(np.mean(errors)) if errors else float("nan"))

        bar_colors = [colors[i % len(colors)] for i in range(len(subset_names))]
        bar_labels = [subset_labels[s] for s in subset_names]

        x = np.arange(len(subset_names))
        rects = ax.bar(x, bar_vals, color=bar_colors, alpha=0.85, width=0.6)
        ax.axhline(true_mag, color="firebrick", linewidth=1.4, linestyle="--",
                   alpha=0.8, label=f"Perturbation  ({true_mag:.2f})")
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(panel_title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.85)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        for rect, val in zip(rects, bar_vals):
            if not np.isnan(val):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + true_mag * 0.02,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / "recovery_by_subset.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Kinematic stages (presentation plot)
# ---------------------------------------------------------------------------

def plot_kinematic_stages(
    kinematic_stages: dict,
    heliostat_ids: list,
    output_dir: pathlib.Path,
) -> None:
    """
    One figure per heliostat: grouped bar chart showing parameter values at 3 stages.

    Three bar groups per parameter index: clean (blue), perturbed (red), trained (green).
    Covers 6 parameter groups arranged in a 3×2 subplot grid.
    """
    stage_colors = {"clean": "steelblue", "perturbed": "firebrick", "trained": "seagreen"}
    stage_labels = {"clean": "Clean (pre-perturb)", "perturbed": "Perturbed", "trained": "Trained (recovered)"}

    param_groups = [
        ("rotation_rad",       "Rotation (mrad)",         1000.0),
        ("actuator_angle_rad", "Actuator angle aᵢ (mrad)", 1000.0),
        ("actuator_stroke_m",  "Actuator stroke bᵢ (mm) [frozen]", 1000.0),
        ("actuator_offset_m",  "Actuator offset cᵢ (mm)", 1000.0),
        ("translation_m",      "Translation (mm)",        1000.0),
        ("base_position_m",    "Base position (mm)",      1000.0),
    ]

    for hid in heliostat_ids:
        fig, axes = plt.subplots(3, 2, figsize=(13, 10))
        fig.patch.set_facecolor("white")
        fig.suptitle(f"Kinematic parameter stages — {hid}", fontsize=13, fontweight="bold")

        for ax, (param_key, ylabel, scale) in zip(axes.flat, param_groups):
            width = 0.22
            stages = ["clean", "perturbed", "trained"]
            n_params = None
            for stage in stages:
                vals = kinematic_stages.get(stage, {}).get(hid, {}).get(param_key, [])
                if vals:
                    n_params = len(vals)
                    break
            if n_params is None:
                ax.set_visible(False)
                continue

            x = np.arange(n_params)
            for k, stage in enumerate(stages):
                vals = kinematic_stages.get(stage, {}).get(hid, {}).get(param_key, [0.0] * n_params)
                ax.bar(x + (k - 1) * width, [v * scale for v in vals],
                       width, label=stage_labels[stage], color=stage_colors[stage], alpha=0.80)

            ax.axhline(0, color="black", linewidth=0.6, linestyle="-")
            ax.set_xticks(x)
            ax.set_xticklabels([f"[{i}]" for i in range(n_params)], fontsize=8)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(param_key.replace("_", " "), fontsize=9, fontweight="bold")
            ax.legend(fontsize=7, framealpha=0.85, ncol=3)
            ax.grid(axis="y", alpha=0.3, linestyle="--")

        fig.tight_layout()
        fig.savefig(output_dir / f"kinematic_stages_{hid}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Flux comparison (presentation plot)
# ---------------------------------------------------------------------------

def plot_flux_comparison(
    measured: "np.ndarray",
    predicted: "np.ndarray",
    pixel_loss: float,
    fse_mrad: float,
    heliostat_id: str,
    output_dir: pathlib.Path,
) -> None:
    """
    Save a side-by-side flux comparison: measured | predicted | absolute difference.

    Both images are assumed to be peak-normalised [0, 1] before calling.
    """
    diff = np.abs(predicted - measured)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("white")

    titles   = ["Measured (dataset)", "Predicted (trained scenario)", "Absolute difference"]
    images   = [measured, predicted, diff]
    cmaps    = ["inferno", "inferno", "hot"]

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if "diff" not in title.lower() else None)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fse_str = f"{fse_mrad:.3f} mrad" if not (fse_mrad != fse_mrad) else "NaN"
    fig.suptitle(
        f"Flux comparison — {heliostat_id}   |   "
        f"FSE = {fse_str}   |   pixel L1 = {pixel_loss:.2f}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"flux_comparison_{heliostat_id}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
