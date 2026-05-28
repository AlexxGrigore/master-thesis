"""
Reporting utilities shared across all KR experiments.

Functions extracted from the old five_heliostats_synth/reporting.py — only
the ones actively used by full_63_heli_kin_reconstruct and one_heliostat_train_sizes.
"""
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def plot_stage_convergence(
    history: list,
    output_dir: pathlib.Path,
    stage_name: str,
    loss_label: str,
    filename: str,
) -> None:
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
    ax.set_title("Convergence", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_param_recovery(recovery: dict, output_dir: pathlib.Path) -> None:
    heliostat_ids = list(recovery.keys())

    param_specs = [
        ("rotation",        "perturbation_rad", "abs_residual_rad", "Rotation (mrad)",                   1000.0, "rotation"),
        ("actuator_angle",  "perturbation_rad", "abs_residual_rad", "Actuator angle aᵢ (mrad)",          1000.0, "actuator_angle"),
        ("actuator_stroke", "perturbation_m",   "abs_residual_m",   "Actuator stroke bᵢ (mm) [frozen]",  1000.0, "actuator_stroke"),
        ("actuator_offset", "perturbation_m",   "abs_residual_m",   "Actuator offset cᵢ (mm)",           1000.0, "actuator_offset"),
        ("translation",     "perturbation_m",   "abs_residual_m",   "Translation (mm)",                  1000.0, "translation"),
        ("base_position",   "perturbation_m",   "abs_residual_m",   "Base position (mm)",                1000.0, "base_position"),
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
        ax.bar(x - width / 2, perturb_vals,  width, label="Applied perturbation",         color="steelblue",  alpha=0.85)
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


def plot_per_heliostat_accuracy_table(
    results: dict,
    heliostat_ids: list,
    output_dir: pathlib.Path,
) -> None:
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
            "heliostat":        hid,
            "post_stage1_mrad": round(s1,    4) if s1    is not None else None,
            "post_stage2_mrad": round(s2,    4) if s2    is not None else None,
            "delta_mrad":       round(delta, 4) if delta is not None else None,
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
    row_h_in = 0.30
    header_h_in = 0.50
    fig_h = max(3.0, header_h_in + n_rows * row_h_in + 0.5)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    header_h_frac = header_h_in / fig_h
    row_h_frac    = row_h_in    / fig_h

    tbl = ax.table(cellText=cell_data, colLabels=col_headers, cellLoc="center", loc="center")
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

    ax.set_title("Per-heliostat accuracy: end of stage 1 vs. stage 2",
                 fontsize=10, fontweight="bold", pad=6)
    fig.tight_layout()
    fig.savefig(output_dir / "per_heliostat_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_heliostat_accuracy_histogram(rows: list[dict], output_dir: pathlib.Path) -> None:
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
    ax.axvline(mean_val,   color="crimson",    linestyle="--", linewidth=2.0, label=f"Mean:   {mean_val:.2f} mrad")
    ax.axvline(median_val, color="darkorange", linestyle="-.", linewidth=2.0, label=f"Median: {median_val:.2f} mrad")
    ax.axvspan(mean_val - std_val, mean_val + std_val, alpha=0.10, color="crimson", label=f"±1 std: {std_val:.2f} mrad")

    stats_text = (
        f"$n$ = {n}\nmean = {mean_val:.2f} mrad\nmedian = {median_val:.2f} mrad\n"
        f"std = {std_val:.2f} mrad\nmin = {arr.min():.2f} mrad\nmax = {arr.max():.2f} mrad"
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="grey", alpha=0.85))

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


def _ph_mean_median(section: dict) -> tuple[float, float]:
    """Return (mean, median) of per-heliostat focal-spot errors from an eval section."""
    ph = section.get("per_heliostat", {})
    vals = [
        v.get("focal_spot_error_mrad")
        for v in ph.values()
        if isinstance(v, dict) and v.get("focal_spot_error_mrad") is not None
    ]
    if not vals:
        return float("nan"), float("nan")
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.median(arr))


def write_summary(results: dict, perturbations_json: dict, output_dir: pathlib.Path) -> None:
    with open(output_dir / "summary.json", "w") as f:
        json.dump({"results": results, "perturbations": perturbations_json}, f, indent=2)

    st1  = results.get("post_stage1")  or {}
    trn  = results.get("post_training") or {}
    pre  = results.get("pre_training")  or {}

    trn_ph_mean, trn_ph_median = _ph_mean_median(trn)

    W = 72
    print(f"\n{'=' * W}")
    print(f"  {'Checkpoint':<36} {'Mean/hel':>9}  {'Med/hel':>9}  n")
    print(f"  {'-' * 36} {'-' * 9}  {'-' * 9}  -")
    if pre:
        pre_ph_mean, pre_ph_median = _ph_mean_median(pre)
        print(f"  {'Pre-training (clean scenario)':<36} {pre_ph_mean:>9.3f}  {pre_ph_median:>9.3f}  {pre.get('num_samples', '?')}")
    if st1:
        st1_ph_mean, st1_ph_median = _ph_mean_median(st1)
        print(f"  {'Post-stage1 (alignment, best-val)':<36} {st1_ph_mean:>9.3f}  {st1_ph_median:>9.3f}  {st1.get('num_samples', '?')}")
    if trn:
        print(f"  {'Post-training (stage-2 best-val)':<36} {trn_ph_mean:>9.3f}  {trn_ph_median:>9.3f}  {trn.get('num_samples', '?')}")
        print(f"  Min / Max (post-training)          : {trn.get('min_mrad', float('nan')):.3f} / {trn.get('max_mrad', float('nan')):.3f} mrad")
        if trn.get("num_nan_samples"):
            print(f"  NaN samples (post-training)        : {trn['num_nan_samples']}  ids={trn.get('nan_heliostat_ids', [])}")
    else:
        print(f"  {'Post-training (stage-2 best-val)':<36} {'skipped':>9}  {'skipped':>9}  —")
    print(f"  Training time                      : {results.get('train_time_min', 0):.1f} min")
    print(f"{'=' * W}\n")


def plot_flux_comparison(
    measured: "np.ndarray",
    predicted: "np.ndarray",
    pixel_loss: float,
    fse_mrad: float,
    heliostat_id: str,
    output_dir: pathlib.Path,
) -> None:
    diff = np.abs(predicted - measured)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("white")

    for ax, img, title, cmap in zip(
        axes,
        [measured, predicted, diff],
        ["Measured (dataset)", "Predicted (trained scenario)", "Absolute difference"],
        ["inferno", "inferno", "hot"],
    ):
        im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if "diff" not in title.lower() else None)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fse_str = f"{fse_mrad:.3f} mrad" if fse_mrad == fse_mrad else "NaN"
    fig.suptitle(
        f"Flux comparison — {heliostat_id}   |   FSE = {fse_str}   |   pixel L1 = {pixel_loss:.2f}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"flux_comparison_{heliostat_id}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
