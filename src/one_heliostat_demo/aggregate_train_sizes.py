"""
Aggregate train-size sensitivity results from run_all_train_sizes.py.

Produces:
  - mrad_vs_train_size.png      — one line per heliostat (after_stage2 mrad vs train size)
  - accuracy_mean_std.png       — field mean ± std mrad vs train size (shaded band)
  - loss_curves.png             — stage1 + stage2 train/val loss mean ± std across heliostats,
                                  one colored line per train size
  - comparison_table.txt        — ASCII table: heliostat × train size → mrad

Can be re-run standalone on any existing output directory.

Usage
-----
    python aggregate_train_sizes.py --output-dir outputs/one_hel_demo_train_sizes_<ts>
    python aggregate_train_sizes.py --dirs outputs/.../AC36 outputs/.../BE35 --out outputs/cmp/
"""

import argparse
import csv
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_KNOWN_DISTANCES = {
    "AC36": 34,
    "AG33": 54,
    "AO34": 90,
    "AW36": 139,
    "BE35": 210,
}

COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_summary(hid_dir: pathlib.Path) -> dict:
    path = hid_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"No summary.json in {hid_dir}")
    with open(path) as f:
        s = json.load(f)
    s["_hid_dir"] = hid_dir   # injected for convergence CSV lookups
    return s


def _load_convergence(csv_path: pathlib.Path) -> dict:
    """
    Read convergence_history.csv and return stage1/stage2 loss arrays.

    Returns
    -------
    dict with keys "stage1" and "stage2", each holding:
        {"train": list[float | None], "val": list[float | None]}
    Epochs are in order; None values indicate missing/NaN entries.
    """
    out = {"stage1": {"train": [], "val": []}, "stage2": {"train": [], "val": []}}
    if not csv_path.exists():
        return out
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stage = row.get("stage", "")
            if stage not in ("stage1", "stage2"):
                continue
            def _float(v):
                try:
                    x = float(v)
                    return None if np.isnan(x) else x
                except (TypeError, ValueError):
                    return None
            out[stage]["train"].append(_float(row.get("train_loss")))
            out[stage]["val"].append(_float(row.get("val_loss")))
    return out


def _collect(output_dir: pathlib.Path, heliostat_ids: list[str] | None) -> list[dict]:
    if heliostat_ids is not None:
        dirs = [output_dir / hid for hid in heliostat_ids]
    else:
        dirs = [d for d in sorted(output_dir.iterdir())
                if d.is_dir() and (d / "summary.json").exists()]
    summaries = []
    for d in dirs:
        try:
            summaries.append(_load_summary(d))
        except FileNotFoundError as e:
            print(f"[WARNING] {e}")
    summaries.sort(key=lambda s: _KNOWN_DISTANCES.get(s["heliostat_id"], 999))
    return summaries


# ---------------------------------------------------------------------------
# Plot 1 — one line per heliostat
# ---------------------------------------------------------------------------

def _split_label(summaries: list[dict]) -> str:
    types = {s.get("split_type", "balanced") for s in summaries}
    return "/".join(sorted(types))


def _plot_per_heliostat(summaries: list[dict], out_dir: pathlib.Path) -> None:
    split = _split_label(summaries)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white")

    for i, s in enumerate(summaries):
        hid   = s["heliostat_id"]
        dist  = _KNOWN_DISTANCES.get(hid, "?")
        sizes = [n for n in s["train_sizes"] if str(n) in s["results"]]
        mrad  = [s["results"][str(n)]["after_stage2"]["mrad_mean"] for n in sizes]

        color  = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        ax.plot(sizes, mrad, marker=marker, color=color, linewidth=2,
                label=f"{hid}  ({dist} m)")

        first_key = str(sizes[0])
        pre_mrad  = s["results"][first_key]["pre_training"]["mrad_mean"]
        ax.axhline(pre_mrad, color=color, linestyle=":", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Training samples", fontsize=12)
    ax.set_ylabel("After-stage-2 mean FSE (mrad)", fontsize=12)
    ax.set_title(f"Train-size sensitivity — per heliostat  [{split} split]", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    out = out_dir / "mrad_vs_train_size.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 — field mean ± std vs train size
# ---------------------------------------------------------------------------

def _plot_accuracy_mean_std(summaries: list[dict], out_dir: pathlib.Path) -> None:
    split = _split_label(summaries)
    all_sizes = sorted({n for s in summaries for n in s["train_sizes"]})

    post_by_size: dict[int, list[float]] = {n: [] for n in all_sizes}
    pre_by_size:  dict[int, list[float]] = {n: [] for n in all_sizes}

    for s in summaries:
        for n in all_sizes:
            key = str(n)
            if key in s["results"]:
                post_by_size[n].append(s["results"][key]["after_stage2"]["mrad_mean"])
                pre_by_size[n].append(s["results"][key]["pre_training"]["mrad_mean"])

    sizes      = [n for n in all_sizes if post_by_size[n]]
    post_mean  = np.array([np.mean(post_by_size[n]) for n in sizes])
    post_std   = np.array([np.std(post_by_size[n])  for n in sizes])
    pre_mean   = np.array([np.mean(pre_by_size[n])  for n in sizes])
    pre_std    = np.array([np.std(pre_by_size[n])   for n in sizes])

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    ax.plot(sizes, post_mean, marker="o", color="steelblue", linewidth=2,
            label="after stage 2 (mean)")
    ax.fill_between(sizes, post_mean - post_std, post_mean + post_std,
                    color="steelblue", alpha=0.2, label="±1 std")

    ax.plot(sizes, pre_mean, marker="s", color="tomato", linewidth=1.5,
            linestyle="--", label="pre-training (mean)")
    ax.fill_between(sizes, pre_mean - pre_std, pre_mean + pre_std,
                    color="tomato", alpha=0.15)

    ax.set_xlabel("Training samples", fontsize=12)
    ax.set_ylabel("Mean FSE (mrad)", fontsize=12)
    ax.set_title(
        f"Field accuracy vs training samples  [{split} split]  (n={len(summaries)} heliostats)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    out = out_dir / "accuracy_mean_std.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3 — loss curves (train + val, mean ± std, one line per train size)
# ---------------------------------------------------------------------------

def _plot_loss_curves(summaries: list[dict], out_dir: pathlib.Path) -> None:
    all_sizes = sorted({n for s in summaries for n in s["train_sizes"]
                        if str(n) in s["results"]})

    cmap   = matplotlib.colormaps["plasma"].resampled(len(all_sizes))
    colors = [cmap(i) for i in range(len(all_sizes))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")
    stage_cfg = [
        ("stage1", axes[0], "Stage 1 — AlignmentLoss"),
        ("stage2", axes[1], "Stage 2 — FocalSpotLoss"),
    ]

    for si, (stage_key, ax, title) in enumerate(stage_cfg):
        has_val = False

        for ci, n in enumerate(all_sizes):
            # Collect loss arrays from all heliostats for this train size.
            train_lists: list[list[float]] = []
            val_lists:   list[list[float]] = []

            for s in summaries:
                if str(n) not in s["results"]:
                    continue
                hid_dir  = pathlib.Path(s["_hid_dir"])
                csv_path = hid_dir / f"train_size_{n}" / "convergence_history.csv"
                conv     = _load_convergence(csv_path)
                t_vals   = conv[stage_key]["train"]
                v_vals   = conv[stage_key]["val"]
                if t_vals:
                    train_lists.append([v if v is not None else float("nan") for v in t_vals])
                if v_vals and any(v is not None for v in v_vals):
                    val_lists.append([v if v is not None else float("nan") for v in v_vals])

            if not train_lists:
                continue

            # Align to shortest length (in case epoch counts differ).
            min_len = min(len(x) for x in train_lists)
            t_arr   = np.array([x[:min_len] for x in train_lists])  # [H, T]
            t_mean  = np.nanmean(t_arr, axis=0)
            t_std   = np.nanstd(t_arr,  axis=0)
            epochs  = np.arange(1, min_len + 1)

            color = colors[ci]
            label = f"n={n}"
            ax.plot(epochs, t_mean, color=color, linewidth=1.8, label=label)
            ax.fill_between(epochs, t_mean - t_std, t_mean + t_std,
                            color=color, alpha=0.15)

            if val_lists:
                min_vlen = min(len(x) for x in val_lists)
                v_arr    = np.array([x[:min_vlen] for x in val_lists])
                v_mean   = np.nanmean(v_arr, axis=0)
                v_std    = np.nanstd(v_arr,  axis=0)
                vepochs  = np.arange(1, min_vlen + 1)
                ax.plot(vepochs, v_mean, color=color, linewidth=1.8, linestyle="--")
                ax.fill_between(vepochs, v_mean - v_std, v_mean + v_std,
                                color=color, alpha=0.10)
                has_val = True

        ax.set_xlabel("Epoch (within stage)", fontsize=11)
        y_label = "AlignmentLoss [rad²]" if stage_key == "stage1" else "FocalSpotLoss [m²]"
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if has_val:
                import matplotlib.lines as mlines
                solid = mlines.Line2D([], [], color="gray", linewidth=1.5, label="train")
                dash  = mlines.Line2D([], [], color="gray", linewidth=1.5,
                                      linestyle="--", label="val")
                ax.legend(handles=handles + [solid, dash],
                          labels=labels + ["train (solid)", "val (dashed)"],
                          fontsize=8, ncol=2)
            else:
                ax.legend(fontsize=9, ncol=2)

    split = _split_label(summaries)
    fig.suptitle(
        f"Loss convergence by train size  [{split} split]  (mean ± std, n={len(summaries)} heliostats)",
        fontsize=13,
    )
    fig.tight_layout()
    out = out_dir / "loss_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def _write_table(summaries: list[dict], out_dir: pathlib.Path) -> None:
    all_sizes = sorted({n for s in summaries for n in s["train_sizes"]})
    split = _split_label(summaries)
    col_w = 10

    header = f"{'Heliostat':<10} {'Dist(m)':>7}  " + "".join(f"{n:>{col_w}}" for n in all_sizes)
    sep    = "-" * len(header)
    lines  = [
        f"One-heliostat train-size sensitivity — after_stage2 test mean mrad  [{split} split]",
        sep, header, sep,
    ]
    for s in summaries:
        hid  = s["heliostat_id"]
        dist = _KNOWN_DISTANCES.get(hid, "?")
        row  = f"{hid:<10} {dist:>7}  "
        for n in all_sizes:
            key = str(n)
            if key in s["results"]:
                val = s["results"][key]["after_stage2"]["mrad_mean"]
                row += f"{val:>{col_w}.4f}"
            else:
                row += f"{'N/A':>{col_w}}"
        lines.append(row)
    lines += [sep, ""]
    out = out_dir / "comparison_table.txt"
    out.write_text("\n".join(lines))
    print(f"Saved: {out}")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate(
    output_dir: pathlib.Path,
    heliostat_ids: list[str] | None = None,
    out_dir: pathlib.Path | None = None,
) -> None:
    """
    Read per-heliostat summary.json files and write all comparison outputs.

    Parameters
    ----------
    output_dir   : root of run_all_train_sizes.py output (contains {hid}/ subdirs)
    heliostat_ids: if given, only these heliostats are read; otherwise auto-discovered
    out_dir      : where to write outputs (default: output_dir/comparison/)
    """
    output_dir = pathlib.Path(output_dir)
    out_dir    = pathlib.Path(out_dir) if out_dir else output_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = _collect(output_dir, heliostat_ids)
    if not summaries:
        print("[WARNING] No summaries found — nothing to aggregate.")
        return

    print(f"Aggregating {len(summaries)} heliostat(s): {[s['heliostat_id'] for s in summaries]}")
    _plot_per_heliostat(summaries, out_dir)
    _plot_accuracy_mean_std(summaries, out_dir)
    _plot_loss_curves(summaries, out_dir)
    _write_table(summaries, out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate train-size sensitivity results.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--output-dir", type=pathlib.Path,
        help="Root dir of run_all_train_sizes.py output; auto-discovers {hid}/summary.json.",
    )
    group.add_argument(
        "--dirs", nargs="+", type=pathlib.Path,
        help="Explicit list of per-heliostat dirs (each must contain summary.json).",
    )
    p.add_argument(
        "--out", type=pathlib.Path, default=None,
        help="Output directory for plots/table (default: <output-dir>/comparison/).",
    )
    args = p.parse_args()

    if args.dirs:
        summaries = []
        for d in args.dirs:
            try:
                summaries.append(_load_summary(d.resolve()))
            except FileNotFoundError as e:
                print(f"[WARNING] {e}")
        summaries.sort(key=lambda s: _KNOWN_DISTANCES.get(s["heliostat_id"], 999))
        out_dir = pathlib.Path(args.out) if args.out else pathlib.Path(".") / "comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        if summaries:
            _plot_per_heliostat(summaries, out_dir)
            _plot_accuracy_mean_std(summaries, out_dir)
            _plot_loss_curves(summaries, out_dir)
            _write_table(summaries, out_dir)
        else:
            print("No valid summaries found.")
    else:
        out_dir = pathlib.Path(args.out) if args.out else None
        aggregate(output_dir=args.output_dir, out_dir=out_dir)


if __name__ == "__main__":
    main()
