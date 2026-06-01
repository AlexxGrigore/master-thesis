"""
Compare train-size sensitivity across 5 heliostats at different field distances.

Reads one summary.json per heliostat output directory, then produces:
  - comparison_mrad_vs_train_size.png  — 5 lines (one per heliostat) of post-training
                                         test mrad vs training sample count
  - comparison_table.txt               — ASCII table: heliostat × train size → mrad

Usage
-----
    python compare_heliostats.py \\
        --dirs outputs/one_hel_train_sizes_AC36 \\
               outputs/one_hel_train_sizes_AG33 \\
               outputs/one_hel_train_sizes_AO34 \\
               outputs/one_hel_train_sizes_AW36 \\
               outputs/one_hel_train_sizes_BE35 \\
        --out  outputs/one_hel_comparison

    # or let it auto-discover under a common parent:
    python compare_heliostats.py --parent outputs/ --out outputs/one_hel_comparison
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Distances from tower (m) for each heliostat ID (for legend labels).
_KNOWN_DISTANCES = {
    "AC36": 34,
    "AG33": 54,
    "AO34": 90,
    "AW36": 139,
    "BE35": 210,
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
MARKERS = ["o", "s", "^", "D", "v"]


def load_summary(run_dir: pathlib.Path) -> dict:
    path = run_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"No summary.json in {run_dir}")
    with open(path) as f:
        return json.load(f)


def collect_runs(dirs: list[pathlib.Path]) -> list[dict]:
    runs = []
    for d in dirs:
        summary = load_summary(d)
        runs.append(summary)
    runs.sort(key=lambda s: _KNOWN_DISTANCES.get(s["heliostat_id"], 999))
    return runs


def plot_comparison(runs: list[dict], out_dir: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white")

    for i, summary in enumerate(runs):
        hid   = summary["heliostat_id"]
        dist  = _KNOWN_DISTANCES.get(hid, "?")
        sizes = summary["train_sizes"]
        mrad  = [
            summary["results"][str(n)]["post_training"]["test"]["mean_mrad"]
            for n in sizes
        ]
        label = f"{hid}  ({dist} m)"
        ax.plot(
            sizes, mrad,
            marker=MARKERS[i % len(MARKERS)],
            color=COLORS[i % len(COLORS)],
            linewidth=2,
            label=label,
        )

    ax.axhline(1.5, color="green",  linestyle="--", linewidth=1, alpha=0.6, label="1.5 mrad target")
    ax.axhline(2.5, color="goldenrod", linestyle="--", linewidth=1, alpha=0.6, label="2.5 mrad threshold")

    ax.set_xlabel("Training samples", fontsize=12)
    ax.set_ylabel("Post-training test mean FSE (mrad)", fontsize=12)
    ax.set_title("Train-size sensitivity — 5 heliostats across the field", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    out = out_dir / "comparison_mrad_vs_train_size.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def write_table(runs: list[dict], out_dir: pathlib.Path) -> None:
    all_sizes = sorted({n for s in runs for n in s["train_sizes"]})
    col_w = 10

    header = f"{'Heliostat':<10} {'Dist(m)':>7}  " + "".join(f"{n:>{col_w}}" for n in all_sizes)
    sep    = "-" * len(header)

    lines = [
        "One-heliostat train-size comparison — post-training test mean mrad",
        sep,
        header,
        sep,
    ]

    for summary in runs:
        hid  = summary["heliostat_id"]
        dist = _KNOWN_DISTANCES.get(hid, "?")
        row  = f"{hid:<10} {dist:>7}  "
        for n in all_sizes:
            key = str(n)
            if key in summary["results"]:
                val = summary["results"][key]["post_training"]["test"]["mean_mrad"]
                row += f"{val:>{col_w}.4f}"
            else:
                row += f"{'N/A':>{col_w}}"
        lines.append(row)

    lines.append(sep)
    lines.append("")

    out = out_dir / "comparison_table.txt"
    out.write_text("\n".join(lines))
    print(f"Saved: {out}")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one-heliostat train-size runs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dirs", nargs="+", type=pathlib.Path,
        help="Explicit list of run output directories (each must contain summary.json).",
    )
    group.add_argument(
        "--parent", type=pathlib.Path,
        help="Parent directory; auto-discovers sub-dirs named one_hel_train_sizes_*.",
    )
    parser.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("."),
        help="Output directory for the comparison plots/tables.",
    )
    args = parser.parse_args()

    if args.dirs:
        dirs = [d.resolve() for d in args.dirs]
    else:
        dirs = sorted(args.parent.resolve().glob("one_hel_train_sizes_*"))
        if not dirs:
            raise SystemExit(f"No one_hel_train_sizes_* directories found under {args.parent}")

    args.out.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(dirs)
    print(f"Loaded {len(runs)} runs: {[r['heliostat_id'] for r in runs]}")

    plot_comparison(runs, args.out)
    write_table(runs, args.out)


if __name__ == "__main__":
    main()
