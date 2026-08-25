"""Field-wide sweep of Stage-1 robust loss reductions.

Question
--------
Not *what* we measure (the residual is always the forward-aim normal error) but
*how the per-sample residuals are aggregated*. Least squares ("l2") lets a few
bad calibration samples dominate; robust reductions cap or discard them.

An earlier single-heliostat study (AA23) found robust reductions trade mean for
median: trimmed-40% drove the median to 1.70 mrad (beating Mathias's 1.87) while
the centroid mean degraded 5.49 -> 7.47. This runs the same comparison across the
whole field so we can see whether that trade-off holds generally.

Stage 1 only (no ray tracing in the objective), so each config over all 63
heliostats costs ~6 minutes.

Usage
-----
  python src/one_heliostat_demo/robust_loss_sweep.py \
      --output-dir outputs/new_mapping_function/robust_loss_sweep
  python src/one_heliostat_demo/robust_loss_sweep.py --skip-runs   # re-analyse only
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

_here = pathlib.Path(__file__).resolve().parent
BASE_DIR = _here.parent.parent

# (label, extra CLI args) — mirrors the AA23 study's configurations.
CONFIGS = [
    ("l2",             ["--stage1-reduction", "l2"]),
    ("huber_d1",       ["--stage1-reduction", "huber",   "--huber-delta", "1.0"]),
    ("huber_d3",       ["--stage1-reduction", "huber",   "--huber-delta", "3.0"]),
    ("soft_l1_d1.5",   ["--stage1-reduction", "soft_l1", "--huber-delta", "1.5"]),
    ("trimmed_25",     ["--stage1-reduction", "trimmed", "--trim-fraction", "0.25"]),
    ("trimmed_40",     ["--stage1-reduction", "trimmed", "--trim-fraction", "0.40"]),
]

METRICS = [
    ("direction_mrad_mean", "dir mean"),
    ("direction_mrad_median", "dir median"),
    ("centroid_mrad_mean", "cen mean"),
    ("centroid_mrad_median", "cen med"),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-1 robust loss reduction sweep.")
    p.add_argument("--output-dir", type=pathlib.Path,
                   default=BASE_DIR / "outputs" / "new_mapping_function" / "robust_loss_sweep")
    p.add_argument("--heliostat-ids", nargs="+", default=None,
                   help="Restrict to these heliostats (default: all 63)")
    p.add_argument("--stage1-epochs", type=int, default=500)
    p.add_argument("--skip-runs", action="store_true",
                   help="Skip training; only rebuild the comparison from existing dirs")
    return p.parse_args()


def run_config(label: str, extra: list[str], out_dir: pathlib.Path,
               args: argparse.Namespace) -> None:
    cmd = [
        sys.executable, str(_here / "run_all.py"),
        "--data-mode", "real", "--skip-stage2", "--no-plots",
        "--stage1-epochs", str(args.stage1_epochs),
        "--output-dir", str(out_dir),
        *extra,
    ]
    if args.heliostat_ids:
        cmd += ["--heliostat-ids", *args.heliostat_ids]
    print(f"\n=== {label} ===\n{' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(out_dir.parent / f"{label}.log", "w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)
    print(f"  {label} done in {(time.time() - t0) / 60:.1f} min", flush=True)


def collect(out_dir: pathlib.Path) -> dict:
    """Per-heliostat after_stage1 metrics from a run directory."""
    summary = out_dir / "summary.json"
    entries = []
    if summary.exists():
        entries = json.load(open(summary)).get("heliostats", [])
    if not entries:
        entries = [json.load(open(p)) for p in sorted(out_dir.glob("*/results.json"))]
    return {
        e["heliostat_id"]: e["after_stage1"]
        for e in entries
        if e.get("status") in (None, "ok") and e.get("after_stage1")
    }


def mathias_per_heliostat() -> dict:
    """Test-split mean of his pointing_error_mrad, per heliostat."""
    import csv
    path = BASE_DIR / "outputs" / "mathias_validation" / "merged_measurements.csv"
    if not path.exists():
        return {}
    per: dict[str, list[float]] = {}
    for row in csv.DictReader(open(path)):
        if row["split"] == "test":
            per.setdefault(row["heliostat_id"], []).append(float(row["pointing_error_mrad"]))
    return {h: float(np.mean(v)) for h, v in per.items()}


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, extra in CONFIGS:
        out_dir = args.output_dir / label
        if not args.skip_runs:
            run_config(label, extra, out_dir, args)
        got = collect(out_dir)
        if got:
            results[label] = got
        else:
            print(f"  (no results for {label})", flush=True)

    if not results:
        raise SystemExit("No results collected.")

    common = sorted(set.intersection(*(set(v) for v in results.values())))
    print(f"\n{len(common)} heliostats common to all {len(results)} configs")

    # Field-level summary: median/mean ACROSS heliostats of each per-heliostat metric.
    rows = []
    for label in results:
        r = {"config": label}
        for key, _ in METRICS:
            vals = np.array([results[label][h][key] for h in common], dtype=float)
            r[f"{key}__median"] = float(np.nanmedian(vals))
            r[f"{key}__mean"] = float(np.nanmean(vals))
        rows.append(r)

    his = mathias_per_heliostat()
    shared = [h for h in common if h in his]
    his_med = float(np.median([his[h] for h in shared])) if shared else None

    with open(args.output_dir / "sweep_results.json", "w") as fh:
        json.dump({"n_heliostats": len(common), "heliostats": common,
                   "field_summary": rows,
                   "per_heliostat": {k: {h: results[k][h] for h in common} for k in results}},
                  fh, indent=2)

    # ------------------------------- plot -------------------------------
    labels = [r["config"] for r in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

    ax = axes[0]
    for key, name in METRICS:
        ax.plot(x, [r[f"{key}__median"] for r in rows], "o-", label=f"field median of {name}")
    if his_med is not None:
        ax.axhline(his_med, ls="--", c="k", lw=1,
                   label=f"Mathias (median {his_med:.2f}, n={len(shared)})")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("mrad"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Field-level accuracy by Stage-1 reduction", fontsize=10)

    ax = axes[1]
    base = rows[0]
    for key, name in METRICS:
        rel = [100 * (r[f"{key}__median"] - base[f"{key}__median"]) / base[f"{key}__median"]
               for r in rows]
        ax.plot(x, rel, "o-", label=name)
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("% change vs l2  (negative = better)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Relative to the l2 baseline — the mean/median trade-off", fontsize=10)

    fig.suptitle(
        f"Stage-1 robust loss reductions — {len(common)} heliostats, real data\n"
        "same residual, different aggregation; robust modes trade mean/centroid for median",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(args.output_dir / "robust_loss_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------ markdown ------------------------------
    with open(args.output_dir / "ROBUST_LOSS_SWEEP.md", "w") as fh:
        fh.write(f"# Stage-1 robust loss reductions — field sweep ({len(common)} heliostats)\n\n")
        fh.write("Same observable (forward-aim normal residual), different **aggregation**.\n"
                 "Values are the field median (across heliostats) of each per-heliostat metric.\n\n")
        fh.write("| config | " + " | ".join(n for _, n in METRICS) + " |\n")
        fh.write("|---" * (len(METRICS) + 1) + "|\n")
        for r in rows:
            fh.write(f"| {r['config']} | "
                     + " | ".join(f"{r[f'{k}__median']:.2f}" for k, _ in METRICS) + " |\n")
        if his_med is not None:
            fh.write(f"| _Mathias (n={len(shared)})_ | {his_med:.2f} | — | — | — |\n")
        fh.write("\n## Field mean (across heliostats)\n\n")
        fh.write("| config | " + " | ".join(n for _, n in METRICS) + " |\n")
        fh.write("|---" * (len(METRICS) + 1) + "|\n")
        for r in rows:
            fh.write(f"| {r['config']} | "
                     + " | ".join(f"{r[f'{k}__mean']:.2f}" for k, _ in METRICS) + " |\n")
        fh.write("\nPlot: `robust_loss_sweep.png`. Per-heliostat values: `sweep_results.json`.\n")

    print("\nField median across heliostats:")
    hdr = f"{'config':<16}" + "".join(f"{n:>12}" for _, n in METRICS)
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['config']:<16}" + "".join(f"{r[f'{k}__median']:>12.2f}" for k, _ in METRICS))
    if his_med is not None:
        print(f"{'Mathias':<16}{his_med:>12.2f}")
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
