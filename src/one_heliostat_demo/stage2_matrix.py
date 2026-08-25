"""Stage-2 focal-spot experiment matrix: loss reduction x trainable parameter set.

Two questions, run as a 3x2 factorial over the whole field:

1. **Loss reduction** — does robustly aggregating the focal-spot residual help,
   as it did for Stage 1? (l2 / huber / soft_l1; trimmed is excluded because the
   Stage-1 sweep showed it blows up.)
2. **Parameter set** — is Stage 2 better off moving everything ("all", the
   historical setup, and the ONLY place translation / base position / offset /
   pivot are ever trained) or restricted to Stage 1's free set
   ("orientation_only"), which stops it drifting the pointing fit?

Every arm starts from the SAME Stage-1 checkpoints, so after_stage1 is identical
across configs and any difference is attributable to Stage 2 alone.

Usage
-----
  python src/one_heliostat_demo/stage2_matrix.py \
      --checkpoint-dir outputs/new_mapping_function/robust_loss_sweep/soft_l1_d1.5 \
      --output-dir     outputs/new_mapping_function/stage2_matrix
  python src/one_heliostat_demo/stage2_matrix.py --skip-runs   # re-analyse only
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

REDUCTIONS = ["l2", "huber", "soft_l1"]
PARAM_SETS = ["all", "orientation_only"]
DELTA_MRAD = 3.0

METRICS = [
    ("direction_mrad_mean", "dir mean"),
    ("direction_mrad_median", "dir median"),
    ("centroid_mrad_mean", "cen mean"),
    ("centroid_mrad_median", "cen med"),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-2 focal-spot experiment matrix.")
    p.add_argument("--checkpoint-dir", type=pathlib.Path,
                   default=BASE_DIR / "outputs" / "new_mapping_function"
                           / "robust_loss_sweep" / "soft_l1_d1.5",
                   help="Run dir holding <HID>/stage1_checkpoint.pt for every heliostat")
    p.add_argument("--output-dir", type=pathlib.Path,
                   default=BASE_DIR / "outputs" / "new_mapping_function" / "stage2_matrix")
    p.add_argument("--heliostat-ids", nargs="+", default=None)
    p.add_argument("--stage2-epochs", type=int, default=120)
    p.add_argument("--skip-runs", action="store_true")
    return p.parse_args()


def config_label(reduction: str, param_set: str) -> str:
    return f"{reduction}__{param_set}"


def run_config(reduction: str, param_set: str, out_dir: pathlib.Path,
               args: argparse.Namespace) -> None:
    cmd = [
        sys.executable, str(_here / "run_all.py"),
        "--data-mode", "real", "--no-plots",
        "--stage1-checkpoint-dir", str(args.checkpoint_dir),
        "--stage2-epochs", str(args.stage2_epochs),
        "--stage2-loss", "focal_spot",
        "--stage2-reduction", reduction,
        "--stage2-huber-delta-mrad", str(DELTA_MRAD),
        "--stage2-param-set", param_set,
        "--output-dir", str(out_dir),
    ]
    if args.heliostat_ids:
        cmd += ["--heliostat-ids", *args.heliostat_ids]
    label = config_label(reduction, param_set)
    print(f"\n=== {label} ===", flush=True)
    t0 = time.time()
    with open(out_dir.parent / f"{label}.log", "w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)
    print(f"  {label} done in {(time.time() - t0) / 60:.1f} min", flush=True)


def collect(out_dir: pathlib.Path) -> dict:
    summary = out_dir / "summary.json"
    entries = []
    if summary.exists():
        entries = json.load(open(summary)).get("heliostats", [])
    if not entries:
        entries = [json.load(open(p)) for p in sorted(out_dir.glob("*/results.json"))]
    return {
        e["heliostat_id"]: {"s1": e["after_stage1"], "s2": e["after_stage2"]}
        for e in entries
        if e.get("status") in (None, "ok") and e.get("after_stage2")
    }


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for reduction in REDUCTIONS:
        for param_set in PARAM_SETS:
            label = config_label(reduction, param_set)
            out_dir = args.output_dir / label
            if not args.skip_runs:
                run_config(reduction, param_set, out_dir, args)
            got = collect(out_dir)
            if got:
                results[label] = got
            else:
                print(f"  (no results for {label})", flush=True)

    if not results:
        raise SystemExit("No results collected.")

    common = sorted(set.intersection(*(set(v) for v in results.values())))
    print(f"\n{len(common)} heliostats common to all {len(results)} configs")

    # after_stage1 is identical across configs (same checkpoints) — the reference.
    ref = results[next(iter(results))]
    rows = []
    for label, res in results.items():
        r = {"config": label}
        for key, _ in METRICS:
            s1 = np.array([ref[h]["s1"][key] for h in common], dtype=float)
            s2 = np.array([res[h]["s2"][key] for h in common], dtype=float)
            r[f"{key}__s1_median"] = float(np.nanmedian(s1))
            r[f"{key}__s2_median"] = float(np.nanmedian(s2))
            r[f"{key}__delta_median"] = float(np.nanmedian(s2 - s1))
            r[f"{key}__improved"] = int((s2 < s1).sum())
            r[f"{key}__blowups"] = int((s2 > s1 * 2 + 2).sum())
        rows.append(r)

    with open(args.output_dir / "matrix_results.json", "w") as fh:
        json.dump({"n_heliostats": len(common), "heliostats": common,
                   "checkpoint_dir": str(args.checkpoint_dir),
                   "summary": rows,
                   "per_heliostat": {k: {h: results[k][h] for h in common}
                                     for k in results}}, fh, indent=2)

    # ------------------------------- plot -------------------------------
    labels = [r["config"] for r in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))

    ax = axes[0]
    width = 0.2
    for i, (key, name) in enumerate(METRICS):
        ax.bar(x + (i - 1.5) * width, [r[f"{key}__s2_median"] for r in rows],
               width, label=f"S2 {name}")
    for i, (key, _) in enumerate(METRICS):
        ax.axhline(rows[0][f"{key}__s1_median"], ls="--", lw=0.9,
                   color=f"C{i}", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("field median [mrad]")
    ax.set_title("After Stage 2 (dashed = Stage-1 starting point)", fontsize=10)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)

    ax = axes[1]
    for i, (key, name) in enumerate(METRICS):
        ax.plot(x, [r[f"{key}__delta_median"] for r in rows], "o-", label=name)
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("median (Stage2 − Stage1) [mrad]   negative = Stage 2 helped")
    ax.set_title("Did Stage 2 actually improve on its starting point?", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(
        f"Stage-2 focal-spot matrix — {len(common)} heliostats, real data\n"
        f"loss reduction x trainable parameter set, all from the same Stage-1 checkpoints "
        f"({args.checkpoint_dir.name})",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(args.output_dir / "stage2_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------ markdown ------------------------------
    with open(args.output_dir / "STAGE2_MATRIX.md", "w") as fh:
        fh.write(f"# Stage-2 focal-spot matrix — {len(common)} heliostats\n\n")
        fh.write(f"All arms start from the same Stage-1 checkpoints "
                 f"(`{args.checkpoint_dir}`), so `after_stage1` is identical and every\n"
                 f"difference below is Stage 2 alone.\n\n")
        fh.write("## Stage-1 starting point (field median)\n\n| metric | mrad |\n|---|---|\n")
        for key, name in METRICS:
            fh.write(f"| {name} | {rows[0][f'{key}__s1_median']:.2f} |\n")
        fh.write("\n## After Stage 2 (field median)\n\n")
        fh.write("| config | " + " | ".join(n for _, n in METRICS) + " |\n")
        fh.write("|---" * (len(METRICS) + 1) + "|\n")
        for r in rows:
            fh.write(f"| {r['config']} | "
                     + " | ".join(f"{r[f'{k}__s2_median']:.2f}" for k, _ in METRICS) + " |\n")
        fh.write("\n## Change vs Stage 1 (median of per-heliostat delta; negative = better)\n\n")
        fh.write("| config | " + " | ".join(n for _, n in METRICS)
                 + " | improved (cen mean) | blow-ups |\n")
        fh.write("|---" * (len(METRICS) + 3) + "|\n")
        for r in rows:
            fh.write(f"| {r['config']} | "
                     + " | ".join(f"{r[f'{k}__delta_median']:+.3f}" for k, _ in METRICS)
                     + f" | {r['centroid_mrad_mean__improved']}/{len(common)}"
                     + f" | {r['centroid_mrad_mean__blowups']} |\n")
        fh.write("\nPlot: `stage2_matrix.png`. Per-heliostat: `matrix_results.json`.\n")

    print("\nField median after Stage 2 (delta vs Stage 1 in brackets):")
    hdr = f"{'config':<26}" + "".join(f"{n:>18}" for _, n in METRICS)
    print(hdr); print("-" * len(hdr))
    for r in rows:
        line = f"{r['config']:<26}"
        for key, _ in METRICS:
            line += f"{r[f'{key}__s2_median']:>10.2f}({r[f'{key}__delta_median']:+.2f})"
        print(line)
    print(f"\nStage-1 reference: " + "  ".join(
        f"{n} {rows[0][f'{k}__s1_median']:.2f}" for k, n in METRICS))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
