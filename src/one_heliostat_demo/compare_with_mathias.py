"""Compare our Stage-1 field results against Mathias's optimizer.

Metric alignment (important)
----------------------------
Mathias's ``pointing_error_mrad`` is the angle between the PREDICTED and the
MEASURED reflected direction from the heliostat position — a direction-type
error. Our like-for-like quantity is therefore ``direction_mrad_*`` (kinematic
pointing, excludes surface), NOT ``centroid_mrad_*`` (ray-traced focal-spot
landing, includes surface).

An earlier comparison (outputs/mathias_validation/make_comparison.py) plotted
our ``mrad_mean`` — an alias of the CENTROID metric — against his direction
metric. That mismatch inflated the apparent gap. This script reports BOTH of
our metrics side by side so the comparison is honest and the difference between
them is visible.

Per-heliostat aggregation follows his: mean (and median) over the TEST split.

Usage
-----
  python src/one_heliostat_demo/compare_with_mathias.py \
      --results-dir outputs/new_mapping_function/all63_stage1 \
      --output-dir  outputs/new_mapping_function/all63_stage1/mathias_comparison
"""

import argparse
import csv
import json
import pathlib
import statistics as st

import matplotlib.pyplot as plt
import numpy as np

_here = pathlib.Path(__file__).resolve().parent
BASE_DIR = _here.parent.parent  # master-thesis/


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare our Stage-1 results with Mathias's.")
    p.add_argument("--results-dir", type=pathlib.Path, required=True,
                   help="Run directory containing summary.json (or per-heliostat results.json)")
    p.add_argument("--mathias-csv", type=pathlib.Path,
                   default=BASE_DIR / "outputs" / "mathias_validation" / "merged_measurements.csv")
    p.add_argument("--output-dir", type=pathlib.Path, default=None)
    p.add_argument("--stage", default="after_stage1",
                   choices=["after_stage1", "after_stage2", "pre_training"],
                   help="Which of our stages to compare (default: after_stage1)")
    return p.parse_args()


def load_ours(results_dir: pathlib.Path, stage: str) -> dict:
    """Per-heliostat metrics from summary.json, falling back to results.json files."""
    out = {}
    summary = results_dir / "summary.json"
    entries = []
    if summary.exists():
        entries = json.load(open(summary)).get("heliostats", [])
    if not entries:
        for rj in sorted(results_dir.glob("*/results.json")):
            entries.append(json.load(open(rj)))
    for e in entries:
        if e.get("status") not in (None, "ok"):
            continue
        hid, sec = e.get("heliostat_id"), e.get(stage)
        if not hid or not sec:
            continue
        out[hid] = {
            "direction_mean": sec.get("direction_mrad_mean"),
            "direction_median": sec.get("direction_mrad_median"),
            "centroid_mean": sec.get("centroid_mrad_mean", sec.get("mrad_mean")),
            "centroid_median": sec.get("centroid_mrad_median", sec.get("mrad_median")),
            "pre_centroid_mean": (e.get("pre_training") or {}).get("centroid_mrad_mean"),
            "pre_direction_mean": (e.get("pre_training") or {}).get("direction_mrad_mean"),
        }
    return out


def load_mathias(csv_path: pathlib.Path) -> dict:
    """Per-heliostat TEST-split mean/median of his pointing_error_mrad."""
    per = {}
    for row in csv.DictReader(open(csv_path)):
        if row["split"] != "test":
            continue
        per.setdefault(row["heliostat_id"], []).append(float(row["pointing_error_mrad"]))
    return {
        h: {"mean": st.mean(v), "median": st.median(v), "n": len(v)}
        for h, v in per.items()
    }


def main() -> None:
    args = _parse_args()
    out_dir = args.output_dir or (args.results_dir / "mathias_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    ours = load_ours(args.results_dir, args.stage)
    his = load_mathias(args.mathias_csv)
    common = sorted(set(ours) & set(his))
    if not common:
        raise SystemExit(f"No overlapping heliostats between {args.results_dir} and Mathias's set")

    rows = []
    for h in common:
        o, m = ours[h], his[h]
        rows.append({
            "heliostat_id": h,
            "our_direction_mean": o["direction_mean"],
            "our_direction_median": o["direction_median"],
            "our_centroid_mean": o["centroid_mean"],
            "our_centroid_median": o["centroid_median"],
            "mathias_mean": m["mean"],
            "mathias_median": m["median"],
            "n_test_his": m["n"],
            "delta_direction_mean": (o["direction_mean"] - m["mean"])
                                    if o["direction_mean"] is not None else None,
            "our_pre_direction_mean": o["pre_direction_mean"],
        })
    rows.sort(key=lambda r: (r["delta_direction_mean"] is None, r["delta_direction_mean"]))

    with open(out_dir / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    od = [r["our_direction_mean"] for r in rows if r["our_direction_mean"] is not None]
    oc = [r["our_centroid_mean"] for r in rows if r["our_centroid_mean"] is not None]
    mm = [r["mathias_mean"] for r in rows]
    wins = sum(1 for r in rows
               if r["our_direction_mean"] is not None and r["our_direction_mean"] < r["mathias_mean"])

    # ------------------------------- plot -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5),
                             gridspec_kw={"width_ratios": [1, 1.35]})

    ax = axes[0]
    x = np.array([r["our_direction_mean"] for r in rows], dtype=float)
    y = np.array(mm, dtype=float)
    lim = [0.7 * min(np.nanmin(x), y.min()), 1.4 * max(np.nanmax(x), y.max())]
    ax.plot(lim, lim, "k-", lw=1, label="parity")
    ax.fill_between(lim, lim, [lim[1]] * 2, color="tab:green", alpha=0.07)
    ax.fill_between(lim, [lim[0]] * 2, lim, color="tab:red", alpha=0.07)
    ax.scatter(x, y, s=42, c="tab:blue", edgecolor="k", linewidth=0.5, zorder=3)
    for r in rows:
        if r["our_direction_mean"] is None:
            continue
        worse = r["our_direction_mean"] > 2 * r["mathias_mean"]
        better = r["mathias_mean"] > 2 * r["our_direction_mean"]
        if worse or better:
            ax.annotate(r["heliostat_id"], (r["our_direction_mean"], r["mathias_mean"]),
                        fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("OUR Stage 1 — test direction error [mrad]")
    ax.set_ylabel("MATHIAS — test pointing error [mrad]")
    ax.set_title(f"Per-heliostat, like-for-like (direction)\n"
                 f"green = we win ({wins}/{len(rows)})", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    ax = axes[1]
    idx = np.arange(len(rows))
    ax.bar(idx - 0.28, [r["our_direction_mean"] for r in rows], 0.28,
           label=f"ours: direction (median {np.nanmedian(od):.1f})", color="tab:blue")
    ax.bar(idx, [r["our_centroid_mean"] for r in rows], 0.28,
           label=f"ours: centroid (median {np.nanmedian(oc):.1f})", color="tab:cyan")
    ax.bar(idx + 0.28, mm, 0.28,
           label=f"Mathias (median {np.median(mm):.1f})", color="tab:orange")
    ax.set_xticks(idx)
    ax.set_xticklabels([r["heliostat_id"] for r in rows], rotation=90, fontsize=6.5)
    ax.set_yscale("log"); ax.set_ylabel("test error [mrad]")
    ax.set_title("Per heliostat, sorted by (ours − Mathias) on the direction metric",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)

    fig.suptitle(
        f"Our Stage 1 ({args.stage}) vs Mathias — {len(rows)} shared heliostats, TEST split\n"
        "his pointing_error_mrad is a DIRECTION metric, so 'ours: direction' is the like-for-like bar",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "stage1_vs_mathias.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------ markdown ------------------------------
    def fmt(v, nd=2):
        return "—" if v is None else f"{v:.{nd}f}"

    with open(out_dir / "STAGE1_VS_MATHIAS.md", "w") as fh:
        fh.write(f"# Our Stage 1 vs Mathias — {len(rows)} shared heliostats (test split)\n\n")
        fh.write(f"Source: `{args.results_dir}` (`{args.stage}`) vs `{args.mathias_csv.name}`.\n\n")
        fh.write("**Metric note.** His `pointing_error_mrad` is the angle between predicted and\n"
                 "measured reflected direction, so our **direction** metric is the like-for-like\n"
                 "one. Our centroid metric (ray-traced focal-spot landing, includes surface) is\n"
                 "shown alongside because reporting only one is misleading.\n\n")
        fh.write("## Summary (mean over each heliostat's test measurements)\n\n")
        fh.write("| | median | mean | min | max |\n|---|---|---|---|---|\n")
        for name, vals in [("ours — direction (like-for-like)", od),
                           ("ours — centroid", oc),
                           ("Mathias", mm)]:
            fh.write(f"| {name} | {np.nanmedian(vals):.2f} | {np.nanmean(vals):.2f} | "
                     f"{np.nanmin(vals):.2f} | {np.nanmax(vals):.2f} |\n")
        fh.write(f"\nWe are better on **{wins}/{len(rows)}** heliostats "
                 f"(direction metric).\n\n")
        fh.write("## Per heliostat\n\n")
        fh.write("| heliostat | ours dir mean | ours dir med | ours centroid mean | "
                 "Mathias mean | Mathias med | Δ (ours−his, dir) | our pre-training dir |\n")
        fh.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['heliostat_id']} | {fmt(r['our_direction_mean'])} | "
                     f"{fmt(r['our_direction_median'])} | {fmt(r['our_centroid_mean'])} | "
                     f"{fmt(r['mathias_mean'])} | {fmt(r['mathias_median'])} | "
                     f"{fmt(r['delta_direction_mean'])} | {fmt(r['our_pre_direction_mean'])} |\n")
        fh.write("\n## Caveat\n\n"
                 "His numbers were computed on our exported split (verified identical labels at\n"
                 "export time). This run re-derives the split with the PAINT DatasetSplitter; if\n"
                 "any config affecting the split changed since the export, the test sets may not\n"
                 "be measurement-for-measurement identical.\n")

    print(f"\n{len(rows)} shared heliostats")
    print(f"  ours  direction : median {np.nanmedian(od):.2f}  mean {np.nanmean(od):.2f} mrad")
    print(f"  ours  centroid  : median {np.nanmedian(oc):.2f}  mean {np.nanmean(oc):.2f} mrad")
    print(f"  Mathias         : median {np.median(mm):.2f}  mean {np.mean(mm):.2f} mrad")
    print(f"  we win on {wins}/{len(rows)}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
