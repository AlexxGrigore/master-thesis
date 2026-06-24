"""
Train-size sensitivity sweep across distance and perturbation seeds (synthetic).

For 5 heliostats spanning the field (closest -> farthest from the target) and a
range of training-set sizes, train the full two-stage pipeline once per
perturbation realisation (the 5 datasets in datasets/synthetic/5_datasets).

    5 heliostats  x  9 train sizes  x  5 perturbations  =  225 runs

The 5 perturbations give a mean +/- std accuracy for every (heliostat, train_size)
cell, so the accuracy-vs-train-size curves carry error bars.

Outputs
-------
    <output_dir>/
        runs/{hid}/train_size_{n}/dataset_{k}/   (results.json, histories; no plots)
        results_long.csv          one row per run
        per_heliostat.csv         mean/std over 5 perturbations, per (hid, train_size)
        accuracy_vs_trainsize.png lines per heliostat (labelled by distance), error bars
        accuracy_by_distance.png  heatmap distance x train_size
        summary.json

Usage
-----
    python run_trainsize_distance_sweep.py
    python run_trainsize_distance_sweep.py --train-sizes 1 5 10 50 100
    python run_trainsize_distance_sweep.py --heliostats AB33 BF39 --smoke-test
"""

import argparse
import csv
import gc
import json
import logging
import pathlib
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

_here = pathlib.Path(__file__).resolve().parent   # one_heliostat_demo/
_src  = _here.parent                               # src/
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_here))

from single_heliostat import config as cfg          # noqa: E402
from single_heliostat import train as tr            # noqa: E402

import torch
from artist.util import get_device, set_logger_config

log = logging.getLogger(__name__)

# 5 heliostats spanning the distance range (even spacing in metres, closest->farthest).
# Distances from the all62 synthetic run; see selection note in the thesis log.
DEFAULT_HELIOSTATS = [
    ("AB33", 53.6),
    ("AN35", 97.6),
    ("AW36", 148.5),
    ("BA42", 182.2),
    ("BF39", 228.1),
]
DEFAULT_TRAIN_SIZES = [1, 3, 5, 10, 20, 30, 50, 75, 100]
N_DATASETS = 5


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets-dir", type=pathlib.Path,
                   default=cfg.BASE_DIR / "datasets" / "synthetic" / "5_datasets",
                   help="Root holding dataset_1 .. dataset_5.")
    p.add_argument("--heliostats", nargs="+", default=None,
                   help="Heliostat IDs (default: the 5 distance-spanning ones).")
    p.add_argument("--train-sizes", nargs="+", type=int, default=None,
                   help=f"Train sizes to sweep (default: {DEFAULT_TRAIN_SIZES}).")
    p.add_argument("--output-dir", type=pathlib.Path, default=None)
    p.add_argument("--stage1-epochs", type=int, default=100)
    p.add_argument("--stage2-epochs", type=int, default=100)
    p.add_argument("--val-size", type=int, default=30,
                   help="Validation/test size; pool needs train_size + 2*val_size.")
    p.add_argument("--sampling-seed", type=int, default=42)
    p.add_argument("--skip-stage2", action="store_true")
    p.add_argument("--smoke-test", action="store_true",
                   help="2 heliostats, sizes [1,5], 2 datasets, 3/3 epochs.")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip training; just rebuild aggregates/plots from results_long.csv.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Aggregation + plotting                                                      #
# --------------------------------------------------------------------------- #
def _aggregate(long_rows, heliostats, train_sizes, out_dir):
    """Build per-heliostat mean/std over perturbations and the figures."""
    dist = {h: d for h, d in heliostats}
    hids = [h for h, _ in heliostats]

    # cell[(hid, ts)] -> list of mrad over datasets
    cell = {}
    for r in long_rows:
        cell.setdefault((r["heliostat"], r["train_size"]), []).append(r["mrad_mean"])

    # per_heliostat.csv
    per_rows = []
    for h in hids:
        for ts in train_sizes:
            vals = np.array(cell.get((h, ts), []), dtype=float)
            vals = vals[~np.isnan(vals)]
            per_rows.append({
                "heliostat": h, "distance_m": dist[h], "train_size": ts,
                "n_seeds": len(vals),
                "mrad_mean": float(vals.mean()) if len(vals) else float("nan"),
                "mrad_std":  float(vals.std())  if len(vals) else float("nan"),
                "mrad_min":  float(vals.min())  if len(vals) else float("nan"),
                "mrad_max":  float(vals.max())  if len(vals) else float("nan"),
            })
    with open(out_dir / "per_heliostat.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        w.writeheader(); w.writerows(per_rows)

    pm = {(r["heliostat"], r["train_size"]): r for r in per_rows}

    # ---- Figure 1: accuracy vs train size, one line per heliostat ---------- #
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(hids)))
    for h, c in zip(hids, colors):
        means = np.array([pm[(h, ts)]["mrad_mean"] for ts in train_sizes])
        stds  = np.array([pm[(h, ts)]["mrad_std"]  for ts in train_sizes])
        ax.errorbar(train_sizes, means, yerr=stds, marker="o", lw=1.8, capsize=3,
                    color=c, label=f"{h}  ({dist[h]:.0f} m)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(train_sizes); ax.set_xticklabels([str(t) for t in train_sizes])
    ax.set_xlabel("Training samples"); ax.set_ylabel("Focal-spot error [mrad]  (test, mean ± std over 5 perturbations)")
    ax.set_title("Accuracy vs training-set size, by heliostat distance")
    ax.grid(True, which="both", alpha=0.3); ax.legend(title="heliostat (distance)")
    fig.tight_layout(); fig.savefig(out_dir / "accuracy_vs_trainsize.png", dpi=150)
    plt.close(fig)

    # ---- Figure 2: heatmap distance x train_size --------------------------- #
    M = np.array([[pm[(h, ts)]["mrad_mean"] for ts in train_sizes] for h in hids])
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(M, aspect="auto", cmap="viridis_r",
                   norm=__import__("matplotlib").colors.LogNorm())
    ax.set_xticks(range(len(train_sizes))); ax.set_xticklabels(train_sizes)
    ax.set_yticks(range(len(hids)))
    ax.set_yticklabels([f"{h} ({dist[h]:.0f} m)" for h in hids])
    ax.set_xlabel("Training samples"); ax.set_ylabel("heliostat (distance)")
    ax.set_title("Mean focal-spot error [mrad]")
    for i in range(len(hids)):
        for j in range(len(train_sizes)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > np.nanmedian(M) else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="mrad")
    fig.tight_layout(); fig.savefig(out_dir / "accuracy_by_distance.png", dpi=150)
    plt.close(fig)

    summary = {
        "heliostats": [{"id": h, "distance_m": dist[h]} for h in hids],
        "train_sizes": train_sizes,
        "n_perturbations": N_DATASETS,
        "per_heliostat": per_rows,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Aggregates + figures written to {out_dir}")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    args = _parse_args()
    set_logger_config()
    device = get_device()

    heliostats = DEFAULT_HELIOSTATS
    if args.heliostats:
        dmap = dict(DEFAULT_HELIOSTATS)
        heliostats = [(h, dmap.get(h, float("nan"))) for h in args.heliostats]
    train_sizes = args.train_sizes or DEFAULT_TRAIN_SIZES
    n_datasets = N_DATASETS

    if args.smoke_test:
        heliostats = heliostats[:2]
        train_sizes = [1, 5]
        n_datasets = 2
        args.stage1_epochs = 3
        args.stage2_epochs = 3

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (
        cfg.BASE_DIR / "outputs" / "new_mapping_function"
        / f"trainsize_distance_sweep_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"; runs_dir.mkdir(exist_ok=True)
    long_csv = out_dir / "results_long.csv"

    # config for synthetic full-pipeline runs
    cfg.DATA_MODE = "synthetic"
    cfg.SPLITTER_TYPE = "balanced"
    cfg.SPLITTER_VAL_SIZE = args.val_size
    cfg.STAGE1_EPOCHS = args.stage1_epochs
    cfg.STAGE2_EPOCHS = args.stage2_epochs

    if args.aggregate_only:
        long_rows = []
        with open(long_csv) as f:
            for r in csv.DictReader(f):
                long_rows.append({"heliostat": r["heliostat"],
                                  "train_size": int(r["train_size"]),
                                  "mrad_mean": float(r["mrad_mean"])})
        _aggregate(long_rows, heliostats, train_sizes, out_dir)
        return

    log.info(f"Output        : {out_dir}")
    log.info(f"Heliostats    : {[h for h,_ in heliostats]}")
    log.info(f"Train sizes   : {train_sizes}")
    log.info(f"Perturbations : {n_datasets}  (datasets_dir={args.datasets_dir})")
    log.info(f"Epochs        : stage1={args.stage1_epochs}  stage2={args.stage2_epochs}")
    log.info(f"val/test size : {args.val_size}")
    total_runs = len(heliostats) * len(train_sizes) * n_datasets
    log.info(f"Total runs    : {total_runs}")

    long_rows = []
    # write header up-front so partial progress is inspectable
    with open(long_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["heliostat", "distance_m", "train_size", "dataset", "hel_dist_m",
             "pre_mrad", "s1_mrad", "mrad_mean", "mrad_median", "minutes"]
        )

    run_i = 0
    t_start = time.time()
    for hid, dist_known in heliostats:
        for ts in train_sizes:
            for k in range(1, n_datasets + 1):
                run_i += 1
                ds_dir = args.datasets_dir / f"dataset_{k}" / "dataset"
                sub = runs_dir / hid / f"train_size_{ts}" / f"dataset_{k}"
                t0 = time.time()
                tag = f"[{run_i}/{total_runs}] {hid} ts={ts} ds={k}"
                try:
                    res = tr.run(
                        heliostat_id=hid, dataset_dir=ds_dir, output_dir=sub,
                        cfg=cfg, device=device, train_size=ts,
                        sampling_seed=args.sampling_seed,
                        skip_stage2=args.skip_stage2, make_plots=False,
                    )
                    mins = (time.time() - t0) / 60.0
                    row = {
                        "heliostat": hid, "distance_m": dist_known, "train_size": ts,
                        "dataset": k, "hel_dist_m": res["hel_dist_m"],
                        "pre_mrad": res["pre_training"]["mrad_mean"],
                        "s1_mrad":  res["after_stage1"]["mrad_mean"],
                        "mrad_mean":   res["after_stage2"]["mrad_mean"],
                        "mrad_median": res["after_stage2"]["mrad_median"],
                        "minutes": round(mins, 2),
                    }
                    long_rows.append(row)
                    with open(long_csv, "a", newline="") as f:
                        csv.writer(f).writerow([
                            row["heliostat"], row["distance_m"], row["train_size"],
                            row["dataset"], row["hel_dist_m"], row["pre_mrad"],
                            row["s1_mrad"], row["mrad_mean"], row["mrad_median"],
                            row["minutes"],
                        ])
                    log.info(f"{tag}: pre={row['pre_mrad']:.2f} -> s2={row['mrad_mean']:.3f} mrad ({mins:.1f} min)")
                except Exception as exc:
                    log.error(f"{tag} FAILED: {exc}", exc_info=True)
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    log.info(f"All runs done in {(time.time()-t_start)/60.0:.1f} min.")
    _aggregate(long_rows, heliostats, train_sizes, out_dir)


if __name__ == "__main__":
    main()
