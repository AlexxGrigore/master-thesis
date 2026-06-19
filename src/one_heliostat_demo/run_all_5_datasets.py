"""
Run the full training pipeline on each of the 5 synthetic datasets and
aggregate results across seeds.

For each dataset the same training loop as run_all.py is executed.
After all datasets are done the results are averaged per heliostat and
the following cross-seed outputs are produced:

    <output_dir>/
        dataset_1/ ... dataset_5/          — per-dataset training results
            {hid}/results.json
            {hid}/convergence_history.csv
            ...
        aggregated/
            results_per_heliostat.csv      — per-seed + mean + std columns
            field_summary.json             — field-level scalar stats
            mean_accuracy_histogram.png    — histogram of mean mrad per heliostat
            field_view_mean.png            — field map coloured by mean mrad
            field_view_mean_detailed.png   — same with finer accuracy bands
            boxplot_per_heliostat.png      — box plot: one box per heliostat, 5 seed points

Usage
-----
    python run_all_5_datasets.py --datasets-dir datasets/synthetic/5_datasets
    python run_all_5_datasets.py --datasets-dir datasets/synthetic/5_datasets \\
        --skip-stage2 --stage1-epochs 100 --split-type balanced
    python run_all_5_datasets.py --smoke-test
"""

import argparse
import csv
import json
import logging
import pathlib
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

_here = pathlib.Path(__file__).resolve().parent   # one_heliostat_demo/
_src  = _here.parent                               # src/
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_here))

import single_heliostat.train as tr
from single_heliostat import config as cfg
from aggregate_results import (
    _ACCURACY_BANDS,
    _band_color,
    _plot_field_view_detailed,
    _SCENARIO_H5,
)

from artist.util import constants as _const, get_device, set_logger_config
from artist.util import setup_distributed_environment

log = logging.getLogger(__name__)


ALL_HELIOSTAT_IDS = [
    "AA23", "AA24", "AA25", "AA49",
    "AB26", "AB33", "AB43", "AB50",
    "AC24", "AC25", "AC27", "AC33", "AC35", "AC36", "AC39", "AC41", "AC47", "AC48",
    "AD39", "AD40",
    "AE23", "AE24", "AE29", "AE30", "AE32",
    "AF37", "AF38", "AF40", "AF44",
    "AG25", "AG27", "AG31", "AG33",
    "AH30",
    "AI36",
    "AJ37",
    "AK29", "AK32",
    "AM25", "AM38",
    "AN35",
    "AO32", "AO34",
    "AP29", "AP43",
    "AQ24",
    "AW36",
    "AX39",
    "AY36", "AY37", "AY39", "AY42", "AY43", "AY44",
    "AZ27", "AZ41",
    "BA28", "BA35", "BA42",
    "BD39",
    "BE25", "BE35",
    "BF39",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train on 5 synthetic datasets and aggregate cross-seed results."
    )
    p.add_argument(
        "--datasets-dir", type=pathlib.Path, default=None,
        help="Root of the 5_datasets directory (default: datasets/synthetic/5_datasets/)",
    )
    p.add_argument(
        "--n-datasets", type=int, default=5,
        help="Number of datasets to train on (default: 5)",
    )
    p.add_argument(
        "--output-dir", type=pathlib.Path, default=None,
        help="Where to write all results (default: auto-named under outputs/)",
    )
    p.add_argument(
        "--heliostat-ids", nargs="+", default=None, metavar="ID",
        help="Subset of heliostat IDs (default: all 63)",
    )
    p.add_argument(
        "--split-type", choices=["balanced", "azimuth"], default="balanced",
        help="DatasetSplitter strategy (default: balanced)",
    )
    p.add_argument(
        "--skip-stage2", action="store_true",
        help="Skip Stage 2 (FocalSpotLoss) — only run Stage 1 (AlignmentLoss)",
    )
    p.add_argument(
        "--stage1-epochs", type=int, default=None,
        help="Override cfg.STAGE1_EPOCHS",
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Quick test: first 3 heliostats, minimal epochs",
    )
    p.add_argument(
        "--daic", action="store_true",
        help="Use DAIC cluster paths instead of local paths.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Cross-seed aggregation
# ---------------------------------------------------------------------------

def _aggregate_across_seeds(
    per_dataset_results: list[dict],
    stage: str = "after_stage2",
) -> dict[str, dict]:
    """
    Given a list of per-dataset result dicts {hid: results_json},
    return {hid: {mean_mrad, std_mrad, per_seed_mrad, ...}} averaged over seeds.
    """
    all_hids = sorted({h for d in per_dataset_results for h in d})
    aggregated: dict[str, dict] = {}

    for hid in all_hids:
        seed_vals = [
            d[hid][stage]["mrad_mean"]
            for d in per_dataset_results
            if hid in d and stage in d[hid]
        ]
        if not seed_vals:
            continue
        arr = np.array(seed_vals)
        aggregated[hid] = {
            "per_seed_mrad":   arr.tolist(),
            "mean_mrad":       float(arr.mean()),
            "std_mrad":        float(arr.std()),
            "median_mrad":     float(np.median(arr)),
            "n_seeds":         len(arr),
            # also carry hel_dist_m from any available result
            "hel_dist_m": next(
                (d[hid].get("hel_dist_m", float("nan"))
                 for d in per_dataset_results if hid in d),
                float("nan"),
            ),
            # compatibility key expected by aggregate_results plot functions
            "after_stage2": {"mrad_mean": float(arr.mean())},
        }
    return aggregated


# ---------------------------------------------------------------------------
# New cross-seed plots
# ---------------------------------------------------------------------------

def _plot_mean_accuracy_histogram(
    aggregated: dict[str, dict],
    agg_dir: pathlib.Path,
    stage_label: str = "after_stage2",
) -> None:
    """Histogram of mean-over-seeds mrad, one bar per bin, coloured by accuracy band."""
    if not aggregated:
        return

    mean_vals = np.array([v["mean_mrad"] for v in aggregated.values()])
    field_mean   = float(np.mean(mean_vals))
    field_median = float(np.median(mean_vals))

    fig, ax = plt.subplots(figsize=(8, 5))

    n_bins  = max(8, len(mean_vals) // 5)
    counts, bin_edges, patches = ax.hist(
        mean_vals, bins=n_bins, edgecolor="white", alpha=0.85, zorder=2,
        color="steelblue",
    )

    # Colour each bar by the accuracy band of its bin centre.
    for patch, left, right in zip(patches, bin_edges[:-1], bin_edges[1:]):
        patch.set_facecolor(_band_color((left + right) / 2))

    ax.axvline(field_mean,   color="navy",   ls="--", lw=1.5,
               label=f"Mean = {field_mean:.3f} mrad")
    ax.axvline(field_median, color="purple", ls=":",  lw=1.5,
               label=f"Median = {field_median:.3f} mrad")

    from matplotlib.patches import Patch
    band_legend = [
        Patch(facecolor=color, label=label)
        for _, color, label in _ACCURACY_BANDS
    ]
    ax.legend(handles=[
        plt.Line2D([], [], color="navy",   ls="--", lw=1.5, label=f"Mean = {field_mean:.3f} mrad"),
        plt.Line2D([], [], color="purple", ls=":",  lw=1.5, label=f"Median = {field_median:.3f} mrad"),
    ] + band_legend, fontsize=8)

    ax.set_xlabel("Mean mrad over seeds (per heliostat)")
    ax.set_ylabel("Number of heliostats")
    ax.set_title(
        f"Per-heliostat accuracy distribution  "
        f"(N={len(mean_vals)} heliostats, mean over {next(iter(aggregated.values()))['n_seeds']} seeds)"
    )
    ax.grid(True, alpha=0.3, zorder=1)
    fig.tight_layout()

    out = agg_dir / "mean_accuracy_histogram.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"  → {out}")


def _plot_boxplot_per_heliostat(
    aggregated: dict[str, dict],
    agg_dir: pathlib.Path,
) -> None:
    """Box plot: one box per heliostat sorted by mean mrad, 5 seed points shown."""
    if not aggregated:
        return

    hids_sorted = sorted(aggregated, key=lambda h: aggregated[h]["mean_mrad"])
    data        = [aggregated[h]["per_seed_mrad"] for h in hids_sorted]
    means       = [aggregated[h]["mean_mrad"]     for h in hids_sorted]

    fig, ax = plt.subplots(figsize=(max(10, len(hids_sorted) * 0.24), 5))

    bp = ax.boxplot(data, patch_artist=True, widths=0.6, zorder=2,
                    medianprops=dict(color="black", lw=1.5))

    for patch, mean_val in zip(bp["boxes"], means):
        patch.set_facecolor(_band_color(mean_val))
        patch.set_alpha(0.75)

    # Scatter individual seed points.
    for i, vals in enumerate(data, start=1):
        ax.scatter([i] * len(vals), vals, color="black", s=14, zorder=4, alpha=0.6)

    ax.set_xticks(range(1, len(hids_sorted) + 1))
    ax.set_xticklabels(hids_sorted, rotation=90, fontsize=7)
    ax.set_ylabel("mrad")
    ax.set_title(
        f"Per-heliostat accuracy across {next(iter(aggregated.values()))['n_seeds']} seeds "
        f"(sorted by mean, coloured by accuracy band)"
    )
    ax.grid(True, axis="y", alpha=0.3, zorder=1)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=color, label=label) for _, color, label in _ACCURACY_BANDS],
        fontsize=8, loc="upper left",
    )
    fig.tight_layout()

    out = agg_dir / "boxplot_per_heliostat.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"  → {out}")


def _write_cross_seed_csv(
    aggregated: dict[str, dict],
    n_datasets: int,
    agg_dir: pathlib.Path,
) -> None:
    rows = sorted(aggregated.items(), key=lambda kv: kv[1]["mean_mrad"])

    fieldnames = (
        ["heliostat_id", "dist_m"]
        + [f"seed_{d}_mrad" for d in range(1, n_datasets + 1)]
        + ["mean_mrad", "std_mrad", "median_mrad"]
    )

    with open(agg_dir / "results_per_heliostat.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for hid, v in rows:
            row = {
                "heliostat_id": hid,
                "dist_m":       round(v["hel_dist_m"], 1),
                "mean_mrad":    round(v["mean_mrad"],   4),
                "std_mrad":     round(v["std_mrad"],    4),
                "median_mrad":  round(v["median_mrad"], 4),
            }
            for i, s in enumerate(v["per_seed_mrad"], start=1):
                row[f"seed_{i}_mrad"] = round(s, 4)
            # fill missing seeds with empty
            for d in range(len(v["per_seed_mrad"]) + 1, n_datasets + 1):
                row[f"seed_{d}_mrad"] = ""
            w.writerow(row)

    log.info(f"  → {agg_dir / 'results_per_heliostat.csv'}")


def _write_field_summary(
    aggregated: dict[str, dict],
    n_datasets: int,
    agg_dir: pathlib.Path,
) -> None:
    means = np.array([v["mean_mrad"] for v in aggregated.values()])
    stds  = np.array([v["std_mrad"]  for v in aggregated.values()])

    worst_hid = max(aggregated, key=lambda h: aggregated[h]["mean_mrad"])
    best_hid  = min(aggregated, key=lambda h: aggregated[h]["mean_mrad"])

    summary = {
        "n_heliostats":         len(aggregated),
        "n_seeds":              n_datasets,
        "field_mean_mrad":      round(float(means.mean()),   4),
        "field_median_mrad":    round(float(np.median(means)), 4),
        "field_std_mrad":       round(float(means.std()),    4),
        "mean_seed_std_mrad":   round(float(stds.mean()),    4),
        "worst_heliostat":      worst_hid,
        "worst_mean_mrad":      round(aggregated[worst_hid]["mean_mrad"], 4),
        "best_heliostat":       best_hid,
        "best_mean_mrad":       round(aggregated[best_hid]["mean_mrad"],  4),
    }

    with open(agg_dir / "field_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 55)
    print(f"  Cross-seed field summary  ({n_datasets} seeds, {len(aggregated)} heliostats)")
    print("=" * 55)
    for k, v in summary.items():
        print(f"  {k:<28}: {v}")
    print("=" * 55)
    print()


# ---------------------------------------------------------------------------
# Training loop for one dataset
# ---------------------------------------------------------------------------

def _run_one_dataset(
    dataset_idx: int,
    dataset_dir: pathlib.Path,
    output_dir: pathlib.Path,
    heliostat_ids: list[str],
    device,
    args,
) -> dict[str, dict]:
    """Train all heliostats on one dataset. Returns {hid: results_json}."""
    results: dict[str, dict] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"=== Dataset {dataset_idx} — {dataset_dir} ===")

    for hid in tqdm(heliostat_ids, desc=f"Dataset {dataset_idx}", unit="hel", dynamic_ncols=True):
        hid_dir = output_dir / hid

        # Skip if already done (allows resuming interrupted runs).
        rf = hid_dir / "results.json"
        if rf.exists():
            log.info(f"  {hid}: result already exists — skipping")
            with open(rf) as f:
                results[hid] = json.load(f)
            continue

        t0 = time.time()
        try:
            r = tr.run(
                heliostat_id=hid,
                dataset_dir=dataset_dir,
                output_dir=hid_dir,
                cfg=cfg,
                device=device,
                skip_stage2=args.skip_stage2,
            )
            results[hid] = r
            elapsed = (time.time() - t0) / 60
            s2 = r.get("after_stage2", r.get("after_stage1", {}))
            log.info(
                f"  {hid} done in {elapsed:.1f} min  "
                f"mrad_mean={s2.get('mrad_mean', float('nan')):.4f}"
            )
        except Exception as exc:
            log.error(f"  {hid} FAILED: {exc}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.daic:
        cfg.BASE_DIR = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
        cfg.PAINT_DIR = pathlib.Path(
            "/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint"
        )
        cfg.SCENARIO_PATH_TEMPLATE = str(
            cfg.BASE_DIR / "scenarios" / "one_heliostat_scenarios" / "{heliostat_id}" / "scenario.h5"
        )
        _synth_root = cfg.PAINT_DIR / "synthetic"
    else:
        _synth_root = cfg.BASE_DIR / "datasets" / "synthetic"

    # Resolve datasets root
    datasets_dir = args.datasets_dir or (_synth_root / "5_datasets")
    datasets_dir = pathlib.Path(datasets_dir)

    heliostat_ids = args.heliostat_ids or ALL_HELIOSTAT_IDS
    if args.smoke_test:
        heliostat_ids           = heliostat_ids[:3]
        args.n_datasets         = 2
        cfg.STAGE1_EPOCHS       = 5
        cfg.STAGE2_EPOCHS       = 5
        args.skip_stage2        = True

    if args.stage1_epochs is not None:
        cfg.STAGE1_EPOCHS = args.stage1_epochs

    cfg.DATA_MODE      = "synthetic"
    cfg.SPLITTER_TYPE  = args.split_type

    # Filter to heliostats with a scenario on disk.
    valid_ids = [
        hid for hid in heliostat_ids
        if pathlib.Path(cfg.SCENARIO_PATH_TEMPLATE.format(heliostat_id=hid)).exists()
    ]
    log.info(f"Valid heliostats: {len(valid_ids)} / {len(heliostat_ids)}")

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        cfg.BASE_DIR / "outputs" / f"5_datasets_run_{timestamp}"
    )
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info(f"Output dir    : {output_dir}")
    log.info(f"Datasets root : {datasets_dir}")
    log.info(f"N datasets    : {args.n_datasets}")
    log.info(f"Skip stage 2  : {args.skip_stage2}")
    log.info(f"Stage1 epochs : {cfg.STAGE1_EPOCHS}")

    device = get_device()

    with setup_distributed_environment(number_of_heliostat_groups=1, device=device) as ddp:
        device = ddp[_const.device]
        log.info(f"Device: {device}")

        per_dataset_results: list[dict] = []

        for d in range(1, args.n_datasets + 1):
            dataset_path = datasets_dir / f"dataset_{d}" / "dataset"
            if not dataset_path.exists():
                log.error(f"Dataset {d} not found: {dataset_path}")
                continue

            cfg.SYNTHETIC_DATASET_DIR = str(dataset_path)

            results_d = _run_one_dataset(
                dataset_idx=d,
                dataset_dir=dataset_path,
                output_dir=output_dir / f"dataset_{d}",
                heliostat_ids=valid_ids,
                device=device,
                args=args,
            )
            per_dataset_results.append(results_d)

            # Save per-dataset results list for resumability.
            with open(output_dir / f"dataset_{d}" / "results_all.json", "w") as f:
                json.dump(results_d, f, indent=2)

    # ------------------------------------------------------------------
    # Cross-seed aggregation
    # ------------------------------------------------------------------
    stage = "after_stage1" if args.skip_stage2 else "after_stage2"

    agg_dir = output_dir / "aggregated"
    agg_dir.mkdir(exist_ok=True)

    aggregated = _aggregate_across_seeds(per_dataset_results, stage=stage)

    if not aggregated:
        log.error("No results to aggregate.")
        return

    log.info(f"Aggregating {len(aggregated)} heliostats across {len(per_dataset_results)} seeds …")

    _write_cross_seed_csv(aggregated, args.n_datasets, agg_dir)
    _write_field_summary(aggregated, len(per_dataset_results), agg_dir)
    _plot_mean_accuracy_histogram(aggregated, agg_dir)
    _plot_boxplot_per_heliostat(aggregated, agg_dir)

    # Field view using existing aggregate_results functions (reuse colour bands).
    _plot_field_view_detailed(aggregated, agg_dir)

    log.info(f"Done. All outputs → {output_dir}")
    print(f"\nOutputs → {output_dir}\n")


if __name__ == "__main__":
    main()
