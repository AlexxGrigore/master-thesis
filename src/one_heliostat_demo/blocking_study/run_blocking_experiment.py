"""Experiment S driver — does training WITH blocking beat training without?

Per heliostat, two arms train on the SAME blocking-generated dataset
(generate_blocking_dataset.py), through the identical two-stage pipeline:

  A0_blocking_off — today's pipeline: ray tracer never sees the neighbours.
                    The 0.2-1 mrad centroid bias blocking injected into the data
                    must be absorbed (wrongly) into the kinematic parameters.
  A1_blocking_on  — neighbours present, aimed at each sample's own target.
                    The model can explain the clipped flux and stay honest.

Both arms also get a CROSS evaluation (same parameters, opposite blocking
setting) — stored in results.json as after_stage2_cross_blocking.

Outputs (all under outputs/new_mapping_function/blocking_study/experiment_s/):
  {HID}/{arm}/results.json, convergence_history.csv, plots/...   (train.py)
  {HID}/comparison.json  +  comparison_*.png                     (this driver)
  summary.json + summary_*.png  across heliostats

Usage
-----
    python run_blocking_experiment.py AY36 BA35 BE35 AC33
    python run_blocking_experiment.py AY36 --smoke-test      # pipeline check
    python run_blocking_experiment.py AY36 --compare-only    # redo JSON+plots
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pathlib
import sys
import time

_here = pathlib.Path(__file__).resolve().parent          # blocking_study/
_src = _here.parents[1]                                   # src/
_sh = _src / "one_heliostat_demo" / "single_heliostat"
for _p in (str(_src), str(_sh), str(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import config as cfg  # noqa: E402  (single_heliostat config)
import train as tr  # noqa: E402
from artist.util import set_logger_config  # noqa: E402

log = logging.getLogger(__name__)

_ROOT = _here.parents[2]                                  # master-thesis/
DATASET_DIR = _ROOT / "datasets" / "synthetic" / "blocking_dataset" / "dataset"
SCENARIO_ROOT = _ROOT / "scenarios" / "neighbourhoods"
OUT_ROOT = (
    _ROOT / "outputs" / "new_mapping_function" / "blocking_study" / "experiment_s"
)

ARMS = {"A0_blocking_off": False, "A1_blocking_on": True}


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #

def run_arms(heliostat_id: str, smoke: bool, device: torch.device) -> None:
    scenario_path = SCENARIO_ROOT / heliostat_id / "scenario.h5"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing scenario: {scenario_path}")
    if not (DATASET_DIR / "train" / heliostat_id).exists():
        raise FileNotFoundError(
            f"Missing blocking dataset for {heliostat_id} under {DATASET_DIR}. "
            "Run generate_blocking_dataset.py first."
        )

    # Settled regime (user decision 2026-07-26): geometric init + soft_l1 both
    # stages, no ray-traced trails during Stage 1, trails every epoch in Stage 2
    # (every 5 epochs for the blocking arm — per-sample trails are the cost
    # driver there, and 60 points still resolve the trail shape).
    cfg.DATA_MODE = "synthetic"
    cfg.STAGE1_EPOCHS = 500
    cfg.STAGE2_EPOCHS = 300
    cfg.STAGE1_TRAIL_PLOTS = False
    cfg.GEOMETRIC_INIT = True
    cfg.STAGE1_REDUCTION = "soft_l1"
    cfg.STAGE2_REDUCTION = "soft_l1"
    cfg.STAGE2_LOSS = "focal_spot"
    # User decision 2026-07-26: ALL raytracing at 10 rays/surface point on
    # 25x25 surface points (TRAIN_RAYS/SURFACE_POINTS_PER_FACET defaults
    # already match); eval uses the same protocol so both arms share identical
    # Monte-Carlo conditions.
    cfg.DISPLAY_RAYS = 10
    if smoke:
        cfg.STAGE1_EPOCHS = 5
        cfg.STAGE2_EPOCHS = 3
        cfg.GEOMETRIC_INIT_EPOCHS = 20
        cfg.TRAIN_RAYS = 5
        cfg.DISPLAY_RAYS = 10
        cfg.SPLITTER_TRAIN_SIZE = 20
        cfg.SPLITTER_VAL_SIZE = 10

    for arm_name, blocking in ARMS.items():
        out_dir = OUT_ROOT / heliostat_id / arm_name
        if (out_dir / "results.json").exists():
            log.info(f"[SKIP] {heliostat_id} {arm_name} — results.json exists")
            continue
        cfg.MINI_BATCH_SIZE = 1 if blocking else 25
        cfg.PLOT_EVERY = 5 if blocking else 1
        log.info(f"=== {heliostat_id} — {arm_name} ===")
        t0 = time.time()
        tr.run(
            heliostat_id,
            DATASET_DIR,
            out_dir,
            cfg,
            device,
            scenario_path=scenario_path,
            blocking=blocking,
        )
        log.info(f"    done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


# --------------------------------------------------------------------------- #
# Comparison                                                                   #
# --------------------------------------------------------------------------- #

_STAGES = ["pre_training", "after_stage1", "after_stage2"]
_STAGE_LABELS = {"pre_training": "pre", "after_stage1": "S1", "after_stage2": "S2"}


def _load_arm(hid: str, arm: str) -> dict:
    with open(OUT_ROOT / hid / arm / "results.json") as fh:
        return json.load(fh)


def _load_convergence(hid: str, arm: str) -> list[dict]:
    path = OUT_ROOT / hid / arm / "convergence_history.csv"
    if not path.exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def compare(heliostat_ids: list[str]) -> dict:
    """Build per-heliostat comparison.json + plots and the field summary."""
    summary = {"heliostats": {}}
    for hid in heliostat_ids:
        arms = {a: _load_arm(hid, a) for a in ARMS if (OUT_ROOT / hid / a / "results.json").exists()}
        if len(arms) < 2:
            log.warning(f"{hid}: need both arms to compare, have {list(arms)}")
            continue
        gen_report_path = OUT_ROOT / hid / "generation_report.json"
        gen = json.load(open(gen_report_path)) if gen_report_path.exists() else {}

        entry = {
            "heliostat_id": hid,
            "hel_dist_m": arms["A0_blocking_off"]["hel_dist_m"],
            "blocked_fraction": gen.get("blocked_fraction"),
            "centroid_shift_mrad": gen.get("centroid_shift_mrad"),
            "metrics": {
                stage: {
                    arm: {
                        "centroid_mean": r[stage]["centroid_mrad_mean"],
                        "centroid_median": r[stage]["centroid_mrad_median"],
                        "direction_mean": r[stage].get("direction_mrad_mean"),
                        "direction_median": r[stage].get("direction_mrad_median"),
                    }
                    for arm, r in arms.items()
                }
                for stage in _STAGES
            },
            "cross_eval": {
                arm: r.get("after_stage2_cross_blocking") for arm, r in arms.items()
            },
            "delta_s2_centroid_mean": (
                arms["A1_blocking_on"]["after_stage2"]["centroid_mrad_mean"]
                - arms["A0_blocking_off"]["after_stage2"]["centroid_mrad_mean"]
            ),
            "delta_s2_centroid_median": (
                arms["A1_blocking_on"]["after_stage2"]["centroid_mrad_median"]
                - arms["A0_blocking_off"]["after_stage2"]["centroid_mrad_median"]
            ),
            "delta_s2_direction_median": (
                arms["A1_blocking_on"]["after_stage2"].get("direction_mrad_median", float("nan"))
                - arms["A0_blocking_off"]["after_stage2"].get("direction_mrad_median", float("nan"))
            ),
            "per_sample_test": {
                arm: r["per_sample_test"] for arm, r in arms.items()
            },
            "convergence": {
                arm: _load_convergence(hid, arm) for arm in ARMS
            },
            "total_time_min": {arm: r["total_time_min"] for arm, r in arms.items()},
        }
        out_hid = OUT_ROOT / hid
        with open(out_hid / "comparison.json", "w") as fh:
            json.dump(entry, fh, indent=2)
        summary["heliostats"][hid] = {
            k: entry[k] for k in (
                "hel_dist_m", "blocked_fraction", "centroid_shift_mrad",
                "metrics", "cross_eval",
                "delta_s2_centroid_mean", "delta_s2_centroid_median",
                "delta_s2_direction_median",
            )
        }
        _plot_comparison(hid, entry, out_hid)
        log.info(f"{hid}: comparison.json + plots -> {out_hid}")

    with open(OUT_ROOT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    if len(summary["heliostats"]) >= 2:
        _plot_summary(summary, OUT_ROOT)
    return summary


# --------------------------------------------------------------------------- #
# Plots                                                                        #
# --------------------------------------------------------------------------- #

_ARM_STYLE = {
    "A0_blocking_off": {"color": "#c0392b", "label": "A0 — blocking OFF (baseline)"},
    "A1_blocking_on": {"color": "#2471a3", "label": "A1 — blocking ON"},
}


def _plot_comparison(hid: str, entry: dict, out_dir: pathlib.Path) -> None:
    blocked = (entry.get("blocked_fraction") or {}).get("median")
    shift = (entry.get("centroid_shift_mrad") or {}).get("median")
    subtitle = (
        f"median blocked {blocked * 100:.1f}% · median injected shift {shift:.2f} mrad"
        if blocked is not None and shift is not None else ""
    )

    # 1 — metric bars: centroid + direction, mean and median, per stage per arm
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    x = np.arange(len(_STAGES))
    width = 0.36
    for ax, (metric, title) in zip(axes, [("centroid", "centroid error"), ("direction", "direction error")]):
        for j, (arm, style) in enumerate(_ARM_STYLE.items()):
            means = [entry["metrics"][s][arm][f"{metric}_mean"] for s in _STAGES]
            meds = [entry["metrics"][s][arm][f"{metric}_median"] for s in _STAGES]
            b1 = ax.bar(x + (j - 0.5) * width - width / 4 + 0.0, means,
                        width / 2, color=style["color"], alpha=0.95,
                        label=style["label"] + " (mean)" if metric == "centroid" else None)
            ax.bar(x + (j - 0.5) * width + width / 4, meds, width / 2,
                   color=style["color"], alpha=0.45,
                   label=style["label"] + " (median)" if metric == "centroid" else None)
            for rect, v in zip(b1, means):
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([_STAGE_LABELS[s] for s in _STAGES])
        ax.set_title(title)
        ax.set_ylabel("mrad")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{hid} — test accuracy per stage, A0 vs A1\n{subtitle}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_bars.png", dpi=160)
    plt.close(fig)

    # 2 — ECDF of per-sample test centroid errors, both arms
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for arm, style in _ARM_STYLE.items():
        errs = np.sort(np.array(entry["per_sample_test"][arm]["s2_centroid_mrad"], dtype=float))
        ecdf = np.arange(1, len(errs) + 1) / len(errs)
        ax.step(errs, ecdf, where="post", color=style["color"], label=style["label"], lw=1.8)
        med = float(np.median(errs))
        ax.axvline(med, color=style["color"], ls=":", lw=1)
        ax.text(med, 0.05, f" {med:.2f}", color=style["color"], fontsize=8, rotation=90, va="bottom")
    ax.set_xlabel("test centroid error [mrad]")
    ax.set_ylabel("fraction of samples ≤ x")
    ax.set_title(f"{hid} — per-sample accuracy after Stage 2\n{subtitle}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_ecdf.png", dpi=160)
    plt.close(fig)

    # 3 — convergence overlay (ray-traced train centroid mrad per captured epoch)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for arm, style in _ARM_STYLE.items():
        rows = entry["convergence"][arm]
        eps = [_f(r["epoch"]) for r in rows]
        tr_mrad = [_f(r["mrad_train_mean"]) for r in rows]
        val_mrad = [_f(r["mrad_val_mean"]) for r in rows]
        ax.plot(eps, tr_mrad, color=style["color"], lw=1.4, label=style["label"] + " — train")
        ax.plot(eps, val_mrad, color=style["color"], lw=1.2, ls="--", alpha=0.7,
                label=style["label"] + " — val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("centroid error [mrad]")
    ax.set_yscale("log")
    ax.set_title(f"{hid} — convergence (ray-traced centroid error)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_convergence.png", dpi=160)
    plt.close(fig)

    # 4 — paired per-sample scatter: A0 error vs A1 error (points below diagonal = A1 wins)
    a0 = np.array(entry["per_sample_test"]["A0_blocking_off"]["s2_centroid_mrad"], dtype=float)
    a1 = np.array(entry["per_sample_test"]["A1_blocking_on"]["s2_centroid_mrad"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    lim = max(a0.max(), a1.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="grey", lw=1, ls="--", label="no change")
    ax.scatter(a0, a1, s=22, alpha=0.75, color="#2471a3", edgecolors="none")
    better = float((a1 < a0).mean() * 100)
    ax.set_xlabel("A0 (blocking off) centroid error [mrad]")
    ax.set_ylabel("A1 (blocking on) centroid error [mrad]")
    ax.set_title(f"{hid} — paired samples after Stage 2\n"
                 f"A1 better on {better:.0f}% of samples · {subtitle}", fontsize=10)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_paired_scatter.png", dpi=160)
    plt.close(fig)


def _plot_summary(summary: dict, out_dir: pathlib.Path) -> None:
    hids = list(summary["heliostats"])
    deltas = [summary["heliostats"][h]["delta_s2_centroid_median"] for h in hids]
    blocked = [
        ((summary["heliostats"][h].get("blocked_fraction") or {}).get("median") or 0.0) * 100
        for h in hids
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = ["#2471a3" if d < 0 else "#c0392b" for d in deltas]
    y = np.arange(len(hids))
    ax.barh(y, deltas, color=colors, alpha=0.85)
    for yi, d in zip(y, deltas):
        ax.text(d + (0.002 if d >= 0 else -0.002), yi, f"{d:+.3f} mrad",
                va="center", ha="left" if d >= 0 else "right", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{h}\nblocked {b:.1f}%" for h, b in zip(hids, blocked)],
                       fontsize=9)
    ax.margins(x=0.22)
    ax.set_xlabel("Δ median test centroid error (A1 − A0) [mrad]  —  negative = blocking-aware training wins")
    ax.set_title("Experiment S — effect of modelling blocking, per heliostat")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_delta.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("heliostat_ids", nargs="+")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("artist", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not args.compare_only:
        device = torch.device("cpu")
        for hid in args.heliostat_ids:
            run_arms(hid, args.smoke_test, device)

    compare(args.heliostat_ids)


if __name__ == "__main__":
    main()
