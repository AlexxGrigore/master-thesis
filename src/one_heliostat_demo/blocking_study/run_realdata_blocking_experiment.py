"""Experiment F-real driver — full-field blocking for AY36 on REAL PAINT data.

Same arm pair as run_fullfield_blocking_experiment.py (Experiment F), but the
training data is the REAL PAINT benchmark for AY36 instead of the synthetic
full-field blocking dataset:

  R0_blocking_off — today's pipeline: ray tracer never sees the field.
  R1_blocking_on  — 15-heliostat scenario (AY36 + 14 census blockers); per
                    sample the blockers aim at solar_tower_juelich_lower
                    (cfg.BLOCKER_TARGET_NAME), while the STUDIED heliostat
                    aims at each real sample's own recorded target.

Data: PAINT benchmark ``benchmark_split-balanced_train-50_validation-20``
with its OWN fixed 50/20/20 train/val/test assignment honored verbatim
(cfg.USE_FIXED_SPLIT = True, train.py _load_fixed_split_real; val/test are
swapped per the project-wide SWAP_VAL_TEST convention, so the benchmark's
VALIDATION split is the final-eval test set). The active-pixel quality filter
still applies per split. Real data has no ground-truth kinematics — evaluation
is the standard centroid/direction mrad error against measured centroids.

Schedule: EXACTLY 500 epochs Stage 1 (forward-aim) + 300 epochs Stage 2
(focal-spot) — set here on the imported config module, config.py untouched.
train.py has NO early stopping (both stage loops run the full configured
range; the only in-loop mechanism is ReduceLROnPlateau LR annealing, which
never cuts a stage short), so no early-stop override is needed. The plateau
schedulers are left at their config defaults, identical to the synthetic runs.
AUTO_MOTOR_OFFSET stays False: OPTIMIZE_ACTUATOR_STROKE=True (config default)
already trains the same encoder-zero correction space and the config forbids
enabling both (double correction). This matches the standard real-data
pipeline.

Outputs (all under outputs/new_mapping_function/blocking_study/experiment_full_field/AY36_real/):
  R0_blocking_off/{results.json, convergence_history.csv, plots/...}   (train.py)
  R1_blocking_on/{results.json, convergence_history.csv, plots/...}    (train.py)
  comparison.json + comparison_*.png                                   (this driver)
  per_sample_plot_data.json + per_sample_bars.png                      (this driver)
  summary.json
Log: experiment_full_field/training_real.log

Usage
-----
    python run_realdata_blocking_experiment.py            # both arms, then compare
    python run_realdata_blocking_experiment.py --smoke-test
    python run_realdata_blocking_experiment.py --compare-only
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

# --------------------------------------------------------------------------- #
# Script-local workaround (train.py is shared and must NOT be edited):        #
# tr._load_fixed_split_real._filter_active builds the active mask as           #
# torch.tensor([n_keep]) — correct for single-heliostat scenarios only. In    #
# the 15-heliostat full-field scenario the mask must be a one-hot of length    #
# n_hel holding the sample count at the studied row (same convention as       #
# tr._pool_and_split._slice). Wrap the loader and rebuild the masks after the #
# fact; the data itself is untouched.                                          #
# --------------------------------------------------------------------------- #
_orig_load_fixed_split_real = tr._load_fixed_split_real


def _load_fixed_split_real_multihel(heliostat_id, cfg_, hg, scenario, device):
    train_d, val_d, test_d, pool = _orig_load_fixed_split_real(
        heliostat_id, cfg_, hg, scenario, device
    )
    n_hel = hg.number_of_heliostats
    if n_hel == 1:
        return train_d, val_d, test_d, pool
    hel_idx = hg.names.index(heliostat_id)

    def _fix(d):
        if d is None:
            return None
        n = d[0].shape[0]
        return (d[0], d[1], d[2], d[3],
                tr._one_hot_active(hel_idx, n, n_hel, device), d[5])

    return _fix(train_d), _fix(val_d), _fix(test_d), pool


tr._load_fixed_split_real = _load_fixed_split_real_multihel

log = logging.getLogger(__name__)

_ROOT = _here.parents[2]                                  # master-thesis/
SCENARIO_PATH = (
    _ROOT / "scenarios" / "neighbourhoods_fullfield" / "AY36" / "scenario.h5"
)
OUT_ROOT = (
    _ROOT / "outputs" / "new_mapping_function" / "blocking_study"
    / "experiment_full_field"
)
OUT_HID = OUT_ROOT / "AY36_real"
BENCHMARK_NAME = "benchmark_split-balanced_train-50_validation-20"
BLOCKER_TARGET_NAME = "solar_tower_juelich_lower"

STAGE1_EPOCHS = 500
STAGE2_EPOCHS = 300

ARMS = {"R0_blocking_off": False, "R1_blocking_on": True}

_ARM_STYLE = {
    "R0_blocking_off": {"color": "#c0392b", "label": "R0 — blocking OFF (real)"},
    "R1_blocking_on": {"color": "#2471a3", "label": "R1 — blocking ON (real)"},
}

_STAGES = ["pre_training", "after_stage1", "after_stage2"]
_STAGE_LABELS = {"pre_training": "pre", "after_stage1": "S1", "after_stage2": "S2"}


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #

def _configure(smoke: bool) -> None:
    """Set every cfg value this experiment relies on (additive, script-local).

    Mirrors the synthetic Experiment-F regime, plus the real-data settings
    (benchmark, fixed split) and the longer 500/300 schedule.
    """
    # Real PAINT data via the benchmark's OWN fixed 50/20/20 split.
    cfg.DATA_MODE = "real"
    cfg.BENCHMARK_NAME = BENCHMARK_NAME
    cfg.BENCHMARK_CSV = cfg.PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
    cfg.CALIBRATION_DIR = cfg.PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
    cfg.REAL_FLUX_DIR = cfg.PAINT_DIR / BENCHMARK_NAME / "flux_image"
    cfg.USE_FIXED_SPLIT = True
    cfg.TARGET_FILTER = None            # real samples aim at their recorded targets
    # AUTO_MOTOR_OFFSET deliberately left False: OPTIMIZE_ACTUATOR_STROKE=True
    # (config default) trains the same encoder-zero correction; enabling both
    # double-corrects (config.py note).

    # Training regime — same settled settings as the synthetic Experiment F.
    cfg.GEOMETRIC_INIT = True
    cfg.STAGE1_REDUCTION = "soft_l1"
    cfg.STAGE2_REDUCTION = "soft_l1"
    cfg.STAGE2_LOSS = "focal_spot"
    cfg.STAGE1_TRAIL_PLOTS = False
    cfg.DISPLAY_RAYS = 10
    # Experiment F convention: passive blockers aim at the fixed lower target.
    cfg.BLOCKER_TARGET_NAME = BLOCKER_TARGET_NAME

    # The approved schedule: EXACTLY 500 + 300 epochs. train.py has no early
    # stopping, so setting the epoch counts is sufficient for full-length stages.
    cfg.STAGE1_EPOCHS = STAGE1_EPOCHS
    cfg.STAGE2_EPOCHS = STAGE2_EPOCHS

    if smoke:
        cfg.STAGE1_EPOCHS = 5
        cfg.STAGE2_EPOCHS = 3
        cfg.GEOMETRIC_INIT_EPOCHS = 20
        cfg.TRAIN_RAYS = 5
        cfg.DISPLAY_RAYS = 10


def run_arms(heliostat_id: str, smoke: bool, device: torch.device,
             out_hid: pathlib.Path, arms: dict | None = None) -> None:
    arms = ARMS if arms is None else arms
    if not SCENARIO_PATH.exists():
        raise FileNotFoundError(f"Missing scenario: {SCENARIO_PATH}")
    if not cfg.BENCHMARK_CSV.exists():
        raise FileNotFoundError(f"Missing benchmark CSV: {cfg.BENCHMARK_CSV}")

    _configure(smoke)

    for arm_name, blocking in arms.items():
        out_dir = out_hid / arm_name
        if (out_dir / "results.json").exists():
            log.info(f"[SKIP] {heliostat_id} {arm_name} — results.json exists")
            continue
        cfg.MINI_BATCH_SIZE = 1 if blocking else 25
        cfg.PLOT_EVERY = 5 if blocking else 1
        log.info(f"=== {heliostat_id} REAL — {arm_name} "
                 f"(S1 {cfg.STAGE1_EPOCHS} + S2 {cfg.STAGE2_EPOCHS} epochs) ===")
        t0 = time.time()
        tr.run(
            heliostat_id,
            # dataset_dir is unused in real mode but run() requires a path.
            cfg.PAINT_DIR / BENCHMARK_NAME,
            out_dir,
            cfg,
            device,
            scenario_path=SCENARIO_PATH,
            blocking=blocking,
        )
        log.info(f"    done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


# --------------------------------------------------------------------------- #
# Comparison                                                                   #
# --------------------------------------------------------------------------- #

def _load_arm(out_hid: pathlib.Path, arm: str) -> dict:
    with open(out_hid / arm / "results.json") as fh:
        return json.load(fh)


def _load_convergence(out_hid: pathlib.Path, arm: str) -> list[dict]:
    path = out_hid / arm / "convergence_history.csv"
    if not path.exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def compare(heliostat_id: str, out_hid: pathlib.Path) -> dict | None:
    """Build comparison.json + plots and per-sample bar data for the R0/R1 pair."""
    arm_off, arm_on = list(ARMS)
    arms = {a: _load_arm(out_hid, a) for a in ARMS
            if (out_hid / a / "results.json").exists()}
    if len(arms) < 2:
        log.warning(f"{heliostat_id}: need both arms {list(ARMS)} to compare, "
                    f"have {list(arms)}")
        return None

    entry = {
        "heliostat_id": heliostat_id,
        "experiment": "full_field_real",
        "data_mode": "real",
        "benchmark": BENCHMARK_NAME,
        "split": "benchmark fixed 50/20/20 (USE_FIXED_SPLIT, val/test swapped)",
        "hel_dist_m": arms[arm_off]["hel_dist_m"],
        "n_train": arms[arm_off]["n_train"],
        "n_val": arms[arm_off]["n_val"],
        "n_test": arms[arm_off]["n_test"],
        "blocker_target": BLOCKER_TARGET_NAME,
        "stage1_epochs": STAGE1_EPOCHS,
        "stage2_epochs": STAGE2_EPOCHS,
        "stage2_loss": "focal_spot",
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
            arms[arm_on]["after_stage2"]["centroid_mrad_mean"]
            - arms[arm_off]["after_stage2"]["centroid_mrad_mean"]
        ),
        "delta_s2_centroid_median": (
            arms[arm_on]["after_stage2"]["centroid_mrad_median"]
            - arms[arm_off]["after_stage2"]["centroid_mrad_median"]
        ),
        "delta_s2_direction_median": (
            arms[arm_on]["after_stage2"].get("direction_mrad_median", float("nan"))
            - arms[arm_off]["after_stage2"].get("direction_mrad_median", float("nan"))
        ),
        "per_sample_test": {
            arm: r["per_sample_test"] for arm, r in arms.items()
        },
        "convergence": {
            arm: _load_convergence(out_hid, arm) for arm in ARMS
        },
        "total_time_min": {arm: r["total_time_min"] for arm, r in arms.items()},
    }
    with open(out_hid / "comparison.json", "w") as fh:
        json.dump(entry, fh, indent=2)
    _plot_comparison(heliostat_id, entry, out_hid)
    try:
        _per_sample_bars(heliostat_id, entry, out_hid)
    except Exception as exc:  # noqa: BLE001 — bar data must not block the comparison
        log.warning(f"{heliostat_id}: per-sample bar data failed: {exc}")
    log.info(f"{heliostat_id}: comparison.json + plots -> {out_hid}")
    return entry


def summarize(heliostat_id: str, out_hid: pathlib.Path) -> dict:
    """Write summary.json over both arms + summary_arms.png."""
    arms = {a: _load_arm(out_hid, a) for a in ARMS
            if (out_hid / a / "results.json").exists()}
    summary = {
        "experiment": "full_field_real",
        "data_mode": "real",
        "benchmark": BENCHMARK_NAME,
        "blocker_target": BLOCKER_TARGET_NAME,
        "stage1_epochs": STAGE1_EPOCHS,
        "stage2_epochs": STAGE2_EPOCHS,
        "heliostats": {},
    }
    if arms:
        per_arm = {}
        for a, r in arms.items():
            s2 = r["after_stage2"]
            per_arm[a] = {
                "stage2_loss": "focal_spot",
                "blocking": ARMS[a],
                "centroid_mrad_mean": s2["centroid_mrad_mean"],
                "centroid_mrad_median": s2["centroid_mrad_median"],
                "direction_mrad_mean": s2.get("direction_mrad_mean"),
                "direction_mrad_median": s2.get("direction_mrad_median"),
                "cross_eval": r.get("after_stage2_cross_blocking"),
                "total_time_min": r["total_time_min"],
            }
        summary["heliostats"][heliostat_id] = {
            "hel_dist_m": arms[next(iter(arms))]["hel_dist_m"],
            "n_train": arms[next(iter(arms))]["n_train"],
            "n_val": arms[next(iter(arms))]["n_val"],
            "n_test": arms[next(iter(arms))]["n_test"],
            "arms": per_arm,
        }
        _plot_summary_arms(heliostat_id, summary["heliostats"][heliostat_id], out_hid)
    with open(out_hid / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"summary.json ({sum(len(v['arms']) for v in summary['heliostats'].values())} arms) "
             f"-> {out_hid / 'summary.json'}")
    return summary


def _plot_summary_arms(hid: str, entry: dict, out_dir: pathlib.Path) -> None:
    arms = entry["arms"]
    names = list(arms)
    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(6.0, 1.9 * len(names)), 4.8))
    for j, (metric, label) in enumerate([
        ("centroid_mrad_mean", "centroid mean"),
        ("centroid_mrad_median", "centroid median"),
        ("direction_mrad_median", "direction median"),
    ]):
        vals = [arms[a][metric] for a in names]
        bars = ax.bar(x + (j - 1) * width, vals, width, label=label, alpha=0.9)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{a}\n({'blocking on' if arms[a]['blocking'] else 'blocking off'})" for a in names],
        fontsize=8,
    )
    ax.set_ylabel("mrad")
    ax.set_title(f"{hid} — REAL data, after Stage 2 (500+300 epochs)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_arms.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Per-sample side-by-side bar data (real data: no known blocked fraction)     #
# --------------------------------------------------------------------------- #

def _per_sample_bars(hid: str, entry: dict, out_dir: pathlib.Path) -> None:
    arm_off, arm_on = list(ARMS)
    r0 = np.array(entry["per_sample_test"][arm_off]["s2_centroid_mrad"], dtype=float)
    r1 = np.array(entry["per_sample_test"][arm_on]["s2_centroid_mrad"], dtype=float)
    if len(r0) != len(r1):
        raise ValueError(f"{hid}: length mismatch r0={len(r0)} r1={len(r1)}")

    merged = {
        "heliostat_id": hid,
        "experiment": "full_field_real",
        "data_mode": "real",
        "benchmark": BENCHMARK_NAME,
        "blocker_target": BLOCKER_TARGET_NAME,
        "stage2_loss": entry.get("stage2_loss"),
        "samples": [
            {
                "sample_index": int(i),
                # Real data carries no synthetic blocked-fraction / injected-shift
                # ground truth — explicitly null.
                "blocked_fraction_percent": None,
                "injected_shift_mrad": None,
                "r0_centroid_mrad": float(r0[i]),
                "r1_centroid_mrad": float(r1[i]),
                "delta_r0_minus_r1_mrad": float(r0[i] - r1[i]),
            }
            for i in range(len(r0))
        ],
        "r0_mean": float(r0.mean()), "r1_mean": float(r1.mean()),
        "r0_median": float(np.median(r0)), "r1_median": float(np.median(r1)),
    }
    with open(out_dir / "per_sample_plot_data.json", "w") as fh:
        json.dump(merged, fh, indent=2)

    order = np.argsort(r0)
    x = np.arange(len(order))
    width = 0.42
    c0 = _ARM_STYLE[arm_off]["color"]
    c1 = _ARM_STYLE[arm_on]["color"]
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.6], "hspace": 0.08},
    )
    ax.bar(x - width / 2, r0[order], width, color=c0, alpha=0.9,
           label=_ARM_STYLE[arm_off]["label"])
    ax.bar(x + width / 2, r1[order], width, color=c1, alpha=0.9,
           label=_ARM_STYLE[arm_on]["label"])
    ax.axhline(r0.mean(), color=c0, ls=":", lw=1.2)
    ax.axhline(r1.mean(), color=c1, ls=":", lw=1.2)
    ax.set_ylabel("test centroid error [mrad]")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(
        f"{hid} — REAL data, full-field blocking, per-sample test error after Stage 2\n"
        f"R0 mean {r0.mean():.3f} / median {np.median(r0):.3f} mrad   ·   "
        f"R1 mean {r1.mean():.3f} / median {np.median(r1):.3f} mrad   ·   "
        f"R1 better on {(r1 < r0).mean() * 100:.0f}% of samples",
        fontsize=10.5,
    )
    delta = r0 - r1
    colors = np.where(delta[order] >= 0, c1, c0)
    ax2.bar(x, delta[order], 0.8, color=colors, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Δ (R0 − R1) [mrad]")
    ax2.set_xlabel("test sample (sorted by R0 error →)")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "per_sample_bars.png", dpi=160)
    plt.close(fig)
    log.info(f"{hid}: per_sample_bars.png + per_sample_plot_data.json -> {out_dir}")


# --------------------------------------------------------------------------- #
# Plots                                                                        #
# --------------------------------------------------------------------------- #

def _plot_comparison(hid: str, entry: dict, out_dir: pathlib.Path) -> None:
    arm_off, arm_on = list(ARMS)
    subtitle = (f"REAL PAINT {BENCHMARK_NAME} · fixed 50/20/20 split · "
                f"blockers → {BLOCKER_TARGET_NAME}")

    # 1 — metric bars: centroid + direction, mean and median, per stage per arm
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    x = np.arange(len(_STAGES))
    width = 0.36
    for ax, (metric, title) in zip(axes, [("centroid", "centroid error"), ("direction", "direction error")]):
        for j, (arm, style) in enumerate(_ARM_STYLE.items()):
            means = [entry["metrics"][s][arm][f"{metric}_mean"] for s in _STAGES]
            meds = [entry["metrics"][s][arm][f"{metric}_median"] for s in _STAGES]
            b1 = ax.bar(x + (j - 0.5) * width - width / 4, means,
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
    fig.suptitle(f"{hid} — REAL full-field blocking, test accuracy per stage, R0 vs R1\n{subtitle}",
                 fontsize=11)
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
    ax.set_title(f"{hid} — REAL per-sample accuracy after Stage 2\n{subtitle}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_ecdf.png", dpi=160)
    plt.close(fig)

    # 3 — convergence overlay
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for arm, style in _ARM_STYLE.items():
        rows = entry["convergence"][arm]
        eps = [_f(r["epoch"]) for r in rows]
        tr_mrad = [_f(r["mrad_train_mean"]) for r in rows]
        val_mrad = [_f(r["mrad_val_mean"]) for r in rows]
        ax.plot(eps, tr_mrad, color=style["color"], lw=1.4, label=style["label"] + " — train")
        ax.plot(eps, val_mrad, color=style["color"], lw=1.2, ls="--", alpha=0.7,
                label=style["label"] + " — val")
    ax.axvline(STAGE1_EPOCHS, color="gray", ls=":", lw=1.0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("centroid error [mrad]")
    ax.set_yscale("log")
    ax.set_title(f"{hid} — REAL convergence (ray-traced centroid error)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_convergence.png", dpi=160)
    plt.close(fig)

    # 4 — paired per-sample scatter (points below diagonal = blocking-on arm wins)
    r0 = np.array(entry["per_sample_test"][arm_off]["s2_centroid_mrad"], dtype=float)
    r1 = np.array(entry["per_sample_test"][arm_on]["s2_centroid_mrad"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    lim = max(r0.max(), r1.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="grey", lw=1, ls="--", label="no change")
    ax.scatter(r0, r1, s=22, alpha=0.75, color=_ARM_STYLE[arm_on]["color"], edgecolors="none")
    better = float((r1 < r0).mean() * 100)
    ax.set_xlabel(f"{arm_off} centroid error [mrad]")
    ax.set_ylabel(f"{arm_on} centroid error [mrad]")
    ax.set_title(f"{hid} — REAL paired samples after Stage 2\n"
                 f"R1 better on {better:.0f}% of samples", fontsize=10)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_paired_scatter.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--heliostat-id", default="AY36")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--arm", choices=list(ARMS), default=None,
                        help="run a single arm (default: both)")
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("artist", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(OUT_ROOT / "training_real.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    out_hid = OUT_ROOT / ("AY36_real_smoke" if args.smoke_test else "AY36_real")

    if not args.compare_only:
        device = torch.device("cpu")  # ARTIST only supports CPU on macOS (get_device)
        arms = {args.arm: ARMS[args.arm]} if args.arm is not None else None
        run_arms(args.heliostat_id, args.smoke_test, device, out_hid, arms=arms)

    if not args.smoke_test and args.arm is None:
        compare(args.heliostat_id, out_hid)
        summarize(args.heliostat_id, out_hid)


if __name__ == "__main__":
    main()
