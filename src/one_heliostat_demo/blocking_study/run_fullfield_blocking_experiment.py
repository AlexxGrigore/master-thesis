"""Experiment F driver — full-field (1277-heliostat) blocking for AY36.

Same question as Experiment S (run_blocking_experiment.py), but the blocking
environment is the WHOLE field: 1276 passive blockers, aimed per sample at
``solar_tower_juelich_lower`` under that sample's sun direction (the generation
convention of generate_fullfield_blocking_dataset.py). Two arms train on the
SAME full-field blocking dataset through the identical two-stage pipeline:

  B0_blocking_off — today's pipeline: ray tracer never sees the field.
  B1_blocking_on  — reduced full-field scenario (AY36 + the 14 census blockers);
                    per sample the blockers aim at solar_tower_juelich_lower
                    (cfg.BLOCKER_TARGET_NAME).

Both arms also get a CROSS evaluation (same parameters, opposite blocking
setting) — stored in results.json as after_stage2_cross_blocking.

Training schedule: EXACTLY 100 epochs Stage 1 + 100 epochs Stage 2 (the config
module defaults are already 100/100; set explicitly here, no global change).
Geometric init is the same closed-form seed as Experiment S.

Outputs (all under outputs/new_mapping_function/blocking_study/experiment_full_field/):
  AY36/{arm}/results.json, convergence_history.csv, plots/...      (train.py)
  AY36/comparison.json + comparison_*.png                          (this driver)
  AY36/per_sample_plot_data.json + per_sample_bars.png             (this driver)
  summary.json

Usage
-----
    python run_fullfield_blocking_experiment.py AY36
    python run_fullfield_blocking_experiment.py AY36 --smoke-test
    python run_fullfield_blocking_experiment.py AY36 --compare-only
    python run_fullfield_blocking_experiment.py AY36 --stage2-loss contour
        # follow-up: C0/C1 arms with the Wortberg upper-contour Stage-2 loss;
        # outputs get a '_contour' suffix, the B0/B1 files stay untouched
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
DATASET_DIR = _ROOT / "datasets" / "synthetic" / "fullfield_blocking_dataset" / "dataset"
SCENARIO_PATH = (
    _ROOT / "scenarios" / "neighbourhoods_fullfield" / "AY36" / "scenario.h5"
)
OUT_ROOT = (
    _ROOT / "outputs" / "new_mapping_function" / "blocking_study" / "experiment_full_field"
)
BLOCKER_TARGET_NAME = "solar_tower_juelich_lower"

ARMS = {"B0_blocking_off": False, "B1_blocking_on": True}
# Follow-up experiment (contour Stage-2 loss, approved 2026-07-28): same dataset,
# same blocking conventions as B0/B1, but Stage 2 optimizes the Wortberg
# upper-contour loss (cfg.STAGE2_LOSS = "contour", config values untouched).
ARMS_CONTOUR = {"C0_blocking_off_contour": False, "C1_blocking_on_contour": True}


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #

def run_arms(heliostat_id: str, smoke: bool, device: torch.device,
             stage2_loss: str = "focal_spot") -> None:
    arms = ARMS if stage2_loss == "focal_spot" else ARMS_CONTOUR
    if not SCENARIO_PATH.exists():
        raise FileNotFoundError(f"Missing scenario: {SCENARIO_PATH}")
    if not (DATASET_DIR / "train" / heliostat_id).exists():
        raise FileNotFoundError(
            f"Missing full-field blocking dataset for {heliostat_id} under {DATASET_DIR}. "
            "Run generate_fullfield_blocking_dataset.py first."
        )

    # Same settled regime as Experiment S (user decision 2026-07-26), except the
    # schedule: EXACTLY 100 epochs per stage for Experiment F.
    cfg.DATA_MODE = "synthetic"
    cfg.STAGE1_EPOCHS = 100
    cfg.STAGE2_EPOCHS = 100
    cfg.STAGE1_TRAIL_PLOTS = False
    cfg.GEOMETRIC_INIT = True
    cfg.STAGE1_REDUCTION = "soft_l1"
    cfg.STAGE2_REDUCTION = "soft_l1"
    cfg.STAGE2_LOSS = stage2_loss
    cfg.DISPLAY_RAYS = 10
    # Experiment F: passive blockers aim at the fixed generation target, not at
    # each sample's own target (additive train.py hook; unset = Experiment S).
    cfg.BLOCKER_TARGET_NAME = BLOCKER_TARGET_NAME
    if smoke:
        cfg.STAGE1_EPOCHS = 5
        cfg.STAGE2_EPOCHS = 3
        cfg.GEOMETRIC_INIT_EPOCHS = 20
        cfg.TRAIN_RAYS = 5
        cfg.DISPLAY_RAYS = 10
        cfg.SPLITTER_TRAIN_SIZE = 20
        cfg.SPLITTER_VAL_SIZE = 10

    for arm_name, blocking in arms.items():
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
            scenario_path=SCENARIO_PATH,
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


def compare(heliostat_ids: list[str], arms_dict: dict = ARMS, suffix: str = "",
            key_prefix: str = "b") -> dict:
    """Build per-heliostat comparison{suffix}.json + plots and per-sample bar data.

    ``arms_dict`` maps {arm_name: blocking_flag} for one blocking pair (B0/B1 or
    C0/C1); the FIRST key is the blocking-off arm, the SECOND the blocking-on arm.
    ``suffix`` differentiates the contour follow-up outputs ("_contour") from the
    original focal-spot ones (""), which are never touched.
    """
    arm_off, arm_on = list(arms_dict)
    comparisons = {}
    for hid in heliostat_ids:
        arms = {a: _load_arm(hid, a) for a in arms_dict
                if (OUT_ROOT / hid / a / "results.json").exists()}
        if len(arms) < 2:
            log.warning(f"{hid}: need both arms {list(arms_dict)} to compare, have {list(arms)}")
            continue
        gen_report_path = OUT_ROOT / hid / "generation_report.json"
        gen = json.load(open(gen_report_path)) if gen_report_path.exists() else {}

        entry = {
            "heliostat_id": hid,
            "hel_dist_m": arms[arm_off]["hel_dist_m"],
            "n_heliostats": gen.get("n_heliostats"),
            "blocker_aim": gen.get("blocker_aim"),
            "blocked_fraction": gen.get("blocked_fraction"),
            "centroid_shift_mrad": gen.get("centroid_shift_mrad"),
            "stage2_loss": "contour" if suffix == "_contour" else "focal_spot",
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
                arm: _load_convergence(hid, arm) for arm in arms_dict
            },
            "total_time_min": {arm: r["total_time_min"] for arm, r in arms.items()},
        }
        out_hid = OUT_ROOT / hid
        with open(out_hid / f"comparison{suffix}.json", "w") as fh:
            json.dump(entry, fh, indent=2)
        _plot_comparison(hid, entry, out_hid, arms_dict, suffix=suffix)
        try:
            _per_sample_bars(hid, entry, gen, out_hid, arms_dict,
                             device=torch.device("cpu"), suffix=suffix,
                             key_prefix=key_prefix)
        except Exception as exc:  # noqa: BLE001 — bar data must not block the comparison
            log.warning(f"{hid}: per-sample bar data failed: {exc}")
        comparisons[hid] = entry
        log.info(f"{hid}: comparison{suffix}.json + plots -> {out_hid}")
    return comparisons


def summarize(heliostat_ids: list[str]) -> dict:
    """Rewrite summary.json + summary_arms.png over ALL arms that have results
    (B0/B1 focal-spot and C0/C1 contour), so the four arms compare side by side."""
    all_arms = {**ARMS, **ARMS_CONTOUR}
    summary = {"experiment": "full_field", "blocker_target": BLOCKER_TARGET_NAME,
               "heliostats": {}}
    gen_report_path = None
    for hid in heliostat_ids:
        arms = {a: _load_arm(hid, a) for a in all_arms
                if (OUT_ROOT / hid / a / "results.json").exists()}
        if not arms:
            continue
        gen_report_path = OUT_ROOT / hid / "generation_report.json"
        gen = json.load(open(gen_report_path)) if gen_report_path.exists() else {}
        per_arm = {}
        for a, r in arms.items():
            s2 = r["after_stage2"]
            per_arm[a] = {
                "stage2_loss": "contour" if a.endswith("_contour") else "focal_spot",
                "blocking": all_arms[a],
                "centroid_mrad_mean": s2["centroid_mrad_mean"],
                "centroid_mrad_median": s2["centroid_mrad_median"],
                "direction_mrad_mean": s2.get("direction_mrad_mean"),
                "direction_mrad_median": s2.get("direction_mrad_median"),
                "cross_eval": r.get("after_stage2_cross_blocking"),
                "total_time_min": r["total_time_min"],
            }
        summary["heliostats"][hid] = {
            "hel_dist_m": arms[next(iter(arms))]["hel_dist_m"],
            "n_heliostats": gen.get("n_heliostats"),
            "blocker_aim": gen.get("blocker_aim"),
            "blocked_fraction": gen.get("blocked_fraction"),
            "centroid_shift_mrad": gen.get("centroid_shift_mrad"),
            "arms": per_arm,
        }
        _plot_summary_arms(hid, summary["heliostats"][hid], OUT_ROOT / hid)
    with open(OUT_ROOT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"summary.json ({sum(len(v['arms']) for v in summary['heliostats'].values())} arms) "
             f"-> {OUT_ROOT / 'summary.json'}")
    return summary


def _plot_summary_arms(hid: str, entry: dict, out_dir: pathlib.Path) -> None:
    """Grouped bars: after-Stage-2 centroid (mean+median) and direction median,
    one group per arm (B0, B1, C0, C1)."""
    arms = entry["arms"]
    names = list(arms)
    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(7.5, 1.9 * len(names)), 4.8))
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
        [f"{a}\n({'blocking on' if arms[a]['blocking'] else 'blocking off'}, "
         f"{arms[a]['stage2_loss']})" for a in names],
        fontsize=8,
    )
    blocked = (entry.get("blocked_fraction") or {}).get("median")
    subtitle = (f"median blocked {blocked * 100:.1f}%" if blocked is not None else "")
    ax.set_ylabel("mrad")
    ax.set_title(f"{hid} — after Stage 2, all Experiment-F arms\n{subtitle}", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_arms.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Per-sample side-by-side bar data (same schema as experiment_s, keys b0/b1)   #
# --------------------------------------------------------------------------- #

def _evaluated_test_set(hid: str, device: torch.device):
    """Rebuild the exact test set train.py evaluated on (same code path as
    plot_per_sample_bars.py, but with the full-field scenario/dataset)."""
    scenario, hg, _dist, hel_idx = tr._load_scenario(hid, cfg, device, SCENARIO_PATH)
    n_hel = hg.number_of_heliostats
    splits = [
        tr._load_split(hid, DATASET_DIR, s, hg, scenario, device)
        for s in ("train", "val", "test")
    ]
    (_trn, _va, test_tuple, _pool) = tr._pool_and_split(
        hid, *splits,
        getattr(cfg, "SPLITTER_TRAIN_SIZE", 100), cfg, device,
        swap_val_test=getattr(cfg, "SWAP_VAL_TEST", True),
        hel_idx=hel_idx, n_hel=n_hel,
    )
    return test_tuple[2].cpu().numpy(), test_tuple[5].cpu().numpy()


def _per_sample_bars(hid: str, entry: dict, gen: dict, out_dir: pathlib.Path,
                     arms_dict: dict, device: torch.device, suffix: str = "",
                     key_prefix: str = "b") -> None:
    arm_off, arm_on = list(arms_dict)
    k0, k1 = f"{key_prefix}0", f"{key_prefix}1"
    b0 = np.array(entry["per_sample_test"][arm_off]["s2_centroid_mrad"], dtype=float)
    b1 = np.array(entry["per_sample_test"][arm_on]["s2_centroid_mrad"], dtype=float)
    test_rays, test_targets = _evaluated_test_set(hid, device)
    if len(test_rays) != len(b0) or len(b0) != len(b1):
        raise ValueError(
            f"{hid}: length mismatch evaluated={len(test_rays)} b0={len(b0)} b1={len(b1)}"
        )

    pool: dict[tuple, list] = {}
    for s in gen.get("samples", []):
        r = s["incident_ray_direction"]
        pool.setdefault(
            (round(r[0], 6), round(r[1], 6), round(r[2], 6), int(s["target_area_index"])),
            [],
        ).append(s)
    blocked, shift = [], []
    for i in range(len(test_rays)):
        r = test_rays[i]
        key = (round(float(r[0]), 6), round(float(r[1]), 6), round(float(r[2]), 6),
               int(test_targets[i]))
        e = pool[key].pop(0)
        blocked.append(float(e["blocked_fraction"]))
        shift.append(float(e.get("centroid_shift_mrad", float("nan"))))
    blocked = np.array(blocked) * 100.0
    shift = np.array(shift)

    merged = {
        "heliostat_id": hid,
        "experiment": "full_field",
        "blocker_target": BLOCKER_TARGET_NAME,
        "stage2_loss": entry.get("stage2_loss"),
        "samples": [
            {
                "sample_index": int(i),
                "blocked_fraction_percent": float(blocked[i]),
                "injected_shift_mrad": float(shift[i]),
                f"{k0}_centroid_mrad": float(b0[i]),
                f"{k1}_centroid_mrad": float(b1[i]),
                f"delta_{k0}_minus_{k1}_mrad": float(b0[i] - b1[i]),
            }
            for i in range(len(b0))
        ],
        f"{k0}_mean": float(b0.mean()), f"{k1}_mean": float(b1.mean()),
        f"{k0}_median": float(np.median(b0)), f"{k1}_median": float(np.median(b1)),
    }
    with open(out_dir / f"per_sample_plot_data{suffix}.json", "w") as fh:
        json.dump(merged, fh, indent=2)

    order = np.argsort(blocked)
    x = np.arange(len(order))
    width = 0.42
    c0 = _ARM_STYLE[arm_off]["color"]
    c1 = _ARM_STYLE[arm_on]["color"]
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.6], "hspace": 0.08},
    )
    ax.bar(x - width / 2, b0[order], width, color=c0, alpha=0.9,
           label=_ARM_STYLE[arm_off]["label"])
    ax.bar(x + width / 2, b1[order], width, color=c1, alpha=0.9,
           label=_ARM_STYLE[arm_on]["label"])
    ax.axhline(b0.mean(), color=c0, ls=":", lw=1.2)
    ax.axhline(b1.mean(), color=c1, ls=":", lw=1.2)
    ax.set_ylabel("test centroid error [mrad]")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(
        f"{hid} — full-field blocking ({entry.get('stage2_loss')} loss), per-sample test error after Stage 2\n"
        f"{k0.upper()} mean {b0.mean():.3f} / median {np.median(b0):.3f} mrad   ·   "
        f"{k1.upper()} mean {b1.mean():.3f} / median {np.median(b1):.3f} mrad   ·   "
        f"{k1.upper()} better on {(b1 < b0).mean() * 100:.0f}% of samples",
        fontsize=10.5,
    )
    delta = b0 - b1
    colors = np.where(delta[order] >= 0, c1, c0)
    ax2.bar(x, delta[order], 0.8, color=colors, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel(f"Δ ({k0.upper()} − {k1.upper()}) [mrad]")
    ax2.set_xlabel("test sample (sorted by blocked fraction →)")
    ax2.grid(axis="y", alpha=0.3)
    ax2b = ax2.twinx()
    ax2b.plot(x, blocked[order], color="grey", lw=1.2, ls="--", alpha=0.8)
    ax2b.set_ylabel("blocked [%]", color="grey")
    ax2b.tick_params(axis="y", labelcolor="grey")
    ax2b.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / f"per_sample_bars{suffix}.png", dpi=160)
    plt.close(fig)
    log.info(f"{hid}: per_sample_bars{suffix}.png + per_sample_plot_data{suffix}.json -> {out_dir}")


# --------------------------------------------------------------------------- #
# Plots                                                                        #
# --------------------------------------------------------------------------- #

_ARM_STYLE = {
    "B0_blocking_off": {"color": "#c0392b", "label": "B0 — blocking OFF (focal-spot)"},
    "B1_blocking_on": {"color": "#2471a3", "label": "B1 — blocking ON (focal-spot)"},
    "C0_blocking_off_contour": {"color": "#e67e22", "label": "C0 — blocking OFF (contour)"},
    "C1_blocking_on_contour": {"color": "#148f77", "label": "C1 — blocking ON (contour)"},
}


def _plot_comparison(hid: str, entry: dict, out_dir: pathlib.Path,
                     arms_dict: dict, suffix: str = "") -> None:
    arm_off, arm_on = list(arms_dict)
    styles = {a: _ARM_STYLE[a] for a in arms_dict}
    pair_label = f"{arm_off.split('_')[0]} vs {arm_on.split('_')[0]}"
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
        for j, (arm, style) in enumerate(styles.items()):
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
    fig.suptitle(f"{hid} — full-field blocking, test accuracy per stage, {pair_label}\n{subtitle}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"comparison_bars{suffix}.png", dpi=160)
    plt.close(fig)

    # 2 — ECDF of per-sample test centroid errors, both arms
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for arm, style in styles.items():
        errs = np.sort(np.array(entry["per_sample_test"][arm]["s2_centroid_mrad"], dtype=float))
        ecdf = np.arange(1, len(errs) + 1) / len(errs)
        ax.step(errs, ecdf, where="post", color=style["color"], label=style["label"], lw=1.8)
        med = float(np.median(errs))
        ax.axvline(med, color=style["color"], ls=":", lw=1)
        ax.text(med, 0.05, f" {med:.2f}", color=style["color"], fontsize=8, rotation=90, va="bottom")
    ax.set_xlabel("test centroid error [mrad]")
    ax.set_ylabel("fraction of samples ≤ x")
    ax.set_title(f"{hid} — per-sample accuracy after Stage 2 ({entry.get('stage2_loss')})\n{subtitle}",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"comparison_ecdf{suffix}.png", dpi=160)
    plt.close(fig)

    # 3 — convergence overlay (ray-traced train centroid mrad per captured epoch)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for arm, style in styles.items():
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
    ax.set_title(f"{hid} — convergence (ray-traced centroid error, {entry.get('stage2_loss')})",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"comparison_convergence{suffix}.png", dpi=160)
    plt.close(fig)

    # 4 — paired per-sample scatter (points below diagonal = blocking-on arm wins)
    b0 = np.array(entry["per_sample_test"][arm_off]["s2_centroid_mrad"], dtype=float)
    b1 = np.array(entry["per_sample_test"][arm_on]["s2_centroid_mrad"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    lim = max(b0.max(), b1.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="grey", lw=1, ls="--", label="no change")
    ax.scatter(b0, b1, s=22, alpha=0.75, color=styles[arm_on]["color"], edgecolors="none")
    better = float((b1 < b0).mean() * 100)
    ax.set_xlabel(f"{arm_off} centroid error [mrad]")
    ax.set_ylabel(f"{arm_on} centroid error [mrad]")
    ax.set_title(f"{hid} — paired samples after Stage 2\n"
                 f"{arm_on.split('_')[0]} better on {better:.0f}% of samples · {subtitle}",
                 fontsize=10)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / f"comparison_paired_scatter{suffix}.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #

_LOSS_SPECS = {
    "focal_spot": {"arms": ARMS, "suffix": "", "key_prefix": "b"},
    "contour": {"arms": ARMS_CONTOUR, "suffix": "_contour", "key_prefix": "c"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("heliostat_ids", nargs="+", default=["AY36"])
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--stage2-loss", choices=list(_LOSS_SPECS), default="focal_spot",
                        help="focal_spot = B0/B1 (default, original outputs); "
                             "contour = C0/C1 follow-up arms (outputs get a "
                             "'_contour' suffix; existing files untouched).")
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("artist", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    spec = _LOSS_SPECS[args.stage2_loss]
    if not args.compare_only:
        device = torch.device("cpu")  # ARTIST only supports CPU on macOS (get_device)
        for hid in args.heliostat_ids:
            run_arms(hid, args.smoke_test, device, stage2_loss=args.stage2_loss)

    compare(args.heliostat_ids, arms_dict=spec["arms"], suffix=spec["suffix"],
            key_prefix=spec["key_prefix"])
    summarize(args.heliostat_ids)


if __name__ == "__main__":
    main()
