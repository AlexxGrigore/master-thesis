"""Per-sample A0-vs-A1 bar plots for Experiment S (reads saved JSONs only).

For each heliostat, plots every evaluated test sample's Stage-2 centroid error
for both arms (A0 blocking off / A1 blocking on), sorted by the sample's
blocked fraction, plus a delta-vs-blocked panel that shows WHICH sun positions
gain from blocking-aware training.

IMPORTANT (split bookkeeping): train.py does NOT evaluate the dataset's own
test/ folder — it pools train+val+test, filters for active pixels, re-splits
with the PAINT DatasetSplitter (balanced) and swaps val/test. This script
therefore rebuilds the EVALUATED test set through the exact same code path
(tr._load_split + tr._pool_and_split with the same cfg defaults) and matches
each evaluated sample to its generation-report entry by (incident ray
direction, target area index) — identity matching, never index guessing.

Outputs per heliostat: per_sample_bars.png + per_sample_plot_data.json.

    python plot_per_sample_bars.py AY36 BA35 BE35 AC33
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys

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

import config as cfg  # noqa: E402
import train as tr  # noqa: E402

log = logging.getLogger(__name__)

_ROOT = _here.parents[2]  # master-thesis/
DATASET_DIR = _ROOT / "datasets" / "synthetic" / "blocking_dataset" / "dataset"
SCENARIO_ROOT = _ROOT / "scenarios" / "neighbourhoods"
OUT_ROOT = _ROOT / "outputs" / "new_mapping_function" / "blocking_study" / "experiment_s"

ARM_STYLE = {
    "A0_blocking_off": {"color": "#c0392b", "label": "A0 — blocking OFF"},
    "A1_blocking_on": {"color": "#2471a3", "label": "A1 — blocking ON"},
}


def _evaluated_test_set(hid: str, device: torch.device):
    """Rebuild the exact test set train.py evaluated on (same code path)."""
    scenario, hg, _dist, hel_idx = tr._load_scenario(
        hid, cfg, device, SCENARIO_ROOT / hid / "scenario.h5"
    )
    n_hel = hg.number_of_heliostats
    splits = [
        tr._load_split(hid, DATASET_DIR, s, hg, scenario, device)
        for s in ("train", "val", "test")
    ]
    (_tr, _va, test_tuple, _pool) = tr._pool_and_split(
        hid, *splits,
        getattr(cfg, "SPLITTER_TRAIN_SIZE", 100), cfg, device,
        swap_val_test=getattr(cfg, "SWAP_VAL_TEST", True),
        hel_idx=hel_idx, n_hel=n_hel,
    )
    test_rays = test_tuple[2].cpu().numpy()
    test_targets = test_tuple[5].cpu().numpy()
    return test_rays, test_targets


def _match_report_entries(gen: dict, test_rays, test_targets, hid: str):
    """Per evaluated test sample: blocked fraction + injected shift, matched
    to generation-report entries by (ray direction, target index)."""
    pool: dict[tuple, dict] = {}
    for s in gen["samples"]:
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
        entries = pool.get(key)
        if not entries:
            raise KeyError(f"{hid}: no generation-report entry for test sample {i} {key}")
        entry = entries.pop(0)  # pop handles duplicate sun positions
        blocked.append(float(entry["blocked_fraction"]))
        shift.append(float(entry["centroid_shift_mrad"]))
    return np.array(blocked) * 100.0, np.array(shift)


def plot_heliostat(hid: str, device: torch.device) -> None:
    out_dir = OUT_ROOT / hid
    comp = json.load(open(out_dir / "comparison.json"))
    gen = json.load(open(out_dir / "generation_report.json"))

    a0 = np.array(comp["per_sample_test"]["A0_blocking_off"]["s2_centroid_mrad"], dtype=float)
    a1 = np.array(comp["per_sample_test"]["A1_blocking_on"]["s2_centroid_mrad"], dtype=float)

    test_rays, test_targets = _evaluated_test_set(hid, device)
    if len(test_rays) != len(a0) or len(a0) != len(a1):
        raise ValueError(
            f"{hid}: length mismatch evaluated={len(test_rays)} a0={len(a0)} a1={len(a1)}"
        )
    blocked, shift = _match_report_entries(gen, test_rays, test_targets, hid)

    # Merged plotting data — restyle without touching training outputs.
    merged = {
        "heliostat_id": hid,
        "samples": [
            {
                "sample_index": int(i),
                "blocked_fraction_percent": float(blocked[i]),
                "injected_shift_mrad": float(shift[i]),
                "a0_centroid_mrad": float(a0[i]),
                "a1_centroid_mrad": float(a1[i]),
                "delta_a0_minus_a1_mrad": float(a0[i] - a1[i]),
            }
            for i in range(len(a0))
        ],
        "a0_mean": float(a0.mean()), "a1_mean": float(a1.mean()),
        "a0_median": float(np.median(a0)), "a1_median": float(np.median(a1)),
    }
    with open(out_dir / "per_sample_plot_data.json", "w") as fh:
        json.dump(merged, fh, indent=2)

    order = np.argsort(blocked)  # unblocked left -> heavily blocked right
    x = np.arange(len(order))
    width = 0.42

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.6], "hspace": 0.08},
    )
    ax.bar(x - width / 2, a0[order], width, color=ARM_STYLE["A0_blocking_off"]["color"],
           alpha=0.9, label=ARM_STYLE["A0_blocking_off"]["label"])
    ax.bar(x + width / 2, a1[order], width, color=ARM_STYLE["A1_blocking_on"]["color"],
           alpha=0.9, label=ARM_STYLE["A1_blocking_on"]["label"])
    ax.axhline(a0.mean(), color=ARM_STYLE["A0_blocking_off"]["color"], ls=":", lw=1.2)
    ax.axhline(a1.mean(), color=ARM_STYLE["A1_blocking_on"]["color"], ls=":", lw=1.2)
    ax.set_ylabel("test centroid error [mrad]")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(
        f"{hid} — per-sample test error after Stage 2 (samples sorted by blocked fraction)\n"
        f"A0 mean {a0.mean():.3f} / median {np.median(a0):.3f} mrad   ·   "
        f"A1 mean {a1.mean():.3f} / median {np.median(a1):.3f} mrad   ·   "
        f"A1 better on {(a1 < a0).mean() * 100:.0f}% of samples",
        fontsize=10.5,
    )

    delta = a0 - a1  # positive = A1 wins
    colors = np.where(delta[order] >= 0, "#2471a3", "#c0392b")
    ax2.bar(x, delta[order], 0.8, color=colors, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Δ (A0 − A1) [mrad]")
    ax2.set_xlabel("test sample (sorted by blocked fraction →)")
    ax2.grid(axis="y", alpha=0.3)
    # Blocked-fraction profile as a secondary line so the sort order is readable.
    ax2b = ax2.twinx()
    ax2b.plot(x, blocked[order], color="grey", lw=1.2, ls="--", alpha=0.8)
    ax2b.set_ylabel("blocked [%]", color="grey")
    ax2b.tick_params(axis="y", labelcolor="grey")
    ax2b.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_dir / "per_sample_bars.png", dpi=160)
    plt.close(fig)
    log.info(f"{hid}: per_sample_bars.png + per_sample_plot_data.json -> {out_dir}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("artist").setLevel(logging.WARNING)
    device = torch.device("cpu")
    for hid in sys.argv[1:]:
        plot_heliostat(hid, device)


if __name__ == "__main__":
    main()
