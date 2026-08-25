"""
Build the STANDARD Stage-1 baseline — the reference every future run is measured against.

Source of truth: the soft_l1 arm of the robust-loss sweep, which is the configuration
`config.py` ships today (STAGE1_REDUCTION = "soft_l1", STAGE1_HUBER_DELTA = 1.5), so a
fresh run reproduces these numbers. Stage 2 is a no-op in that sweep — verified here, not
assumed — so these are Stage-1-only results.

Both metrics are reported everywhere, always:
  * direction  — angle between the reflected beam and the direction to the observed
                 centroid. Kinematics only, no surface. THIS is the metric Mathias
                 reports, so it is the only one a head-to-head may use.
  * centroid   — ray-traced focal-spot centroid vs the observed centroid. Includes
                 mirror surface (canting/shape) spread.
Reporting one alone is what once made a gap to Mathias look real when it was a metric
mismatch.

    python src/standard_results/build_standard.py
"""

import argparse
import json
import pathlib
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
SWEEP = ROOT / "outputs" / "new_mapping_function" / "robust_loss_sweep"
MATHIAS = ROOT / "mathias_results" / "pointing_accuracy_evaluation"

# Documented data faults, excluded from headline field statistics but still listed.
# Their beams land on target while the recorded motors claim ~60 mrad of error, so no
# parameter set can fit them. See outputs/new_mapping_function/ac33_time_jump/.
DATA_FAULTS = ["AC33", "AW36"]

VARIANTS = ["l2", "huber_d1", "huber_d3", "soft_l1_d1.5", "trimmed_25", "trimmed_40"]
STANDARD = "soft_l1_d1.5"


def load_ours(variant: str) -> pd.DataFrame:
    rows = []
    for res in sorted((SWEEP / variant).glob("*/results.json")):
        r = json.loads(res.read_text())
        s1, s2 = r["after_stage1"], r["after_stage2"]
        rows.append(dict(
            heliostat=r["heliostat_id"],
            dist_m=r["hel_dist_m"], n_test=r["n_test"],
            direction_mrad_median=s1["direction_mrad_median"],
            direction_mrad_mean=s1["direction_mrad_mean"],
            centroid_mrad_median=s1["centroid_mrad_median"],
            centroid_mrad_mean=s1["centroid_mrad_mean"],
            pre_direction_mrad_median=r["pre_training"]["direction_mrad_median"],
            pre_centroid_mrad_median=r["pre_training"]["centroid_mrad_median"],
            stage2_changed=bool(s2 != s1),
        ))
    return pd.DataFrame(rows).sort_values("heliostat").reset_index(drop=True)


def load_mathias() -> pd.DataFrame:
    d = pd.read_csv(MATHIAS / "pointing_accuracy_by_heliostat.csv")
    d = d[d["split"] == "test"]
    return d.rename(columns={
        "heliostat_id": "heliostat",
        "mean_mrad": "mathias_direction_mrad_mean",
        "median_mrad": "mathias_direction_mrad_median",
        "count": "mathias_n_test",
    })[["heliostat", "mathias_direction_mrad_mean",
        "mathias_direction_mrad_median", "mathias_n_test"]]


def stats_block(d: pd.DataFrame, label: str) -> dict:
    out = {"set": label, "n": len(d)}
    for c in ["direction_mrad_median", "direction_mrad_mean",
              "centroid_mrad_median", "centroid_mrad_mean"]:
        out[f"{c}__field_median"] = float(d[c].median())
        out[f"{c}__field_mean"] = float(d[c].mean())
    out["worst_centroid_median"] = float(d["centroid_mrad_median"].max())
    out["worst_heliostat"] = d.loc[d["centroid_mrad_median"].idxmax(), "heliostat"]
    return out


def plot_bars(d: pd.DataFrame, out: pathlib.Path):
    """Per-heliostat accuracy, both metrics, median and mean side by side."""
    s = d.sort_values("direction_mrad_median")
    y = np.arange(len(s))
    fig, axes = plt.subplots(1, 2, figsize=(15, 14), sharey=True)
    h = 0.4
    for ax, kind in zip(axes, ("median", "mean")):
        ax.barh(y - h / 2, s[f"direction_mrad_{kind}"], h,
                color="tab:blue", label="direction (kinematics only)")
        ax.barh(y + h / 2, s[f"centroid_mrad_{kind}"], h,
                color="tab:orange", label="centroid (ray-traced, incl. surface)")
        for i, hel in enumerate(s["heliostat"]):
            if hel in DATA_FAULTS:
                ax.axhspan(i - 0.5, i + 0.5, color="red", alpha=0.10)
        ax.set_xscale("log")
        ax.set_xlabel(f"test {kind} error [mrad]")
        ax.set_title(f"per-heliostat {kind}", fontsize=11)
        ax.grid(alpha=0.3, axis="x")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(s["heliostat"], fontsize=6.5)
    axes[0].set_ylim(-1, len(s))
    fig.suptitle(
        "STANDARD Stage-1 baseline — soft_l1, 63 heliostats, test split.\n"
        "Both metrics always shown: direction excludes mirror-surface spread, centroid "
        "includes it — which is why centroid is consistently the larger of the two.\n"
        "Red rows are the two documented data faults (AC33, AW36), excluded from headline "
        "field statistics.",
        fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out / "accuracy_bars.png", dpi=130)
    plt.close(fig)


def plot_mathias(d: pd.DataFrame, out: pathlib.Path):
    """Ours vs Mathias on the 41 he reports — DIRECTION metric only (his metric)."""
    m = d.dropna(subset=["mathias_direction_mrad_mean"])
    XMAX = 16.0          # both methods' bulk lives well under this
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, kind in zip(axes, ("median", "mean")):
        ours = m[f"direction_mrad_{kind}"].values
        his = m[f"mathias_direction_mrad_{kind}"].values
        # Statistics use the FULL data; only the display is clipped, with the
        # clipped heliostats named so nothing is quietly hidden.
        bins = np.linspace(0, XMAX, 33)
        ax.hist(np.clip(ours, 0, XMAX - 1e-6), bins=bins, alpha=0.55,
                color="tab:blue", label="ours (Stage 1, soft_l1)")
        ax.hist(np.clip(his, 0, XMAX - 1e-6), bins=bins, alpha=0.55,
                color="tab:green", label="Mathias")
        lines = [
            (np.median(ours), "tab:blue", "-", "our median"),
            (np.mean(ours), "tab:blue", "--", "our mean"),
            (np.median(his), "tab:green", "-", "Mathias median"),
            (np.mean(his), "tab:green", "--", "Mathias mean"),
        ]
        for v, c, ls, _ in lines:
            ax.axvline(min(v, XMAX), color=c, ls=ls, lw=2)
        ax.text(0.97, 0.62,
                "\n".join(f"{lab:<15s}{v:6.2f}" for v, _, _, lab in lines),
                transform=ax.transAxes, ha="right", va="top",
                family="monospace", fontsize=9,
                bbox=dict(fc="white", ec="0.7", alpha=0.9))
        over = sorted(set(m.loc[ours > XMAX, "heliostat"]) |
                      set(m.loc[his > XMAX, "heliostat"]))
        wins = int((ours < his).sum())
        ax.set_xlim(0, XMAX)
        ax.set_xlabel(f"per-heliostat test {kind} DIRECTION error [mrad]"
                      + (f"   (beyond axis: {', '.join(over)})" if over else ""))
        ax.set_ylabel("heliostats")
        ax.set_title(f"{kind}: we are better on {wins}/{len(m)} heliostats", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Ours vs Mathias — the {len(m)} heliostats he reports, test split, DIRECTION metric.\n"
        "Direction is the only valid basis for this comparison: his pointing error is an "
        "angle between predicted and measured reflected direction.\n"
        "Comparing our centroid metric against it is what produced the phantom 14.6-vs-4.6 gap.",
        fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out / "mathias_comparison_hist.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "outputs" / "new_mapping_function" / "standard_results")
    args = ap.parse_args()
    out = args.out
    (out / "parameters").mkdir(parents=True, exist_ok=True)

    ours = load_ours(STANDARD)
    assert not ours["stage2_changed"].any(), \
        "Stage 2 altered results — this is not a Stage-1-only baseline"
    d = ours.merge(load_mathias(), on="heliostat", how="left")
    d["data_fault"] = d["heliostat"].isin(DATA_FAULTS)
    d["in_mathias_set"] = d["mathias_direction_mrad_mean"].notna()
    d.to_csv(out / "standard_results.csv", index=False)

    for hel in d["heliostat"]:
        src = SWEEP / STANDARD / hel / "kinematic_parameters.json"
        if src.exists():
            shutil.copy(src, out / "parameters" / f"{hel}.json")

    # Variant comparison, so "something better" is judged against all four metrics.
    rows = []
    for v in VARIANTS:
        a = load_ours(v)
        rows.append(stats_block(a[~a["heliostat"].isin(DATA_FAULTS)], v))
    pd.DataFrame(rows).to_csv(out / "variant_comparison.csv", index=False)

    summary = dict(
        source=f"outputs/new_mapping_function/robust_loss_sweep/{STANDARD}",
        stage="after_stage1 (Stage 2 verified to be a no-op)",
        all_63=stats_block(d, "all 63"),
        excl_faults=stats_block(d[~d["data_fault"]], "61, excl. AC33+AW36"),
    )
    m = d[d["in_mathias_set"]]
    for kind in ("median", "mean"):
        summary[f"mathias_{kind}"] = dict(
            n=len(m),
            ours=float(m[f"direction_mrad_{kind}"].median()),
            mathias=float(m[f"mathias_direction_mrad_{kind}"].median()),
            we_win=int((m[f"direction_mrad_{kind}"]
                        < m[f"mathias_direction_mrad_{kind}"]).sum()),
        )
    (out / "standard_summary.json").write_text(json.dumps(summary, indent=2))

    plot_bars(d, out)
    plot_mathias(d, out)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
