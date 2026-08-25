"""
Driver for the split-criterion study — see residual_clustering.py for the method.

    python src/split_analysis/run_study.py --heliostat-ids AC25 AC33
"""

import argparse
import json
import logging
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent))
sys.path.insert(0, str(_here.parent / "one_heliostat_demo" / "single_heliostat"))

from artist.util import get_device, set_logger_config  # noqa: E402

import residual_clustering as rc  # noqa: E402

log = logging.getLogger(__name__)

DEFAULT_CKPT_ROOT = (
    _here.parent.parent / "outputs" / "new_mapping_function"
    / "robust_loss_sweep" / "soft_l1_d1.5"
)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_timeline(df, X, cps, hel, out):
    """Residual vs chronological order — the view that exposes drift and jumps."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, layout="constrained")
    t = df["dt"].values
    cp_t = [t[c] for c in cps]
    for ax, j, lbl in zip(axes[:2], (0, 1), ("axis 1", "axis 2")):
        for tgt, sub in df.groupby("target_name"):
            ax.scatter(t[sub.index], X[sub.index, j], s=14, alpha=0.75, label=tgt)
        # Per-segment mean: what a per-segment parameter set would absorb.
        for a, b in zip([0] + cps, cps + [len(df)]):
            ax.plot([t[a], t[b - 1]], [X[a:b, j].mean()] * 2, color="k", lw=2.2)
        for x in cp_t:
            ax.axvline(x, color="r", ls="--", lw=1.2)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_ylabel(f"Δα {lbl}  [mrad]")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=3, loc="upper left")
    sc = axes[2].scatter(t, df["dir_mrad"], c=df["hour_local"], cmap="twilight_shifted", s=14)
    for x in cp_t:
        axes[2].axvline(x, color="r", ls="--", lw=1.2)
    axes[2].set_ylabel("beam error [mrad]")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("measurement date")
    axes[2].grid(alpha=0.3)
    # Span all three axes so every panel is shrunk equally and the changepoint
    # lines stay vertically aligned across panels.
    fig.colorbar(sc, ax=list(axes), label="hour of day (local)", pad=0.01)
    when = ", ".join(pd.Timestamp(x).strftime("%Y-%m-%d") for x in cp_t) or "none"
    fig.suptitle(
        f"{hel} — residual of the FITTED single-parameter-set model, over time.\n"
        "Δα is the per-axis joint-angle correction each sample still wants: a flat band means one "
        "parameter set suffices, a step or trend means it does not.\n"
        f"Black = per-segment mean, red = BIC-selected changepoints ({when}).",
        fontsize=9.5,
    )
    fig.savefig(out / f"{hel}_residual_timeline.png", dpi=130)
    plt.close(fig)


def plot_criteria_scatter(df, X, hel, out):
    """The residual plane, coloured by every candidate criterion in turn."""
    n = len(rc.CRITERIA)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    lim = np.abs(X).max() * 1.1
    for ax, (col, kind, label) in zip(axes, rc.CRITERIA):
        v = df[col].values
        if kind == "cat":
            for lv in pd.unique(v):
                m = v == lv
                ax.scatter(X[m, 0], X[m, 1], s=16, alpha=0.8, label=str(lv)[:18])
            ax.legend(fontsize=6, loc="best")
        else:
            sc = ax.scatter(X[:, 0], X[:, 1], c=v.astype(float), cmap="viridis", s=16)
            plt.colorbar(sc, ax=ax, pad=0.01)
        ax.set_title(label, fontsize=9)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Δα axis 1 [mrad]", fontsize=8)
        ax.set_ylabel("Δα axis 2 [mrad]", fontsize=8)
        ax.grid(alpha=0.25)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(
        f"{hel} — residual plane coloured by each candidate split criterion. "
        "A criterion is a good split if its colours separate into distinct blobs.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / f"{hel}_criteria_scatter.png", dpi=130)
    plt.close(fig)


def plot_ranking(res, joint, df, hel, out):
    """Out-of-sample structure per criterion, and what survives confounding."""
    d = pd.DataFrame(res).sort_values("r2_full", ascending=True)
    y = np.arange(len(d))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8),
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})

    h = 0.38
    axes[0].barh(y - h / 2, d["r2_step"], h, color="tab:blue",
                 label="best 2-way split  (= two parameter sets)")
    axes[0].barh(y + h / 2, d["r2_full"], h, color="tab:orange",
                 label="forest on the criterion  (= any number of sets)")
    axes[0].axvline(0, color="k", lw=1)
    axes[0].axvline(joint["r2_joint"], color="tab:red", ls="--",
                    label=f"all criteria jointly = {joint['r2_joint']:.2f}  (the ceiling)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(d["label"], fontsize=8)
    axes[0].set_xlabel("out-of-sample R²  (residual variance removed)")
    axes[0].set_title("Marginal — but confounded", fontsize=10)
    axes[0].legend(fontsize=7, loc="lower right")
    axes[0].grid(alpha=0.3, axis="x")

    incr = d["r2_incr"].fillna(0.0)
    cols = ["tab:green" if v > 0.02 else "tab:gray" for v in incr]
    axes[1].barh(y, incr, color=cols)
    axes[1].axvline(0, color="k", lw=1)
    for yy, (v, lab) in enumerate(zip(d["r2_incr"], d["criterion"])):
        if np.isnan(v):
            axes[1].text(0.005, yy, "in the base set", va="center", fontsize=7,
                         color="tab:gray", style="italic")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("ΔR² added on top of sun azimuth + elevation")
    axes[1].set_title("Incremental — what is genuinely NEW\n"
                      f"(geometry alone already gives R²={joint['r2_geometry']:.2f})",
                      fontsize=10)
    axes[1].grid(alpha=0.3, axis="x")

    # Criterion cross-correlation: shows WHY the marginal column is confounded.
    cont = [c for c, k, _ in rc.CRITERIA if k == "cont"]
    C = np.abs(np.corrcoef(df[cont].values.astype(float).T))
    im = axes[2].imshow(C, cmap="magma", vmin=0, vmax=1)
    axes[2].set_xticks(range(len(cont)))
    axes[2].set_xticklabels(cont, rotation=90, fontsize=7)
    axes[2].set_yticks(range(len(cont)))
    axes[2].set_yticklabels(cont, fontsize=7)
    axes[2].set_title("|correlation| between criteria", fontsize=10)
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    fig.suptitle(
        f"{hel} — left: blue ≈ orange means the structure is a STEP (two parameter sets suffice); "
        "orange ≫ blue means it is graded.\nMiddle is the column to trust: a tracking heliostat's "
        "motor positions, clock time and season are all functions of the sun position,\n"
        "so only a positive ΔR² there is information a workspace-dependent model term would not "
        "already capture.",
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(out / f"{hel}_split_ranking.png", dpi=130)
    plt.close(fig)


def plot_gmm(df, X, labels, bics, res, hel, out):
    """Unsupervised clusters, and what each discovered cluster corresponds to."""
    k = len(np.unique(labels))
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    for g in np.unique(labels):
        m = labels == g
        ax.scatter(X[m, 0], X[m, 1], s=18, alpha=0.8, label=f"cluster {g} (n={m.sum()})")
    ax.legend(fontsize=8)
    ax.set_xlabel("Δα axis 1 [mrad]")
    ax.set_ylabel("Δα axis 2 [mrad]")
    ax.set_title(f"GMM on the residual — BIC picks k={k}", fontsize=10)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(range(1, len(bics) + 1), bics, "o-")
    ax.axvline(k, color="r", ls="--")
    ax.set_xlabel("k")
    ax.set_ylabel("BIC (lower = better)")
    ax.set_title("Model selection", fontsize=10)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    d = pd.DataFrame(res).sort_values("cluster_align")
    ax.barh(np.arange(len(d)), d["cluster_align"], color="tab:purple")
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d["label"], fontsize=8)
    ax.set_xlabel("agreement with the discovered clusters  [0..1]")
    ax.set_title("What ARE the clusters?", fontsize=10)
    ax.grid(alpha=0.3, axis="x")

    pretty = {c: l for c, _, l in rc.CRITERIA}
    for i, col in enumerate(["hour_local", "t_index", "sun_elevation"]):
        ax = fig.add_subplot(gs[1, i])
        for g in np.unique(labels):
            ax.hist(df[col].values[labels == g], bins=20, alpha=0.6, label=f"c{g}")
        ax.set_xlabel(pretty[col])
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"{hel} — unsupervised view: cluster the residual first, ask what the "
        "clusters correspond to second.", fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / f"{hel}_gmm_clusters.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyse(hel: str, cfg, ckpt_root: pathlib.Path, out: pathlib.Path, device):
    log.info(f"===== {hel} =====")
    scenario, hg, cents, rays, motors, meta = rc.load_pool(hel, cfg, device)
    meta = rc.attach_timestamps(meta, cfg)

    ckpt = ckpt_root / hel / "stage1_checkpoint.pt"
    base_pos = rc.restore_checkpoint(hg.kinematics, ckpt, hel, device)
    r_fit = rc.residuals(hg, cents, rays, motors, base_pos, device)

    # Work in chronological order throughout, so df rows and residual rows stay
    # aligned and "t_index" is a meaningful calendar axis.
    chrono = np.argsort(meta["dt"].values, kind="stable")
    df = meta.iloc[chrono].reset_index(drop=True)
    X = r_fit["dalpha"][chrono]
    W = r_fit["omega"][chrono]
    df["dir_mrad"] = r_fit["dir_mrad"][chrono]
    mp = motors.cpu().numpy()[chrono]
    df["motor_1"], df["motor_2"] = mp[:, 0], mp[:, 1]
    sun = -rays[:, :3].cpu().numpy()[chrono]
    df["sun_azimuth"] = np.degrees(np.arctan2(sun[:, 0], sun[:, 1])) % 360.0
    df["sun_elevation"] = np.degrees(np.arcsin(np.clip(sun[:, 2], -1, 1)))
    df["hour_local"] = 12.0 + (df["sun_azimuth"] - 180.0) / 15.0
    df["day_of_year"] = df["dt"].dt.dayofyear
    df["t_index"] = np.arange(len(df))
    # Slew direction: sign of the motor move from the previous measurement in
    # time. If backlash / hysteresis is the ceiling, this is the criterion.
    for ax_i in (1, 2):
        d = np.diff(df[f"motor_{ax_i}"].values, prepend=df[f"motor_{ax_i}"].values[0])
        df[f"slew_{ax_i}"] = np.where(d >= 0, "up", "down")

    log.info(f"  fitted-model residual: beam median {np.median(df['dir_mrad']):.2f} mrad, "
             f"|Δα| median {np.median(np.linalg.norm(X, axis=1)):.2f} mrad")

    rng = np.random.default_rng(0)
    k, labels, bics = rc.gmm_clusters(X)
    log.info(f"  GMM: BIC selects k={k}")

    # Fold at the measurement-day level everywhere: samples from one session are
    # minutes apart at nearly identical sun and motor positions, and would leak.
    groups = df["dt"].dt.floor("D").astype(str).values
    base_F = df[rc.GEOMETRY_BASE].values.astype(float)
    base_r2, base_mrad, _ = rc.cv_forest_r2(X, base_F, groups, rng=rng)
    log.info(f"  geometry base (sun az+el): R²={base_r2:+.3f} -> {base_mrad:.2f} mrad")

    res = []
    for col, kind, label in rc.CRITERIA:
        v = df[col].values
        eta2, lab, desc = rc.best_two_way_split(X, v, kind)
        p = rc.permutation_p(X, eta2, lab, n=1000, rng=rng)
        cv = rc.cv_scores(X, v, kind, groups, base_F, base_r2, rng=rng)
        if col in rc.GEOMETRY_BASE:
            # Already in the base set, so its increment over the base is 0 by
            # construction — reporting it would read as "adds nothing".
            cv["r2_incr"] = float("nan")
        align = rc.criterion_vs_clusters(v, kind, labels)
        res.append(dict(criterion=col, label=label, eta2=eta2, split=desc,
                        p_perm=p, cluster_align=align, **cv))
        log.info(f"  {label:32s} R²step={cv['r2_step']:+.3f} R²full={cv['r2_full']:+.3f} "
                 f"R²incr={cv['r2_incr']:+.3f}  {cv['mrad_base']:.2f}->{cv['mrad_full']:.2f} mrad"
                 f"  align={align:.2f}  [{desc}]")

    joint = rc.cv_joint_ceiling(
        X, df, ["hour_local", "t_index", "day_of_year", "sun_azimuth",
                "sun_elevation", "motor_1", "motor_2"], groups, rng=rng)
    joint["r2_geometry"] = base_r2
    joint["mrad_geometry"] = base_mrad
    log.info(f"  ALL criteria jointly: R²={joint['r2_joint']:+.3f}  "
             f"{joint['mrad_base']:.2f}->{joint['mrad_joint']:.2f} mrad")

    cps, cp_bics = rc.changepoints(X)
    seg_edges = [0] + cps + [len(df)]
    segments = [
        dict(start=str(df["dt"].iloc[a].date()), end=str(df["dt"].iloc[b - 1].date()),
             n=int(b - a),
             beam_mrad_median=float(np.median(df["dir_mrad"].values[a:b])),
             dalpha_mean=[float(v) for v in X[a:b].mean(0)])
        for a, b in zip(seg_edges[:-1], seg_edges[1:])
    ]
    log.info(f"  changepoints: {len(cps)} -> "
             + " | ".join(f"{s['start']}..{s['end']} n={s['n']} "
                          f"Δα=({s['dalpha_mean'][0]:+.1f},{s['dalpha_mean'][1]:+.1f})"
                          for s in segments))

    out.mkdir(parents=True, exist_ok=True)
    plot_timeline(df, X, cps, hel, out)
    plot_criteria_scatter(df, X, hel, out)
    plot_ranking(res, joint, df, hel, out)
    plot_gmm(df, X, labels, bics, res, hel, out)

    per_sample = df.copy()
    per_sample["dalpha_1"], per_sample["dalpha_2"] = X[:, 0], X[:, 1]
    per_sample["omega_e"], per_sample["omega_n"], per_sample["omega_u"] = W[:, 0], W[:, 1], W[:, 2]
    per_sample["gmm_cluster"] = labels
    per_sample.to_csv(out / f"{hel}_per_sample.csv", index=False)

    summary = dict(
        heliostat=hel, n_samples=len(df), gmm_k=k, bics=list(map(float, bics)),
        fitted_beam_mrad_median=float(np.median(df["dir_mrad"])),
        fitted_beam_mrad_mean=float(np.mean(df["dir_mrad"])),
        dalpha_mrad_median=float(np.median(np.linalg.norm(X, axis=1))),
        joint=joint, criteria=res, segments=segments,
    )
    (out / f"{hel}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heliostat-ids", nargs="+", default=["AC25", "AC33"],
                    help="'all' = every heliostat with a Stage-1 checkpoint under --ckpt-root")
    ap.add_argument("--ckpt-root", type=pathlib.Path, default=DEFAULT_CKPT_ROOT)
    ap.add_argument("--output-dir", type=pathlib.Path,
                    default=_here.parent.parent / "outputs" / "new_mapping_function"
                    / "split_criteria_study")
    args = ap.parse_args()

    set_logger_config()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(message)s"))
    for lg in (log, rc.log):
        lg.setLevel(logging.INFO)
        lg.addHandler(h)
        lg.propagate = False
    device = get_device()
    import config as cfg  # single_heliostat/config.py

    ids = args.heliostat_ids
    if len(ids) == 1 and ids[0].lower() == "all":
        ids = sorted(p.parent.name for p in args.ckpt_root.glob("*/stage1_checkpoint.pt"))
        log.info(f"{len(ids)} heliostats with a Stage-1 checkpoint")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, failed = [], []
    for i, h in enumerate(ids, 1):
        try:
            summaries.append(analyse(h, cfg, args.ckpt_root, args.output_dir, device))
        except Exception as exc:                      # one bad heliostat must not
            log.warning(f"  {h}: FAILED — {exc!r}")   # abort a 63-heliostat sweep
            failed.append(dict(heliostat=h, error=repr(exc)))
        log.info(f"--- {i}/{len(ids)} done ---")
    (args.output_dir / "summary_all.json").write_text(
        json.dumps(dict(summaries=summaries, failed=failed), indent=2))
    log.info(f"Wrote {args.output_dir}  ({len(summaries)} ok, {len(failed)} failed)")


if __name__ == "__main__":
    main()
