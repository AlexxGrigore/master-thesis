"""
Field-wide aggregation of the split-criterion study.

Classifies every heliostat into one of four regimes and answers the question the
per-heliostat runs cannot: is the workspace effect the SAME across the field?
If it is, one shared model term beats 63 private ones.

    python src/split_analysis/aggregate_field.py \
        --study-dir outputs/new_mapping_function/split_criteria_field
"""

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Classification thresholds. R² here is day-grouped cross-validated, so these are
# "real structure" levels, not in-sample fit.
R2_WEAK = 0.15          # below this, a criterion is not worth acting on
GAIN_WEAK = 0.20        # mrad; below this, conditioning is not worth building for
CLASSES = ["time-segmented", "workspace-limited", "mixed", "diffuse", "noise-limited"]
COLORS = dict(zip(CLASSES,
                  ["tab:red", "tab:blue", "tab:purple", "tab:orange", "tab:gray"]))


def classify(geo: float, cal: float, joint_r2: float, gain_mrad: float) -> str:
    """geo = R² of sun position alone; cal = ΔR² of calendar date on top of it.

    Deliberately uses TWO statistics that disagree in a known way. R² is
    variance-based and so is dominated by a handful of outlier samples; the
    median-mrad gain describes the bulk. A heavy-tailed heliostat can have
    R² ≈ 0 while conditioning still sharpens most of its samples — calling that
    "noise-limited" (as the first version of this script did) is wrong. Only
    when BOTH say there is nothing is it really noise.
    """
    if joint_r2 < R2_WEAK and gain_mrad < GAIN_WEAK:
        return "noise-limited"
    if cal >= R2_WEAK and geo >= R2_WEAK:
        return "mixed"
    if cal >= R2_WEAK:
        return "time-segmented"
    if geo >= R2_WEAK:
        return "workspace-limited"
    # Conditioning helps, but no single criterion carries it.
    return "diffuse"


def day_level_slope(csv: pathlib.Path):
    """Slope of Δα against sun elevation, per axis, in mrad/deg.

    Fitted on DAY MEANS, not raw samples: a session contributes a dozen images at
    almost the same elevation, and treating them as independent would shrink the
    standard error by ~sqrt(12) and make every heliostat look significant.
    """
    df = pd.read_csv(csv, parse_dates=["dt"])
    df["day"] = df["dt"].dt.floor("D")
    g = df.groupby("day").agg(el=("sun_elevation", "mean"),
                              d1=("dalpha_1", "mean"), d2=("dalpha_2", "mean"))
    if len(g) < 8:
        return None
    out = {}
    x = g["el"].values
    for ax, col in ((1, "d1"), (2, "d2")):
        y = g[col].values
        b, a = np.polyfit(x, y, 1)
        resid = y - (a + b * x)
        sxx = ((x - x.mean()) ** 2).sum()
        se = np.sqrt((resid ** 2).sum() / max(len(x) - 2, 1) / max(sxx, 1e-12))
        out[f"slope_{ax}"] = float(b)
        out[f"slope_{ax}_se"] = float(se)
    out["n_days"] = int(len(g))
    return out


def three_baselines(csv: pathlib.Path, n_folds=5, n_rep=4, seed=0):
    """Decompose the prize into the part that is a global bias and the part that
    is genuinely conditioning.

    Three predictions of Δα, all scored the same way (median beam mrad) and on
    the SAME day-grouped folds, so the differences are comparable:

      B0  Δα = 0        — the deployed model. This is the honest starting point:
                          the fitted parameters ARE the prediction, so the
                          residual at zero is the current error.
      B1  Δα = const    — refit one global offset, per-axis median on the
                          training folds. Improving from B0 to B1 means the
                          fitted model carries a small residual BIAS; it has
                          nothing to do with splitting. (soft_l1 sits at a robust
                          centre, so its residual mean is not zero.)
      B2  forest on all criteria — conditioning on external factors.

    The quantity that actually answers "should I fit separate parameter sets" is
    B1 -> B2. Reporting B0 -> B2, as the first version of this script did,
    credits the split with removing a bias that a plain refit would remove.
    """
    from sklearn.ensemble import RandomForestRegressor

    df = pd.read_csv(csv, parse_dates=["dt"])
    X = df[["dalpha_1", "dalpha_2"]].values
    cols = ["hour_local", "t_index", "day_of_year", "sun_azimuth",
            "sun_elevation", "motor_1", "motor_2"]
    F = df[cols].values.astype(float)
    days = df["dt"].dt.floor("D").astype(str).values
    rng = np.random.default_rng(seed)

    p1, p2, w = np.zeros_like(X), np.zeros_like(X), np.zeros(len(X))
    for r in range(n_rep):
        ug = rng.permutation(np.unique(days))
        for chunk in np.array_split(ug, min(n_folds, len(ug))):
            te = np.flatnonzero(np.isin(days, chunk))
            tr = np.flatnonzero(~np.isin(days, chunk))
            if len(tr) < 20 or len(te) == 0:
                continue
            p1[te] += np.median(X[tr], axis=0)
            p2[te] += RandomForestRegressor(
                n_estimators=200, min_samples_leaf=5, random_state=seed + r
            ).fit(F[tr], X[tr]).predict(F[te])
            w[te] += 1
    ok = w > 0
    wv = np.where(w == 0, 1, w)[:, None]
    med = lambda A: float(2 * np.median(np.linalg.norm(A, axis=1)))
    return dict(
        b0_current=med(X[ok]),
        b1_refit_offset=med(X[ok] - (p1 / wv)[ok]),
        b2_conditioned=med(X[ok] - (p2 / wv)[ok]),
    )


def load(study_dir: pathlib.Path) -> pd.DataFrame:
    rows = []
    for js in sorted(study_dir.glob("*_summary.json")):
        s = json.loads(js.read_text())
        hel = s["heliostat"]
        crit = {c["criterion"]: c for c in s["criteria"]}
        geo = s["joint"]["r2_geometry"]
        cal = crit["t_index"]["r2_incr"]
        row = dict(
            heliostat=hel, n=s["n_samples"],
            beam_mrad=s["fitted_beam_mrad_median"],
            r2_geometry=geo, r2_calendar_incr=cal,
            r2_calendar_step=crit["t_index"]["r2_step"],
            r2_calendar_full=crit["t_index"]["r2_full"],
            r2_joint=s["joint"]["r2_joint"],
            mrad_base=s["joint"]["mrad_base"], mrad_joint=s["joint"]["mrad_joint"],
            r2_timeofday_incr=crit["hour_local"]["r2_incr"],
            r2_target_incr=crit["target_name"]["r2_incr"],
            r2_slew1_incr=crit["slew_1"]["r2_incr"],
            n_segments=len(s["segments"]),
        )
        csv = study_dir / f"{hel}_per_sample.csv"
        sl = day_level_slope(csv)
        if sl:
            row.update(sl)
        row.update(three_baselines(csv))
        rows.append(row)
    d = pd.DataFrame(rows)
    d["gain_conditioning"] = d["b1_refit_offset"] - d["b2_conditioned"]
    d["regime"] = [
        classify(r.r2_geometry, r.r2_calendar_incr, r.r2_joint, r.gain_conditioning)
        for r in d.itertuples()
    ]
    return d


def plot_classification(d: pd.DataFrame, out: pathlib.Path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4),
                             gridspec_kw={"width_ratios": [1.3, 1, 1]})

    ax = axes[0]
    for reg, sub in d.groupby("regime"):
        ax.scatter(sub["r2_geometry"], sub["r2_calendar_incr"],
                   s=25 + 3 * np.sqrt(sub["beam_mrad"]) * 4,
                   color=COLORS[reg], alpha=0.8, label=f"{reg} (n={len(sub)})")
    for _, r in d.iterrows():
        if r["beam_mrad"] > 10 or r["r2_calendar_incr"] > 0.3 or r["r2_geometry"] > 0.45:
            ax.annotate(r["heliostat"], (r["r2_geometry"], r["r2_calendar_incr"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axvline(R2_WEAK, color="k", ls=":", lw=1)
    ax.axhline(R2_WEAK, color="k", ls=":", lw=1)
    ax.set_xlabel("R² of sun position alone  →  workspace effect")
    ax.set_ylabel("ΔR² of calendar date on top  →  time regime")
    ax.set_title("Every heliostat placed in the plane\n(marker size = current beam error)",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    cnt = d["regime"].value_counts().reindex(CLASSES).fillna(0)
    ax.bar(range(len(cnt)), cnt.values, color=[COLORS[c] for c in cnt.index])
    ax.set_xticks(range(len(cnt)))
    ax.set_xticklabels(cnt.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("heliostats")
    for i, v in enumerate(cnt.values):
        ax.text(i, v + 0.4, int(v), ha="center", fontsize=9)
    ax.set_title(f"How the field splits  (n={len(d)})", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    dead = ["r2_timeofday_incr", "r2_target_incr", "r2_slew1_incr", "r2_calendar_incr"]
    names = ["time of day", "aim target", "slew direction", "calendar date"]
    ax.boxplot([d[c].dropna() for c in dead], tick_labels=names, showfliers=False)
    for i, c in enumerate(dead, 1):
        v = d[c].dropna()
        ax.scatter(np.random.default_rng(0).normal(i, 0.06, len(v)), v, s=9,
                   alpha=0.5, color="k")
    ax.axhline(0, color="k", lw=1)
    ax.axhline(R2_WEAK, color="r", ls=":", lw=1)
    ax.set_ylabel("ΔR² over sun position, all 63 heliostats")
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_title("Does any criterion hold up field-wide?", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Split-criterion study across the field — which heliostats need which fix",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "field_classification.png", dpi=130)
    plt.close(fig)


def plot_ceiling(d: pd.DataFrame, out: pathlib.Path):
    s = d.sort_values("b0_current", ascending=False)
    y = np.arange(len(s))
    fig, axes = plt.subplots(1, 2, figsize=(15, 13),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    ax.barh(y, s["b0_current"], color="lightgray", label="B0  now (deployed model)")
    ax.barh(y, s["b1_refit_offset"], color="silver",
            label="B1  after refitting one global offset (not a split)")
    ax.barh(y, s["b2_conditioned"], color=[COLORS[r] for r in s["regime"]],
            label="B2  conditioned on every criterion")
    ax.set_yticks(y)
    ax.set_yticklabels(s["heliostat"], fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("median beam error [mrad], day-grouped cross-validated")
    ax.set_ylim(-1, len(s))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    ax.set_title("Per heliostat: B0 → B1 is a bias a plain refit removes,\n"
                 "B1 → B2 is what CONDITIONING actually buys.\n"
                 "Colour = regime (red time-segmented, blue workspace, "
                 "purple mixed, grey noise)", fontsize=10)

    ax = axes[1]
    gain = (s["b1_refit_offset"] - s["b2_conditioned"])
    ax.barh(y, gain, color=[COLORS[r] for r in s["regime"]])
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_ylim(-1, len(s))
    ax.set_xlabel("mrad bought by conditioning alone  (B1 − B2)")
    ax.set_title(f"The real prize\nmedian {gain.median():+.2f} mrad, "
                 f"positive on {int((gain > 0).sum())}/{len(s)}", fontsize=10)
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(out / "field_ceiling.png", dpi=130)
    plt.close(fig)


def plot_shared_slope(d: pd.DataFrame, out: pathlib.Path):
    """Is the elevation effect the SAME across heliostats? -> shared model term."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, axis in zip(axes[:2], (1, 2)):
        s = d.dropna(subset=[f"slope_{axis}"]).sort_values(f"slope_{axis}")
        y = np.arange(len(s))
        ax.errorbar(s[f"slope_{axis}"], y, xerr=1.96 * s[f"slope_{axis}_se"],
                    fmt="o", ms=3, lw=0.9, color="tab:blue")
        ax.axvline(0, color="k", lw=1.2)
        med = s[f"slope_{axis}"].median()
        ax.axvline(med, color="tab:red", ls="--",
                   label=f"field median = {med:+.3f}")
        n_pos = int((s[f"slope_{axis}"] > 0).sum())
        ax.set_yticks(y)
        ax.set_yticklabels(s["heliostat"], fontsize=5.5)
        ax.set_xlabel(f"dΔα{axis} / d(sun elevation)   [mrad/deg]")
        ax.set_title(f"Axis {axis}: {n_pos}/{len(s)} positive", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="x")

    ax = axes[2]
    for axis, c in ((1, "tab:blue"), (2, "tab:orange")):
        v = d[f"slope_{axis}"].dropna()
        ax.hist(v, bins=20, alpha=0.6, color=c, label=f"axis {axis}")
    ax.axvline(0, color="k", lw=1.2)
    ax.set_xlabel("slope [mrad/deg]")
    ax.set_ylabel("heliostats")
    ax.set_title("A distribution centred off zero means a SHARED\n"
                 "systematic; centred on zero means it is private", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Does the workspace effect point the same way on every heliostat?\n"
                 "Slopes fitted on day means (a session is one point, not twelve).",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "field_shared_slope.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()

    d = load(args.study_dir)
    if d.empty:
        raise SystemExit(f"No *_summary.json in {args.study_dir}")
    d = d.sort_values("beam_mrad", ascending=False)
    d.to_csv(args.study_dir / "field_table.csv", index=False)

    plot_classification(d, args.study_dir)
    plot_ceiling(d, args.study_dir)
    plot_shared_slope(d, args.study_dir)

    print(f"n = {len(d)} heliostats\n")
    print(d["regime"].value_counts().to_string(), "\n")
    print(f"{'regime':20s} {'n':>3s} {'B0 now':>8s} {'B1 refit':>9s} "
          f"{'B2 cond':>8s} {'B1-B2':>7s}")
    for reg in CLASSES + ["ALL"]:
        s = d if reg == "ALL" else d[d.regime == reg]
        if s.empty:
            continue
        print(f"{reg:20s} {len(s):3d} {s['b0_current'].median():8.2f} "
              f"{s['b1_refit_offset'].median():9.2f} {s['b2_conditioned'].median():8.2f} "
              f"{(s['b1_refit_offset'] - s['b2_conditioned']).median():+7.2f}")
    print()
    for reg in CLASSES:
        s = d[d.regime == reg]
        if not s.empty:
            print(f"{reg}: {' '.join(s['heliostat'])}\n")
    for axis in (1, 2):
        v = d[f"slope_{axis}"].dropna()
        sig = (v.abs() > 1.96 * d.loc[v.index, f"slope_{axis}_se"]).sum()
        print(f"elevation slope axis {axis}: median {v.median():+.4f} mrad/deg, "
              f"{int((v > 0).sum())}/{len(v)} positive, {sig} significant at 95%")
    print(f"\nWrote {args.study_dir}")


if __name__ == "__main__":
    main()
