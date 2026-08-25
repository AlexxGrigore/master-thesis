"""
Split-criterion diagnosis — does the irreducible Stage-1 residual CLUSTER?

Motivation
----------
Single-heliostat training has hit an accuracy ceiling (soft_l1 Stage 1). The
hypothesis under test: a *single* kinematic parameter set cannot explain all
calibration samples, and fitting SEPARATE parameter sets per sample subgroup
would break the ceiling. Before spending training runs, this script asks the
cheap question first: **is there structure left in the residual, and along
which axis does it split?**

Residual definition
-------------------
For every calibration sample i we have the recorded motor position m_i, the sun
direction s_i, and the observed focal-spot centroid c_i. Two normals follow:

    n_model(i)   = forward kinematics at m_i under the FITTED parameters θ*
    n_desired(i) = bisector(to_sun, to_c_i)          (pure geometry, no model)

A perfect model would have n_model == n_desired for every sample. What is left
is expressed in two spaces, both in mrad:

    dalpha  [N,2]  motor / joint space — the per-axis JOINT-ANGLE correction
                   that would fix this sample. Solved by least squares from the
                   autograd Jacobian dn/dm, then converted steps -> rad. This is
                   exactly the space an a_i (home-angle) parameter lives in, so
                   "fit a second parameter set" == "allow dalpha to differ
                   between groups". PRIMARY clustering space.
    omega   [N,3]  world-frame mount-rotation vector — the minimal rotation
                   taking n_model onto n_desired. This is the space of Mathias's
                   free 4-angle mount orientation. SECONDARY.

Both are model residuals, not raw errors: the global fit has already absorbed
everything a single parameter set can absorb. Any remaining CLUSTER is, by
construction, something only a split can capture.

What is measured
----------------
For each candidate split criterion (time of day, calendar date, season, sun
azimuth/elevation, motor positions, aim target, slew direction):

  A. eta^2   — fraction of residual variance explained by the best 2-way split
               on that criterion, with a permutation null (is it real?).
  B. CV gain — the honest one. Fit a constant per-group offset on a random half,
               evaluate on the other half, repeat. Reports the mrad the split
               actually buys OUT OF SAMPLE. A criterion that only wins on A is
               fitting noise.
  C. GMM     — unsupervised Gaussian mixture on the residual itself (BIC over
               k=1..4). If the data clusters at all, this finds it WITHOUT being
               told what to look for; each criterion is then scored on how well
               it predicts the discovered labels (AUC / Cramer's V). This is the
               strongest evidence, because it cannot be talked into a split that
               is not there.

Usage
-----
    python src/split_analysis/residual_clustering.py \
        --heliostat-ids AC25 AC33 \
        --output-dir outputs/new_mapping_function/split_criteria_study
"""

import argparse
import json
import logging
import pathlib
import sys
import warnings

import h5py
import numpy as np
import pandas as pd
import torch

_here = pathlib.Path(__file__).resolve().parent          # src/split_analysis/
_src = _here.parent                                      # src/
_root = _src.parent                                      # master-thesis/
_paint = _root.parent / "PAINT"
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_src / "one_heliostat_demo" / "single_heliostat"))
if _paint.exists():
    sys.path.insert(0, str(_paint))

from artist.io.paint_calibration_parser import PaintCalibrationDataParser  # noqa: E402
from artist.scenario.scenario import Scenario                              # noqa: E402
from artist.util import get_device, indices, set_logger_config              # noqa: E402

from utils.evaluation import build_heliostat_data_mapping                   # noqa: E402

log = logging.getLogger(__name__)

# Candidate split criteria. (column, kind, pretty label)
#   "cont" — continuous, split by scanning every threshold
#   "cat"  — categorical, split by grouping levels
CRITERIA = [
    ("hour_local",    "cont", "Time of day [h]"),
    ("t_index",       "cont", "Calendar date (chronological)"),
    ("day_of_year",   "cont", "Season (day of year)"),
    ("sun_azimuth",   "cont", "Sun azimuth [deg]"),
    ("sun_elevation", "cont", "Sun elevation [deg]"),
    ("motor_1",       "cont", "Motor axis 1 [steps]"),
    ("motor_2",       "cont", "Motor axis 2 [steps]"),
    ("target_name",   "cat",  "Aim target"),
    ("slew_1",        "cat",  "Slew direction axis 1"),
    ("slew_2",        "cat",  "Slew direction axis 2"),
]


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def load_pool(heliostat_id: str, cfg, device):
    """Load every PAINT benchmark sample for one heliostat, WITH its sample Id.

    ``build_heliostat_data_mapping`` returns calibration paths in CSV row order
    and ``PaintCalibrationDataParser`` consumes that list in order, so the i-th
    row of the returned tensors is the i-th path — that is what lets us recover
    the measurement Id (and therefore the timestamp) for each sample.
    """
    scenario_path = pathlib.Path(
        cfg.SCENARIO_PATH_TEMPLATE.format(heliostat_id=heliostat_id)
    )
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario not found: {scenario_path}")
    with h5py.File(scenario_path, "r") as fh:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=fh,
            device=device,
            number_of_surface_points_per_facet=torch.tensor(
                [cfg.SURFACE_POINTS_PER_FACET, cfg.SURFACE_POINTS_PER_FACET]
            ),
        )
    hg = scenario.heliostat_field.heliostat_groups[0]

    parser = PaintCalibrationDataParser(
        centroid_extraction_method=getattr(cfg, "CENTROID_METHOD", "UTIS")
    )
    flux, cents, rays, motors, ids, splits, targets = [], [], [], [], [], [], []
    for split in ("train", "validation", "test"):
        full = build_heliostat_data_mapping(
            pathlib.Path(cfg.BENCHMARK_CSV),
            pathlib.Path(cfg.CALIBRATION_DIR),
            pathlib.Path(cfg.REAL_FLUX_DIR),
            split,
        )
        mine = [(h, c, f) for h, c, f in full if h == heliostat_id]
        if not mine:
            continue
        cal_paths = mine[0][1]
        f, c, r, m, _mask, _tgt = parser.parse_data_for_reconstruction(
            heliostat_data_mapping=mine,
            heliostat_group=hg,
            scenario=scenario,
            device=device,
        )
        flux.append(f)
        cents.append(c)
        rays.append(r)
        motors.append(m)
        ids += [int(p.name.split("-")[0]) for p in cal_paths]
        splits += [split] * len(cal_paths)
        targets += [json.load(open(p))["target_name"] for p in cal_paths]

    flux = torch.cat(flux)
    cents = torch.cat(cents)
    rays = torch.cat(rays)
    motors = torch.cat(motors)

    # Same active-pixel filter the training pool applies, so the sample set
    # analysed here is the sample set the model was fitted on.
    min_pct = getattr(cfg, "MIN_ACTIVE_PIXEL_PERCENT", 2.0)
    pct = torch.tensor(
        [float((flux[i] > 0.01).sum()) / float(flux[i].numel()) * 100.0
         for i in range(flux.shape[0])]
    )
    keep = (pct >= min_pct).cpu().numpy()
    log.info(f"  {heliostat_id}: pool {int(keep.sum())}/{len(keep)} after active-pixel filter")

    meta = pd.DataFrame({
        "id": np.array(ids)[keep],
        "split": np.array(splits)[keep],
        "target_name": np.array(targets)[keep],
    })
    k = torch.tensor(np.flatnonzero(keep), device=device)
    return scenario, hg, cents[k], rays[k], motors[k], meta


def attach_timestamps(meta: pd.DataFrame, cfg) -> pd.DataFrame:
    """Join the PAINT calibration metadata CSV to recover DateTime per sample."""
    md = pd.read_csv(_root / "datasets" / "paint" / "metadata"
                     / "calibration_metadata_all_heliostats.csv",
                     usecols=["Id", "DateTime"])
    md = md.rename(columns={"Id": "id"})
    out = meta.merge(md, on="id", how="left")
    if out["DateTime"].isna().any():
        log.warning(f"  {int(out['DateTime'].isna().sum())} samples without a timestamp")
    out["dt"] = pd.to_datetime(out["DateTime"], utc=True)
    return out


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------

def restore_checkpoint(kinematic, ckpt_path: pathlib.Path, heliostat_id: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("heliostat_id") not in (None, heliostat_id):
        raise ValueError(f"{ckpt_path} belongs to {ckpt.get('heliostat_id')!r}")
    kinematic.translation_deviation_parameters.data.copy_(ckpt["translation"].to(device))
    kinematic.rotation_deviation_parameters.data.copy_(ckpt["rotation"].to(device))
    kinematic.actuators.optimizable_parameters.data.copy_(ckpt["act_angle"].to(device))
    kinematic.actuators.non_optimizable_parameters.data.copy_(ckpt["act_offset"].to(device))
    return ckpt["base_pos"].to(device)


def _model_normal(kinematic, motors, device):
    normal_local = torch.tensor([0.0, -1.0, 0.0, 0.0], device=device)
    orient = kinematic._compute_orientations_from_motor_positions(motors.to(device), device)
    return torch.nn.functional.normalize((orient @ normal_local)[:, :3], dim=-1)


def _steps_per_rad(kin, m):
    """[N,2] motor positions -> [N,2] motor steps per radian of joint angle."""
    nop = kin.actuators.non_optimizable_parameters
    op = kin.actuators.optimizable_parameters
    inc = nop[0, indices.actuator_increment]
    c = nop[0, indices.actuator_offset]
    r = nop[0, indices.actuator_pivot_radius]
    b = op[0, indices.actuator_initial_stroke_length]
    s = b + m / inc
    u = ((c ** 2 + r ** 2 - s ** 2) / (2 * c * r)).clamp(-1 + 1e-9, 1 - 1e-9)
    return (c * r * torch.sqrt(1 - u ** 2) / s) * inc


def residuals(hg, cents, rays, motors, base_pos, device):
    """Per-sample residual of the fitted model, in three representations.

    Returns dict with
        dalpha    [N,2] mrad — joint-angle correction that fixes the sample
        omega     [N,3] mrad — world mount-rotation that fixes the sample
        dir_mrad  [N]   mrad — beam pointing error (2x the normal error)
    """
    kinematic = hg.kinematics
    n_act = torch.tensor([motors.shape[0]], device=device)

    hg.activate_heliostats(active_heliostats_mask=n_act, device=device)
    pad = torch.zeros(motors.shape[0], 1, device=device)
    rep = base_pos.repeat_interleave(n_act, dim=0)
    kinematic.active_heliostat_positions = (
        kinematic.active_heliostat_positions + torch.cat([rep, pad], dim=1)
    )
    origins = kinematic.active_heliostat_positions[:, :3].detach()

    # Desired normal: bisector of (to sun, to observed centroid). Pure geometry.
    to_sun = torch.nn.functional.normalize(-rays[:, :3], dim=-1)
    to_tgt = torch.nn.functional.normalize(cents[:, :3] - origins, dim=-1)
    n_des = torch.nn.functional.normalize(to_sun + to_tgt, dim=-1)

    # Model normal + its Jacobian w.r.t. the motor positions.
    m = motors.clone().detach().requires_grad_(True)
    n_mod = _model_normal(kinematic, m, device)
    J = torch.stack([
        torch.autograd.grad(n_mod[:, c].sum(), m, retain_graph=True)[0]
        for c in range(3)
    ], dim=1)                                    # [N,3,2]
    n_mod = n_mod.detach()

    # Least-squares motor delta that maps n_mod onto n_des, then steps -> mrad.
    dn = (n_des - n_mod).unsqueeze(-1)           # [N,3,1]
    dm = torch.linalg.lstsq(J.detach(), dn).solution.squeeze(-1)   # [N,2] steps
    with torch.no_grad():
        spr = _steps_per_rad(kinematic, motors)
    dalpha = (dm / spr * 1000.0).detach().cpu().numpy()

    # World-frame minimal rotation n_mod -> n_des (axis-angle, mrad).
    cross = torch.cross(n_mod, n_des, dim=-1)
    dot = (n_mod * n_des).sum(-1).clamp(-1.0, 1.0)
    ang = torch.arccos(dot)
    axis = torch.nn.functional.normalize(cross, dim=-1)
    omega = (axis * ang.unsqueeze(-1) * 1000.0).cpu().numpy()

    # Beam direction error: reflect the sun about the model normal.
    d_in = torch.nn.functional.normalize(rays[:, :3], dim=-1)
    refl = torch.nn.functional.normalize(
        d_in - 2 * (d_in * n_mod).sum(-1, keepdim=True) * n_mod, dim=-1
    )
    meas = torch.nn.functional.normalize(cents[:, :3] - origins, dim=-1)
    dir_mrad = (torch.arccos((refl * meas).sum(-1).clamp(-1, 1)) * 1000.0).cpu().numpy()

    return {"dalpha": dalpha, "omega": omega, "dir_mrad": dir_mrad}


# ---------------------------------------------------------------------------
# Split scoring
# ---------------------------------------------------------------------------

def _eta_squared(X: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of total residual variance explained by the grouping."""
    ss_tot = ((X - X.mean(0)) ** 2).sum()
    if ss_tot <= 0:
        return 0.0
    ss_within = sum(
        ((X[labels == g] - X[labels == g].mean(0)) ** 2).sum()
        for g in np.unique(labels)
    )
    return float(1.0 - ss_within / ss_tot)


def best_two_way_split(X: np.ndarray, values: np.ndarray, kind: str, min_group=15):
    """Best 2-way split of the residual X by a criterion. Returns (eta2, labels, desc)."""
    if kind == "cat":
        levels = pd.unique(values)
        if len(levels) < 2:
            return 0.0, None, "single level"
        best = (0.0, None, "")
        # For >2 levels, try every one-vs-rest split.
        cands = ([(levels[0],)] if len(levels) == 2
                 else [(lv,) for lv in levels])
        for lv in cands:
            lab = np.isin(values, lv).astype(int)
            if min(lab.sum(), (1 - lab).sum()) < min_group:
                continue
            e = _eta_squared(X, lab)
            if e > best[0]:
                best = (e, lab, f"{lv[0]} vs rest")
        return best

    v = values.astype(float)
    order = np.argsort(v)
    best = (0.0, None, "")
    uniq = np.unique(v)
    if len(uniq) < 2:
        return 0.0, None, "constant"
    # Scan candidate thresholds (midpoints), respecting the min group size.
    for thr in uniq[1:]:
        lab = (v >= thr).astype(int)
        if min(lab.sum(), (1 - lab).sum()) < min_group:
            continue
        e = _eta_squared(X, lab)
        if e > best[0]:
            best = (e, lab, f">= {thr:.4g}")
    del order
    return best


def permutation_p(X: np.ndarray, eta2: float, labels: np.ndarray, n=2000, rng=None) -> float:
    """How often does a RANDOM split of the same size reach this eta^2?

    Guards against the obvious trap: scanning every threshold on 200 samples
    finds *some* split with a decent eta^2 even in pure noise.
    """
    if labels is None:
        return 1.0
    rng = rng or np.random.default_rng(0)
    k = int(labels.sum())
    n_s = len(labels)
    hits = 0
    for _ in range(n):
        lab = np.zeros(n_s, dtype=int)
        lab[rng.choice(n_s, k, replace=False)] = 1
        if _eta_squared(X, lab) >= eta2:
            hits += 1
    return (hits + 1) / (n + 1)


def _r2(y_true: np.ndarray, y_pred: np.ndarray, y_base: np.ndarray) -> float:
    """Variance-weighted R² against a constant-baseline prediction."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_base) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _folds(n, k, rng):
    idx = rng.permutation(n)
    return [(np.setdiff1d(idx, f), f) for f in np.array_split(idx, k)]


def _folds_grouped(groups: np.ndarray, k, rng):
    """Fold at the MEASUREMENT-DAY level, never within a day.

    PAINT calibrations come in sessions: a dozen images minutes apart, at almost
    identical sun and motor positions. With plain shuffled folds a sample's
    nearest neighbour in *any* criterion is usually its own session-mate, so
    every criterion — calendar date most of all — scores as "predictive" by
    memorising the session. Holding out whole days removes that shortcut.
    """
    ug = rng.permutation(np.unique(groups))
    return [
        (np.flatnonzero(~np.isin(groups, c)), np.flatnonzero(np.isin(groups, c)))
        for c in np.array_split(ug, min(k, len(ug)))
    ]


def cv_forest_r2(X: np.ndarray, F: np.ndarray, groups: np.ndarray,
                 n_folds=5, n_rep=4, rng=None, seed=0):
    """Day-grouped cross-validated R² of a random forest predicting the residual.

    A forest rather than k-NN: it is scale-free, tolerates irrelevant columns
    (so adding a feature cannot hurt merely through the curse of dimensionality)
    and is piecewise constant — which is exactly "condition the parameter set on
    these variables". Reading: R² here is the fraction of residual variance that
    conditioning on F could remove, out of sample.
    """
    from sklearn.ensemble import RandomForestRegressor

    rng = rng or np.random.default_rng(0)
    n = len(X)
    pred, base_p, w = np.zeros_like(X), np.zeros_like(X), np.zeros(n)
    for r in range(n_rep):
        for tr, te in _folds_grouped(groups, n_folds, rng):
            if len(tr) < 20 or len(te) == 0:
                continue
            base_p[te] += X[tr].mean(0)
            # n_jobs stays sequential on purpose: at ~200 samples a joblib pool
            # costs more to spin up and tear down than the trees cost to grow
            # (profiled: 59 s of pool churn out of 96 s per heliostat). Results
            # are unaffected — trees are seeded from random_state either way.
            rf = RandomForestRegressor(
                n_estimators=200, min_samples_leaf=5, random_state=seed + r
            ).fit(F[tr], X[tr])
            pred[te] += rf.predict(F[te])
            w[te] += 1
    ok = w > 0
    w = np.where(w == 0, 1, w)[:, None]
    pred, base_p = pred / w, base_p / w
    return (_r2(X[ok], pred[ok], base_p[ok]),
            float(2 * np.median(np.linalg.norm(X[ok] - pred[ok], axis=1))),
            float(2 * np.median(np.linalg.norm(X[ok] - base_p[ok], axis=1))))


def _as_matrix(values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "cont":
        return values.astype(float).reshape(-1, 1)
    return pd.get_dummies(pd.Series(values)).values.astype(float)


# Sun azimuth and elevation fix the heliostat's whole geometric state: for a
# tracking heliostat the motor positions, the time of day and the season are all
# deterministic functions of them (plus the target). Anything explained by this
# base set is a WORKSPACE effect — a missing model term — not a reason to split.
GEOMETRY_BASE = ["sun_azimuth", "sun_elevation"]


def cv_scores(X: np.ndarray, values: np.ndarray, kind: str, groups: np.ndarray,
              base_F: np.ndarray, base_r2: float,
              n_folds=5, n_rep=4, min_group=15, rng=None) -> dict:
    """What this criterion is worth, out of sample, on day-grouped folds.

    Three numbers, answering three different questions:

      r2_step — one constant offset per group under the best 2-way threshold.
                Literally "fit two parameter sets and pick between them by this
                rule". The thing the user asked about.
      r2_full — a forest conditioned on the criterion: the best that ANY number
                of groups on this axis could do. r2_full >> r2_step means two
                parameter sets are not enough and the dependence is graded.
      r2_incr — the forest's gain from adding this criterion ON TOP of sun
                azimuth + elevation. This is the number that survives
                confounding: motor positions, time of day and season are all
                deterministic functions of the sun position for a tracking
                heliostat, so they score high on r2_full for free. Only a
                positive r2_incr is genuinely new information.
    """
    rng = rng or np.random.default_rng(0)
    n = len(X)
    F = _as_matrix(values, kind)

    r2_full, mrad_full, mrad_base = cv_forest_r2(X, F, groups, n_folds, n_rep, rng)
    r2_both, _, _ = cv_forest_r2(
        X, np.hstack([base_F, F]), groups, n_folds, n_rep, rng)

    pred_step, pred_base, w = np.zeros_like(X), np.zeros_like(X), np.zeros(n)
    for _ in range(n_rep):
        for tr, te in _folds_grouped(groups, n_folds, rng):
            if len(tr) < 20 or len(te) == 0:
                continue
            base = X[tr].mean(0)
            pred_base[te] += base
            p = np.tile(base, (len(te), 1))
            _, lab_tr, _ = best_two_way_split(X[tr], values[tr], kind,
                                              min_group=min_group)
            if lab_tr is not None:
                if kind == "cat":
                    lv = list(set(values[tr][lab_tr == 1]))
                    lab_te = np.isin(values[te], lv).astype(int)
                else:
                    thr = values[tr].astype(float)[lab_tr == 1].min()
                    lab_te = (values[te].astype(float) >= thr).astype(int)
                for g in (0, 1):
                    m_tr, m_te = lab_tr == g, lab_te == g
                    if m_tr.sum() >= 5 and m_te.sum() > 0:
                        p[m_te] = X[tr][m_tr].mean(0)
            pred_step[te] += p
            w[te] += 1
    ok = w > 0
    w = np.where(w == 0, 1, w)[:, None]
    pred_step, pred_base = pred_step / w, pred_base / w

    # Joint-angle mrad -> beam mrad: a mirror rotation doubles in the beam.
    return dict(
        r2_step=_r2(X[ok], pred_step[ok], pred_base[ok]),
        r2_full=r2_full,
        r2_incr=r2_both - base_r2,
        mrad_base=mrad_base,
        mrad_step=float(2 * np.median(np.linalg.norm(X[ok] - pred_step[ok], axis=1))),
        mrad_full=mrad_full,
    )


def cv_joint_ceiling(X: np.ndarray, df: pd.DataFrame, cols: list[str],
                     groups: np.ndarray, n_folds=5, n_rep=4, rng=None) -> dict:
    """Upper bound: how predictable is the residual from ALL criteria together?

    Whatever this model cannot explain is noise — centroid-extraction error,
    tracking jitter, atmospheric refraction, real surface deformation — and no
    amount of splitting or extra parameters will recover it. This number is the
    ceiling of the whole idea, and the honest thing to quote alongside any
    single-criterion result.
    """
    F = df[cols].values.astype(float)
    r2, mrad, mrad_base = cv_forest_r2(X, F, groups, n_folds, n_rep,
                                       rng or np.random.default_rng(0))
    return dict(r2_joint=r2, mrad_base=mrad_base, mrad_joint=mrad)



def gmm_clusters(X: np.ndarray, kmax=4, seed=0):
    """Unsupervised mixture on the residual; BIC picks k. Returns (k, labels, bics)."""
    from sklearn.mixture import GaussianMixture
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    bics, models = [], []
    for k in range(1, kmax + 1):
        gm = GaussianMixture(k, covariance_type="full", n_init=10, random_state=seed).fit(Z)
        bics.append(gm.bic(Z))
        models.append(gm)
    k = int(np.argmin(bics)) + 1
    return k, models[k - 1].predict(Z), bics


def criterion_vs_clusters(values: np.ndarray, kind: str, labels: np.ndarray) -> float:
    """How well does the criterion predict the discovered clusters? [0..1]."""
    if len(np.unique(labels)) < 2:
        return 0.0
    if kind == "cont":
        # Max pairwise AUC, rescaled to 0..1 (0.5 = no information).
        from sklearn.metrics import roc_auc_score
        best = 0.5
        for g in np.unique(labels):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a = roc_auc_score((labels == g).astype(int), values.astype(float))
            best = max(best, max(a, 1 - a))
        return float(2 * (best - 0.5))
    # Cramer's V for categorical.
    tab = pd.crosstab(pd.Series(values), pd.Series(labels)).values
    if min(tab.shape) < 2:
        return 0.0
    n = tab.sum()
    exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / n
    chi2 = (((tab - exp) ** 2) / np.maximum(exp, 1e-9)).sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


# ---------------------------------------------------------------------------
# Where to cut — changepoint detection on the time axis
# ---------------------------------------------------------------------------

def _seg_cost(X: np.ndarray) -> float:
    return float(((X - X.mean(0)) ** 2).sum()) if len(X) else 0.0


def changepoints(X: np.ndarray, max_k=5, min_seg=12):
    """Binary segmentation of the chronologically ordered residual, BIC-selected.

    Turns "split by time" into an actionable recommendation: how many segments,
    and at which sample boundaries. Cost of a segment is the scatter around its
    own mean, so a boundary is placed only where the mean genuinely shifts.
    """
    n, d = X.shape
    bounds = [0, n]
    chosen, bics = [], []
    for k in range(max_k + 1):
        rss = sum(_seg_cost(X[a:b]) for a, b in zip(bounds[:-1], bounds[1:]))
        bics.append(n * d * np.log(max(rss, 1e-12) / (n * d)) + (k + 1) * d * np.log(n))
        best = None
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < 2 * min_seg:
                continue
            base = _seg_cost(X[a:b])
            for c in range(a + min_seg, b - min_seg + 1):
                gain = base - _seg_cost(X[a:c]) - _seg_cost(X[c:b])
                if best is None or gain > best[0]:
                    best = (gain, c)
        if best is None:
            break
        chosen.append(best[1])
        bounds = sorted(bounds + [best[1]])
    k_best = int(np.argmin(bics))
    return sorted(chosen[:k_best]), bics
