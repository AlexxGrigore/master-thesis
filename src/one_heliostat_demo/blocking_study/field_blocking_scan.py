"""Geometric field-wide blocking scan — which heliostat showcases blocking best?

Pure math, NO flux rendering. For every heliostat in the full 1277-heliostat
scenario (all aimed at ``solar_tower_juelich_lower`` — the max-blocking
reference pose, same convention as the AY36 validation experiment), under one
representative sun position (AY36 train sample 0011: az 236.4 deg, el 8.3 deg,
the worst-blocking sample of the sun sweep):

  * cast each candidate's reflected beam (surface points -> lower-target
    rectangle, approximated by the segment to the target center),
  * test intersection against every other heliostat's aimed mirror rectangle
    (plane + bounded 2D extents fitted to its aimed surface points),
  * geometric blocked fraction = fraction of surface points whose segment to
    the target is intercepted before arrival.

The scan subsamples each mirror (~100 points). For the top-10 candidates the
blocked mask is recomputed on the FULL 25x25-per-facet grid (against their
actual blockers only) and checked for contiguity (largest 4-connected component
on the mirror grid covers >= 80 % of blocked points -> a single visible
spatial bite in the flux image rather than salt-and-pepper dimming).

Outputs (new files only, under blocking_validation/field_scan/):
    topdown_field_map.png   field map colored by blocked fraction
    data/blocking_scan.json per-heliostat results + top-10 details
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_here = pathlib.Path(__file__).resolve().parent          # blocking_study/
_src = _here.parents[1]
_sh = _src / "one_heliostat_demo" / "single_heliostat"
for _p in (str(_src), str(_sh), str(_here)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

from artist.util import set_logger_config  # noqa: E402

import validate_blocking_flux as vbf  # noqa: E402

log = logging.getLogger(__name__)

OUT_DIR = vbf.OUT_DIR.parent / "field_scan"
SCAN_POINTS = 100          # subsampled mirror points per heliostat for the scan
CONTIGUITY_MIN_SHARE = 0.8
SUN_SAMPLE_ID = "0011"     # az 236.4 deg, el 8.3 deg (worst AY36 train sample)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def fit_rectangles(surfaces: torch.Tensor) -> dict[str, torch.Tensor]:
    """Fit one bounded plane-rectangle per heliostat to its aimed surface points.

    surfaces: [N, P, 4] world points. Returns centers [N,3], normals [N,3],
    basis_u/basis_v [N,3], half-extents hu/hv [N] (max projection extent).
    """
    pts = surfaces[:, :, :3].double()
    centers = pts.mean(dim=1)
    _, _, vv = torch.svd(pts - centers.unsqueeze(1))
    normals = vv[:, :, 2]
    basis_u = vv[:, :, 0]
    basis_v = vv[:, :, 1]
    rel = pts - centers.unsqueeze(1)
    hu = (rel @ basis_u.unsqueeze(-1)).abs().amax(dim=(1, 2))
    hv = (rel @ basis_v.unsqueeze(-1)).abs().amax(dim=(1, 2))
    return {"c": centers, "n": normals, "u": basis_u, "v": basis_v, "hu": hu, "hv": hv}


def blocked_mask_for_candidate(
    origins: torch.Tensor,
    target_center: torch.Tensor,
    rects: dict[str, torch.Tensor],
    skip_row: int,
    eps: float = 0.05,
) -> torch.Tensor:
    """Boolean mask [P]: is the segment origin -> target_center intercepted?

    Vectorized over (P origins) x (N rectangles). Self row ``skip_row`` and
    intersections closer than ``eps`` to the origin are ignored.
    """
    d = target_center.double() - origins.double()          # [P,3]
    t_target = d.norm(dim=-1, keepdim=True)                # [P,1]
    d_hat = d / t_target                                   # [P,3]

    c = rects["c"]                                         # [N,3]
    n = rects["n"]
    denom = d_hat @ n.T                                    # [P,N]
    numer = ((c - origins.double().unsqueeze(1)) * n).sum(-1)  # [P,N]
    with torch.no_grad():
        t = numer / torch.where(denom.abs() < 1e-12, torch.full_like(denom, torch.inf), denom)
    valid = (t > eps) & (t < (t_target - eps))
    hit_p = origins.double().unsqueeze(1) + t.unsqueeze(-1) * d_hat.unsqueeze(1)  # [P,N,3]
    rel = hit_p - c
    pu = (rel * rects["u"]).sum(-1).abs()
    pv = (rel * rects["v"]).sum(-1).abs()
    inside = (pu <= rects["hu"] + 1e-9) & (pv <= rects["hv"] + 1e-9)
    hits = valid & inside
    hits[:, skip_row] = False
    return hits.any(dim=1)


def largest_component_share(mask2d: np.ndarray) -> float:
    """Share of True cells in the largest 4-connected component (BFS)."""
    seen = np.zeros_like(mask2d, dtype=bool)
    total = int(mask2d.sum())
    if total == 0:
        return 0.0
    best = 0
    for seed in zip(*np.nonzero(mask2d)):
        if seen[seed]:
            continue
        stack = [seed]
        seen[seed] = True
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    0 <= ny < mask2d.shape[0]
                    and 0 <= nx < mask2d.shape[1]
                    and mask2d[ny, nx]
                    and not seen[ny, nx]
                ):
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        best = max(best, size)
    return best / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("artist", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    (OUT_DIR / "data").mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    torch.manual_seed(0)

    t0 = time.time()
    scenario, hg, ay36_idx, target_index, aim_center, _ = vbf.load_context(
        device, surface_points_per_facet=25, rays_per_surface_point=1
    )
    names = [str(n) for n in hg.names]
    n_hel = hg.number_of_heliostats
    n_points = hg.surface_points.shape[1]
    points_per_facet = 25 * 25
    n_facets = n_points // points_per_facet
    log.info(f"{n_hel} heliostats x {n_points} points ({n_facets} facets x 25x25)")

    # Representative sun: AY36 train sample 0011 (worst blocking of the sweep).
    props = json.load(
        open(vbf.DATASET_DIR / "train" / "AY36" / SUN_SAMPLE_ID / "calibration_properties.json")
    )
    d = np.asarray(props["incident_ray_direction"][:3], dtype=float)
    d = d / np.linalg.norm(d)
    sun = torch.tensor(d.tolist() + [0.0], dtype=torch.float, device=device)
    sun_vec = -d
    az = float(np.degrees(np.arctan2(sun_vec[0], sun_vec[1])) % 360.0)
    el = float(np.degrees(np.arcsin(np.clip(sun_vec[2], -1, 1))))
    log.info(f"sun sample {SUN_SAMPLE_ID}: az={az:.1f} deg, el={el:.1f} deg")

    # Aim the WHOLE field at the lower target once (max-blocking pose).
    surfaces = vbf.field_surfaces_with_custom_normals(hg, sun, aim_center, None, device)
    rects = fit_rectangles(surfaces.cpu())
    target_center = aim_center[:3].detach().cpu()
    tower_xy = np.array([float(target_center[0]), float(target_center[1])])
    positions = hg.positions.detach().cpu()[:, :3]

    # ---------------- scan (subsampled) ----------------
    blocked_frac = np.zeros(n_hel)
    # 2D subsample: every 5th point in both facet directions (4 facets x 5x5
    # = 100 points); a flat stride aliases on the facet grid.
    sub_idx = (
        np.arange(n_points).reshape(n_facets, 25, 25)[:, ::5, ::5].ravel()
    )
    for i in range(n_hel):
        pts = surfaces[i, sub_idx, :3].cpu()
        mask = blocked_mask_for_candidate(pts, target_center, rects, skip_row=i)
        blocked_frac[i] = float(mask.float().mean())
        if (i + 1) % 200 == 0:
            log.info(f"scanned {i + 1}/{n_hel} ({time.time() - t0:.0f}s)")
    log.info(f"scan done in {time.time() - t0:.0f}s")

    order = np.argsort(-blocked_frac)
    # Refine more candidates than needed, then re-rank by the full-grid value
    # (the subsampled scan can alias on the facet grid). Always include AY36.
    refine_rows = sorted(set(order[:15].tolist()) | {ay36_idx})

    # ---------------- top-10 refinement (full grid + contiguity) ----------------
    details = {}
    for i in refine_rows:
        pts = surfaces[i, :, :3].cpu()
        mask = blocked_mask_for_candidate(pts, target_center, rects, skip_row=i)
        frac_full = float(mask.float().mean())
        # blockers: recompute per-rectangle hits on the blocked subset
        blockers = []
        if mask.any():
            d_vec = target_center.double() - pts[mask].double()
            tt = d_vec.norm(dim=-1, keepdim=True)
            d_hat = d_vec / tt
            for j in range(n_hel):
                if j == i:
                    continue
                n = rects["n"][j]
                denom = d_hat @ n
                numer = ((rects["c"][j] - pts[mask].double()) * n).sum(-1)
                t = numer / torch.where(denom.abs() < 1e-12, torch.full_like(denom, torch.inf), denom)
                valid = (t > 0.05) & (t < (tt.squeeze(-1) - 0.05))
                if not valid.any():
                    continue
                hit_p = pts[mask].double()[valid] + t[valid].unsqueeze(-1) * d_hat[valid]
                rel = hit_p - rects["c"][j]
                pu = (rel * rects["u"][j]).sum(-1).abs()
                pv = (rel * rects["v"][j]).sum(-1).abs()
                nhits = int(((pu <= rects["hu"][j]) & (pv <= rects["hv"][j])).sum())
                if nhits > 0:
                    blockers.append({"name": names[j], "n_blocked_points": nhits})
            blockers.sort(key=lambda b: -b["n_blocked_points"])
        # contiguity per facet on the 25x25 grid
        mask_np = mask.numpy().reshape(n_facets, 25, 25)
        shares = [
            largest_component_share(mask_np[f]) for f in range(n_facets) if mask_np[f].any()
        ]
        contiguous = bool(shares and min(shares) >= CONTIGUITY_MIN_SHARE)
        dist = float(torch.norm(positions[i] - target_center))
        details[names[i]] = {
            "row": int(i),
            "blocked_fraction_full_grid": frac_full,
            "n_blockers": len(blockers),
            "blockers": blockers,
            "contiguous_shadow": contiguous,
            "contiguity_min_component_share": round(min(shares), 3) if shares else None,
            "tower_distance_m": round(dist, 1),
        }
        log.info(
            f"top {names[i]}: full-grid blocked={frac_full:.4f}, blockers="
            f"{[b['name'] for b in blockers]}, contiguous={contiguous}, dist={dist:.0f} m"
        )

    # Re-rank refined candidates by the full-grid blocked fraction.
    top10 = sorted(refine_rows, key=lambda r: -details[names[r]]["blocked_fraction_full_grid"])[:10]

    # ---------------- outputs ----------------
    result = {
        "config": {
            "scenario": str(vbf.SCENARIO_PATH),
            "target_name": vbf.TARGET_NAME,
            "target_area_index": int(target_index),
            "sun_sample": SUN_SAMPLE_ID,
            "sun_azimuth_deg": round(az, 2),
            "sun_elevation_deg": round(el, 2),
            "surface_points_per_facet": 25,
            "scan_points_per_mirror": SCAN_POINTS,
            "contiguity_min_share": CONTIGUITY_MIN_SHARE,
            "pose": "all heliostats aimed at solar_tower_juelich_lower (max-blocking)",
            "method": "geometric segment/rectangle intersection, no flux rendering",
        },
        "blocked_fraction_scan": {names[i]: round(float(blocked_frac[i]), 5) for i in range(n_hel)},
        "top10": [
            {"name": names[i], "blocked_fraction_scan": round(float(blocked_frac[i]), 5),
             **details[names[i]]}
            for i in top10
        ],
        "ay36_reference": {
            "blocked_fraction_scan": round(float(blocked_frac[ay36_idx]), 5),
            "blocked_fraction_full_grid": details[names[ay36_idx]]["blocked_fraction_full_grid"],
            "contiguous_shadow": details[names[ay36_idx]]["contiguous_shadow"],
            "blockers": [b["name"] for b in details[names[ay36_idx]]["blockers"]],
            "rank_full_grid": int(
                (np.array([details[names[r]]["blocked_fraction_full_grid"] for r in refine_rows])
                 > details[names[ay36_idx]]["blocked_fraction_full_grid"]).sum()
            ) + 1,
            "note": "rank among the refined candidates only",
        },
    }
    (OUT_DIR / "data" / "blocking_scan.json").write_text(json.dumps(result, indent=1))

    # ---------------- field map ----------------
    xy = positions.numpy()
    fig, ax = plt.subplots(figsize=(13, 13))
    sc = ax.scatter(
        xy[:, 0], xy[:, 1], c=blocked_frac * 100, cmap="hot_r", s=18,
        vmin=0, vmax=max(1.0, float(blocked_frac.max() * 100)),
    )
    fig.colorbar(sc, ax=ax, label="geometric blocked fraction [%]", shrink=0.8)
    ax.scatter(*tower_xy, marker="^", s=250, c="black", zorder=5, label="tower (lower target)")
    ax.scatter(*xy[ay36_idx, :2], marker="*", s=350, c="blue", zorder=6, label="AY36 (reference)")
    label_offsets = [(10, 10), (10, -34), (-80, 14)]
    for k, i in enumerate(top10[:3]):
        pct = details[names[i]]["blocked_fraction_full_grid"] * 100
        ax.annotate(
            f"#{k + 1} {names[i]}\n{pct:.1f} %",
            (xy[i, 0], xy[i, 1]),
            textcoords="offset points", xytext=label_offsets[k], fontsize=11, weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85),
        )
        ax.scatter(*xy[i, :2], marker="o", s=180, facecolors="none",
                   edgecolors="green", linewidths=2, zorder=6)
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title(
        f"Geometric blocking scan — {n_hel} heliostats aimed at {vbf.TARGET_NAME}\n"
        f"sun sample {SUN_SAMPLE_ID} (az {az:.1f} deg, el {el:.1f} deg), "
        f"segment->target vs aimed mirror rectangles"
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "topdown_field_map.png", dpi=150)
    plt.close(fig)

    # ---------------- top-10 table ----------------
    print("\ntop-10 blocking candidates")
    print(f"{'rank':<5}{'name':<8}{'blocked%':>9}{'#blk':>6}{'contig':>8}{'dist[m]':>9}  blockers")
    for k, i in enumerate(top10):
        det = details[names[i]]
        print(
            f"{k + 1:<5}{names[i]:<8}{det['blocked_fraction_full_grid'] * 100:>9.2f}"
            f"{det['n_blockers']:>6}{str(det['contiguous_shadow']):>8}"
            f"{det['tower_distance_m']:>9.0f}  "
            f"{[b['name'] for b in det['blockers']]}"
        )
    print(f"\nAY36 reference: scan {blocked_frac[ay36_idx] * 100:.2f} %, full-grid "
          f"{details[names[ay36_idx]]['blocked_fraction_full_grid'] * 100:.2f} %, "
          f"blockers {[b['name'] for b in details[names[ay36_idx]]['blockers']]}")
    log.info(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
