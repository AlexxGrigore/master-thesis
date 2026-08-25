"""Mathematical blocker census: vertical-mirror shadow corridor, no ray tracing.

Assumption: every heliostat in the field stands with its mirror VERTICAL
(normal horizontal). A heliostat B can block incident sunlight for the studied
heliostat A only if B sits inside A's shadow corridor: the slanted prism swept
by A's mirror along the sun direction, projected onto the ground plane.

Per sample (sun direction d, light travel direction, d_u < 0):

    s   = horizontal unit vector of d            (shadow cast direction)
    r   = |d_h| / (-d_u)                          (ground length per unit height)
    A's mirror vertical span [z_lo, z_hi] casts a ground shadow band
        lambda in [z_lo * r, z_hi * r]            (distance from A along s)
    B at horizontal offset v from A blocks iff
        lambda = v . s  in  [z_lo*r - tol, z_hi*r + tol]
        |kappa| = |v - lambda*s|  <=  (w_A + w_B)/2 + margin

plus a vertical-overlap check: the ray from A's mirror at height z reaches
height z - lambda / r at the blocker's longitudinal position; this band must
overlap B's vertical mirror span.

Union over all 199 sampled sun positions of the AY36 blocking dataset.

Outputs: CSV + JSON + a field map PNG under
outputs/new_mapping_function/blocking_study/vertical_shadow/AY36/
"""
from __future__ import annotations

import json
import pathlib
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_here = pathlib.Path(__file__).resolve().parent
_ROOT = _here.parents[2]
_ARTIST = _ROOT.parent / "ARTIST"
for _p in (str(_ROOT / "src"), str(_ARTIST)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from artist.scenario.scenario import Scenario  # noqa: E402

HID = "AY36"
SCENARIO = _ROOT / "scenarios" / "full_benchmark_ideal" / "ideal_1277.h5"
REPORT = (
    _ROOT / "outputs" / "new_mapping_function" / "blocking_study"
    / "experiment_s" / HID / "generation_report.json"
)
OUT = _ROOT / "outputs" / "new_mapping_function" / "blocking_study" / "vertical_shadow" / HID
MARGIN_M = 0.5  # lateral safety margin on top of mirror half-widths


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    with h5py.File(SCENARIO) as handle:
        scenario = Scenario.load_scenario_from_hdf5(handle, device=device)
    group = scenario.heliostat_field.heliostat_groups[0]
    names = [str(n) for n in group.names]
    positions = group.positions.detach().cpu().numpy()[:, :3]  # [N,3] ENU
    ia = names.index(HID)

    # Mirror dimensions from AY36's own surface point cloud (nominal pose):
    # lateral width and vertical height of the facet envelope.
    # surface_points are LOCAL: x = width, y = height, z ~ 0 (near-planar).
    pts = group.surface_points[ia].detach().cpu().numpy()[:, :3]
    half_w = float(np.abs(pts[:, 0]).max())
    half_h = float(np.abs(pts[:, 1]).max())
    z_pivot = float(positions[ia, 2])
    print(f"{HID}: pivot z={z_pivot:.2f} m, mirror half-width~{half_w:.2f} m, half-height~{half_h:.2f} m")

    sun_dirs = np.array(
        [s["incident_ray_direction"][:3] for s in json.load(open(REPORT))["samples"]]
    )  # [199,3] light travel direction

    offsets = positions[:, :2] - positions[ia, :2]           # [N,2]
    z_all = positions[:, 2]

    counts = np.zeros(len(names), dtype=int)
    per_sample_lists: list[list[str]] = []

    for d in sun_dirs:
        d = d / np.linalg.norm(d)
        dh = d[:2]
        nrm = np.linalg.norm(dh)
        s = dh / nrm                                       # light travel / shadow cast dir
        inv_r = (-d[2]) / nrm                              # height drop per ground metre
        blockers = []
        for i, name in enumerate(names):
            if i == ia:
                continue
            v = offsets[i]
            lam = float(v @ s)                             # >0 = downstream of A
            t = -lam                                       # upstream distance B->A
            if t < half_w:                                 # B must be upstream of A
                continue
            kap = abs(float(v[0] * s[1] - v[1] * s[0]))    # |v x s|
            if kap > 2 * half_w + MARGIN_M:
                continue
            # A ray leaving B's mirror at height z_b descends inv_r per metre.
            # At the B->A gap t it is at z_b - t*inv_r; shading of A happens iff
            # that band overlaps A's vertical mirror span (with mirror width
            # loosening t by +/- half_w).
            t_lo, t_hi = max(t - half_w, 0.0), t + half_w
            z_band = [z_all[i] - half_h - t_hi * inv_r,
                      z_all[i] + half_h - t_lo * inv_r]
            if z_band[1] < z_pivot - half_h or z_band[0] > z_pivot + half_h:
                continue
            blockers.append(name)
        per_sample_lists.append(blockers)
        for b in blockers:
            counts[names.index(b)] += 1

    union = [n for n in names if counts[names.index(n)] > 0 and n != HID]
    union_sorted = sorted(union, key=lambda n: -counts[names.index(n)])
    result = {
        "heliostat_id": HID,
        "assumption": "all mirrors vertical (normal horizontal); incident-side shading only",
        "n_sun_positions": len(sun_dirs),
        "n_blockers_union": len(union_sorted),
        "blockers": [
            {"name": n, "n_samples_blocked": int(counts[names.index(n)]),
             "east_m": float(positions[names.index(n), 0]),
             "north_m": float(positions[names.index(n), 1])}
            for n in union_sorted
        ],
        "per_sample_blockers": per_sample_lists,
    }
    (OUT / "vertical_shadow_blockers.json").write_text(json.dumps(result, indent=1))

    # ---- field map ------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(positions[:, 0], positions[:, 1], s=6, c="lightgrey", label="field (no block)")
    idx_u = [names.index(n) for n in union_sorted]
    sc = ax.scatter(positions[idx_u, 0], positions[idx_u, 1],
                    c=counts[idx_u], cmap="hot_r", s=40, vmin=1, vmax=counts.max(),
                    label=f"blocks ≥1 sample (n={len(union_sorted)})")
    ax.scatter(*positions[ia, :2], marker="*", s=400, c="blue", zorder=5, label=HID)
    for n in union_sorted[:15]:
        i = names.index(n)
        ax.annotate(f"{n}\n{counts[i]}", (positions[i, 0], positions[i, 1]),
                    fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("east [m]"); ax.set_ylabel("north [m]")
    ax.set_title(f"{HID} — vertical-mirror shadow census, union over {len(sun_dirs)} sun positions\n"
                 f"color = #samples blocked (incident side)")
    ax.legend(); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    fig.colorbar(sc, label="# samples blocked")
    fig.tight_layout()
    fig.savefig(OUT / "vertical_shadow_map.png", dpi=150)

    print(f"union: {len(union_sorted)} blockers over {len(sun_dirs)} sun positions")
    print("top 15:", [(n, int(counts[names.index(n)])) for n in union_sorted[:15]])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
