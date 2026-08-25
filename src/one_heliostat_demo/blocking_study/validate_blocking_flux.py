"""Validate ARTIST blocking visually: two animations + grid plots for AY36.

Scenario: full 1277-heliostat benchmark field (blocking enabled), AY36 always
aimed at the solar-tower-lower target (``solar_tower_juelich_lower``).

Animation 1 — SUN SWEEP
    All 1277 heliostats (AY36 included) aimed at ``solar_tower_juelich_lower``
    under each sun position of the AY36 TRAIN split of the full-field blocking
    dataset, ordered east -> west (sorted by sun azimuth). Per frame: flux image
    at the lower target with blocking ON + measured blocked-ray fraction.

Animation 2 — BLOCKER POSE SWEEP (vertical -> horizontal)
    Fixed sun position: the TRAIN sample with the highest measured blocked
    fraction (from animation 1; cross-checked against the long-experiment test
    fractions, worst approx 0.263). AY36 stays aimed at the lower target. The
    heliostats that geometrically sit inside the AY36 -> lower-target ray cone
    (the only ones that can block its reflected light) are tilted step by step
    from VERTICAL (mirror plane vertical, normal horizontal, facing the sun's
    azimuth) to HORIZONTAL (mirror plane horizontal, normal = zenith/stow),
    while every other heliostat stays aimed at the lower target. Per frame:
    flux + measured blocked-ray fraction.

Blocker orientations are set through ARTIST's public aim-point API only: for a
desired mirror normal n and light-travel direction d, the reflection law gives
the required reflection direction r = d - 2 (d . n) n, and aiming the heliostat
at ``position + 1000 * r`` makes ARTIST converge to n. ARTIST itself is never
modified; blocking goes through ``brute_blocking.exact_blocking`` (see
blocking_utils.py / brute_blocking.py for why the stock LBVH filter is unsafe).

Frames are rendered once to ``data/frames_*/`` as PNGs; the GIFs are assembled
from those PNGs, so a different frame duration only requires re-running with
``--assemble-only`` (no re-rendering).

Usage
-----
    python validate_blocking_flux.py                 # full run (anim 1 + 2)
    python validate_blocking_flux.py --max-frames 3  # smoke test
    python validate_blocking_flux.py --assemble-only # rebuild gifs/grids only
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

_here = pathlib.Path(__file__).resolve().parent          # blocking_study/
_src = _here.parents[1]                                   # src/
_sh = _src / "one_heliostat_demo" / "single_heliostat"    # single_heliostat/
for _p in (str(_src), str(_sh), str(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer  # noqa: E402
from artist.scene.sun import Sun  # noqa: E402
from artist.util import set_logger_config  # noqa: E402

import train as tr  # noqa: E402
from blocking_utils import forward_pass_blocking, one_hot_mask  # noqa: E402
from brute_blocking import exact_blocking  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FRAME_DURATION_S = 0.01        # gif frame duration (change me + --assemble-only)
TARGET_NAME = "solar_tower_juelich_lower"
RAYS_PER_SURFACE_POINT = 8     # 8 x 625 surface points = 5000 rays per frame
POSE_STEPS = 15                # vertical -> horizontal interpolation steps
RANDOM_SEED = 0
BLOCKER_MARGIN_M = 0.75        # extra margin for the geometric cone test
GRID_FRAMES = 16               # representative frames in the grid plots

_ROOT = _here.parents[2]       # master-thesis/
SCENARIO_PATH = (
    _ROOT / "scenarios" / "full_benchmark_ideal" / "ideal_1277_AY36_deflectometry.h5"
)
DATASET_DIR = _ROOT / "datasets" / "synthetic" / "fullfield_blocking_dataset" / "dataset"


def configure(heliostat_id: str, out_name: str | None = None) -> None:
    """Point all module-level output paths at one showcase heliostat.

    ``out_name`` overrides the output-folder name (default: ``heliostat_id``)
    so variant runs (e.g. point-source sun) don't clobber the reference run.
    """
    global HID, OUT_DIR, GIF_DIR, PLOT_DIR, DATA_DIR, FRAMES_SUN, FRAMES_POSE, SUMMARY_JSON
    HID = heliostat_id
    OUT_DIR = (
        _ROOT / "outputs" / "new_mapping_function" / "blocking_study"
        / "blocking_validation" / (out_name or HID)
    )
    GIF_DIR = OUT_DIR / "gifs"
    PLOT_DIR = OUT_DIR / "plots"
    DATA_DIR = OUT_DIR / "data"
    FRAMES_SUN = DATA_DIR / "frames_sun_sweep"
    FRAMES_POSE = DATA_DIR / "frames_pose_sweep"
    SUMMARY_JSON = DATA_DIR / "run_summary.json"


configure("AY36")
# AY36 long-experiment reference (fixed: that experiment only exists for AY36).
TEST_FRACTIONS_JSON = (
    _ROOT / "outputs" / "new_mapping_function" / "blocking_study"
    / "experiment_full_field_long" / "AY36" / "test_blocked_fractions.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_context(device: torch.device, surface_points_per_facet: int = 25,
                 rays_per_surface_point: int = RAYS_PER_SURFACE_POINT,
                 point_source_covariance: float | None = None):
    """Load scenario + group, resolve AY36 and the lower target, set ray count.

    ``point_source_covariance``, when given, replaces the scenario's sun with
    a near-delta-function light source (same 2D-normal distribution family,
    variance shrunk to this value instead of the physical ~2.1 mrad std dev)
    -- the idealized geometric-optics limit used to check whether blocking
    looks like a spatial "bite" once sunshape blur is removed.
    """
    cfg = SimpleNamespace(
        SCENARIO_PATH_TEMPLATE=str(SCENARIO_PATH),
        SURFACE_POINTS_PER_FACET=surface_points_per_facet,
    )
    scenario, hg, hel_dist_m, hel_idx = tr._load_scenario(
        HID, cfg, device, scenario_path=SCENARIO_PATH
    )
    scenario.set_number_of_rays(rays_per_surface_point)
    if point_source_covariance is not None:
        old_sun = scenario.light_sources.light_source_list[0]
        old_cov = old_sun.distribution_parameters.get("covariance")
        new_sun = Sun(
            number_of_rays=rays_per_surface_point,
            distribution_parameters={
                "distribution_type": "normal",
                "mean": 0.0,
                "covariance": point_source_covariance,
            },
            device=device,
        )
        scenario.light_sources.light_source_list[0] = new_sun
        log.info(
            f"POINT-SOURCE SUN: covariance {old_cov} -> {point_source_covariance} "
            f"(std dev {np.sqrt(old_cov) * 1000 if old_cov else float('nan'):.3f} -> "
            f"{np.sqrt(point_source_covariance) * 1000:.6f} mrad)"
        )
    target_index = scenario.solar_tower.target_name_to_index[TARGET_NAME]
    aim_center = scenario.solar_tower.get_centers_of_target_areas(
        target_area_indices=torch.tensor([target_index], device=device), device=device
    )[0]
    log.info(
        f"target '{TARGET_NAME}' -> index {target_index} "
        f"(center {aim_center[:3].tolist()}), rays/point={rays_per_surface_point}, "
        f"surface points/facet={surface_points_per_facet}"
    )
    return scenario, hg, hel_idx, target_index, aim_center, hel_dist_m


def load_train_sun_positions() -> list[dict]:
    """All TRAIN sun positions, sorted east -> west (ascending sun azimuth).

    Always read from train/AY36: the dataset only exists for AY36, and sun
    directions are heliostat-independent.
    """
    split_dir = DATASET_DIR / "train" / "AY36"
    samples = []
    for sample_dir in sorted(split_dir.iterdir()):
        if not (sample_dir.is_dir() and sample_dir.name.isdigit()):
            continue
        props = json.load(open(sample_dir / "calibration_properties.json"))
        d = np.asarray(props["incident_ray_direction"][:3], dtype=float)
        d = d / np.linalg.norm(d)
        sun = -d  # direction from the heliostat toward the sun
        az = float(np.degrees(np.arctan2(sun[0], sun[1])) % 360.0)  # 0=N, 90=E
        el = float(np.degrees(np.arcsin(np.clip(sun[2], -1, 1))))
        samples.append(
            {
                "sample_id": sample_dir.name,
                "incident_ray_direction": d.tolist(),
                "sun_azimuth_deg": az,
                "sun_elevation_deg": el,
                "motor_position": props.get("motor_position"),
                "dataset_target_area_index": props.get("target_area_index"),
            }
        )
    samples.sort(key=lambda s: s["sun_azimuth_deg"])
    return samples


def field_surfaces_with_custom_normals(
    hg,
    incident_ray_direction: torch.Tensor,
    default_aim_point: torch.Tensor,
    custom_normals: dict[int, torch.Tensor] | None,
    device: torch.device,
) -> torch.Tensor:
    """World surface points of the whole group, aimed.

    Every heliostat is aimed at ``default_aim_point`` except the rows in
    ``custom_normals`` (row index -> desired world mirror normal [3]), which are
    oriented to that normal via the reflection law and a synthetic aim point
    (ARTIST's public aim API only). Returns ``[n_hel, n_points, 4]``.
    """
    n_hel = hg.number_of_heliostats
    d3 = incident_ray_direction[:3].float()
    aim_points = default_aim_point.repeat(n_hel, 1).clone()
    if custom_normals:
        for row, normal in custom_normals.items():
            n = normal.float() / normal.float().norm()
            reflection = d3 - 2.0 * torch.dot(d3, n) * n  # r = d - 2(d.n)n
            origin = hg.positions[row, :3].float()
            aim_points[row, :3] = origin + 1000.0 * reflection
    mask = torch.ones(n_hel, dtype=torch.long, device=device)
    with torch.no_grad():
        hg.activate_heliostats(active_heliostats_mask=mask, device=device)
        hg.align_surfaces_with_incident_ray_directions(
            aim_points=aim_points,
            incident_ray_directions=incident_ray_direction.expand(n_hel, -1),
            active_heliostats_mask=mask,
            device=device,
        )
        surfaces = hg.active_surface_points.detach().clone()
    return surfaces


def trace_with_surfaces(
    scenario,
    hg,
    hel_idx: int,
    incident_ray_direction: torch.Tensor,
    target_index: int,
    aim_point: torch.Tensor,
    surfaces: torch.Tensor,
    device: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """One blocking trace of AY36 against precomputed neighbour surfaces.

    Returns (flux bitmap [H, W], blocked fraction).
    """
    n_hel = hg.number_of_heliostats
    sun = incident_ray_direction.view(1, 4)
    mask = one_hot_mask(hel_idx, 1, n_hel, device)
    with torch.no_grad():
        hg.activate_heliostats(active_heliostats_mask=mask, device=device)
        hg.align_surfaces_with_incident_ray_directions(
            aim_points=aim_point.view(1, 4),
            incident_ray_directions=sun,
            active_heliostats_mask=mask,
            device=device,
        )
        ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=hg,
            blocking_active=False,
            world_size=1,
            rank=0,
            batch_size=1,
            random_seed=RANDOM_SEED,
        )
        ray_tracer.blocking_active = True
        ray_tracer.blocking_heliostat_surfaces_active = surfaces.to(device)
        with exact_blocking():
            flux, _, _, blocking_factor = ray_tracer.trace_rays(
                incident_ray_directions=sun,
                active_heliostats_mask=mask,
                target_area_indices=torch.tensor([target_index], device=device),
                device=device,
            )
    return flux[0].detach().cpu(), float(1.0 - blocking_factor.item())


def render_frame(
    flux: torch.Tensor,
    path: pathlib.Path,
    title_lines: list[str],
    vmax: float | None = None,
) -> None:
    """Render one annotated flux frame to PNG.

    ``vmax=None`` normalizes per frame (autoscale — hides absolute flux loss!);
    pass a shared value across an animation to keep brightness comparable.
    """
    img = flux.numpy()
    if vmax is None:
        vmax = float(img.max()) if img.max() > 0 else 1.0
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=100)
    im = ax.imshow(img, cmap="inferno", origin="upper", vmin=0.0, vmax=vmax)
    ax.set_title("\n".join(title_lines), fontsize=10)
    ax.set_xlabel("bitmap e [px]", fontsize=8)
    ax.set_ylabel("bitmap u [px]", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="flux [a.u.]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def assemble_gif(frames_dir: pathlib.Path, gif_path: pathlib.Path) -> int:
    """Assemble a gif from the PNGs in ``frames_dir`` (no re-rendering)."""
    pngs = sorted(frames_dir.glob("frame_*.png"))
    if not pngs:
        raise FileNotFoundError(f"no frames in {frames_dir}")
    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in pngs]
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=int(FRAME_DURATION_S * 1000),
        loop=0,
    )
    return len(pngs)


def make_grid(frames_dir: pathlib.Path, plot_path: pathlib.Path, caption: str) -> None:
    """Wide montage of GRID_FRAMES evenly spaced representative frames."""
    pngs = sorted(frames_dir.glob("frame_*.png"))
    picks = np.linspace(0, len(pngs) - 1, min(GRID_FRAMES, len(pngs))).astype(int)
    picks = sorted(set(picks.tolist()))
    ncols = 4
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, idx in zip(axes, picks):
        ax.imshow(plt.imread(pngs[idx]))
        ax.set_title(f"frame {idx:03d}", fontsize=11)
        ax.axis("off")
    for ax in axes[len(picks):]:
        ax.axis("off")
    fig.suptitle(caption, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(plot_path, dpi=110)
    plt.close(fig)


def point_to_segments_distance(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Min distance from point p [3] to each segment a[i] -> b[j] (all pairs)."""
    na, nb = a.shape[0], b.shape[0]
    a = a.unsqueeze(1).expand(na, nb, 3).reshape(-1, 3)
    b = b.unsqueeze(0).expand(na, nb, 3).reshape(-1, 3)
    ab = b - a
    t = ((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)
    t = t.clamp(0.0, 1.0)
    closest = a + t.unsqueeze(-1) * ab
    return torch.norm(closest - p, dim=-1).min()


def identify_blockers(
    scenario,
    hg,
    hel_idx: int,
    incident_ray_direction: torch.Tensor,
    target_index: int,
    aim_point: torch.Tensor,
    device: torch.device,
) -> list[int]:
    """Geometric cone test: which heliostats can block AY36's reflected cone.

    The blocking cone is the set of segments from AY36's (aimed) mirror surface
    points to the lower-target rectangle. A heliostat is a potential blocker if
    its pivot (plus its mirror bounding radius + margin) comes within that cone.
    """
    n_hel = hg.number_of_heliostats
    sun = incident_ray_direction.view(1, 4)
    mask = one_hot_mask(hel_idx, 1, n_hel, device)
    with torch.no_grad():
        hg.activate_heliostats(active_heliostats_mask=mask, device=device)
        hg.align_surfaces_with_incident_ray_directions(
            aim_points=aim_point.view(1, 4),
            incident_ray_directions=sun,
            active_heliostats_mask=mask,
            device=device,
        )
        ay36_pts = hg.active_surface_points[0, :, :3].detach().cpu()
    ay36_pts = ay36_pts[:: max(1, ay36_pts.shape[0] // 100)]  # ~100 samples

    planar = scenario.solar_tower.target_areas[0]  # planar target areas
    local = target_index  # lower target is planar
    center = planar.centers[local, :3].detach().cpu()
    normal = planar.normals[local, :3].detach().cpu()
    dims = planar.dimensions[local].detach().cpu()
    up = torch.tensor([0.0, 0.0, 1.0])
    e_axis = torch.linalg.cross(up, normal)
    e_axis = e_axis / e_axis.norm()
    u_axis = torch.linalg.cross(normal, e_axis)
    u_axis = u_axis / u_axis.norm()
    he, hu = float(dims[0]) / 2.0, float(dims[1]) / 2.0
    target_samples = [center]
    for se in (-1.0, 0.0, 1.0):
        for su in (-1.0, 0.0, 1.0):
            target_samples.append(center + se * he * e_axis + su * hu * u_axis)
    target_samples = torch.stack(target_samples)

    local_pts = hg.surface_points.detach().cpu()  # [n_hel, n_pts, 4]
    radii = torch.linalg.norm(local_pts[:, :, :2], dim=-1).max(dim=1).values
    positions = hg.positions.detach().cpu()[:, :3]

    blockers = []
    for i in range(n_hel):
        if i == hel_idx:
            continue
        dist = point_to_segments_distance(positions[i], ay36_pts, target_samples)
        if float(dist) < float(radii[i]) + BLOCKER_MARGIN_M:
            blockers.append(i)
    return blockers


# ---------------------------------------------------------------------------
# Animation 1 — sun sweep
# ---------------------------------------------------------------------------

def run_sun_sweep(
    scenario, hg, hel_idx, target_index, aim_center, samples, device, max_frames=None
) -> list[dict]:
    FRAMES_SUN.mkdir(parents=True, exist_ok=True)
    n_hel = hg.number_of_heliostats
    aim = aim_center.view(1, 4)
    records = []
    frames = samples if max_frames is None else samples[:max_frames]
    for frame_i, sample in enumerate(frames):
        t0 = time.time()
        sun = torch.tensor(
            sample["incident_ray_direction"] + [0.0], dtype=torch.float, device=device
        )
        _, flux, blocked = forward_pass_blocking(
            scenario,
            hg,
            hel_idx,
            sun.view(1, 4),
            torch.tensor([target_index], device=device),
            device,
            aim_points=aim,
            random_seed=RANDOM_SEED,
            target_index_override=target_index,
        )
        total_flux = float(flux[0].sum())
        rec = {
            "frame": frame_i,
            "sample_id": sample["sample_id"],
            "incident_ray_direction": sample["incident_ray_direction"],
            "sun_azimuth_deg": sample["sun_azimuth_deg"],
            "sun_elevation_deg": sample["sun_elevation_deg"],
            "blocked_fraction": blocked[0],
            "total_flux": total_flux,
        }
        records.append(rec)
        render_frame(
            flux[0],
            FRAMES_SUN / f"frame_{frame_i:03d}.png",
            [
                f"{HID} -> {TARGET_NAME} | sun sweep frame {frame_i + 1}/{len(frames)}",
                f"sample {sample['sample_id']}  az={sample['sun_azimuth_deg']:.1f} deg  "
                f"el={sample['sun_elevation_deg']:.1f} deg",
                f"blocked = {blocked[0] * 100:.2f} %   total flux = {total_flux:.2f}",
            ],
        )
        log.info(
            f"[sun] frame {frame_i + 1}/{len(frames)} sample {sample['sample_id']} "
            f"blocked={blocked[0]:.4f} flux={total_flux:.3f} ({time.time() - t0:.1f}s)"
        )
    return records


# ---------------------------------------------------------------------------
# Animation 2 — blocker pose sweep
# ---------------------------------------------------------------------------

def achieved_normal(surfaces: torch.Tensor, row: int, desired: torch.Tensor) -> float:
    """Angle [deg] between the fitted plane normal of ``row`` and ``desired``.

    The normal is the smallest-singular-value direction of the surface point
    cloud — exactly what the blocking primitive fit sees.
    """
    pts = surfaces[row, :, :3].double()
    pts = pts - pts.mean(dim=0, keepdim=True)
    _, _, v = torch.svd(pts)
    normal = v[:, -1]
    normal = normal / normal.norm()
    if torch.dot(normal, desired.double()) < 0:
        normal = -normal
    cos = torch.dot(normal, desired.double()).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.arccos(cos)))


# rotation_about_axis / manually_rotated_surfaces moved to blocking_utils.py
# (forward_pass_blocking/aimed_neighbour_surfaces need them too, and this
# module already imports FROM blocking_utils, so defining them there and
# importing back here would be circular). Re-exported for every existing
# `from validate_blocking_flux import rotation_about_axis, ...` caller.
from blocking_utils import manually_rotated_surfaces, rotation_about_axis  # noqa: E402,F401


def run_pose_sweep(
    scenario, hg, hel_idx, target_index, aim_center, sun_record, device, pose_steps
) -> dict:
    FRAMES_POSE.mkdir(parents=True, exist_ok=True)
    sun = torch.tensor(
        sun_record["incident_ray_direction"] + [0.0], dtype=torch.float, device=device
    )
    d3 = sun[:3].detach().cpu()

    blocker_rows = identify_blockers(
        scenario, hg, hel_idx, sun, target_index, aim_center, device
    )
    blocker_names = [str(hg.names[i]) for i in blocker_rows]
    log.info(f"[pose] {len(blocker_rows)} cone blockers: {blocker_names}")

    # Vertical normal: horizontal, facing the sun's azimuth. Horizontal: zenith.
    sun_h = -d3[:2]
    n_vertical = torch.tensor([sun_h[0], sun_h[1], 0.0]) / torch.norm(sun_h)
    n_horizontal = torch.tensor([0.0, 0.0, 1.0])

    # Exact orientation path: vertical-pose rotation R_v (local frame: x = width,
    # y = height -> world up, z = normal -> n_vertical), tilted about the
    # horizontal axis k = n_vertical x zenith by tilt * 90 deg.
    up = torch.tensor([0.0, 0.0, 1.0])
    y_w = up
    z_w = n_vertical
    x_w = torch.linalg.cross(y_w, z_w)
    x_w = x_w / x_w.norm()
    r_vertical = torch.stack([x_w, y_w, z_w], dim=1)  # columns = world basis
    tilt_axis = torch.linalg.cross(n_vertical, n_horizontal)
    tilt_axis = tilt_axis / tilt_axis.norm()

    # Base surfaces: every heliostat aimed at the lower target (computed once);
    # blocker rows are replaced by exact rotations per step.
    base_surfaces = field_surfaces_with_custom_normals(
        hg, sun, aim_center, None, device
    ).cpu()

    records = []
    fluxes = []
    for step in range(pose_steps):
        t0 = time.time()
        tilt = step / (pose_steps - 1) if pose_steps > 1 else 1.0
        angle = tilt * (np.pi / 2.0)
        rotation = rotation_about_axis(tilt_axis, angle) @ r_vertical
        normal = float(np.cos(angle)) * n_vertical + float(np.sin(angle)) * n_horizontal
        surfaces = base_surfaces.clone()
        for row in blocker_rows:
            surfaces[row] = manually_rotated_surfaces(hg, row, rotation)
        # Diagnostic: the fitted blocking-plane normal should match exactly now.
        achieved = {
            str(hg.names[row]): round(achieved_normal(surfaces, row, normal), 2)
            for row in blocker_rows
        }
        flux, blocked = trace_with_surfaces(
            scenario, hg, hel_idx, sun, target_index, aim_center, surfaces, device
        )
        total_flux = float(flux.sum())
        fluxes.append(flux)
        records.append(
            {
                "frame": step,
                "tilt_fraction": tilt,
                "blocked_fraction": blocked,
                "total_flux": total_flux,
                "achieved_normal_deviation_deg": achieved,
            }
        )
        log.info(
            f"[pose] step {step + 1}/{pose_steps} tilt={tilt:.2f} "
            f"blocked={blocked:.4f} flux={total_flux:.3f} ({time.time() - t0:.1f}s)"
        )

    # Render all frames on a SHARED intensity scale: per-frame autoscaling
    # normalizes each spot to its own maximum and would exactly hide the
    # ~25 % flux loss from blocking.
    shared_vmax = max(float(f.max()) for f in fluxes)
    for step, (flux, rec) in enumerate(zip(fluxes, records)):
        tilt = rec["tilt_fraction"]
        render_frame(
            flux,
            FRAMES_POSE / f"frame_{step:03d}.png",
            [
                f"{HID} -> {TARGET_NAME} | pose sweep step {step + 1}/{pose_steps}",
                f"sample {sun_record['sample_id']}  az={sun_record['sun_azimuth_deg']:.1f} deg  "
                f"el={sun_record['sun_elevation_deg']:.1f} deg",
                f"tilt {tilt * 100:.0f} % | blocked {rec['blocked_fraction'] * 100:.2f} % | "
                f"flux {rec['total_flux']:.0f} (shared scale)",
            ],
            vmax=shared_vmax,
        )
    return {"blocker_rows": blocker_rows, "blocker_names": blocker_names,
            "records": records, "shared_vmax": shared_vmax, "fluxes": fluxes}


def bite_or_dimming_diagnostics(flux_blocked: torch.Tensor, flux_unblocked: torch.Tensor) -> dict:
    """Compare the most- and least-blocked frames of a pose sweep.

    Same protocol as the BH58 "bite or dimming" analysis: pixel-wise ratio
    inside the (unblocked) spot, top/bottom and left/right half means, and the
    centroid shift in pixels. A clean geometric "bite" would show a sharply
    bimodal ratio map (pixels either ~untouched or ~zeroed); pure dimming
    shows a smooth, unimodal ratio distribution close to a single global
    scale factor.
    """
    a = flux_blocked.numpy().astype(float)
    b = flux_unblocked.numpy().astype(float)
    thresh = 0.01 * b.max()
    mask = b > thresh
    ratio = np.full_like(b, np.nan)
    ratio[mask] = a[mask] / b[mask]
    h = a.shape[0]
    top_mask = mask.copy(); top_mask[h // 2 :, :] = False
    bot_mask = mask.copy(); bot_mask[: h // 2, :] = False

    def com(img: np.ndarray) -> tuple[float, float]:
        ys, xs = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
        s = img.sum()
        return (float((img * ys).sum() / s), float((img * xs).sum() / s))

    cu = com(b)
    cb = com(a) if a.sum() > 0 else (float("nan"), float("nan"))
    return {
        "ratio_mean": float(np.nanmean(ratio)),
        "ratio_p10": float(np.nanpercentile(ratio, 10)),
        "ratio_p90": float(np.nanpercentile(ratio, 90)),
        "ratio_std": float(np.nanstd(ratio)),
        "top_half_ratio_mean": float(np.nanmean(ratio[top_mask])) if top_mask.any() else None,
        "bottom_half_ratio_mean": float(np.nanmean(ratio[bot_mask])) if bot_mask.any() else None,
        "n_pixels_lit_unblocked": int(mask.sum()),
        "n_pixels_zeroed_by_blocking": int(((a <= thresh) & mask).sum()),
        "frac_pixels_zeroed": float(((a <= thresh) & mask).sum()) / max(1, int(mask.sum())),
        "centroid_unblocked_px": cu,
        "centroid_blocked_px": cb,
        "centroid_shift_px": float(np.hypot(cu[0] - cb[0], cu[1] - cb[1])) if a.sum() > 0 else None,
    }


def plot_flux_curve(records: list[dict], plot_path: pathlib.Path, caption: str) -> None:
    tilts = [r["tilt_fraction"] for r in records]
    blocked = [r["blocked_fraction"] for r in records]
    flux = [r["total_flux"] for r in records]
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(tilts, flux, "o-", color="darkred", label="total flux on target")
    ax1.set_xlabel("blocker tilt fraction (0 = vertical, 1 = horizontal/stow)")
    ax1.set_ylabel("total flux [a.u.]", color="darkred")
    ax1.tick_params(axis="y", labelcolor="darkred")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(tilts, [100 * b for b in blocked], "s--", color="navy", label="blocked fraction")
    ax2.set_ylabel("blocked ray fraction [%]", color="navy")
    ax2.tick_params(axis="y", labelcolor="navy")
    ax1.set_title(caption)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="limit sun-sweep frames (smoke test)")
    parser.add_argument("--pose-steps", type=int, default=POSE_STEPS)
    parser.add_argument("--pose-only", action="store_true",
                        help="reuse sun-sweep results from run_summary.json, redo pose sweep")
    parser.add_argument("--assemble-only", action="store_true",
                        help="only re-assemble gifs/grid plots from existing frames")
    parser.add_argument("--surface-points", type=int, default=25,
                        help="surface points per facet axis (N x N per facet)")
    parser.add_argument("--rays", type=int, default=RAYS_PER_SURFACE_POINT,
                        help="rays per surface point")
    parser.add_argument("--heliostat-id", default="AY36",
                        help="showcase heliostat (outputs go to blocking_validation/<id>/)")
    parser.add_argument("--sun-sample", default=None,
                        help="pose sweep: use this dataset sample id's sun position directly "
                             "(from train/AY36 — sun directions are heliostat-independent) "
                             "instead of the argmax over a sun sweep")
    parser.add_argument("--point-source-sun", action="store_true",
                        help="idealized geometric-optics limit: replace the scenario's sun "
                             "(physical sunshape std ~2.1 mrad) with a near-delta-function "
                             "source, to check whether blocking looks like a spatial bite "
                             "once sunshape blur is removed. Writes to "
                             "blocking_validation/<id>_point_source/ instead of <id>/.")
    parser.add_argument("--point-source-std-mrad", type=float, default=0.001,
                        help="std dev [mrad] of the point-source sun (default: 0.001 mrad, "
                             "~2000x narrower than the physical ~2.1 mrad sunshape)")
    args = parser.parse_args()
    out_name = f"{args.heliostat_id}_point_source" if args.point_source_sun else args.heliostat_id
    configure(args.heliostat_id, out_name=out_name)

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("artist", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    for d in (GIF_DIR, PLOT_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if args.assemble_only:
        if not args.pose_only:
            n1 = assemble_gif(FRAMES_SUN, GIF_DIR / "sun_sweep.gif")
            log.info(f"sun gif: {n1} frames")
        n2 = assemble_gif(FRAMES_POSE, GIF_DIR / "pose_sweep.gif")
        log.info(f"pose gif: {n2} frames ({FRAME_DURATION_S}s each)")
        return

    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cpu")
    point_source_covariance = (
        (args.point_source_std_mrad * 1e-3) ** 2 if args.point_source_sun else None
    )
    scenario, hg, hel_idx, target_index, aim_center, hel_dist_m = load_context(
        device, surface_points_per_facet=args.surface_points,
        rays_per_surface_point=args.rays,
        point_source_covariance=point_source_covariance,
    )
    samples = load_train_sun_positions()
    log.info(f"{len(samples)} train sun positions, az "
             f"{samples[0]['sun_azimuth_deg']:.1f} -> {samples[-1]['sun_azimuth_deg']:.1f} deg")

    # ---------------- animation 1 ----------------
    sun_records = None
    previous = None
    if args.sun_sample is not None:
        log.info(f"pose sweep uses sun of sample {args.sun_sample} directly")
    elif args.pose_only:
        if not SUMMARY_JSON.exists():
            raise FileNotFoundError("--pose-only needs an existing run_summary.json")
        previous = json.load(open(SUMMARY_JSON))
        sun_records = previous["sun_sweep"]["frames"]
        log.info(f"reusing {len(sun_records)} sun-sweep frames from {SUMMARY_JSON}")
    else:
        t0 = time.time()
        sun_records = run_sun_sweep(
            scenario, hg, hel_idx, target_index, aim_center, samples, device, args.max_frames
        )
        log.info(f"sun sweep done in {(time.time() - t0) / 60:.1f} min")

    # ---------------- animation 2 ----------------
    if args.sun_sample is not None:
        best = next(s for s in samples if s["sample_id"] == args.sun_sample)
        best = {**best, "blocked_fraction": None}
    else:
        best = max(sun_records, key=lambda r: r["blocked_fraction"])
    test_worst = None
    if TEST_FRACTIONS_JSON.exists():
        test_fracs = json.load(open(TEST_FRACTIONS_JSON))["blocked_fractions"]
        test_worst = float(max(test_fracs))
    log.info(
        f"[pose] chosen sun = train sample {best['sample_id']} "
        f"(blocked={best['blocked_fraction']}); test worst = {test_worst}"
    )
    t0 = time.time()
    pose = run_pose_sweep(
        scenario, hg, hel_idx, target_index, aim_center, best, device, args.pose_steps
    )
    log.info(f"pose sweep done in {(time.time() - t0) / 60:.1f} min")

    # ---------------- outputs ----------------
    n1 = None
    if sun_records is not None and not args.pose_only:
        n1 = assemble_gif(FRAMES_SUN, GIF_DIR / "sun_sweep.gif")
        make_grid(
            FRAMES_SUN, PLOT_DIR / "sun_sweep_grid.png",
            f"{HID} sun sweep (east -> west), aimed at {TARGET_NAME}, blocking ON",
        )
    elif previous is not None:
        n1 = previous["sun_sweep"]["n_frames"]
    n2 = assemble_gif(FRAMES_POSE, GIF_DIR / "pose_sweep.gif")
    make_grid(
        FRAMES_POSE, PLOT_DIR / "pose_sweep_grid.png",
        f"{HID} blocker pose sweep (vertical -> horizontal), sample {best['sample_id']}"
        " — shared intensity scale",
    )
    plot_flux_curve(
        pose["records"], PLOT_DIR / "pose_sweep_flux_curve.png",
        f"{HID} pose sweep — total flux and blocked fraction vs blocker tilt",
    )

    summary = {
        "config": {
            "heliostat_id": HID,
            "target_name": TARGET_NAME,
            "target_area_index": int(target_index),
            "scenario": str(SCENARIO_PATH),
            "sun_sample_source": str(DATASET_DIR / "train" / "AY36"),
            "rays_per_surface_point": args.rays,
            "surface_points_per_facet": args.surface_points,
            "random_seed": RANDOM_SEED,
            "frame_duration_s": FRAME_DURATION_S,
            "pose_steps": args.pose_steps,
            "pose_frames_shared_intensity_scale": True,
            "pose_shared_vmax": pose.get("shared_vmax"),
            "blocker_margin_m": BLOCKER_MARGIN_M,
            "hel_dist_to_tower_m": hel_dist_m,
            "blocking": "exact_blocking (brute force filter), blocking_active=True",
            "point_source_sun": args.point_source_sun,
            "point_source_std_mrad": args.point_source_std_mrad if args.point_source_sun else None,
            "physical_sunshape_std_mrad": None if not args.point_source_sun else round(
                float(np.sqrt(4.3681e-06) * 1000), 4
            ),
        },
        "pose_sweep": {
            "n_frames": n2,
            "chosen_sample": best,
            "test_worst_blocked_fraction_reference": test_worst,
            "n_blockers": len(pose["blocker_rows"]),
            "blocker_names": pose["blocker_names"],
            "frames": pose["records"],
            "blocked_fraction_min": min(r["blocked_fraction"] for r in pose["records"]),
            "blocked_fraction_max": max(r["blocked_fraction"] for r in pose["records"]),
        },
    }
    if sun_records is not None:
        summary["sun_sweep"] = {
            "n_frames": n1,
            "frames": sun_records,
            "blocked_fraction_min": min(r["blocked_fraction"] for r in sun_records),
            "blocked_fraction_max": max(r["blocked_fraction"] for r in sun_records),
        }

    # Bite-or-dimming diagnostic: most-blocked (vertical, step 0) vs
    # least-blocked (horizontal, last step) frame -- same protocol as the
    # blurred-sun BH58 analysis, so the two are directly comparable.
    diag = bite_or_dimming_diagnostics(pose["fluxes"][0], pose["fluxes"][-1])
    summary["bite_or_dimming_diagnostics"] = diag
    log.info(f"bite/dimming diagnostics: {diag}")

    SUMMARY_JSON.write_text(json.dumps(summary, indent=1))
    log.info(f"wrote {SUMMARY_JSON}")
    log.info(f"gifs: {GIF_DIR}  plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
