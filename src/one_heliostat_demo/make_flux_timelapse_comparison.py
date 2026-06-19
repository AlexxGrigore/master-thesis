"""Generate per-heliostat flux timelapse GIFs comparing:
  grid1_real_paint.gif        — real PAINT measured flux images
  grid2_ideal_kinematics.gif  — ARTIST simulation, ideal (unperturbed) kinematics
  grid3_perturbed_kinematics.gif — ARTIST simulation, GT-perturbed kinematics

All three use the same sun positions (all PAINT samples pooled across train/val/test),
sorted sunrise→sunset by azimuth.

Usage:
    python make_flux_timelapse_comparison.py
    python make_flux_timelapse_comparison.py --heliostat-ids AA23 BE35
    python make_flux_timelapse_comparison.py --frame-duration 50
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_here   = pathlib.Path(__file__).resolve().parent
_root   = _here.parent.parent
_src    = _here.parent
_paint  = _root / "PAINT"
_artist = _root / "ARTIST"
for _p in [str(_here), str(_src), str(_paint), str(_artist)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from single_heliostat import config as cfg                                    # noqa: E402
from utils.evaluation import build_heliostat_data_mapping                     # noqa: E402
from utils.synth_data import (apply_perturbations, reset_perturbations,       # noqa: E402
                               _forward_pass)
from artist.io.paint_calibration_parser import PaintCalibrationDataParser     # noqa: E402
from artist.scenario.scenario import Scenario                                 # noqa: E402

DEFAULT_PERTURBATIONS = (_root / "datasets" / "synthetic" /
                         "balanced_dataset" / "dataset" / "perturbations.json")
DEFAULT_OUTPUT  = _root / "outputs" / "flux_timelapse_comparison"
SCENARIOS_DIR   = _root / "scenarios" / "one_heliostat_scenarios"
SPLITS          = ["train", "validation", "test"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_uint8(flux: torch.Tensor) -> np.ndarray:
    arr = flux.cpu().float().numpy()
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    return (arr * 255).astype(np.uint8)


def _label_frame(img: Image.Image, az: float, el: float,
                 color=(100, 220, 100)) -> Image.Image:
    img = img.convert("RGB")
    W, H = img.size
    scale = max(1, 150 // max(W, 1))
    if scale > 1:
        img = img.resize((W * scale, H * scale), Image.NEAREST)
    W, H = img.size
    draw = ImageDraw.Draw(img)
    text = f"az={az:.0f}° el={el:.0f}°"
    bbox = draw.textbbox((0, 0), text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([2, H - th - 6, tw + 6, H - 2], fill=(0, 0, 0))
    draw.text((4, H - th - 4), text, fill=color)
    return img


def _save_gif(frames: list[Image.Image], path: pathlib.Path,
              duration_ms: int) -> None:
    if not frames:
        return
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)


def _save_grid_png(frames: list[Image.Image], path: pathlib.Path,
                   n_cols: int = 20) -> None:
    """Arrange all frames in a grid (left-to-right, top-to-bottom) and save as PNG."""
    if not frames:
        return
    n_cols = min(n_cols, len(frames))
    n_rows = (len(frames) + n_cols - 1) // n_cols
    fw, fh = frames[0].size
    gap = 2
    grid = Image.new("RGB",
                     (n_cols * (fw + gap) - gap, n_rows * (fh + gap) - gap),
                     color=(30, 30, 30))
    for idx, frame in enumerate(frames):
        row, col = divmod(idx, n_cols)
        grid.paste(frame.convert("RGB"), (col * (fw + gap), row * (fh + gap)))
    grid.save(path)


def _load_scenario(hel_id: str, device: torch.device):
    scenario_path = SCENARIOS_DIR / hel_id / "scenario.h5"
    with h5py.File(scenario_path, "r") as f:
        scenario = Scenario.load_scenario_from_hdf5(scenario_file=f, device=device)
    hg = scenario.heliostat_field.heliostat_groups[0]
    return scenario, hg


def _perturbation_to_tensors(pert_json: dict) -> dict:
    return {
        "rotation":        torch.tensor(pert_json["rotation_rad"]).unsqueeze(0),
        "actuator_angle":  torch.tensor(pert_json["actuator_angle_rad"]).unsqueeze(0),
        "actuator_stroke": torch.tensor(pert_json["actuator_stroke_m"]).unsqueeze(0),
        "actuator_offset": torch.tensor(pert_json["actuator_offset_m"]).unsqueeze(0),
        "translation":     torch.tensor(pert_json["translation_m"]).unsqueeze(0),
        "base_position":   torch.tensor(pert_json["base_position_m"]).unsqueeze(0),
    }


# ---------------------------------------------------------------------------
# PAINT data loading
# ---------------------------------------------------------------------------

def _load_paint_samples(hel_id: str, scenario, hg, device: torch.device,
                        benchmark_csv: pathlib.Path | None = None):
    """Pool all PAINT splits for one heliostat, sort sunrise→sunset.

    Returns:
        rays        [N, 4]
        active_mask [1]   (value = N)
        target_mask [N]
        base_pos    [1, 3] zeros
        az_list     list[float]
        el_list     list[float]
        flux_paths  list[Path]
    """
    parser = PaintCalibrationDataParser(
        centroid_extraction_method=getattr(cfg, "CENTROID_METHOD", "UTIS"),
    )

    all_rays, all_targets, all_flux_paths = [], [], []
    all_az, all_el = [], []

    csv = benchmark_csv or pathlib.Path(cfg.BENCHMARK_CSV)
    benchmark_name = csv.stem
    cal_dir  = pathlib.Path(cfg.PAINT_DIR) / benchmark_name / "calibration_properties"
    flux_dir = pathlib.Path(cfg.PAINT_DIR) / benchmark_name / "flux_image"
    if not cal_dir.exists():
        cal_dir  = pathlib.Path(cfg.CALIBRATION_DIR)
        flux_dir = pathlib.Path(cfg.REAL_FLUX_DIR)

    for paint_split in SPLITS:
        mapping = build_heliostat_data_mapping(
            csv,
            cal_dir,
            flux_dir,
            paint_split,
        )
        hel_mapping = [(h, c, f) for h, c, f in mapping if h == hel_id]
        if not hel_mapping:
            continue

        _, _, rays, _, _, target_mask = parser.parse_data_for_reconstruction(
            heliostat_data_mapping=hel_mapping,
            heliostat_group=hg,
            scenario=scenario,
            device=device,
        )

        cal_paths  = hel_mapping[0][1]
        flux_paths = hel_mapping[0][2]

        for i, (cp, fp) in enumerate(zip(cal_paths, flux_paths)):
            with open(cp) as f:
                props = json.load(f)
            all_az.append(float(props.get("sun_azimuth", 0.0)))
            all_el.append(float(props.get("sun_elevation", 0.0)))
            all_rays.append(rays[i])
            all_targets.append(target_mask[i])
            all_flux_paths.append(pathlib.Path(fp))

    if not all_rays:
        return None, None, None, None, None, None, None

    order = sorted(range(len(all_az)), key=lambda i: all_az[i])
    N = len(order)
    return (
        torch.stack([all_rays[i]    for i in order]).to(device),   # [N,4]
        torch.tensor([N], dtype=torch.long, device=device),        # active_mask
        torch.stack([all_targets[i] for i in order]).to(device),   # [N]
        torch.zeros(1, 3, device=device),                          # base_pos_delta
        [all_az[i]         for i in order],
        [all_el[i]         for i in order],
        [all_flux_paths[i] for i in order],
    )


# ---------------------------------------------------------------------------
# Per-heliostat pipeline
# ---------------------------------------------------------------------------

def process_heliostat(
    hel_id: str,
    perturbations_json: dict,
    output_dir: pathlib.Path,
    device: torch.device,
    frame_duration_ms: int,
    benchmark_csv: pathlib.Path | None = None,
) -> None:
    hel_out = output_dir / hel_id
    hel_out.mkdir(parents=True, exist_ok=True)

    # Load scenario once — reused for PAINT parsing and forward passes
    scenario, hg = _load_scenario(hel_id, device)

    rays, active_mask, target_mask, base_pos, az_list, el_list, flux_paths = \
        _load_paint_samples(hel_id, scenario, hg, device, benchmark_csv=benchmark_csv)

    if rays is None:
        print(f"  {hel_id}: no PAINT samples, skipping")
        return
    print(f"  {hel_id}: {rays.shape[0]} samples", flush=True)

    # ── Grid 1: real PAINT images ──────────────────────────────────────────
    frames_real = []
    for fp, az, el in zip(flux_paths, az_list, el_list):
        img = Image.open(fp).convert("L")
        frames_real.append(_label_frame(img, az, el, color=(100, 220, 100)))
    _save_gif(frames_real, hel_out / "grid1_real_paint.gif", frame_duration_ms)
    _save_grid_png(frames_real, hel_out / "grid1_real_paint.png")

    # ── Grid 2: ideal kinematics ───────────────────────────────────────────
    _, flux_ideal = _forward_pass(scenario, hg, rays, active_mask, target_mask,
                                  base_pos, device)
    frames_ideal = []
    for i, (az, el) in enumerate(zip(az_list, el_list)):
        img = Image.fromarray(_to_uint8(flux_ideal[i]), mode="L")
        frames_ideal.append(_label_frame(img, az, el, color=(100, 180, 255)))
    _save_gif(frames_ideal, hel_out / "grid2_ideal_kinematics.gif", frame_duration_ms)
    _save_grid_png(frames_ideal, hel_out / "grid2_ideal_kinematics.png")

    # ── Grid 3: perturbed kinematics ───────────────────────────────────────
    if hel_id not in perturbations_json:
        print(f"  {hel_id}: no perturbation entry, skipping Grid 3")
        return

    # Reload fresh scenario so Grid 2 state doesn't bleed into Grid 3
    scenario3, hg3 = _load_scenario(hel_id, device)
    pert    = _perturbation_to_tensors(perturbations_json[hel_id])
    snap    = apply_perturbations(hg3.kinematics, pert, device)
    _, flux_pert = _forward_pass(scenario3, hg3, rays, active_mask, target_mask,
                                 base_pos, device)
    reset_perturbations(hg3.kinematics, snap)

    frames_pert = []
    for i, (az, el) in enumerate(zip(az_list, el_list)):
        img = Image.fromarray(_to_uint8(flux_pert[i]), mode="L")
        frames_pert.append(_label_frame(img, az, el, color=(255, 160, 80)))
    _save_gif(frames_pert, hel_out / "grid3_perturbed_kinematics.gif", frame_duration_ms)
    _save_grid_png(frames_pert, hel_out / "grid3_perturbed_kinematics.png")

    print(f"  {hel_id}: saved 3 GIFs + 3 PNGs → {hel_out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--heliostat-ids",   nargs="+", default=None)
    p.add_argument("--perturbations",   type=pathlib.Path, default=DEFAULT_PERTURBATIONS)
    p.add_argument("--output-dir",      type=pathlib.Path, default=DEFAULT_OUTPUT)
    p.add_argument("--frame-duration",  type=int, default=50,
                   help="GIF frame duration in ms (default 50)")
    p.add_argument("--benchmark-csv",   type=pathlib.Path, default=None,
                   help="Override benchmark CSV (default: cfg.BENCHMARK_CSV)")
    args = p.parse_args()

    device = torch.device("cpu")

    with open(args.perturbations) as f:
        perturbations_json = json.load(f)

    hel_ids = args.heliostat_ids or sorted(perturbations_json.keys())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output    : {args.output_dir}")
    print(f"Heliostats: {len(hel_ids)}  |  frame duration: {args.frame_duration} ms\n")

    for i, hel_id in enumerate(hel_ids, 1):
        print(f"[{i}/{len(hel_ids)}] {hel_id}")
        try:
            process_heliostat(hel_id, perturbations_json, args.output_dir,
                              device, args.frame_duration, args.benchmark_csv)
        except Exception as e:
            import traceback
            print(f"  {hel_id}: ERROR — {e}")
            traceback.print_exc()

    print(f"\nDone. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
