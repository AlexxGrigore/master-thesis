"""Per-target flux timelapses: real PAINT vs. ideal (unperturbed) kinematics.

For each heliostat the PAINT samples are pooled across all splits, split by
aim target (the calibration data uses 3 different targets:
``solar_tower_juelich_lower``, ``solar_tower_juelich_upper``,
``multi_focus_tower``), and sorted sunrise->sunset by azimuth.

For every heliostat x target we produce:
  {target}_real_vs_ideal.gif   — side-by-side animation, real (left) | ideal (right)
  {target}_contact_sheet.png    — static grid, real row over ideal row

Only two sources are shown (no perturbed kinematics):
  * real   — the measured PAINT flux image
  * ideal  — ARTIST ray tracing with the nominal (unperturbed) kinematics,
             aimed at this target's centre.

Usage:
    python make_flux_timelapse_per_target.py
    python make_flux_timelapse_per_target.py --heliostat-ids AA39 BE35 AC39 AA24
    python make_flux_timelapse_per_target.py --fps 3
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

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
from utils.synth_data import _forward_pass                                    # noqa: E402
from artist.io.paint_calibration_parser import PaintCalibrationDataParser     # noqa: E402
# Reuse frame/gif helpers from the sibling comparison script.
from make_flux_timelapse_comparison import (                                  # noqa: E402
    _to_uint8, _label_frame, _save_gif, _load_scenario,
)

# All 4 heliostats requested live in the train-10 benchmark (the 100/50 one has
# no AA39). Override with --benchmark-csv if needed.
DEFAULT_BENCHMARK = (_root / "datasets" / "paint" / "splits" /
                     "benchmark_split-balanced_train-10_validation-30.csv")
DEFAULT_HELIOSTATS = ["AA39", "BE35", "AC39", "AA24"]
DEFAULT_OUTPUT     = _root / "outputs" / "flux_timelapse_per_target"
SCENARIOS_DIR      = _root / "scenarios" / "one_heliostat_scenarios"
SPLITS             = ["train", "validation", "test"]

# Pretty short labels for the target names.
TARGET_SHORT = {
    "solar_tower_juelich_lower": "lower",
    "solar_tower_juelich_upper": "upper",
    "multi_focus_tower":         "multi_focus",
}


# ---------------------------------------------------------------------------
# PAINT loading (with per-sample target name)
# ---------------------------------------------------------------------------

def _load_paint_samples(hel_id, scenario, hg, device, benchmark_csv):
    """Pool all PAINT splits for one heliostat.

    Returns lists (all in a single natural order, unsorted):
        rays        [N, 4] tensor
        target_mask [N]    tensor (target-area index per sample)
        az_list, el_list, target_names, flux_paths
    """
    parser = PaintCalibrationDataParser(
        centroid_extraction_method=getattr(cfg, "CENTROID_METHOD", "UTIS"),
    )

    csv = pathlib.Path(benchmark_csv)
    benchmark_name = csv.stem
    cal_dir  = _root / "datasets" / "paint" / benchmark_name / "calibration_properties"
    flux_dir = _root / "datasets" / "paint" / benchmark_name / "flux_image"

    all_rays, all_targets = [], []
    all_az, all_el, all_tname, all_flux = [], [], [], []

    for split in SPLITS:
        mapping = build_heliostat_data_mapping(csv, cal_dir, flux_dir, split)
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
            all_tname.append(props.get("target_name", "unknown"))
            all_rays.append(rays[i])
            all_targets.append(target_mask[i])
            all_flux.append(pathlib.Path(fp))

    if not all_rays:
        return None
    return {
        "rays":     torch.stack(all_rays).to(device),
        "target":   torch.stack(all_targets).to(device),
        "az":       all_az,
        "el":       all_el,
        "tname":    all_tname,
        "flux":     all_flux,
    }


# ---------------------------------------------------------------------------
# Frame composition
# ---------------------------------------------------------------------------

def _panel(img_uint8_or_img, az, el, color):
    """Turn a source image into a labelled RGB panel."""
    if isinstance(img_uint8_or_img, Image.Image):
        img = img_uint8_or_img.convert("L")
    else:
        img = Image.fromarray(img_uint8_or_img, mode="L")
    return _label_frame(img, az, el, color=color)


def _compose_side_by_side(real_panel, ideal_panel, header):
    """Stack REAL | IDEAL horizontally with a header bar and column titles."""
    gap = 8
    bar = 22
    W = real_panel.width + gap + ideal_panel.width
    H = bar + max(real_panel.height, ideal_panel.height)
    canvas = Image.new("RGB", (W, H), (18, 18, 18))
    canvas.paste(real_panel,  (0, bar))
    canvas.paste(ideal_panel, (real_panel.width + gap, bar))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), header, fill=(230, 230, 230))
    draw.text((6, bar + 2), "REAL", fill=(120, 235, 120))
    draw.text((real_panel.width + gap + 6, bar + 2), "IDEAL", fill=(120, 190, 255))
    return canvas


def _contact_sheet(real_panels, ideal_panels, header, n_cols=None):
    """Two rows (real over ideal) laid out left->right, plus a header bar."""
    n = len(real_panels)
    n_cols = n_cols or n
    fw, fh = real_panels[0].size
    gap, bar, rowgap = 2, 22, 16
    W = n_cols * (fw + gap) - gap
    H = bar + fh + rowgap + fh
    canvas = Image.new("RGB", (W, H), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), header, fill=(230, 230, 230))
    draw.text((4, bar - 1), "REAL", fill=(120, 235, 120))
    y_real = bar
    y_ideal = bar + fh + rowgap
    for i, (rp, ip) in enumerate(zip(real_panels, ideal_panels)):
        x = i * (fw + gap)
        canvas.paste(rp, (x, y_real))
        canvas.paste(ip, (x, y_ideal))
    draw.text((4, y_ideal - 1), "IDEAL", fill=(120, 190, 255))
    return canvas


# ---------------------------------------------------------------------------
# Per-heliostat pipeline
# ---------------------------------------------------------------------------

def process_heliostat(hel_id, output_dir, device, fps, benchmark_csv):
    hel_out = output_dir / hel_id
    hel_out.mkdir(parents=True, exist_ok=True)

    scenario, hg = _load_scenario(hel_id, device)
    data = _load_paint_samples(hel_id, scenario, hg, device, benchmark_csv)
    if data is None:
        print(f"  {hel_id}: no PAINT samples, skipping")
        return

    n = data["rays"].shape[0]
    print(f"  {hel_id}: {n} samples total", flush=True)

    # One ideal-kinematics forward pass for ALL samples of this heliostat.
    active_mask = torch.tensor([n], dtype=torch.long, device=device)
    base_pos    = torch.zeros(1, 3, device=device)
    _, flux_ideal = _forward_pass(
        scenario, hg, data["rays"], active_mask, data["target"], base_pos, device
    )

    # Group sample indices by target name.
    by_target = defaultdict(list)
    for i, t in enumerate(data["tname"]):
        by_target[t].append(i)

    frame_ms = int(round(1000.0 / max(fps, 0.1)))

    for tname, idxs in sorted(by_target.items()):
        # Sort this target's samples sunrise->sunset by azimuth.
        idxs = sorted(idxs, key=lambda i: data["az"][i])
        short = TARGET_SHORT.get(tname, tname)

        real_panels, ideal_panels, sbs_frames = [], [], []
        for i in idxs:
            az, el = data["az"][i], data["el"][i]
            rp = _panel(Image.open(data["flux"][i]), az, el, color=(120, 235, 120))
            ip = _panel(_to_uint8(flux_ideal[i]),     az, el, color=(120, 190, 255))
            real_panels.append(rp)
            ideal_panels.append(ip)
            header = f"{hel_id}  |  {short}  |  n={len(idxs)}"
            sbs_frames.append(_compose_side_by_side(rp, ip, header))

        gif_path = hel_out / f"{short}_real_vs_ideal.gif"
        png_path = hel_out / f"{short}_contact_sheet.png"
        _save_gif(sbs_frames, gif_path, frame_ms)
        _contact_sheet(real_panels, ideal_panels,
                       f"{hel_id}  |  target: {short}  |  {len(idxs)} samples "
                       f"(sunrise->sunset)").save(png_path)
        print(f"    {short:12s} {len(idxs):3d} frames -> {gif_path.name}, {png_path.name}")

    print(f"  {hel_id}: done -> {hel_out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heliostat-ids", nargs="+", default=DEFAULT_HELIOSTATS)
    p.add_argument("--benchmark-csv", type=pathlib.Path, default=DEFAULT_BENCHMARK)
    p.add_argument("--output-dir",    type=pathlib.Path, default=DEFAULT_OUTPUT)
    p.add_argument("--fps",           type=float, default=3.0,
                   help="GIF playback speed in frames per second (default 3)")
    args = p.parse_args()

    device = torch.device("cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Benchmark : {args.benchmark_csv.name}")
    print(f"Output    : {args.output_dir}")
    print(f"Heliostats: {args.heliostat_ids}  |  {args.fps} fps\n")

    for i, hel_id in enumerate(args.heliostat_ids, 1):
        print(f"[{i}/{len(args.heliostat_ids)}] {hel_id}")
        try:
            process_heliostat(hel_id, args.output_dir, device, args.fps,
                              args.benchmark_csv)
        except Exception as e:
            import traceback
            print(f"  {hel_id}: ERROR — {e}")
            traceback.print_exc()

    print(f"\nDone. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
