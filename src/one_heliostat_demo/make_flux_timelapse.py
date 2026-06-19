"""Generate per-heliostat flux-image timelapse GIFs from a synthetic dataset.

For each heliostat, collects all flux images (train + val + test splits),
sorts them from sunrise to sunset by azimuth angle, and saves an animated GIF.

Usage:
    python make_flux_timelapse.py                          # all heliostats, balanced_dataset
    python make_flux_timelapse.py --heliostat-ids AA23 BE35
    python make_flux_timelapse.py --dataset-dir ../../datasets/synthetic/balanced_dataset/dataset
    python make_flux_timelapse.py --frame-duration 50     # ms per frame (default 50)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_here = pathlib.Path(__file__).resolve().parent
_root = _here.parent.parent

DEFAULT_DATASET = _root / "datasets" / "synthetic" / "balanced_dataset" / "dataset"
DEFAULT_OUTPUT  = _root / "outputs" / "flux_timelapse"

SPLITS = ["train", "val", "test"]


def _ray_to_az_el(ray: list[float]) -> tuple[float, float]:
    """Convert incident_ray_direction [E, N, U, 0] → (azimuth_deg, elevation_deg).

    incident_ray_direction points FROM sun TO heliostat, so sun direction is -ray.
    """
    sun = [-ray[0], -ray[1], -ray[2]]
    az  = float(np.degrees(np.arctan2(sun[0], sun[1])) % 360.0)
    el  = float(np.degrees(np.arcsin(np.clip(sun[2], -1.0, 1.0))))
    return az, el


def _collect_samples(dataset_dir: pathlib.Path, hel_id: str) -> list[dict]:
    """Return all samples for a heliostat across all splits, sorted sunrise→sunset."""
    samples = []
    for split in SPLITS:
        hel_dir = dataset_dir / split / hel_id
        if not hel_dir.exists():
            continue
        for sample_dir in sorted(hel_dir.iterdir()):
            props_path = sample_dir / "calibration_properties.json"
            flux_path  = sample_dir / "flux_image.png"
            if not props_path.exists() or not flux_path.exists():
                continue
            with open(props_path) as f:
                props = json.load(f)
            az, el = _ray_to_az_el(props["incident_ray_direction"])
            samples.append({"flux_path": flux_path, "az": az, "el": el, "split": split})

    # Sort by azimuth (East=90° → South=180° → West=270°) to get sunrise→sunset
    samples.sort(key=lambda s: s["az"])
    return samples


def _add_label(img: Image.Image, az: float, el: float, split: str) -> Image.Image:
    """Overlay az/el text and split indicator onto a copy of the image."""
    img = img.convert("RGB")
    # Scale up for readability if small
    W, H = img.size
    scale = max(1, 150 // W)
    if scale > 1:
        img = img.resize((W * scale, H * scale), Image.NEAREST)
    W, H = img.size

    draw = ImageDraw.Draw(img)
    text = f"az={az:.0f}° el={el:.0f}°"
    split_colors = {"train": (100, 200, 100), "val": (255, 165, 0), "test": (100, 150, 255)}
    color = split_colors.get(split, (255, 255, 255))

    # Draw text with dark background for readability
    bbox = draw.textbbox((0, 0), text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([2, H - th - 6, 2 + tw + 4, H - 2], fill=(0, 0, 0, 180))
    draw.text((4, H - th - 4), text, fill=color)

    return img


def make_gif(
    dataset_dir: pathlib.Path,
    hel_id: str,
    output_dir: pathlib.Path,
    frame_duration_ms: int = 50,
) -> pathlib.Path | None:
    samples = _collect_samples(dataset_dir, hel_id)
    if not samples:
        print(f"  {hel_id}: no samples found, skipping")
        return None

    frames = []
    for s in samples:
        img = Image.open(s["flux_path"])
        img = _add_label(img, s["az"], s["el"], s["split"])
        frames.append(img)

    out_path = output_dir / f"{hel_id}_timelapse.gif"
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Generate flux timelapse GIFs per heliostat.")
    p.add_argument("--dataset-dir", type=pathlib.Path, default=DEFAULT_DATASET)
    p.add_argument("--output-dir",  type=pathlib.Path, default=DEFAULT_OUTPUT)
    p.add_argument("--heliostat-ids", nargs="+", default=None, metavar="ID")
    p.add_argument("--frame-duration", type=int, default=50,
                   help="Duration per frame in milliseconds (default: 50)")
    args = p.parse_args()

    dataset_dir: pathlib.Path = args.dataset_dir
    output_dir:  pathlib.Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover heliostat IDs from train split
    if args.heliostat_ids:
        hel_ids = args.heliostat_ids
    else:
        hel_ids = sorted(
            d.name for d in (dataset_dir / "train").iterdir() if d.is_dir()
        )

    print(f"Dataset : {dataset_dir}")
    print(f"Output  : {output_dir}")
    print(f"Heliostats: {len(hel_ids)}  |  frame duration: {args.frame_duration} ms")
    print()

    for i, hel_id in enumerate(hel_ids, 1):
        print(f"[{i}/{len(hel_ids)}] {hel_id} ... ", end="", flush=True)
        out = make_gif(dataset_dir, hel_id, output_dir, args.frame_duration)
        if out:
            print(f"saved ({out.name})")
        else:
            print("skipped")

    print(f"\nDone. GIFs saved to {output_dir}")


if __name__ == "__main__":
    main()
