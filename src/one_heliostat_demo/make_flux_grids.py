"""
Build a grid image from all flux samples for every heliostat × split.

For each  datasets/synthetic/dataset/{split}/{hid}/  folder:
  - Reads every {idx:04d}/flux_image.png in sorted order
  - Arranges them in a grid (COLS_PER_ROW columns)
  - Overlays the sample index and active-pixel % on each cell
  - Saves the result as  {split}/{hid}/grid.png

Usage
-----
    python make_flux_grids.py
    python make_flux_grids.py --dataset-dir /path/to/dataset
    python make_flux_grids.py --cols 8
"""

import argparse
import math
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


COLS_PER_ROW = 10
CELL_SIZE    = 256   # pixels — matches flux_image.png native size
LABEL_HEIGHT = 18    # extra pixels below each cell for the text label
BORDER       = 1     # 1-pixel border between cells


def _active_pct(arr: np.ndarray) -> float:
    return float((arr > 0).sum()) / arr.size * 100.0


def _load_font(size: int = 11):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def make_grid(hel_dir: pathlib.Path, cols: int = COLS_PER_ROW) -> pathlib.Path:
    """
    Build and save grid.png for one heliostat folder.
    Returns the path to the saved grid.
    """
    sample_dirs = sorted(
        d for d in hel_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    if not sample_dirs:
        raise ValueError(f"No sample folders found in {hel_dir}")

    n      = len(sample_dirs)
    rows   = math.ceil(n / cols)
    cell_h = CELL_SIZE + LABEL_HEIGHT
    cell_w = CELL_SIZE

    grid_w = cols * (cell_w + BORDER) + BORDER
    grid_h = rows * (cell_h + BORDER) + BORDER

    grid = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(grid)
    font = _load_font(11)

    for i, sample_dir in enumerate(sample_dirs):
        flux_path = sample_dir / "flux_image.png"
        if not flux_path.exists():
            continue

        img = Image.open(flux_path).convert("RGB").resize(
            (cell_w, CELL_SIZE), Image.NEAREST
        )
        arr = np.array(Image.open(flux_path))
        pct = _active_pct(arr)

        col = i % cols
        row = i // cols
        x   = BORDER + col * (cell_w + BORDER)
        y   = BORDER + row * (cell_h + BORDER)

        grid.paste(img, (x, y))

        label = f"{sample_dir.name}  {pct:.0f}%"
        draw.rectangle([x, y + CELL_SIZE, x + cell_w - 1, y + cell_h - 1], fill=(20, 20, 20))
        draw.text((x + 3, y + CELL_SIZE + 2), label, fill=(200, 200, 200), font=font)

    out_path = hel_dir / "grid.png"
    grid.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate flux grid images for all heliostats.")
    parser.add_argument(
        "--dataset-dir", type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[2] / "datasets" / "synthetic" / "dataset",
        help="Root dataset directory containing train/val/test splits.",
    )
    parser.add_argument(
        "--cols", type=int, default=COLS_PER_ROW,
        help=f"Number of columns per row in the grid (default: {COLS_PER_ROW}).",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        help="Splits to process (default: train val test).",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Collect all (split, heliostat_dir) pairs.
    tasks: list[tuple[str, pathlib.Path]] = []
    for split in args.splits:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            print(f"  Skipping {split}: {split_dir} not found")
            continue
        for hel_dir in sorted(split_dir.iterdir()):
            if hel_dir.is_dir():
                tasks.append((split, hel_dir))

    if not tasks:
        print("No heliostat folders found.")
        return

    print(f"Building grids for {len(tasks)} heliostat×split folders in {dataset_dir}")

    ok = failed = 0
    for split, hel_dir in tqdm(tasks, unit="folder", dynamic_ncols=True):
        try:
            out = make_grid(hel_dir, cols=args.cols)
            ok += 1
        except Exception as exc:
            tqdm.write(f"  FAILED {split}/{hel_dir.name}: {exc}")
            failed += 1

    print(f"\nDone — {ok} grids saved, {failed} failed.")


if __name__ == "__main__":
    main()
