#!/usr/bin/env python3
"""
Download the PAINT calibration benchmark dataset.

All data is stored under a single directory (PAINT_DIR):

  PAINT_DIR/
  ├── metadata/                                             <- heliostat metadata CSV
  ├── splits/                                               <- benchmark split CSV
  ├── benchmark_split-<type>_train-<n>_validation-<m>/     <- organized benchmark
  │   ├── calibration_properties/
  │   │   ├── train/
  │   │   ├── test/
  │   │   └── validation/
  │   └── flux_image/
  │       ├── train/
  │       ├── test/
  │       └── validation/
  └── heliostats/                                           <- raw per-heliostat data
      ├── WRI1030197-tower-measurements.json
      ├── AA23/
      │   ├── Properties/
      │   └── Deflectometry/
      └── ...

Run from anywhere:
    python src/download_paint_benchmark.py

All steps are idempotent — already-downloaded data is skipped.
"""

import pathlib

import pandas as pd
import paint.util.paint_mappings as mappings
from paint.data import StacClient
from paint.data.dataset import PaintCalibrationDataset
from paint.data.dataset_splits import DatasetSplitter
from paint.util import set_logger_config

# ── configuration ─────────────────────────────────────────────────────────────

PAINT_DIR = pathlib.Path(__file__).parent.parent / "datasets" / "paint"

SPLIT_TYPE = mappings.BALANCED_SPLIT
TRAIN_SIZE = 10
VAL_SIZE = 30

ITEM_TYPES = [
    mappings.CALIBRATION_PROPERTIES_KEY,
    mappings.CALIBRATION_FLUX_IMAGE_KEY,
]

TOWER_FILE = "WRI1030197-tower-measurements.json"

# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    set_logger_config()

    # ── 1. download calibration metadata ──────────────────────────────────────
    metadata_file = PAINT_DIR / "metadata" / "calibration_metadata_all_heliostats.csv"
    if not metadata_file.exists():
        print("Downloading calibration metadata...")
        client = StacClient(output_dir=PAINT_DIR)
        client.get_heliostat_metadata(collections=[mappings.SAVE_CALIBRATION.lower()])
        print(f"✓ Metadata saved to {metadata_file}")
    else:
        print(f"✓ Metadata already present: {metadata_file}")

    # ── 2. create benchmark splits ────────────────────────────────────────────
    splits_dir = PAINT_DIR / "splits"
    splits_csv = (
        splits_dir
        / f"benchmark_split-{SPLIT_TYPE}_train-{TRAIN_SIZE}_validation-{VAL_SIZE}.csv"
    )

    if splits_csv.exists():
        print(f"✓ Splits already present: {splits_csv}")
        splits_df = pd.read_csv(splits_csv, index_col=mappings.SAVE_ID_INDEX)
    else:
        splits_dir.mkdir(parents=True, exist_ok=True)
        splitter = DatasetSplitter(
            input_file=metadata_file,
            output_dir=splits_dir,
            remove_unused_data=True,
        )
        splits_df = splitter.get_dataset_splits(
            split_type=SPLIT_TYPE,
            training_size=TRAIN_SIZE,
            validation_size=VAL_SIZE,
        )
        print(f"✓ Splits saved to {splits_csv}")

    # ── 3. download benchmark items per item type ──────────────────────────────
    benchmark_name = (
        f"benchmark_split-{SPLIT_TYPE}_train-{TRAIN_SIZE}_validation-{VAL_SIZE}"
    )

    for item_type in ITEM_TYPES:
        item_dir = PAINT_DIR / benchmark_name / item_type
        if item_dir.exists():
            print(f"✓ Already downloaded: {item_type} → {item_dir}")
            continue

        print(f"Downloading {item_type}...")
        # Pass a copy because from_benchmark calls reset_index(inplace=True)
        train_ds, test_ds, val_ds = PaintCalibrationDataset.from_benchmark(
            benchmark_file=splits_df.copy(),
            root_dir=item_dir,
            item_type=item_type,
            download=True,
        )
        print(
            f"✓ {item_type}: "
            f"train={len(train_ds)}, test={len(test_ds)}, val={len(val_ds)}"
        )

    # ── 4. download tower measurements ────────────────────────────────────────
    heliostats_dir = PAINT_DIR / "heliostats"
    tower_file = heliostats_dir / TOWER_FILE

    if tower_file.exists():
        print(f"✓ Tower measurements already present: {tower_file}")
    else:
        heliostats_dir.mkdir(parents=True, exist_ok=True)
        print("Downloading tower measurements...")
        client = StacClient(output_dir=heliostats_dir)
        client.get_tower_measurements()
        print(f"✓ Tower measurements saved to {tower_file}")

    # ── 5. download heliostat Properties + Deflectometry ──────────────────────
    benchmark_heliostats = sorted(splits_df[mappings.HELIOSTAT_ID].unique().tolist())

    missing = [
        name
        for name in benchmark_heliostats
        if not (heliostats_dir / name / mappings.SAVE_PROPERTIES).exists()
    ]

    if not missing:
        print(
            f"✓ Properties + Deflectometry already present "
            f"for all {len(benchmark_heliostats)} benchmark heliostats"
        )
    else:
        print(
            f"Downloading Properties + Deflectometry "
            f"for {len(missing)}/{len(benchmark_heliostats)} heliostats..."
        )
        client = StacClient(output_dir=heliostats_dir)
        client.get_heliostat_data(
            heliostats=missing,
            collections=[
                mappings.SAVE_PROPERTIES.lower(),
                mappings.SAVE_DEFLECTOMETRY.lower(),
            ],
        )
        print(f"✓ Properties + Deflectometry downloaded to {heliostats_dir}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
