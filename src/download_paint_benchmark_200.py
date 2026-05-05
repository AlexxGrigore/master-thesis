#!/usr/bin/env python3
"""
Download a 100-train / 50-val / 50-test PAINT benchmark restricted to
heliostats that have BOTH deflectometry data AND at least 200 calibration
measurements.

From the full dataset (1,893 heliostats, median 139 measurements):
  - 155 heliostats have >= 200 measurements
  -  63 of those also have filled deflectometry h5 files locally

This script produces:
  datasets/paint/splits/benchmark_split-balanced_train-100_validation-50_deflectometry.csv
  datasets/paint/benchmark_split-balanced_train-100_validation-50_deflectometry/
    calibration_properties/{train,test,validation}/
    flux_image/{train,test,validation}/

Deflectometry + Properties for these heliostats are already present in
datasets/paint/heliostats/ (downloaded by the original benchmark script).
Any that are missing are re-downloaded.

Run from anywhere:
    python src/download_paint_benchmark_200.py

All steps are idempotent.
"""

import pathlib
import tempfile

import pandas as pd
import paint.util.paint_mappings as mappings
from paint.data import StacClient
from paint.data.dataset import PaintCalibrationDataset
from paint.data.dataset_splits import DatasetSplitter
from paint.util import set_logger_config

# ── configuration ─────────────────────────────────────────────────────────────

PAINT_DIR       = pathlib.Path(__file__).parent.parent / "datasets" / "paint"
HELIOSTATS_DIR  = PAINT_DIR / "heliostats"
METADATA_FILE   = PAINT_DIR / "metadata" / "calibration_metadata_all_heliostats.csv"

SPLIT_TYPE = mappings.BALANCED_SPLIT
TRAIN_SIZE = 100
VAL_SIZE   = 50
# test = remaining after train + val — with >=200 measurements the minimum is 50

BENCHMARK_NAME = (
    f"benchmark_split-{SPLIT_TYPE}_train-{TRAIN_SIZE}"
    f"_validation-{VAL_SIZE}_deflectometry"
)

ITEM_TYPES = [
    mappings.CALIBRATION_PROPERTIES_KEY,
    mappings.CALIBRATION_FLUX_IMAGE_KEY,
]

MIN_MEASUREMENTS = TRAIN_SIZE + VAL_SIZE + VAL_SIZE   # 200: ensures test >= 50

# ─────────────────────────────────────────────────────────────────────────────


def _heliostats_with_deflectometry() -> set[str]:
    """Return IDs of heliostats that have a locally-downloaded filled deflectometry file."""
    has_defl = set()
    if not HELIOSTATS_DIR.exists():
        return has_defl
    for hid_dir in HELIOSTATS_DIR.iterdir():
        if not hid_dir.is_dir():
            continue
        hid = hid_dir.name
        defl_dir = hid_dir / "Deflectometry"
        if not defl_dir.exists():
            continue
        if any(defl_dir.glob(f"{hid}-filled-*-deflectometry.h5")):
            has_defl.add(hid)
    return has_defl


def main() -> None:
    set_logger_config()

    # ── 0. verify metadata is present ─────────────────────────────────────────
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_FILE}\n"
            "Run download_paint_benchmark.py first to download the metadata."
        )
    print(f"✓ Metadata present: {METADATA_FILE}")

    # ── 1. identify qualifying heliostats ─────────────────────────────────────
    metadata = pd.read_csv(METADATA_FILE)
    counts   = metadata.groupby(mappings.HELIOSTAT_ID).size()

    hids_enough_data  = set(counts[counts >= MIN_MEASUREMENTS].index)
    hids_deflectometry = _heliostats_with_deflectometry()
    qualifying = sorted(hids_enough_data & hids_deflectometry)

    print(f"\nHeliostat selection:")
    print(f"  Total in metadata          : {len(counts)}")
    print(f"  With >= {MIN_MEASUREMENTS} measurements     : {len(hids_enough_data)}")
    print(f"  With deflectometry locally : {len(hids_deflectometry)}")
    print(f"  Qualifying (both)          : {len(qualifying)}")

    if not qualifying:
        raise RuntimeError(
            "No qualifying heliostats found. "
            "Ensure deflectometry data has been downloaded with "
            "download_paint_benchmark.py first."
        )

    # ── 2. create benchmark split from filtered metadata ──────────────────────
    splits_dir = PAINT_DIR / "splits"
    splits_csv = splits_dir / f"{BENCHMARK_NAME}.csv"

    if splits_csv.exists():
        print(f"\n✓ Splits already present: {splits_csv}")
        splits_df = pd.read_csv(splits_csv, index_col=mappings.SAVE_ID_INDEX)
    else:
        splits_dir.mkdir(parents=True, exist_ok=True)

        # Filter full metadata to qualifying heliostats only, then split.
        filtered_metadata = metadata[
            metadata[mappings.HELIOSTAT_ID].isin(qualifying)
        ].copy()
        print(f"\n  Filtered metadata rows: {len(filtered_metadata)}")

        # DatasetSplitter expects a CSV file path, so write a temp file.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, dir=splits_dir
        ) as tmp:
            filtered_metadata.to_csv(tmp, index=False)
            tmp_path = pathlib.Path(tmp.name)

        try:
            splitter = DatasetSplitter(
                input_file=tmp_path,
                output_dir=splits_dir,
                remove_unused_data=True,
            )
            splits_df = splitter.get_dataset_splits(
                split_type=SPLIT_TYPE,
                training_size=TRAIN_SIZE,
                validation_size=VAL_SIZE,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        # DatasetSplitter saves with the auto-generated name; rename to ours.
        auto_name = (
            f"benchmark_split-{SPLIT_TYPE}"
            f"_train-{TRAIN_SIZE}_validation-{VAL_SIZE}.csv"
        )
        auto_path = splits_dir / auto_name
        if auto_path.exists() and not splits_csv.exists():
            auto_path.rename(splits_csv)

        splits_df.to_csv(splits_csv)
        n_hels = splits_df[mappings.HELIOSTAT_ID].nunique()
        print(f"✓ Splits saved: {splits_csv}  ({n_hels} heliostats)")
        counts_by_split = splits_df.groupby(mappings.SPLIT_KEY).size()
        print(f"  {dict(counts_by_split)}")

    # ── 3. download calibration_properties + flux_image ───────────────────────
    for item_type in ITEM_TYPES:
        item_dir = PAINT_DIR / BENCHMARK_NAME / item_type
        if item_dir.exists():
            print(f"\n✓ Already downloaded: {item_type}")
            continue

        print(f"\nDownloading {item_type} ...")
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

    # ── 4. ensure Properties + Deflectometry are present ─────────────────────
    benchmark_heliostats = sorted(splits_df[mappings.HELIOSTAT_ID].unique().tolist())
    missing = [
        hid
        for hid in benchmark_heliostats
        if not (HELIOSTATS_DIR / hid / mappings.SAVE_PROPERTIES).exists()
    ]

    if not missing:
        print(
            f"\n✓ Properties + Deflectometry already present "
            f"for all {len(benchmark_heliostats)} benchmark heliostats."
        )
    else:
        print(f"\nDownloading Properties + Deflectometry for {len(missing)} heliostats...")
        client = StacClient(output_dir=HELIOSTATS_DIR)
        client.get_heliostat_data(
            heliostats=missing,
            collections=[
                mappings.SAVE_PROPERTIES.lower(),
                mappings.SAVE_DEFLECTOMETRY.lower(),
            ],
        )
        print(f"✓ Done → {HELIOSTATS_DIR}")

    print(f"\nDone. Benchmark: {BENCHMARK_NAME}")
    print(f"  Heliostats : {len(benchmark_heliostats)}")
    print(f"  Per heliostat: {TRAIN_SIZE} train / {VAL_SIZE} val / ~{VAL_SIZE} test")


if __name__ == "__main__":
    main()
