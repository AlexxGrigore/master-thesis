#!/usr/bin/env python3
"""
Download a PAINT benchmark restricted to heliostats that have BOTH deflectometry
data AND at least 200 calibration measurements.

From the full dataset (1,893 heliostats, median 139 measurements):
  - 155 heliostats have >= 200 measurements
  -  63 of those also have filled deflectometry h5 files locally

Usage
-----
    python src/download_paint_benchmark_200.py
    python src/download_paint_benchmark_200.py --split-type azimuth
    python src/download_paint_benchmark_200.py --split-type balanced --train-size 100 --val-size 50

Available split types: balanced, azimuth, solstice, high_variance

All steps are idempotent.
"""

import argparse
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

_SPLIT_CHOICES = {
    "balanced":      mappings.BALANCED_SPLIT,
    "azimuth":       mappings.AZIMUTH_SPLIT,
    "solstice":      mappings.SOLSTICE_SPLIT,
    "high_variance": mappings.HIGH_VARIANCE_SPLIT,
}

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download a PAINT benchmark with deflectometry heliostats (>=200 measurements)."
    )
    p.add_argument(
        "--split-type",
        choices=list(_SPLIT_CHOICES),
        default="balanced",
        help="How to split calibration images into train/val/test (default: balanced).",
    )
    p.add_argument(
        "--train-size", type=int, default=100,
        help="Number of training samples per heliostat (default: 100).",
    )
    p.add_argument(
        "--val-size", type=int, default=50,
        help="Number of validation samples per heliostat (default: 50).",
    )
    p.add_argument(
        "--daic", action="store_true",
        help="Use DAIC storage path (/tudelft.net/...) instead of local datasets/paint/.",
    )
    p.add_argument(
        "--no-deflectometry", action="store_true",
        help="Include ALL heliostats with enough measurements, regardless of "
             "deflectometry availability (default: require deflectometry).",
    )
    return p.parse_args()

ITEM_TYPES = [
    mappings.CALIBRATION_PROPERTIES_KEY,
    mappings.CALIBRATION_FLUX_IMAGE_KEY,
]

# ─────────────────────────────────────────────────────────────────────────────


_DAIC_PAINT_DIR = pathlib.Path(
    "/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint"
)


def _heliostats_with_deflectometry(heliostats_dir: pathlib.Path) -> set[str]:
    """Return IDs of heliostats that have a locally-downloaded filled deflectometry file."""
    has_defl = set()
    if not heliostats_dir.exists():
        return has_defl
    for hid_dir in heliostats_dir.iterdir():
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
    args       = _parse_args()
    split_type = _SPLIT_CHOICES[args.split_type]
    train_size = args.train_size
    val_size   = args.val_size
    # test = remaining after train + val — with >=200 measurements the minimum is 50
    min_measurements = train_size + val_size + val_size

    paint_dir     = _DAIC_PAINT_DIR if args.daic else PAINT_DIR
    heliostats_dir = paint_dir / "heliostats"
    metadata_file  = paint_dir / "metadata" / "calibration_metadata_all_heliostats.csv"

    _suffix = "" if args.no_deflectometry else "_deflectometry"
    benchmark_name = (
        f"benchmark_split-{split_type}_train-{train_size}"
        f"_validation-{val_size}{_suffix}"
    )

    set_logger_config()

    # ── 0. verify metadata is present ─────────────────────────────────────────
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_file}\n"
            "Run download_paint_benchmark.py first to download the metadata."
        )
    print(f"✓ Metadata present: {metadata_file}")
    print(f"  Paint dir  : {paint_dir}")
    print(f"  Split type : {split_type}  |  train={train_size}  val={val_size}")

    # ── 1. identify qualifying heliostats ─────────────────────────────────────
    metadata = pd.read_csv(metadata_file)
    counts   = metadata.groupby(mappings.HELIOSTAT_ID).size()

    hids_enough_data   = set(counts[counts >= min_measurements].index)
    if args.no_deflectometry:
        qualifying = sorted(hids_enough_data)
    else:
        hids_deflectometry = _heliostats_with_deflectometry(heliostats_dir)
        qualifying         = sorted(hids_enough_data & hids_deflectometry)

    print(f"\nHeliostat selection:")
    print(f"  Total in metadata          : {len(counts)}")
    print(f"  With >= {min_measurements} measurements     : {len(hids_enough_data)}")
    if args.no_deflectometry:
        print(f"  Deflectometry filter       : DISABLED (--no-deflectometry)")
    else:
        print(f"  With deflectometry locally : {len(hids_deflectometry)}")
    print(f"  Qualifying                 : {len(qualifying)}")

    if not qualifying:
        raise RuntimeError(
            "No qualifying heliostats found. "
            "Ensure deflectometry data has been downloaded with "
            "download_paint_benchmark.py first."
        )

    # ── 2. create benchmark split from filtered metadata ──────────────────────
    splits_dir = paint_dir / "splits"
    splits_csv = splits_dir / f"{benchmark_name}.csv"

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
                split_type=split_type,
                training_size=train_size,
                validation_size=val_size,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        # DatasetSplitter saves with the auto-generated name; rename to ours.
        auto_name = (
            f"benchmark_split-{split_type}"
            f"_train-{train_size}_validation-{val_size}.csv"
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
        item_dir = paint_dir / benchmark_name / item_type
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
        if not (heliostats_dir / hid / mappings.SAVE_PROPERTIES).exists()
    ]

    if not missing:
        print(
            f"\n✓ Properties + Deflectometry already present "
            f"for all {len(benchmark_heliostats)} benchmark heliostats."
        )
    else:
        collections = [mappings.SAVE_PROPERTIES.lower()]
        if not args.no_deflectometry:
            collections.append(mappings.SAVE_DEFLECTOMETRY.lower())
        print(f"\nDownloading {collections} for {len(missing)} heliostats...")
        client = StacClient(output_dir=heliostats_dir)
        client.get_heliostat_data(
            heliostats=missing,
            collections=collections,
        )
        print(f"✓ Done → {heliostats_dir}")

    print(f"\nDone. Benchmark: {benchmark_name}")
    print(f"  Heliostats : {len(benchmark_heliostats)}")
    print(f"  Per heliostat: {train_size} train / {val_size} val / ~{val_size} test")


if __name__ == "__main__":
    main()
