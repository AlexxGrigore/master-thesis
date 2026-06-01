#!/usr/bin/env python3
"""
Training of ARTIST - Dataset Split Creation Script
Converted from Jupyter notebook for standalone execution
"""

import pathlib
import argparse

import paint.util.paint_mappings as mappings
from paint import PAINT_ROOT
from paint.data import StacClient
from paint.data.dataset import PaintCalibrationDataset
from paint.data.dataset_splits import DatasetSplitter
from paint.util import set_logger_config


def main():
    """Main function to create dataset splits for PAINT calibration data"""
    
    # Setup logging
    set_logger_config()
    print("Imports complete")
    
    # ============================================================================
    # Configuration
    # ============================================================================
    
    # Path to the metadata file
    metadata_file = pathlib.Path("../paint_dataset/metadata/calibration_metadata_all_heliostats.csv")
    
    # Output directory for benchmarks
    output_dir = ".." / pathlib.Path.cwd() / "paint_benchmarks"
    
    # Split configuration
    split_type = mappings.BALANCED_SPLIT  # Using balanced split
    train_size = 10  # Number of training samples per heliostat
    val_size = 30    # Number of validation samples per heliostat
    
    # Whether to remove unused metadata
    remove_unused_data = True
    
    # Item types to load - NOW A LIST
    item_types = [
        mappings.CALIBRATION_PROPERTIES_KEY,  # "calibration_properties"
        mappings.CALIBRATION_FLUX_IMAGE_KEY,  # "flux_image"
    ]
    
    print(f"Metadata file: {metadata_file}")
    print(f"Output directory: {output_dir}")
    print(f"Split type: {split_type}")
    print(f"Train size per heliostat: {train_size}")
    print(f"Validation size per heliostat: {val_size}")
    print(f"Item types: {item_types}")
    
    # ============================================================================
    # Check and download metadata if needed
    # ============================================================================
    
    if not metadata_file.exists():
        print(f"Metadata file not found at {metadata_file}")
        print("Downloading metadata...")
        
        # Create STAC client to download the metadata
        output_dir_for_stac = metadata_file.parent
        output_dir_for_stac.mkdir(parents=True, exist_ok=True)
        
        client = StacClient(output_dir=output_dir_for_stac)
        client.get_heliostat_metadata(heliostats=None)
        print("✓ Metadata downloaded")
    else:
        print(f"✓ Metadata file found at {metadata_file}")
    
    # ============================================================================
    # Create dataset splits
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("Creating dataset splits...")
    print("=" * 80)
    
    # Set the output directory for splits
    splits_output_dir = output_dir / "splits"
    splits_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the splitter
    splitter = DatasetSplitter(
        input_file=metadata_file,
        output_dir=splits_output_dir,
        remove_unused_data=remove_unused_data,
    )
    
    # Generate the splits
    splits = splitter.get_dataset_splits(
        split_type=split_type,
        training_size=train_size,
        validation_size=val_size,
    )
    
    print(f"\n✓ Dataset splits created and saved to: {splits_output_dir}")
    
    # ============================================================================
    # Prepare to load the datasets
    # ============================================================================
    
    # Determine the benchmark file name
    dataset_benchmark_file = (
        splits_output_dir
        / f"benchmark_split-{split_type}_train-{train_size}_validation-{val_size}.csv"
    )
    
    print(f"\nBenchmark file: {dataset_benchmark_file}")
    print(f"\nWill create datasets for {len(item_types)} item types:")
    for item_type in item_types:
        print(f"  - {item_type}")
    
    # ============================================================================
    # Initialize PyTorch datasets for all item types
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("Initializing PyTorch datasets for all item types...")
    print("=" * 80)
    
    # Dictionary to store datasets
    datasets = {}
    
    for item_type in item_types:
        print(f"\n--- Processing {item_type} ---")
        
        # Set the dataset output directory for this item type
        # This will create organized train/test/val folders
        dataset_output_dir = (
            output_dir
            / "datasets"
            / f"benchmark_split-{split_type}_train-{train_size}_validation-{val_size}"
            / item_type
        )
        
        # The dataset will copy/link files from your paint_dataset to the organized structure
        print(f"Dataset output directory: {dataset_output_dir}")
        print(f"Will organize data from: paint_dataset/")
        
        # Set download=True the FIRST time to organize the data
        # After that, you can set it to False
        dataset_download = not dataset_output_dir.exists()
        print(f"Need to organize data: {dataset_download}")
        
        # Initialize dataset from benchmark splits
        train_dataset, test_dataset, val_dataset = PaintCalibrationDataset.from_benchmark(
            benchmark_file=dataset_benchmark_file,
            root_dir=dataset_output_dir,
            item_type=item_type,
            download=dataset_download,  # Will organize data into train/test/val structure
        )
        
        # Store in dictionary
        datasets[item_type] = {
            'train': train_dataset,
            'test': test_dataset,
            'val': val_dataset,
        }
        
        print(f"✓ {item_type} datasets initialized!")
        print(f"  Training: {len(train_dataset)}, Test: {len(test_dataset)}, Validation: {len(val_dataset)}")
    
    print("\n" + "=" * 80)
    print("All datasets initialized successfully!")
    
    # ============================================================================
    # Inspect samples
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("Samples from training sets:")
    print("=" * 80)
    
    for item_type in item_types:
        print(f"\n--- {item_type} ---")
        train_dataset = datasets[item_type]['train']
        
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"Sample type: {type(sample)}")
            print(f"Sample keys: {sample.keys() if isinstance(sample, dict) else 'Not a dict'}")
            print(f"Sample content preview:")
            print(sample)
    
    # ============================================================================
    # Summary
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Split type: {split_type}")
    print(f"Training samples per heliostat: {train_size}")
    print(f"Validation samples per heliostat: {val_size}")
    
    for item_type in item_types:
        print(f"\n{item_type}:")
        print(f"  Training: {len(datasets[item_type]['train'])}")
        print(f"  Test: {len(datasets[item_type]['test'])}")
        print(f"  Validation: {len(datasets[item_type]['val'])}")
    
    print(f"\nFiles saved to: {output_dir}")
    print("\n✓ Dataset split creation complete for all item types!")


if __name__ == "__main__":
    main()
