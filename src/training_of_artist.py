import pathlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import paint.util.paint_mappings as mappings
from paint.data.dataset import PaintCalibrationDataset
from paint.util import set_logger_config

# Setup logging
set_logger_config()

# ===== Environment toggle =====
IS_ON_DAIC = True  # Set to False if running locally

# Dataset configuration
BENCHMARK_NAME = "benchmark_split-balanced_train-10_validation-30"

if IS_ON_DAIC:
    matplotlib.use("Agg")  # Non-interactive backend for HPC
    BASE_DIR = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/src/paint_benchmarks")
else:
    BASE_DIR = pathlib.Path("..") / pathlib.Path.cwd() / "paint_benchmarks"

# Item types to load
ITEM_TYPES = [
    mappings.CALIBRATION_PROPERTIES_KEY,  # "calibration_properties"
    mappings.CALIBRATION_FLUX_IMAGE_KEY,  # "flux_image"
]

# Training configuration
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on DAIC: {IS_ON_DAIC}")
print(f"Base directory: {BASE_DIR}")
print(f"Benchmark: {BENCHMARK_NAME}")
print(f"Device: {DEVICE}")

# ===== Load datasets =====
benchmark_file = BASE_DIR / "splits" / f"{BENCHMARK_NAME}.csv"

print(f"\nLoading datasets from: {benchmark_file}\n")

datasets = {}

for item_type in ITEM_TYPES:
    print(f"Loading {item_type} datasets...")

    dataset_dir = BASE_DIR / "datasets" / BENCHMARK_NAME / item_type

    train_dataset, test_dataset, val_dataset = PaintCalibrationDataset.from_benchmark(
        benchmark_file=benchmark_file,
        root_dir=dataset_dir,
        item_type=item_type,
        download=False,
    )

    datasets[item_type] = {
        "train": train_dataset,
        "test": test_dataset,
        "val": val_dataset,
    }

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples\n")

print(f"Successfully loaded {len(datasets)} dataset types!")

# ===== Inspect samples =====
for item_type in ITEM_TYPES:
    print("=" * 80)
    print(f"Inspecting {item_type} - Training set - Sample 0")
    print("=" * 80)

    sample = datasets[item_type]["train"][0]
    print(f"Sample type: {type(sample)}")

    if isinstance(sample, dict):
        print(f"Keys: {list(sample.keys())}")
        print("\nContent:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, dict):
                print(f"  {key}: Dict with keys={list(value.keys())}")
                for k2, v2 in value.items():
                    print(f"    - {k2}: {v2 if not isinstance(v2, (list, dict)) else type(v2)}")
            else:
                print(f"  {key}: {value}")
    elif isinstance(sample, torch.Tensor):
        print(f"Tensor shape: {sample.shape}")
        print(f"Tensor dtype: {sample.dtype}")
        print(f"Min value: {sample.min().item():.4f}")
        print(f"Max value: {sample.max().item():.4f}")
        print(f"Mean value: {sample.mean().item():.4f}")

    print()

# ===== Visualize a flux image =====
if "flux_image" in datasets:
    sample_image = datasets["flux_image"]["train"][0]

    if isinstance(sample_image, torch.Tensor):
        img = sample_image.numpy()

        fig, axes = plt.subplots(1, 1, figsize=(15, 5))
        axes.imshow(img[0], cmap="gray")
        axes.set_title("Flux Image")
        axes.axis("off")
        plt.tight_layout()

        if IS_ON_DAIC:
            plt.savefig("flux_image_sample.png", dpi=150)
            print("Saved flux image sample to flux_image_sample.png")
        else:
            plt.show()

        print(f"Image shape: {img.shape}")
        print(f"Image range: [{img.min():.4f}, {img.max():.4f}]")

# ===== Create DataLoaders =====
print("\nCreating DataLoaders...")
print(f"Batch size: {BATCH_SIZE}")
print(f"Num workers: {NUM_WORKERS}\n")

dataloaders = {}

for item_type in ITEM_TYPES:
    print(f"Creating dataloaders for {item_type}...")

    train_loader = DataLoader(
        datasets[item_type]["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        datasets[item_type]["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        datasets[item_type]["val"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    dataloaders[item_type] = {
        "train": train_loader,
        "test": test_loader,
        "val": val_loader,
    }

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print(f"  Val batches:   {len(val_loader)}\n")

print("DataLoaders created!")

# ===== Test first batch =====
print("\nTesting DataLoaders - First Batch:\n")

for item_type in ITEM_TYPES:
    print(f"{item_type}:")
    train_loader = dataloaders[item_type]["train"]

    batch = next(iter(train_loader))

    if isinstance(batch, torch.Tensor):
        print(f"  Batch shape: {batch.shape}")
        print(f"  Batch dtype: {batch.dtype}")
        print(f"  Batch device: {batch.device}")
    elif isinstance(batch, dict):
        print(f"  Batch is a dictionary with keys: {list(batch.keys())}")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, dict):
                print(f"    {key}: nested dict with keys={list(value.keys())}")
    print()
