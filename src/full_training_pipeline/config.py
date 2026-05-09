"""
Configuration for the full training pipeline.

Edit the variables below, then run:
    python main.py
"""
from __future__ import annotations

import datetime
import pathlib
from dataclasses import asdict, dataclass

import paint.util.paint_mappings as paint_mappings

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

IS_ON_DAIC = False

if IS_ON_DAIC:
    BASE_DIR = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
    PAINT_DIR = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")
else:
    BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
    PAINT_DIR = BASE_DIR / "datasets" / "paint"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

BENCHMARK_NAME = "benchmark_split-balanced_train-100_validation-50_deflectometry"

# "real"      — PAINT calibration images
# "synthetic" — pre-generated data from full_field_200_samples/generate_dataset.py
DATASET_TYPE = "synthetic"

SAMPLE_LIMIT_PER_HELIOSTAT = 100   # max calibration images used per heliostat

CENTROID_METHOD = paint_mappings.UTIS_KEY

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

# "focal_spot" — Euclidean distance between predicted and measured centroids
# "pixel"      — L1 on Gaussian-blurred, peak-normalised flux bitmaps
# "alignment"  — MSE on motor positions in joint-angle space (no ray tracing)
LOSS_TYPE = "focal_spot"

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

MAX_EPOCHS         = 200
LEARNING_RATE      = 1e-3
WEIGHT_DECAY       = 1e-5
RESIDUAL_L2_WEIGHT = 1e-4
GRAD_CLIP_MAX_NORM = 1.0
LR_SCHEDULER_PATIENCE  = 10
LR_SCHEDULER_FACTOR    = 0.5
LR_SCHEDULER_MIN_LR    = 1e-6

# ---------------------------------------------------------------------------
# Ray tracing
# ---------------------------------------------------------------------------

NUMBER_OF_RAYS         = 10
RAY_TRACING_BATCH_SIZE = 32
SURFACE_POINTS_PER_FACET = (25, 25)
BITMAP_RESOLUTION        = (256, 256)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

# Set to None to auto-generate a timestamped directory.
OUTPUT_DIR: pathlib.Path | None = None


# ---------------------------------------------------------------------------
# Internal — do not edit below this line
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    base_dir: pathlib.Path
    paint_dir: pathlib.Path
    benchmark_name: str
    scenario_path: pathlib.Path
    coarse_checkpoint_path: pathlib.Path
    output_dir: pathlib.Path
    benchmark_csv: pathlib.Path
    calibration_properties_dir: pathlib.Path
    flux_image_dir: pathlib.Path
    synthetic_data_base_dir: pathlib.Path
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    dataset_type: str = "real"
    loss_type: str = "focal_spot"
    sample_limit_per_heliostat: int = 100
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    residual_l2_weight: float = 1e-4
    grad_clip_max_norm: float = 1.0
    lr_scheduler_patience: int = 10
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-6
    number_of_rays: int = 10
    ray_tracing_batch_size: int = 32
    surface_points_per_facet: tuple[int, int] = (25, 25)
    bitmap_resolution: tuple[int, int] = (256, 256)
    centroid_method: str = paint_mappings.UTIS_KEY
    is_on_daic: bool = False
    smoke_test: bool = False

    def to_jsonable_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, pathlib.Path):
                payload[key] = str(value)
        return payload


def build_config(
    *,
    smoke_test: bool = False,
    dataset_type: str | None = None,
    is_on_daic: bool | None = None,
) -> PipelineConfig:
    module_dir = pathlib.Path(__file__).parent
    resolved_daic = is_on_daic if is_on_daic is not None else IS_ON_DAIC
    resolved_dataset_type = dataset_type if dataset_type is not None else DATASET_TYPE

    if resolved_daic:
        base_dir = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
        paint_dir = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")
    else:
        base_dir = BASE_DIR
        paint_dir = PAINT_DIR

    _ckpt_name = (
        "kinematic_parameters_synthetic.json"
        if resolved_dataset_type == "synthetic"
        else "kinematic_parameters_real.json"
    )
    coarse_checkpoint_path = module_dir / "coarse_learning_parameters" / _ckpt_name
    scenario_path = base_dir / "scenarios" / "full_field_200_samples_scenario" / "scenario.h5"
    synthetic_data_base_dir = (
        base_dir / "scenarios" / "full_field_200_samples_scenario" / "synthetic_data"
    )

    if OUTPUT_DIR is not None:
        output_dir = OUTPUT_DIR
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            base_dir / "outputs" / f"full_training_pipeline_{timestamp}"
            if resolved_daic
            else base_dir / "outputs" / "local_runs" / f"full_training_pipeline_{timestamp}"
        )

    return PipelineConfig(
        base_dir=base_dir,
        paint_dir=paint_dir,
        benchmark_name=BENCHMARK_NAME,
        scenario_path=scenario_path,
        coarse_checkpoint_path=coarse_checkpoint_path,
        output_dir=output_dir,
        benchmark_csv=paint_dir / "splits" / f"{BENCHMARK_NAME}.csv",
        calibration_properties_dir=paint_dir / BENCHMARK_NAME / "calibration_properties",
        flux_image_dir=paint_dir / BENCHMARK_NAME / "flux_image",
        synthetic_data_base_dir=synthetic_data_base_dir,
        dataset_type=resolved_dataset_type,
        loss_type=LOSS_TYPE,
        sample_limit_per_heliostat=SAMPLE_LIMIT_PER_HELIOSTAT,
        max_epochs=3 if smoke_test else MAX_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        residual_l2_weight=RESIDUAL_L2_WEIGHT,
        grad_clip_max_norm=GRAD_CLIP_MAX_NORM,
        lr_scheduler_patience=LR_SCHEDULER_PATIENCE,
        lr_scheduler_factor=LR_SCHEDULER_FACTOR,
        lr_scheduler_min_lr=LR_SCHEDULER_MIN_LR,
        number_of_rays=NUMBER_OF_RAYS,
        ray_tracing_batch_size=RAY_TRACING_BATCH_SIZE,
        surface_points_per_facet=SURFACE_POINTS_PER_FACET,
        bitmap_resolution=BITMAP_RESOLUTION,
        centroid_method=CENTROID_METHOD,
        is_on_daic=IS_ON_DAIC,
        smoke_test=smoke_test,
    )
