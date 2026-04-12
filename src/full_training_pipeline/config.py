from __future__ import annotations

import datetime
import pathlib
from dataclasses import asdict, dataclass

import paint.util.paint_mappings as paint_mappings


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
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    sample_limit_per_heliostat: int = 10
    max_epochs: int = 20
    learning_rate: float = 5e-3
    weight_decay: float = 1e-5
    residual_l2_weight: float = 1e-4
    number_of_rays: int = 20
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


def build_default_config(
    *,
    is_on_daic: bool = False,
    smoke_test: bool = False,
    max_epochs: int | None = None,
    output_dir: pathlib.Path | None = None,
) -> PipelineConfig:
    module_dir = pathlib.Path(__file__).parent
    src_dir = module_dir.parent

    if is_on_daic:
        base_dir = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
        paint_dir = pathlib.Path(
            "/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint"
        )
    else:
        base_dir = src_dir.parent
        paint_dir = base_dir / "datasets" / "paint"

    benchmark_name = "benchmark_split-balanced_train-10_validation-30"
    scenario_path = (
        base_dir / "scenarios" / "one_heliostat_scenarios" / "scenario1.h5"
        if smoke_test
        else base_dir / "scenarios" / "deflectometry_scenario" / "deflectometry_scenario.h5"
    )
    coarse_checkpoint_path = module_dir / "coarse_learning_parameters" / "kinematic_parameters.json"

    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            base_dir / "outputs" / "local_runs" / f"full_training_pipeline_{timestamp}"
            if not is_on_daic
            else base_dir / "outputs" / f"full_training_pipeline_{timestamp}"
        )

    epoch_count = max_epochs if max_epochs is not None else (2 if smoke_test else 20)
    sample_limit = 2 if smoke_test else 10
    rays = 5 if smoke_test else 20

    return PipelineConfig(
        base_dir=base_dir,
        paint_dir=paint_dir,
        benchmark_name=benchmark_name,
        scenario_path=scenario_path,
        coarse_checkpoint_path=coarse_checkpoint_path,
        output_dir=output_dir,
        benchmark_csv=paint_dir / "splits" / f"{benchmark_name}.csv",
        calibration_properties_dir=paint_dir / benchmark_name / "calibration_properties",
        flux_image_dir=paint_dir / benchmark_name / "flux_image",
        sample_limit_per_heliostat=sample_limit,
        max_epochs=epoch_count,
        number_of_rays=rays,
        is_on_daic=is_on_daic,
        smoke_test=smoke_test,
    )