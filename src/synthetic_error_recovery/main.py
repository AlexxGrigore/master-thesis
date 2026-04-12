"""
Single-Perturbation Recovery Benchmark for Kinematic Reconstruction.

This entrypoint configures one recovery experiment and delegates the actual
benchmark execution to the local experiment and plotting modules.
"""

import pathlib
import sys

_pkg = pathlib.Path(__file__).parent
_src = _pkg.parent
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_pkg))

import datetime
import logging

IS_ON_DAIC = False
SMOKE_TEST = not IS_ON_DAIC
SMOKE_TEST_HELIOSTAT = "AA31"
SMOKE_TEST_SAMPLE_LIMIT = 8

if IS_ON_DAIC:
    import matplotlib

    matplotlib.use("Agg")

import torch
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device

from artist_extensions.kinematic_reconstructors import (
    RotationsActuatorsReconstructor,
    RotationsOnlyReconstructor,
)
from experiment import (
    Perturbation,
    RecoveryExperiment,
    filter_mapping_for_smoke_test,
    run_recovery_benchmark_experiment,
)
from utils.evaluation import build_heliostat_data_mapping

set_logger_config()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

torch.manual_seed(0)
torch.cuda.manual_seed(0)

if IS_ON_DAIC:
    BASE_DIR = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
    PAINT_DIR = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")
else:
    BASE_DIR = pathlib.Path(__file__).parent.parent.parent
    PAINT_DIR = BASE_DIR / "datasets" / "paint"

SCENARIO_PATH = (
    BASE_DIR / "scenarios" / "one_heliostat_scenarios" / "scenario1.h5"
    if SMOKE_TEST
    else BASE_DIR / "scenarios" / "deflectometry_scenario" / "deflectometry_scenario.h5"
)
BENCHMARK_NAME = "benchmark_split-balanced_train-10_validation-30"
BENCHMARK_CSV = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
CALIBRATION_PROPERTIES_DIR = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
FLUX_IMAGE_DIR = PAINT_DIR / BENCHMARK_NAME / "flux_image"

_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = (
    BASE_DIR / "outputs" / "local_runs" / f"recovery_benchmark_{_run_timestamp}"
    if not IS_ON_DAIC
    else BASE_DIR / "outputs" / f"recovery_benchmark_{_run_timestamp}"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_log_fh = logging.FileHandler(OUTPUT_DIR / "recovery_benchmark.log")
_log_fh.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
logging.getLogger().addHandler(_log_fh)

TRAIN_SAMPLE_LIMIT = SMOKE_TEST_SAMPLE_LIMIT if SMOKE_TEST else 10
EVAL_SAMPLE_LIMIT = SMOKE_TEST_SAMPLE_LIMIT if SMOKE_TEST else 30
BITMAP_RESOLUTION = torch.tensor([256, 256])
RAY_TRACING_BATCH_SIZE = 32
SCENARIO_NUM_RAYS = 10
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
EVALUATION_SPLIT = "test"
CENTROID_EXTRACTION_METHOD = "UTIS"

OPTIMIZATION_CONFIG = {
    config_dictionary.initial_learning_rate: 1e-4,
    config_dictionary.tolerance: 1e-6,
    config_dictionary.max_epoch: 11 if SMOKE_TEST else 300,
    config_dictionary.batch_size: 8,
    config_dictionary.log_step: 5 if SMOKE_TEST else 25,
    config_dictionary.early_stopping_window: 10,
    config_dictionary.early_stopping_delta: 1e-5,
    config_dictionary.early_stopping_patience: 400,
    config_dictionary.scheduler: config_dictionary.reduce_on_plateau,
    "scheduler_parameters": {
        config_dictionary.min: 1e-6,
        config_dictionary.reduce_factor: 0.5,
        config_dictionary.patience: 10,
        config_dictionary.threshold: 1e-3,
        config_dictionary.cooldown: 5,
    },
}

EXPERIMENTS = {
    "rotations_only_all_optimized": RecoveryExperiment(
        name="rotations_only_all_optimized",
        reconstructor_cls=RotationsOnlyReconstructor,
        perturbations=(
            Perturbation("rotation", 0.003, index=0, label="first_joint_tilt_n"),
            Perturbation("rotation", 0.003, index=1, label="first_joint_tilt_u"),
            Perturbation("rotation", 0.003, index=2, label="second_joint_tilt_e"),
            Perturbation("rotation", 0.003, index=3, label="second_joint_tilt_n"),
        ),
        description="Perturb all rotation parameters optimized by RotationsOnlyReconstructor.",
    ),
    "rotations_actuators_all_optimized": RecoveryExperiment(
        name="rotations_actuators_all_optimized",
        reconstructor_cls=RotationsActuatorsReconstructor,
        perturbations=(
            Perturbation("rotation", 0.003, index=0, label="first_joint_tilt_n"),
            Perturbation("rotation", 0.003, index=1, label="first_joint_tilt_u"),
            Perturbation("rotation", 0.003, index=2, label="second_joint_tilt_e"),
            Perturbation("rotation", 0.003, index=3, label="second_joint_tilt_n"),
            Perturbation("actuator_angle", 0.003, label="all_actuator_initial_angles"),
            Perturbation("actuator_offset", 0.003, label="all_actuator_offsets"),
        ),
        description="Perturb all parameters optimized by RotationsActuatorsReconstructor.",
    ),
}
ACTIVE_EXPERIMENT_NAME = "rotations_actuators_all_optimized"
ACTIVE_EXPERIMENT = EXPERIMENTS[ACTIVE_EXPERIMENT_NAME]


def main() -> None:
    device = get_device()
    log.info("Device: %s", device)
    log.info("Running on DAIC: %s", IS_ON_DAIC)
    log.info("Smoke test: %s", SMOKE_TEST)
    if torch.cuda.is_available():
        log.info(
            "GPU: %s, %.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    train_mapping = build_heliostat_data_mapping(
        benchmark_csv=BENCHMARK_CSV,
        calibration_properties_dir=CALIBRATION_PROPERTIES_DIR,
        flux_image_dir=FLUX_IMAGE_DIR,
        split=TRAIN_SPLIT,
    )
    validation_mapping = build_heliostat_data_mapping(
        benchmark_csv=BENCHMARK_CSV,
        calibration_properties_dir=CALIBRATION_PROPERTIES_DIR,
        flux_image_dir=FLUX_IMAGE_DIR,
        split=VALIDATION_SPLIT,
    )
    evaluation_mapping = build_heliostat_data_mapping(
        benchmark_csv=BENCHMARK_CSV,
        calibration_properties_dir=CALIBRATION_PROPERTIES_DIR,
        flux_image_dir=FLUX_IMAGE_DIR,
        split=EVALUATION_SPLIT,
    )

    if SMOKE_TEST:
        train_mapping = filter_mapping_for_smoke_test(
            train_mapping,
            heliostat_id=SMOKE_TEST_HELIOSTAT,
            sample_limit=SMOKE_TEST_SAMPLE_LIMIT,
        )
        validation_mapping = filter_mapping_for_smoke_test(
            validation_mapping,
            heliostat_id=SMOKE_TEST_HELIOSTAT,
            sample_limit=SMOKE_TEST_SAMPLE_LIMIT,
        )
        evaluation_mapping = filter_mapping_for_smoke_test(
            evaluation_mapping,
            heliostat_id=SMOKE_TEST_HELIOSTAT,
            sample_limit=SMOKE_TEST_SAMPLE_LIMIT,
        )
        log.info(
            "Smoke test filtered to heliostat %s with %s samples per split.",
            SMOKE_TEST_HELIOSTAT,
            SMOKE_TEST_SAMPLE_LIMIT,
        )

    number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
        scenario_path=SCENARIO_PATH
    )

    summary = run_recovery_benchmark_experiment(
        active_experiment=ACTIVE_EXPERIMENT,
        output_dir=OUTPUT_DIR,
        scenario_path=SCENARIO_PATH,
        scenario_num_rays=SCENARIO_NUM_RAYS,
        number_of_heliostat_groups=number_of_heliostat_groups,
        optimization_config=OPTIMIZATION_CONFIG,
        train_mapping=train_mapping,
        validation_mapping=validation_mapping,
        evaluation_mapping=evaluation_mapping,
        train_sample_limit=TRAIN_SAMPLE_LIMIT,
        eval_sample_limit=EVAL_SAMPLE_LIMIT,
        centroid_extraction_method=CENTROID_EXTRACTION_METHOD,
        bitmap_resolution=BITMAP_RESOLUTION,
        ray_tracing_batch_size=RAY_TRACING_BATCH_SIZE,
        device=device,
    )

    print("\n" + "=" * 70)
    print("RECOVERY BENCHMARK SUMMARY")
    print("=" * 70)
    print(
        f"[{summary['name']}] baseline={summary['baseline_mean_mrad']:.3f} mrad | "
        f"perturbed={summary['perturbed_mean_mrad']:.3f} mrad | "
        f"recovered={summary['recovered_mean_mrad']:.3f} mrad | "
        f"recovered_fraction={summary['mean_recovered_fraction']}"
    )
    print("=" * 70)
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
