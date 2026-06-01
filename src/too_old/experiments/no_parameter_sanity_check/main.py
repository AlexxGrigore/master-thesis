import pathlib
import sys

_pkg = pathlib.Path(__file__).parent
_experiments = _pkg.parent
_src = _experiments.parent
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_experiments))
sys.path.insert(0, str(_pkg))

IS_ON_DAIC = True
SMOKE_TEST = not IS_ON_DAIC
SMOKE_TEST_HELIOSTAT = "AA31"
SMOKE_TEST_SAMPLE_LIMIT = 8

if IS_ON_DAIC:
    import matplotlib

    matplotlib.use("Agg")

import datetime
import logging
import traceback

import torch
import paint.util.paint_mappings as paint_mappings
from artist.core.loss_functions import FocalSpotLoss
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment
from artist_extensions.kinematic_reconstructors import NoParametersReconstructor
from improved_focal_spot_kr.experiment import run_experiment
from utils.evaluation import build_heliostat_data_mapping

torch.manual_seed(42)
torch.cuda.manual_seed(42)

set_logger_config()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

if IS_ON_DAIC:
    BASE_DIR = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
else:
    BASE_DIR = _src.parent

if IS_ON_DAIC:
    PAINT_DIR = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")
else:
    PAINT_DIR = BASE_DIR / "datasets" / "paint"

BENCHMARK_NAME = "benchmark_split-balanced_train-10_validation-30"
SCENARIO_PATH = (
    BASE_DIR / "scenarios" / "one_heliostat_scenarios" / "scenario1.h5"
    if SMOKE_TEST
    else BASE_DIR / "scenarios" / "deflectometry_scenario" / "deflectometry_scenario.h5"
)

_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = (
    BASE_DIR / "outputs" / "local_runs" / f"no_parameter_sanity_check_{_run_timestamp}"
    if not IS_ON_DAIC
    else BASE_DIR / "outputs" / f"no_parameter_sanity_check_{_run_timestamp}"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_log_file_handler = logging.FileHandler(OUTPUT_DIR / "training.log")
_log_file_handler.setFormatter(
    logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
)
logging.getLogger().addHandler(_log_file_handler)

BENCHMARK_CSV = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
CALIBRATION_PROPERTIES_DIR = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
FLUX_IMAGE_DIR = PAINT_DIR / BENCHMARK_NAME / "flux_image"

SAMPLE_LIMIT_PER_HELIOSTAT = SMOKE_TEST_SAMPLE_LIMIT if SMOKE_TEST else 10
CENTROID_METHOD = paint_mappings.UTIS_KEY


def _filter_mapping(mapping):
    return [
        (hid, cal[:SMOKE_TEST_SAMPLE_LIMIT], flux[:SMOKE_TEST_SAMPLE_LIMIT])
        for hid, cal, flux in mapping
        if hid == SMOKE_TEST_HELIOSTAT
    ]


def main() -> None:
    print(f"\nRunning on DAIC: {IS_ON_DAIC}")
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Scenario: {SCENARIO_PATH}")

    device = get_device()
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    train_mapping = build_heliostat_data_mapping(
        benchmark_csv=BENCHMARK_CSV,
        calibration_properties_dir=CALIBRATION_PROPERTIES_DIR,
        flux_image_dir=FLUX_IMAGE_DIR,
        split="train",
    )
    test_mapping = build_heliostat_data_mapping(
        benchmark_csv=BENCHMARK_CSV,
        calibration_properties_dir=CALIBRATION_PROPERTIES_DIR,
        flux_image_dir=FLUX_IMAGE_DIR,
        split="test",
    )
    validation_mapping = build_heliostat_data_mapping(
        benchmark_csv=BENCHMARK_CSV,
        calibration_properties_dir=CALIBRATION_PROPERTIES_DIR,
        flux_image_dir=FLUX_IMAGE_DIR,
        split="validation",
    )

    if SMOKE_TEST:
        train_mapping = _filter_mapping(train_mapping)
        test_mapping = _filter_mapping(test_mapping)
        validation_mapping = _filter_mapping(validation_mapping)
        print(f"[SMOKE TEST] Filtered to heliostat {SMOKE_TEST_HELIOSTAT}")

    print(f"\nTrain:      {len(train_mapping)} heliostats")
    print(f"Validation: {len(validation_mapping)} heliostats")
    print(f"Test:       {len(test_mapping)} heliostats")

    train_data_parser = PaintCalibrationDataParser(
        sample_limit=SAMPLE_LIMIT_PER_HELIOSTAT,
        centroid_extraction_method=CENTROID_METHOD,
    )
    eval_data_parser = PaintCalibrationDataParser(
        sample_limit=SMOKE_TEST_SAMPLE_LIMIT if SMOKE_TEST else 10,
        centroid_extraction_method=CENTROID_METHOD,
    )

    optimization_configuration = {
        config_dictionary.initial_learning_rate: 1e-4,
        config_dictionary.tolerance: 1e-6,
        config_dictionary.max_epoch: 11 if SMOKE_TEST else 300,
        config_dictionary.batch_size: 8,
        config_dictionary.log_step: 5,
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

    number_of_heliostat_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(
        scenario_path=SCENARIO_PATH
    )

    try:
        with setup_distributed_environment(
            number_of_heliostat_groups=number_of_heliostat_groups,
            device=device,
        ) as ddp_setup:
            device = ddp_setup[config_dictionary.device]
            metrics = run_experiment(
                loss_name="sanity_no_parameters",
                loss_fn_factory=lambda scenario: FocalSpotLoss(scenario=scenario),
                reconstructor_cls=NoParametersReconstructor,
                ddp_setup=ddp_setup,
                device=device,
                scenario_path=SCENARIO_PATH,
                train_mapping=train_mapping,
                test_mapping=test_mapping,
                train_data_parser=train_data_parser,
                eval_data_parser=eval_data_parser,
                optimization_configuration=optimization_configuration,
                output_dir=OUTPUT_DIR,
                save_figures=True,
                validation_mapping=validation_mapping,
            )
    except Exception:
        print("\n" + "=" * 60)
        print("ERROR: Sanity-check experiment crashed!")
        print("=" * 60)
        traceback.print_exc()
        raise

    drift = metrics.get("parameter_drift", {})
    print("\n" + "=" * 60)
    print("NO-PARAMETER SANITY CHECK")
    print("=" * 60)
    print(f"Baseline (untrained): {metrics.get('baseline_mean_mrad', float('nan')):.3f} mrad")
    print(f"Final:                {metrics.get('mean_focal_spot_error_mrad', float('nan')):.3f} mrad")
    print(f"Improvement:          {metrics.get('improvement_mrad', float('nan')):+.3f} mrad")
    print(f"Max parameter drift:  {drift.get('global_max_abs_drift', float('nan')):.6e}")
    print(f"Outputs saved to:     {OUTPUT_DIR}")


if __name__ == "__main__":
    main()