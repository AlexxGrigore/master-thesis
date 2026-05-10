from __future__ import annotations

import json
import logging
import pathlib
import sys
import time

from tqdm import tqdm
from dataclasses import dataclass

_pkg = pathlib.Path(__file__).parent
_src = _pkg.parent
sys.path.insert(0, str(_src))

import h5py
import torch
from artist.core.loss_functions import FocalSpotLoss, PixelLoss
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist_extensions.cached_paint_parser import CachedPaintCalibrationDataParser
from artist.util import set_logger_config
from artist.util.environment_setup import get_device

from artist_extensions.loss_functions_ext import AlignmentLoss
from five_heliostats_synth.data import SyntheticDatasetParser
from full_training_pipeline.config import PipelineConfig, build_config
from full_training_pipeline.data import (
    SplitDataBundle,
    build_group_calibration_inputs,
    build_split_bundle,
    build_split_bundle_synth,
)
from full_training_pipeline.evaluate import apply_model_to_scenario, evaluate_model_tracking_accuracy
from full_training_pipeline.features import HeliostatCalibrationInput
from full_training_pipeline.model import SHARED_WORTBERG_PARAMETER_NAMES, build_residual_model
from full_training_pipeline.pipeline import FineErrorLearningPipeline, capture_all_group_parameter_states
from full_training_pipeline.plotting import (
    plot_baseline_vs_corrected_metrics,
    plot_error_histogram,
    plot_linear_weights_heatmap,
    plot_loss_curves,
    plot_per_heliostat_improvement_scatter,
    plot_predicted_residual_boxplot,
    plot_response_curves,
    plot_feature_importance,
)
from utils.checkpointing import load_kinematic_parameters, save_kinematic_parameters
from utils.evaluation import evaluate_flux_accuracy

set_logger_config()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

_LOSS_CONFIGS = {
    "focal_spot": lambda scenario: FocalSpotLoss(scenario=scenario),
    "pixel":      lambda scenario: PixelLoss(scenario=scenario),
    "alignment":  lambda _: AlignmentLoss(),
}


def _build_loss_fn(config: PipelineConfig, scenario):
    if config.loss_type not in _LOSS_CONFIGS:
        raise ValueError(f"Unknown loss_type {config.loss_type!r}. Choose from {list(_LOSS_CONFIGS)}.")
    return _LOSS_CONFIGS[config.loss_type](scenario)


# ---------------------------------------------------------------------------
# Data parsers factory
# ---------------------------------------------------------------------------

@dataclass
class DataParsers:
    train: object
    validation: object
    test: object


def _build_parsers(config: PipelineConfig) -> DataParsers:
    if config.dataset_type == "synthetic":
        return DataParsers(
            train=SyntheticDatasetParser(config.synthetic_data_base_dir / "train"),
            validation=SyntheticDatasetParser(config.synthetic_data_base_dir / "val"),
            test=SyntheticDatasetParser(config.synthetic_data_base_dir / "test"),
        )
    # Separate instances so each split has its own independent cache.
    def _make_parser():
        return CachedPaintCalibrationDataParser(
            sample_limit=config.sample_limit_per_heliostat,
            centroid_extraction_method=config.centroid_method,
        )
    return DataParsers(train=_make_parser(), validation=_make_parser(), test=_make_parser())


# ---------------------------------------------------------------------------
# Bundle factory
# ---------------------------------------------------------------------------

def _build_bundle(
    config: PipelineConfig,
    split: str,
    scenario,
    group_states,
    norm_stats=None,
) -> SplitDataBundle:
    if config.dataset_type == "synthetic":
        split_dir_map = {"train": "train", "validation": "val", "test": "test"}
        return build_split_bundle_synth(
            split_dir=config.synthetic_data_base_dir / split_dir_map[split],
            sample_limit_per_heliostat=config.sample_limit_per_heliostat,
            scenario=scenario,
            group_states=group_states,
            norm_stats=norm_stats,
        )
    return build_split_bundle(
        benchmark_csv=config.benchmark_csv,
        calibration_properties_dir=config.calibration_properties_dir,
        flux_image_dir=config.flux_image_dir,
        split=split,
        sample_limit_per_heliostat=config.sample_limit_per_heliostat,
        centroid_method=config.centroid_method,
        scenario=scenario,
        group_states=group_states,
        norm_stats=norm_stats,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(output_path: pathlib.Path, payload: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)


def _clone_model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _save_model_checkpoint(
    *,
    output_path: pathlib.Path,
    model_state_dict: dict[str, torch.Tensor],
    norm_stats_dict: dict[str, list[float]],
    selected_epoch: int,
    selected_validation_mean_mrad: float,
    selection_tag: str,
) -> None:
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "norm_stats": norm_stats_dict,
            "selected_epoch": selected_epoch,
            "selected_validation_mean_focal_spot_error_mrad": selected_validation_mean_mrad,
            "selection_tag": selection_tag,
        },
        output_path,
    )


def _save_model_checkpoint_json(
    *,
    output_path: pathlib.Path,
    model_state_dict: dict[str, torch.Tensor],
    norm_stats_dict: dict[str, list[float]],
    selected_epoch: int,
    selected_validation_mean_mrad: float,
    selection_tag: str,
) -> None:
    _write_json(
        output_path,
        {
            "selection_tag": selection_tag,
            "selected_epoch": selected_epoch,
            "selected_validation_mean_focal_spot_error_mrad": selected_validation_mean_mrad,
            "parameter_names": list(SHARED_WORTBERG_PARAMETER_NAMES),
            "norm_stats": norm_stats_dict,
            "residual_bounds": model_state_dict["residual_bounds"].tolist(),
            "linear_weight": model_state_dict["linear.weight"].tolist(),
            "linear_bias": model_state_dict["linear.bias"].tolist(),
        },
    )


def _build_training_summary(
    *,
    baseline_validation_metrics: dict[str, object],
    baseline_test_metrics: dict[str, object],
    best_epoch_record: dict[str, float],
    best_validation_metrics: dict[str, object],
    best_test_metrics: dict[str, object],
    last_epoch_record: dict[str, float],
    last_validation_metrics: dict[str, object],
    last_test_metrics: dict[str, object],
) -> dict[str, object]:
    baseline_validation_mean = float(baseline_validation_metrics["mean_focal_spot_error_mrad"])
    best_validation_mean = float(best_validation_metrics["mean_focal_spot_error_mrad"])
    last_validation_mean = float(last_validation_metrics["mean_focal_spot_error_mrad"])
    baseline_test_mean = float(baseline_test_metrics["mean_focal_spot_error_mrad"])
    best_test_mean = float(best_test_metrics["mean_focal_spot_error_mrad"])
    last_test_mean = float(last_test_metrics["mean_focal_spot_error_mrad"])
    return {
        "baseline_validation_mean_focal_spot_error_mrad": baseline_validation_mean,
        "best_epoch": int(best_epoch_record["epoch"]),
        "best_validation_mean_focal_spot_error_mrad": best_validation_mean,
        "best_validation_improvement_vs_baseline_mrad": baseline_validation_mean - best_validation_mean,
        "last_epoch": int(last_epoch_record["epoch"]),
        "last_validation_mean_focal_spot_error_mrad": last_validation_mean,
        "last_validation_improvement_vs_baseline_mrad": baseline_validation_mean - last_validation_mean,
        "baseline_test_mean_focal_spot_error_mrad": baseline_test_mean,
        "best_test_mean_focal_spot_error_mrad": best_test_mean,
        "best_test_improvement_vs_baseline_mrad": baseline_test_mean - best_test_mean,
        "last_test_mean_focal_spot_error_mrad": last_test_mean,
        "last_test_improvement_vs_baseline_mrad": baseline_test_mean - last_test_mean,
        "best_epoch_train_loss_m": float(best_epoch_record["train_loss_m"]),
        "best_epoch_objective": float(best_epoch_record["objective"]),
        "last_epoch_train_loss_m": float(last_epoch_record["train_loss_m"]),
        "last_epoch_objective": float(last_epoch_record["objective"]),
    }


@torch.no_grad()
def _predict_residual_tensor(
    *,
    residual_model: torch.nn.Module,
    group_calibration_inputs: list[list],
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for group_inputs in group_calibration_inputs:
        rows.append(residual_model(group_inputs))
    return torch.cat(rows, dim=0)


def _write_pipeline_markdown(
    *,
    output_path: pathlib.Path,
    config: PipelineConfig,
) -> None:
    from full_training_pipeline.model import INPUT_DIM
    parameter_lines = "\n".join(f"- `{name}`" for name in SHARED_WORTBERG_PARAMETER_NAMES)
    markdown = f"""# Full Training Pipeline

## Overview

This run trains the fine-error-learning baseline for heliostat kinematic correction.
The pipeline starts from a frozen coarse kinematic checkpoint and learns a shared linear
residual model that predicts a 20-dimensional Wortberg-style correction vector for every heliostat.

## Inputs

- Scenario file: `{config.scenario_path}`
- Coarse checkpoint: `{config.coarse_checkpoint_path}`
- Dataset type: `{config.dataset_type}`
- Loss type: `{config.loss_type}`

## Feature Construction

Each heliostat is represented by 19 aggregate statistics computed over all its calibration
measurements, plus its position and coarse kinematic parameters.

Input layout (total {INPUT_DIM}D):
- `heliostat_position` (3D): absolute ENU position from scenario
- `kinematic_params` (20D): coarse checkpoint parameters
- `mean_centroid` (3D): mean ENU flux centroid across all measurements
- `std_centroid` (3D): standard deviation of centroid positions
- `range_centroid` (3D): max - min centroid per axis
- `mean_sun` (3D): mean sun direction unit vector
- `cen_sun_slope` (3D): OLS slope of centroid on sun elevation
- `mean_motor` (2D): mean motor encoder readings
- `std_motor` (2D): spread of motor readings

All features are z-score normalised using training-set statistics.

## Linear Model

- Input dimension: {INPUT_DIM}
- Output dimension: 20
- Shared across all heliostats: yes
- Output activation: `tanh`
- Output scaling: physical residual bounds per parameter
- Learnable parameters: {INPUT_DIM * 20 + 20} (vs ~16K with raw measurement flattening)

The output parameters are:

{parameter_lines}

## Run Configuration

- Dataset type: {config.dataset_type}
- Loss type: {config.loss_type}
- Max epochs: {config.max_epochs}
- Learning rate: {config.learning_rate}
- Weight decay: {config.weight_decay}
- Residual L2 weight: {config.residual_l2_weight}
- Gradient clip max norm: {config.grad_clip_max_norm}
- LR scheduler patience: {config.lr_scheduler_patience}
- LR scheduler factor: {config.lr_scheduler_factor}
- LR scheduler min LR: {config.lr_scheduler_min_lr}
- Number of rays: {config.number_of_rays}
- Sample limit per heliostat: {config.sample_limit_per_heliostat}
- Smoke test: {config.smoke_test}
- Running on DAIC: {config.is_on_daic}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        handle.write(markdown)


def _load_scenario(config: PipelineConfig, device: torch.device) -> Scenario:
    with h5py.File(config.scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            device=device,
            number_of_surface_points_per_facet=torch.tensor(config.surface_points_per_facet),
        )
    scenario.set_number_of_rays(config.number_of_rays)
    load_kinematic_parameters(scenario, config.coarse_checkpoint_path, device)
    return scenario


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run(config: PipelineConfig) -> None:
    if config.max_epochs < 1:
        raise ValueError("max_epochs must be at least 1.")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    plots_dir = config.output_dir / "plots"
    json_dir = config.output_dir / "json"
    models_dir = config.output_dir / "models"
    plots_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(config.output_dir / "training.log")
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    # Silence console completely — everything goes to training.log.
    # set_logger_config() installs a StreamHandler on the "artist" logger (not root).
    # With propagate=True, messages also hit Python's lastResort handler → double print.
    # Fix: disable propagation and mute the artist StreamHandler to CRITICAL.
    _artist_logger = logging.getLogger("artist")
    _artist_logger_propagate_orig = _artist_logger.propagate
    _artist_logger.propagate = False
    _artist_logger.addHandler(file_handler)
    _artist_console_handlers = [
        h for h in _artist_logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    for h in _artist_console_handlers:
        h.setLevel(logging.CRITICAL)
    # Root logger: also suppress any StreamHandlers added externally.
    _console_handlers = [
        h for h in logging.getLogger().handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    for h in _console_handlers:
        h.setLevel(logging.CRITICAL)

    try:
        log.info("Starting full training pipeline.")
        log.info("Dataset type: %s | Loss type: %s", config.dataset_type, config.loss_type)
        _write_json(json_dir / "config.json", config.to_jsonable_dict())
        device = get_device()
        bitmap_resolution = torch.tensor(config.bitmap_resolution)
        log.info("Using device: %s", device)

        # Load scenario first — bundles need heliostat positions and kinematic params.
        tqdm.write("[1/6] Loading scenario...")
        scenario = _load_scenario(config, device)
        group_parameter_states = capture_all_group_parameter_states(scenario, device=device)

        tqdm.write("[2/6] Building train/validation/test bundles...")
        train_bundle = _build_bundle(config, "train", scenario, group_parameter_states)
        validation_bundle = _build_bundle(
            config, "validation", scenario, group_parameter_states,
            norm_stats=train_bundle.norm_stats,
        )
        test_bundle = _build_bundle(
            config, "test", scenario, group_parameter_states,
            norm_stats=train_bundle.norm_stats,
        )

        _write_json(json_dir / "norm_stats.json", train_bundle.norm_stats.to_dict())
        _write_pipeline_markdown(
            output_path=config.output_dir / "pipeline_details.md",
            config=config,
        )

        train_group_cal = build_group_calibration_inputs(scenario, train_bundle.calibration_inputs)
        validation_group_cal = build_group_calibration_inputs(scenario, validation_bundle.calibration_inputs)
        test_group_cal = build_group_calibration_inputs(scenario, test_bundle.calibration_inputs)

        parsers = _build_parsers(config)
        loss_fn = _build_loss_fn(config, scenario)

        residual_model = build_residual_model(config.model_type).to(device)
        pipeline = FineErrorLearningPipeline(
            scenario=scenario,
            residual_model=residual_model,
            group_parameter_states=group_parameter_states,
            bitmap_resolution=bitmap_resolution,
            ray_tracing_batch_size=config.ray_tracing_batch_size,
            loss_fn=loss_fn,
            loss_type=config.loss_type,
        )
        optimizer = torch.optim.Adam(
            residual_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience,
            min_lr=config.lr_scheduler_min_lr,
        )
        history: list[dict[str, float]] = []

        tqdm.write("[3/6] Evaluating baseline (val + test)...")
        baseline_validation_scenario = _load_scenario(config, device)
        baseline_validation_metrics = evaluate_flux_accuracy(
            scenario=baseline_validation_scenario,
            heliostat_data_mapping=validation_bundle.heliostat_data_mapping,
            data_parser=parsers.validation,
            device=device,
            bitmap_resolution=bitmap_resolution,
            ray_tracing_batch_size=config.ray_tracing_batch_size,
        )
        baseline_test_scenario = _load_scenario(config, device)
        baseline_test_metrics = evaluate_flux_accuracy(
            scenario=baseline_test_scenario,
            heliostat_data_mapping=test_bundle.heliostat_data_mapping,
            data_parser=parsers.test,
            device=device,
            bitmap_resolution=bitmap_resolution,
            ray_tracing_batch_size=config.ray_tracing_batch_size,
        )
        _write_json(json_dir / "validation_baseline_metrics.json", baseline_validation_metrics)
        _write_json(json_dir / "test_baseline_metrics.json", baseline_test_metrics)
        tqdm.write(
            f"    baseline val: {baseline_validation_metrics['mean_focal_spot_error_mrad']:.2f} mrad  |  "
            f"test: {baseline_test_metrics['mean_focal_spot_error_mrad']:.2f} mrad"
        )

        best_epoch_record: dict[str, float] | None = None
        best_validation_metrics: dict[str, object] | None = None
        best_model_state_dict: dict[str, torch.Tensor] | None = None
        last_epoch_record: dict[str, float] | None = None
        last_validation_metrics: dict[str, object] | None = None

        tqdm.write("[4/6] Training...")
        early_stopping_patience = max(5, config.max_epochs // 10)
        epochs_without_improvement = 0
        training_t0 = time.time()
        pbar = tqdm(range(config.max_epochs), desc="Training", unit="epoch")
        for epoch in pbar:
            residual_model.train()
            optimizer.zero_grad()

            train_loss, residual_penalty = pipeline.compute_dataset_loss(
                heliostat_data_mapping=train_bundle.heliostat_data_mapping,
                data_parser=parsers.train,
                group_calibration_inputs=train_group_cal,
                device=device,
            )
            objective = train_loss + config.residual_l2_weight * residual_penalty
            objective.backward()
            torch.nn.utils.clip_grad_norm_(residual_model.parameters(), config.grad_clip_max_norm)
            optimizer.step()

            residual_model.eval()
            validation_scenario = _load_scenario(config, device)
            validation_group_states = capture_all_group_parameter_states(validation_scenario, device=device)
            validation_metrics = evaluate_model_tracking_accuracy(
                scenario=validation_scenario,
                residual_model=residual_model,
                group_parameter_states=validation_group_states,
                group_calibration_inputs=validation_group_cal,
                heliostat_data_mapping=validation_bundle.heliostat_data_mapping,
                data_parser=parsers.validation,
                device=device,
                bitmap_resolution=bitmap_resolution,
                ray_tracing_batch_size=config.ray_tracing_batch_size,
            )

            current_lr = optimizer.param_groups[0]["lr"]
            prev_lr = current_lr
            scheduler.step(validation_metrics["mean_focal_spot_error_mrad"])
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < prev_lr:
                log.info(
                    "ReduceLROnPlateau: LR reduced %.2e → %.2e at epoch %d.",
                    prev_lr, new_lr, epoch + 1,
                )

            epoch_record = {
                "epoch": float(epoch),
                "train_loss_m": float(train_loss.item()),
                "residual_l2_penalty": float(residual_penalty.item()),
                "objective": float(objective.item()),
                "learning_rate": current_lr,
                "validation_mean_focal_spot_error_m": float(
                    validation_metrics["mean_focal_spot_error_m"]
                ),
                "validation_mean_focal_spot_error_mrad": float(
                    validation_metrics["mean_focal_spot_error_mrad"]
                ),
                "validation_median_focal_spot_error_mrad": float(
                    validation_metrics["median_focal_spot_error_mrad"]
                ),
            }
            history.append(epoch_record)
            last_epoch_record = epoch_record
            last_validation_metrics = validation_metrics
            pbar.set_postfix({
                "loss": f"{epoch_record['train_loss_m']:.4f}",
                "val_mrad": f"{epoch_record['validation_mean_focal_spot_error_mrad']:.2f}",
            })
            log.info(
                "Epoch %d/%d - train_loss=%.6f m, objective=%.6f, val_mean=%.6f mrad, lr=%.2e",
                epoch + 1,
                config.max_epochs,
                epoch_record["train_loss_m"],
                epoch_record["objective"],
                epoch_record["validation_mean_focal_spot_error_mrad"],
                epoch_record["learning_rate"],
            )

            if (
                best_epoch_record is None
                or epoch_record["validation_mean_focal_spot_error_mrad"]
                < best_epoch_record["validation_mean_focal_spot_error_mrad"]
            ):
                best_epoch_record = dict(epoch_record)
                best_validation_metrics = validation_metrics
                best_model_state_dict = _clone_model_state_dict(residual_model)
                epochs_without_improvement = 0
                log.info(
                    "Selected new best checkpoint at epoch %d with val_mean=%.6f mrad.",
                    epoch + 1,
                    epoch_record["validation_mean_focal_spot_error_mrad"],
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    tqdm.write(
                        f"    early stopping at epoch {epoch + 1} "
                        f"(no improvement for {epochs_without_improvement} epochs)"
                    )
                    pbar.close()
                    break

        training_time_s = time.time() - training_t0

        if (
            best_epoch_record is None
            or best_validation_metrics is None
            or best_model_state_dict is None
            or last_epoch_record is None
            or last_validation_metrics is None
        ):
            raise RuntimeError("Training completed without producing validation metrics.")

        last_model_state_dict = _clone_model_state_dict(residual_model)
        last_test_scenario = _load_scenario(config, device)
        last_test_group_states = capture_all_group_parameter_states(last_test_scenario, device=device)
        last_test_metrics = evaluate_model_tracking_accuracy(
            scenario=last_test_scenario,
            residual_model=residual_model,
            group_parameter_states=last_test_group_states,
            group_calibration_inputs=test_group_cal,
            heliostat_data_mapping=test_bundle.heliostat_data_mapping,
            data_parser=parsers.test,
            device=device,
            bitmap_resolution=bitmap_resolution,
            ray_tracing_batch_size=config.ray_tracing_batch_size,
        )

        _write_json(json_dir / "history.json", history)

        norm_stats_dict = train_bundle.norm_stats.to_dict()
        for tag, state_dict, epoch_record_for_tag in [
            ("best_validation", best_model_state_dict, best_epoch_record),
            ("last_epoch",      last_model_state_dict, last_epoch_record),
        ]:
            suffix = "" if tag == "best_validation" else "_last_epoch"
            _save_model_checkpoint(
                output_path=models_dir / f"linear_residual_model{suffix}.pt",
                model_state_dict=state_dict,
                norm_stats_dict=norm_stats_dict,
                selected_epoch=int(epoch_record_for_tag["epoch"]),
                selected_validation_mean_mrad=float(
                    epoch_record_for_tag["validation_mean_focal_spot_error_mrad"]
                ),
                selection_tag=tag,
            )
            _save_model_checkpoint_json(
                output_path=models_dir / f"linear_residual_model{suffix}.json",
                model_state_dict=state_dict,
                norm_stats_dict=norm_stats_dict,
                selected_epoch=int(epoch_record_for_tag["epoch"]),
                selected_validation_mean_mrad=float(
                    epoch_record_for_tag["validation_mean_focal_spot_error_mrad"]
                ),
                selection_tag=tag,
            )

        residual_model.load_state_dict(best_model_state_dict)
        residual_model.eval()

        best_validation_scenario = _load_scenario(config, device)
        best_validation_group_states = capture_all_group_parameter_states(
            best_validation_scenario, device=device,
        )
        best_validation_metrics = evaluate_model_tracking_accuracy(
            scenario=best_validation_scenario,
            residual_model=residual_model,
            group_parameter_states=best_validation_group_states,
            group_calibration_inputs=validation_group_cal,
            heliostat_data_mapping=validation_bundle.heliostat_data_mapping,
            data_parser=parsers.validation,
            device=device,
            bitmap_resolution=bitmap_resolution,
            ray_tracing_batch_size=config.ray_tracing_batch_size,
        )
        apply_model_to_scenario(
            scenario=best_validation_scenario,
            residual_model=residual_model,
            group_parameter_states=best_validation_group_states,
            group_calibration_inputs=validation_group_cal,
        )

        tqdm.write("[5/6] Evaluating best model on test set...")
        test_eval_t0 = time.time()
        best_test_scenario = _load_scenario(config, device)
        best_test_group_states = capture_all_group_parameter_states(best_test_scenario, device=device)
        best_test_metrics = evaluate_model_tracking_accuracy(
            scenario=best_test_scenario,
            residual_model=residual_model,
            group_parameter_states=best_test_group_states,
            group_calibration_inputs=test_group_cal,
            heliostat_data_mapping=test_bundle.heliostat_data_mapping,
            data_parser=parsers.test,
            device=device,
            bitmap_resolution=bitmap_resolution,
            ray_tracing_batch_size=config.ray_tracing_batch_size,
        )
        test_eval_time_s = time.time() - test_eval_t0

        _write_json(json_dir / "validation_corrected_metrics.json", best_validation_metrics)
        _write_json(json_dir / "validation_corrected_metrics_last_epoch.json", last_validation_metrics)
        _write_json(json_dir / "test_corrected_metrics.json", best_test_metrics)
        _write_json(json_dir / "test_corrected_metrics_last_epoch.json", last_test_metrics)

        overall_time_s = time.time() - overall_t0
        _write_json(
            json_dir / "timing.json",
            {
                "overall_s": round(overall_time_s, 1),
                "overall_min": round(overall_time_s / 60, 2),
                "training_loop_s": round(training_time_s, 1),
                "training_loop_min": round(training_time_s / 60, 2),
                "test_evaluation_s": round(test_eval_time_s, 1),
                "test_evaluation_min": round(test_eval_time_s / 60, 2),
                "peak_gpu_memory_allocated_gb": round(
                    torch.cuda.max_memory_allocated() / 1024 ** 3, 3
                ) if torch.cuda.is_available() else None,
                "peak_gpu_memory_reserved_gb": round(
                    torch.cuda.max_memory_reserved() / 1024 ** 3, 3
                ) if torch.cuda.is_available() else None,
            },
        )
        tqdm.write("[6/6] Saving results and plots...")
        _write_json(
            config.output_dir / "training_summary.json",
            _build_training_summary(
                baseline_validation_metrics=baseline_validation_metrics,
                baseline_test_metrics=baseline_test_metrics,
                best_epoch_record=best_epoch_record,
                best_validation_metrics=best_validation_metrics,
                best_test_metrics=best_test_metrics,
                last_epoch_record=last_epoch_record,
                last_validation_metrics=last_validation_metrics,
                last_test_metrics=last_test_metrics,
            ),
        )

        predicted_residuals = _predict_residual_tensor(
            residual_model=residual_model,
            group_calibration_inputs=test_group_cal,
        )
        all_heliostat_names = [
            name
            for group in scenario.heliostat_field.heliostat_groups
            for name in group.names
        ]
        _write_json(
            json_dir / "predicted_residuals.json",
            {
                "parameter_names": list(SHARED_WORTBERG_PARAMETER_NAMES),
                "residuals": {
                    name: predicted_residuals[i].tolist()
                    for i, name in enumerate(all_heliostat_names)
                },
            },
        )
        plot_loss_curves(
            history=history,
            test_loss_m=float(best_test_metrics["mean_focal_spot_error_m"]),
            output_path=plots_dir / "loss_curve.png",
        )
        plot_baseline_vs_corrected_metrics(
            validation_baseline_metrics=baseline_validation_metrics,
            validation_best_metrics=best_validation_metrics,
            validation_last_metrics=last_validation_metrics,
            test_baseline_metrics=baseline_test_metrics,
            test_best_metrics=best_test_metrics,
            test_last_metrics=last_test_metrics,
            output_path=plots_dir / "baseline_vs_corrected_metrics.png",
        )
        plot_error_histogram(
            baseline_errors_mrad=baseline_test_metrics["all_errors_mrad"],
            corrected_errors_mrad=best_test_metrics["all_errors_mrad"],
            output_path=plots_dir / "error_histogram.png",
        )
        if config.model_type == "linear":
            plot_linear_weights_heatmap(
                linear_weight=best_model_state_dict["linear.weight"],
                linear_bias=best_model_state_dict["linear.bias"],
                parameter_names=SHARED_WORTBERG_PARAMETER_NAMES,
                output_path=plots_dir / "linear_weights_heatmap.png",
            )
        plot_predicted_residual_boxplot(
            predicted_residuals=predicted_residuals,
            parameter_names=SHARED_WORTBERG_PARAMETER_NAMES,
            output_path=plots_dir / "predicted_residual_boxplot.png",
        )
        plot_per_heliostat_improvement_scatter(
            baseline_per_heliostat=baseline_test_metrics["per_heliostat"],
            corrected_per_heliostat=best_test_metrics["per_heliostat"],
            output_path=plots_dir / "per_heliostat_improvement.png",
        )

        plot_response_curves(
            model=residual_model,
            calibration_inputs=train_bundle.calibration_inputs,
            parameter_names=SHARED_WORTBERG_PARAMETER_NAMES,
            output_path=plots_dir / "response_curves.png",
        )

        plot_feature_importance(
            model=residual_model,
            calibration_inputs=train_bundle.calibration_inputs,
            parameter_names=SHARED_WORTBERG_PARAMETER_NAMES,
            output_path=plots_dir / "feature_importance.png",
        )

        save_kinematic_parameters(
            best_validation_scenario,
            models_dir / "corrected_kinematic_parameters_best.json",
        )

        sep = "=" * 60
        tqdm.write(f"\n{sep}")
        tqdm.write("TRAINING COMPLETE")
        tqdm.write(sep)
        tqdm.write(f"  Baseline  val:  {float(baseline_validation_metrics['mean_focal_spot_error_mrad']):.2f} mrad")
        tqdm.write(f"  Best      val:  {float(best_validation_metrics['mean_focal_spot_error_mrad']):.2f} mrad  (epoch {int(best_epoch_record['epoch']) + 1})")
        tqdm.write(f"  Last      val:  {float(last_validation_metrics['mean_focal_spot_error_mrad']):.2f} mrad")
        tqdm.write(f"  Baseline  test: {float(baseline_test_metrics['mean_focal_spot_error_mrad']):.2f} mrad")
        tqdm.write(f"  Best      test: {float(best_test_metrics['mean_focal_spot_error_mrad']):.2f} mrad")
        tqdm.write(f"  Last      test: {float(last_test_metrics['mean_focal_spot_error_mrad']):.2f} mrad")
        tqdm.write(f"  Output:   {config.output_dir}")
        tqdm.write(sep)
    finally:
        for h in _console_handlers:
            h.setLevel(logging.DEBUG)
        for h in _artist_console_handlers:
            h.setLevel(logging.DEBUG)
        _artist_logger.removeHandler(file_handler)
        _artist_logger.propagate = _artist_logger_propagate_orig
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
