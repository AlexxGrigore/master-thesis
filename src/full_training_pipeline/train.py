from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

_pkg = pathlib.Path(__file__).parent
_src = _pkg.parent
# Run this from thesisenv, ideally with the working directory set to src/.
# In that setup, ARTIST and PAINT already resolve to the workspace checkouts.
sys.path.insert(0, str(_src))

import h5py
import torch
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import set_logger_config
from artist.util.environment_setup import get_device

from full_training_pipeline.config import PipelineConfig, build_default_config
from full_training_pipeline.data import build_group_feature_tensors, build_split_bundle
from full_training_pipeline.evaluate import apply_model_to_scenario, evaluate_model_tracking_accuracy
from full_training_pipeline.model import SHARED_WORTBERG_PARAMETER_NAMES, SharedLinearResidualModel
from full_training_pipeline.pipeline import FineErrorLearningPipeline, capture_all_group_parameter_states
from full_training_pipeline.plotting import (
    plot_baseline_vs_corrected_metrics,
    plot_error_histogram,
    plot_linear_weights_heatmap,
    plot_loss_curves,
    plot_per_heliostat_improvement_scatter,
    plot_predicted_residual_boxplot,
)
from utils.checkpointing import load_kinematic_parameters, save_kinematic_parameters
from utils.evaluation import evaluate_flux_accuracy

set_logger_config()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FEL V1 linear residual pipeline.")
    parser.add_argument("--on-daic", action="store_true", help="Use DAIC path defaults.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a minimal local smoke configuration.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the default epoch count.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=None, help="Optional output directory.")
    return parser.parse_args()


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
    feature_names: tuple[str, ...],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    selected_epoch: int,
    selected_validation_mean_mrad: float,
    selection_tag: str,
) -> None:
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "feature_names": feature_names,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
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
    feature_names: tuple[str, ...],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
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
            "feature_names": list(feature_names),
            "parameter_names": list(SHARED_WORTBERG_PARAMETER_NAMES),
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
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


def _predict_residual_tensor(
    *,
    residual_model: torch.nn.Module,
    group_feature_tensors: list[torch.Tensor],
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    with torch.no_grad():
        for group_features in group_feature_tensors:
            rows.append(residual_model(group_features))
    return torch.cat(rows, dim=0)


def _write_pipeline_markdown(
    *,
    output_path: pathlib.Path,
    config: PipelineConfig,
    feature_names: tuple[str, ...],
) -> None:
    feature_lines = "\n".join(f"- `{name}`" for name in feature_names)
    parameter_lines = "\n".join(f"- `{name}`" for name in SHARED_WORTBERG_PARAMETER_NAMES)
    markdown = f"""# Full Training Pipeline

## Overview

This run trains the V1 fine-error-learning baseline for heliostat kinematic correction.
The pipeline starts from a frozen coarse kinematic checkpoint and learns a shared linear residual model that predicts a 20-dimensional Wortberg-style correction vector for every heliostat.

## Inputs

- Scenario file: `{config.scenario_path}`
- Coarse checkpoint: `{config.coarse_checkpoint_path}`
- Benchmark CSV: `{config.benchmark_csv}`
- Calibration properties directory: `{config.calibration_properties_dir}`
- Flux image directory: `{config.flux_image_dir}`
- Train split: `{config.train_split}`
- Validation split: `{config.validation_split}`
- Test split: `{config.test_split}`

## Feature Construction

Each heliostat is represented by an aggregated feature vector computed from its calibration samples in a split.

Per-sample features:

- 3D sun direction derived from elevation and azimuth
- Axis 1 motor position
- Axis 2 motor position

Aggregated heliostat features:

- Mean of each per-sample feature
- Standard deviation of each per-sample feature
- Sample count

Feature names:

{feature_lines}

## Linear Model

The model is a single shared linear layer used for all heliostats.
It maps the normalized heliostat feature vector to a 20-dimensional residual vector.

- Input dimension: {len(feature_names)}
- Output dimension: 20
- Shared across all heliostats: yes
- Output activation: `tanh`
- Output scaling: physical residual bounds per parameter

The output parameters are:

{parameter_lines}

## Training Loop

At each epoch:

1. Load the frozen coarse parameters as the base state.
2. Predict a residual correction per heliostat using the shared linear model.
3. Add the residual to the coarse parameters.
4. Run ARTIST ray tracing with the corrected parameters.
5. Compute focal-spot loss on the training split.
6. Evaluate the current model on the validation split.
7. Keep the checkpoint with the best validation mean tracking error.

## Output Layout

- `training_summary.json`: concise top-level summary
- `training.log`: raw log output
- `pipeline_details.md`: this file
- `plots/`: generated figures
- `json/`: run metadata and metrics
- `models/`: `.pt`, `.json`, and corrected parameter exports

## Run Configuration

- Max epochs: {config.max_epochs}
- Learning rate: {config.learning_rate}
- Weight decay: {config.weight_decay}
- Residual L2 weight: {config.residual_l2_weight}
- Number of rays: {config.number_of_rays}
- Bitmap resolution: {config.bitmap_resolution}
- Surface points per facet: {config.surface_points_per_facet}
- Sample limit per heliostat: {config.sample_limit_per_heliostat}
- Smoke test: {config.smoke_test}
- Running on DAIC: {config.is_on_daic}
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        handle.write(markdown)


def _build_data_parser(config: PipelineConfig) -> PaintCalibrationDataParser:
    return PaintCalibrationDataParser(
        sample_limit=config.sample_limit_per_heliostat,
        centroid_extraction_method=config.centroid_method,
    )


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


def _build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return build_default_config(
        is_on_daic=args.on_daic,
        smoke_test=args.smoke_test,
        max_epochs=args.epochs,
        output_dir=args.output_dir,
    )


def main() -> None:
    args = _parse_args()
    config = _build_config_from_args(args)
    if config.max_epochs < 1:
        raise ValueError("max_epochs must be at least 1.")
    config.output_dir.mkdir(parents=True, exist_ok=True)

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

    try:
        log.info("Starting full training pipeline V1.")
        _write_json(json_dir / "config.json", config.to_jsonable_dict())
        device = get_device()
        bitmap_resolution = torch.tensor(config.bitmap_resolution)
        log.info("Using device: %s", device)

        train_bundle = build_split_bundle(
            benchmark_csv=config.benchmark_csv,
            calibration_properties_dir=config.calibration_properties_dir,
            flux_image_dir=config.flux_image_dir,
            split=config.train_split,
            sample_limit_per_heliostat=config.sample_limit_per_heliostat,
        )
        validation_bundle = build_split_bundle(
            benchmark_csv=config.benchmark_csv,
            calibration_properties_dir=config.calibration_properties_dir,
            flux_image_dir=config.flux_image_dir,
            split=config.validation_split,
            sample_limit_per_heliostat=config.sample_limit_per_heliostat,
            feature_mean=train_bundle.feature_mean,
            feature_std=train_bundle.feature_std,
        )
        test_bundle = build_split_bundle(
            benchmark_csv=config.benchmark_csv,
            calibration_properties_dir=config.calibration_properties_dir,
            flux_image_dir=config.flux_image_dir,
            split=config.test_split,
            sample_limit_per_heliostat=config.sample_limit_per_heliostat,
            feature_mean=train_bundle.feature_mean,
            feature_std=train_bundle.feature_std,
        )

        _write_json(
            json_dir / "feature_normalization.json",
            {
                "feature_names": list(train_bundle.feature_names),
                "feature_mean": train_bundle.feature_mean.tolist(),
                "feature_std": train_bundle.feature_std.tolist(),
            },
        )
        _write_pipeline_markdown(
            output_path=config.output_dir / "pipeline_details.md",
            config=config,
            feature_names=train_bundle.feature_names,
        )

        scenario = _load_scenario(config, device)
        group_parameter_states = capture_all_group_parameter_states(scenario, device=device)
        train_group_features = build_group_feature_tensors(
            scenario,
            feature_summaries=train_bundle.normalized_feature_summaries,
            feature_dim=train_bundle.feature_dim,
            device=device,
        )
        validation_group_features = build_group_feature_tensors(
            scenario,
            feature_summaries=validation_bundle.normalized_feature_summaries,
            feature_dim=validation_bundle.feature_dim,
            device=device,
        )
        test_group_features = build_group_feature_tensors(
            scenario,
            feature_summaries=test_bundle.normalized_feature_summaries,
            feature_dim=test_bundle.feature_dim,
            device=device,
        )

        residual_model = SharedLinearResidualModel(input_dim=train_bundle.feature_dim).to(device)
        pipeline = FineErrorLearningPipeline(
            scenario=scenario,
            residual_model=residual_model,
            group_parameter_states=group_parameter_states,
            bitmap_resolution=bitmap_resolution,
            ray_tracing_batch_size=config.ray_tracing_batch_size,
        )
        optimizer = torch.optim.Adam(
            residual_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        data_parser = _build_data_parser(config)
        history: list[dict[str, float]] = []

        baseline_validation_scenario = _load_scenario(config, device)
        baseline_validation_metrics = evaluate_flux_accuracy(
            scenario=baseline_validation_scenario,
            heliostat_data_mapping=validation_bundle.heliostat_data_mapping,
            data_parser=data_parser,
            device=device,
            bitmap_resolution=bitmap_resolution,
            ray_tracing_batch_size=config.ray_tracing_batch_size,
        )
        baseline_test_scenario = _load_scenario(config, device)
        baseline_test_metrics = evaluate_flux_accuracy(
            scenario=baseline_test_scenario,
            heliostat_data_mapping=test_bundle.heliostat_data_mapping,
            data_parser=data_parser,
            device=device,
            bitmap_resolution=bitmap_resolution,
            ray_tracing_batch_size=config.ray_tracing_batch_size,
        )
        _write_json(json_dir / "validation_baseline_metrics.json", baseline_validation_metrics)
        _write_json(json_dir / "test_baseline_metrics.json", baseline_test_metrics)

        best_epoch_record: dict[str, float] | None = None
        best_validation_metrics: dict[str, object] | None = None
        best_model_state_dict: dict[str, torch.Tensor] | None = None
        last_epoch_record: dict[str, float] | None = None
        last_validation_metrics: dict[str, object] | None = None

        for epoch in range(config.max_epochs):
            residual_model.train()
            optimizer.zero_grad()

            train_loss, residual_penalty = pipeline.compute_dataset_loss(
                heliostat_data_mapping=train_bundle.heliostat_data_mapping,
                data_parser=data_parser,
                group_feature_tensors=train_group_features,
                device=device,
            )
            objective = train_loss + config.residual_l2_weight * residual_penalty
            objective.backward()
            optimizer.step()

            residual_model.eval()
            validation_scenario = _load_scenario(config, device)
            validation_group_states = capture_all_group_parameter_states(validation_scenario, device=device)
            validation_metrics = evaluate_model_tracking_accuracy(
                scenario=validation_scenario,
                residual_model=residual_model,
                group_parameter_states=validation_group_states,
                group_feature_tensors=validation_group_features,
                heliostat_data_mapping=validation_bundle.heliostat_data_mapping,
                data_parser=data_parser,
                device=device,
                bitmap_resolution=bitmap_resolution,
                ray_tracing_batch_size=config.ray_tracing_batch_size,
            )

            epoch_record = {
                "epoch": float(epoch),
                "train_loss_m": float(train_loss.item()),
                "residual_l2_penalty": float(residual_penalty.item()),
                "objective": float(objective.item()),
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
            log.info(
                "Epoch %d/%d - train_loss=%.6f m, objective=%.6f, val_mean=%.6f mrad",
                epoch + 1,
                config.max_epochs,
                epoch_record["train_loss_m"],
                epoch_record["objective"],
                epoch_record["validation_mean_focal_spot_error_mrad"],
            )

            if (
                best_epoch_record is None
                or epoch_record["validation_mean_focal_spot_error_mrad"]
                < best_epoch_record["validation_mean_focal_spot_error_mrad"]
            ):
                best_epoch_record = dict(epoch_record)
                best_validation_metrics = validation_metrics
                best_model_state_dict = _clone_model_state_dict(residual_model)
                log.info(
                    "Selected new best checkpoint at epoch %d with val_mean=%.6f mrad.",
                    epoch + 1,
                    epoch_record["validation_mean_focal_spot_error_mrad"],
                )

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
            group_feature_tensors=test_group_features,
            heliostat_data_mapping=test_bundle.heliostat_data_mapping,
            data_parser=data_parser,
            device=device,
            bitmap_resolution=bitmap_resolution,
            ray_tracing_batch_size=config.ray_tracing_batch_size,
        )

        _write_json(json_dir / "history.json", history)

        _save_model_checkpoint(
            output_path=models_dir / "linear_residual_model.pt",
            model_state_dict=best_model_state_dict,
            feature_names=train_bundle.feature_names,
            feature_mean=train_bundle.feature_mean,
            feature_std=train_bundle.feature_std,
            selected_epoch=int(best_epoch_record["epoch"]),
            selected_validation_mean_mrad=float(
                best_epoch_record["validation_mean_focal_spot_error_mrad"]
            ),
            selection_tag="best_validation",
        )
        _save_model_checkpoint_json(
            output_path=models_dir / "linear_residual_model.json",
            model_state_dict=best_model_state_dict,
            feature_names=train_bundle.feature_names,
            feature_mean=train_bundle.feature_mean,
            feature_std=train_bundle.feature_std,
            selected_epoch=int(best_epoch_record["epoch"]),
            selected_validation_mean_mrad=float(
                best_epoch_record["validation_mean_focal_spot_error_mrad"]
            ),
            selection_tag="best_validation",
        )
        _save_model_checkpoint(
            output_path=models_dir / "linear_residual_model_last_epoch.pt",
            model_state_dict=last_model_state_dict,
            feature_names=train_bundle.feature_names,
            feature_mean=train_bundle.feature_mean,
            feature_std=train_bundle.feature_std,
            selected_epoch=int(last_epoch_record["epoch"]),
            selected_validation_mean_mrad=float(
                last_epoch_record["validation_mean_focal_spot_error_mrad"]
            ),
            selection_tag="last_epoch",
        )
        _save_model_checkpoint_json(
            output_path=models_dir / "linear_residual_model_last_epoch.json",
            model_state_dict=last_model_state_dict,
            feature_names=train_bundle.feature_names,
            feature_mean=train_bundle.feature_mean,
            feature_std=train_bundle.feature_std,
            selected_epoch=int(last_epoch_record["epoch"]),
            selected_validation_mean_mrad=float(
                last_epoch_record["validation_mean_focal_spot_error_mrad"]
            ),
            selection_tag="last_epoch",
        )

        residual_model.load_state_dict(best_model_state_dict)
        residual_model.eval()

        best_validation_scenario = _load_scenario(config, device)
        best_validation_group_states = capture_all_group_parameter_states(
            best_validation_scenario,
            device=device,
        )
        best_validation_metrics = evaluate_model_tracking_accuracy(
            scenario=best_validation_scenario,
            residual_model=residual_model,
            group_parameter_states=best_validation_group_states,
            group_feature_tensors=validation_group_features,
            heliostat_data_mapping=validation_bundle.heliostat_data_mapping,
            data_parser=data_parser,
            device=device,
            bitmap_resolution=bitmap_resolution,
            ray_tracing_batch_size=config.ray_tracing_batch_size,
        )
        apply_model_to_scenario(
            scenario=best_validation_scenario,
            residual_model=residual_model,
            group_parameter_states=best_validation_group_states,
            group_feature_tensors=validation_group_features,
        )

        best_test_scenario = _load_scenario(config, device)
        best_test_group_states = capture_all_group_parameter_states(best_test_scenario, device=device)
        best_test_metrics = evaluate_model_tracking_accuracy(
            scenario=best_test_scenario,
            residual_model=residual_model,
            group_parameter_states=best_test_group_states,
            group_feature_tensors=test_group_features,
            heliostat_data_mapping=test_bundle.heliostat_data_mapping,
            data_parser=data_parser,
            device=device,
            bitmap_resolution=bitmap_resolution,
            ray_tracing_batch_size=config.ray_tracing_batch_size,
        )

        _write_json(json_dir / "validation_corrected_metrics.json", best_validation_metrics)
        _write_json(json_dir / "validation_corrected_metrics_last_epoch.json", last_validation_metrics)
        _write_json(json_dir / "test_corrected_metrics.json", best_test_metrics)
        _write_json(json_dir / "test_corrected_metrics_last_epoch.json", last_test_metrics)
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
            group_feature_tensors=test_group_features,
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
        plot_linear_weights_heatmap(
            linear_weight=best_model_state_dict["linear.weight"],
            linear_bias=best_model_state_dict["linear.bias"],
            feature_names=train_bundle.feature_names,
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

        save_kinematic_parameters(
            best_validation_scenario,
            models_dir / "corrected_kinematic_parameters_best.json",
        )
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


if __name__ == "__main__":
    main()
