from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import logging
import math
import pathlib
import sys
from typing import Any

_pkg = pathlib.Path(__file__).parent
_experiments = _pkg.parent
_src = _experiments.parent
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_experiments))
sys.path.insert(0, str(_pkg))

import h5py
import numpy as np
import torch

from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.loss_functions import FocalSpotLoss
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, index_mapping
from artist.util.environment_setup import setup_distributed_environment
from artist.util.utils import get_center_of_mass

from plotting import (
    plot_accuracy_bucket_comparison,
    plot_accuracy_bucket_pies,
    plot_convergence,
    plot_representative_heliostat_parameter_comparison,
    plot_stage_comparison,
)
from utils.checkpointing import save_kinematic_parameters
from utils.plotting import plot_tracking_error_histogram

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Perturbation:
    parameter: str
    value: float
    index: int | None = None
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "value": self.value,
            "index": self.index,
            "label": self.label,
        }


@dataclass(frozen=True)
class RecoveryExperiment:
    name: str
    reconstructor_cls: type
    perturbations: tuple[Perturbation, ...]
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reconstructor": self.reconstructor_cls.__name__,
            "description": self.description,
            "perturbations": [perturbation.to_dict() for perturbation in self.perturbations],
        }


@dataclass
class KinematicState:
    groups: list[dict[str, torch.Tensor]] = field(default_factory=list)


def write_json(output_path: pathlib.Path, data: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(data, handle, indent=2)


@contextmanager
def stage_log_handler(log_path: pathlib.Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
    logging.getLogger().addHandler(handler)
    try:
        yield
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def filter_mapping_for_smoke_test(
    heliostat_data_mapping,
    heliostat_id: str,
    sample_limit: int,
):
    return [
        (name, calibration_paths[:sample_limit], flux_paths[:sample_limit])
        for name, calibration_paths, flux_paths in heliostat_data_mapping
        if name == heliostat_id
    ]


def load_fresh_scenario(
    device: torch.device,
    scenario_path: pathlib.Path,
    scenario_num_rays: int,
) -> Scenario:
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file,
            device=device,
            number_of_surface_points_per_facet=torch.tensor([25, 25]),
        )
    scenario.set_number_of_rays(scenario_num_rays)
    return scenario


def snapshot_kinematic_state(scenario: Scenario) -> KinematicState:
    groups: list[dict[str, torch.Tensor]] = []
    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        kinematic = heliostat_group.kinematics
        group_state = {
            "translation": kinematic.translation_deviation_parameters.detach().cpu().clone(),
            "rotation": kinematic.rotation_deviation_parameters.detach().cpu().clone(),
            "actuator_optimizable": kinematic.actuators.optimizable_parameters.detach().cpu().clone(),
            "actuator_nonoptimizable": kinematic.actuators.non_optimizable_parameters.detach().cpu().clone(),
        }
        if hasattr(kinematic, "_base_position_deviation"):
            group_state["base_position"] = kinematic._base_position_deviation.detach().cpu().clone()
        groups.append(group_state)
    return KinematicState(groups=groups)


def restore_kinematic_state(scenario: Scenario, state: KinematicState, device: torch.device) -> None:
    with torch.no_grad():
        for heliostat_group, group_state in zip(scenario.heliostat_field.heliostat_groups, state.groups):
            kinematic = heliostat_group.kinematics
            kinematic.translation_deviation_parameters.data = group_state["translation"].to(device).clone()
            kinematic.rotation_deviation_parameters.data = group_state["rotation"].to(device).clone()
            kinematic.actuators.optimizable_parameters.data = (
                group_state["actuator_optimizable"].to(device).clone()
            )
            kinematic.actuators.non_optimizable_parameters.data = (
                group_state["actuator_nonoptimizable"].to(device).clone()
            )
            if "base_position" in group_state:
                kinematic._base_position_deviation = group_state["base_position"].to(device).clone()


def apply_perturbation(scenario: Scenario, perturbation: Perturbation, device: torch.device) -> None:
    with torch.no_grad():
        for heliostat_group in scenario.heliostat_field.heliostat_groups:
            kinematic = heliostat_group.kinematics
            if perturbation.parameter == "rotation":
                kinematic.rotation_deviation_parameters[:, perturbation.index] += perturbation.value
            elif perturbation.parameter == "translation":
                kinematic.translation_deviation_parameters[:, perturbation.index] += perturbation.value
            elif perturbation.parameter == "actuator_angle":
                kinematic.actuators.optimizable_parameters[
                    :, index_mapping.actuator_initial_angle, :
                ] += perturbation.value
            elif perturbation.parameter == "actuator_offset":
                kinematic.actuators.non_optimizable_parameters[
                    :, index_mapping.actuator_offset, :
                ] += perturbation.value
            elif perturbation.parameter == "base_position":
                if not hasattr(kinematic, "_base_position_deviation"):
                    kinematic._base_position_deviation = torch.zeros(
                        kinematic.number_of_heliostats, 3, device=device
                    )
                kinematic._base_position_deviation[:, perturbation.index] += perturbation.value
            else:
                raise ValueError(f"Unsupported perturbation parameter: {perturbation.parameter}")


def apply_experiment_perturbations(
    scenario: Scenario,
    experiment: RecoveryExperiment,
    device: torch.device,
) -> None:
    for perturbation in experiment.perturbations:
        apply_perturbation(scenario, perturbation, device)


def inject_base_position_deviation_if_present(
    heliostat_group,
    active_heliostats_mask: torch.Tensor,
    device: torch.device,
) -> None:
    kinematic = heliostat_group.kinematics
    if not hasattr(kinematic, "_base_position_deviation"):
        return
    repeat_counts = active_heliostats_mask.to(dtype=torch.int64)
    active_base_dev = kinematic._base_position_deviation.repeat_interleave(repeat_counts, dim=0)
    pad = torch.zeros(active_base_dev.shape[0], 1, device=device)
    kinematic.active_heliostat_positions = (
        kinematic.active_heliostat_positions + torch.cat([active_base_dev, pad], dim=1)
    )


@torch.no_grad()
def evaluate_tracking_accuracy(
    scenario: Scenario,
    heliostat_data_mapping,
    data_parser: PaintCalibrationDataParser,
    device: torch.device,
    bitmap_resolution: torch.Tensor,
    ray_tracing_batch_size: int,
) -> dict[str, Any]:
    all_errors_m: list[float] = []
    all_errors_mrad: list[float] = []
    per_heliostat: dict[str, dict[str, float | None]] = {}
    nan_heliostat_ids: set[str] = set()

    reference_target = scenario.target_areas.centers[:, :3].mean(dim=0).to(device)

    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        (
            _measured_flux,
            focal_spots_measured,
            incident_ray_directions,
            _,
            active_heliostats_mask,
            target_area_mask,
        ) = data_parser.parse_data_for_reconstruction(
            heliostat_data_mapping=heliostat_data_mapping,
            heliostat_group=heliostat_group,
            scenario=scenario,
            bitmap_resolution=bitmap_resolution,
            device=device,
        )

        if active_heliostats_mask.sum() == 0:
            continue

        heliostat_group.activate_heliostats(
            active_heliostats_mask=active_heliostats_mask,
            device=device,
        )
        inject_base_position_deviation_if_present(
            heliostat_group=heliostat_group,
            active_heliostats_mask=active_heliostats_mask,
            device=device,
        )
        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=scenario.target_areas.centers[target_area_mask],
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            device=device,
        )

        ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=heliostat_group,
            blocking_active=False,
            batch_size=min(heliostat_group.number_of_active_heliostats, ray_tracing_batch_size),
            bitmap_resolution=bitmap_resolution.to(device),
        )
        predicted_flux = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            target_area_indices=target_area_mask,
            device=device,
        )

        # The RestrictedDistributedSampler truncates to floor(total/N_unique)*N_unique
        # when total samples are not evenly divisible by the number of unique heliostats
        # (e.g. EVAL_SAMPLE_LIMIT=30 but some heliostats have fewer images).
        # predicted_flux therefore has fewer rows than target_area_mask.
        # Use get_sampler_indices() to subset everything to the actually-traced instances.
        sample_indices = ray_tracer.get_sampler_indices()
        target_area_mask_sampled = target_area_mask[sample_indices]
        focal_spots_measured_sampled = focal_spots_measured[sample_indices]

        target_centers = scenario.target_areas.centers[target_area_mask_sampled]
        target_widths = scenario.target_areas.dimensions[target_area_mask_sampled][:, index_mapping.target_area_width]
        target_heights = scenario.target_areas.dimensions[target_area_mask_sampled][:, index_mapping.target_area_height]
        focal_spots_predicted = get_center_of_mass(
            bitmaps=predicted_flux,
            target_centers=target_centers,
            target_widths=target_widths,
            target_heights=target_heights,
            device=device,
        )

        focal_spot_error_m = torch.norm(
            focal_spots_predicted[:, :3] - focal_spots_measured_sampled[:, :3], dim=1
        )
        all_errors_m.extend(focal_spot_error_m.cpu().tolist())

        active_indices = torch.where(active_heliostats_mask > 0)[0]
        active_positions = heliostat_group.positions[active_indices, :3].to(device)
        distances = torch.norm(active_positions - reference_target.unsqueeze(0), dim=1)

        num_active = active_indices.shape[0]
        num_samples = focal_spot_error_m.shape[0]
        samples_per_heliostat = max(num_samples // max(num_active, 1), 1)
        distances_per_sample = distances.repeat_interleave(samples_per_heliostat)[:num_samples]
        focal_spot_error_mrad = (focal_spot_error_m / distances_per_sample) * 1000.0
        focal_spot_error_mrad_np = focal_spot_error_mrad.detach().cpu().numpy()
        all_errors_mrad.extend(focal_spot_error_mrad_np.tolist())

        for i, active_idx in enumerate(active_indices.tolist()):
            heliostat_name = heliostat_group.names[active_idx]
            start = i * samples_per_heliostat
            end = min(start + samples_per_heliostat, num_samples)
            heliostat_errors = focal_spot_error_mrad_np[start:end]
            finite_errors = heliostat_errors[np.isfinite(heliostat_errors)]
            if finite_errors.size == 0:
                nan_heliostat_ids.add(heliostat_name)
                mean_error = None
            else:
                mean_error = float(np.mean(finite_errors))
            per_heliostat[heliostat_name] = {"focal_spot_error_mrad": mean_error}

    valid_errors_m = [value for value in all_errors_m if not math.isnan(value)]
    valid_errors_mrad = [value for value in all_errors_mrad if not math.isnan(value)]

    if not valid_errors_mrad:
        return {
            "mean_focal_spot_error_mrad": float("inf"),
            "median_focal_spot_error_mrad": float("inf"),
            "min_focal_spot_error_mrad": float("inf"),
            "max_focal_spot_error_mrad": float("inf"),
            "mean_focal_spot_error_m": float("inf"),
            "num_samples_evaluated": 0,
            "num_nan_samples": len(all_errors_mrad),
            "nan_heliostat_ids": sorted(nan_heliostat_ids),
            "all_errors_mrad": all_errors_mrad,
            "per_heliostat": per_heliostat,
        }

    return {
        "mean_focal_spot_error_mrad": float(np.mean(valid_errors_mrad)),
        "median_focal_spot_error_mrad": float(np.median(valid_errors_mrad)),
        "min_focal_spot_error_mrad": float(np.min(valid_errors_mrad)),
        "max_focal_spot_error_mrad": float(np.max(valid_errors_mrad)),
        "mean_focal_spot_error_m": float(np.mean(valid_errors_m)),
        "num_samples_evaluated": len(valid_errors_m),
        "num_nan_samples": len(all_errors_mrad) - len(valid_errors_mrad),
        "nan_heliostat_ids": sorted(nan_heliostat_ids),
        "all_errors_mrad": all_errors_mrad,
        "per_heliostat": per_heliostat,
    }


def metrics_to_jsonable(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_focal_spot_error_mrad": metrics["mean_focal_spot_error_mrad"],
        "median_focal_spot_error_mrad": metrics["median_focal_spot_error_mrad"],
        "min_focal_spot_error_mrad": metrics["min_focal_spot_error_mrad"],
        "max_focal_spot_error_mrad": metrics["max_focal_spot_error_mrad"],
        "mean_focal_spot_error_m": metrics["mean_focal_spot_error_m"],
        "num_samples_evaluated": metrics["num_samples_evaluated"],
        "num_nan_samples": metrics["num_nan_samples"],
        "nan_heliostat_ids": metrics["nan_heliostat_ids"],
        "per_heliostat": metrics["per_heliostat"],
    }


def mean_valid_final_loss(final_loss_per_heliostat: torch.Tensor) -> float:
    valid_losses = final_loss_per_heliostat[final_loss_per_heliostat != float("inf")]
    return valid_losses.mean().item() if valid_losses.numel() > 0 else float("inf")


def build_reconstructor_data(train_parser, train_mapping, eval_parser, validation_mapping):
    data = {
        config_dictionary.data_parser: train_parser,
        config_dictionary.heliostat_data_mapping: train_mapping,
    }
    eval_data = {
        config_dictionary.data_parser: eval_parser,
        config_dictionary.heliostat_data_mapping: validation_mapping,
    }
    return data, eval_data


def train_reconstructor(
    scenario: Scenario,
    reconstructor_cls: type,
    ddp_setup,
    device: torch.device,
    train_parser: PaintCalibrationDataParser,
    train_mapping,
    eval_parser: PaintCalibrationDataParser,
    validation_mapping,
    optimization_config: dict[str, Any],
    output_dir: pathlib.Path,
    artifact_prefix: str,
) -> tuple[float, list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data, eval_data = build_reconstructor_data(
        train_parser=train_parser,
        train_mapping=train_mapping,
        eval_parser=eval_parser,
        validation_mapping=validation_mapping,
    )

    with stage_log_handler(output_dir / f"{artifact_prefix}_training.log"):
        reconstructor = reconstructor_cls(
            ddp_setup=ddp_setup,
            scenario=scenario,
            data=data,
            optimization_configuration=optimization_config,
            reconstruction_method=config_dictionary.kinematics_reconstruction_raytracing,
            eval_data=eval_data,
        )
        loss_fn = FocalSpotLoss(scenario=scenario)
        final_loss = reconstructor.reconstruct_kinematics(loss_definition=loss_fn, device=device)
        history = list(reconstructor._convergence_history)

    write_json(output_dir / f"{artifact_prefix}_convergence_history.json", history)
    plot_convergence(
        history=history,
        output_path=output_dir / f"{artifact_prefix}_convergence.png",
        title=f"{reconstructor_cls.__name__} — {artifact_prefix}",
    )
    return mean_valid_final_loss(final_loss), history


def evaluate_and_save(
    scenario: Scenario,
    heliostat_data_mapping,
    data_parser: PaintCalibrationDataParser,
    device: torch.device,
    output_dir: pathlib.Path,
    artifact_prefix: str,
    title: str,
    bitmap_resolution: torch.Tensor,
    ray_tracing_batch_size: int,
) -> dict[str, Any]:
    metrics = evaluate_tracking_accuracy(
        scenario=scenario,
        heliostat_data_mapping=heliostat_data_mapping,
        data_parser=data_parser,
        device=device,
        bitmap_resolution=bitmap_resolution,
        ray_tracing_batch_size=ray_tracing_batch_size,
    )
    write_json(output_dir / f"{artifact_prefix}_metrics.json", metrics_to_jsonable(metrics))
    plot_tracking_error_histogram(
        errors_mrad=metrics["all_errors_mrad"],
        output_path=output_dir / f"{artifact_prefix}_tracking_error_histogram.png",
        title=title,
    )
    return metrics


def compute_recovery_summary(
    baseline_metrics: dict[str, Any],
    perturbed_metrics: dict[str, Any],
    recovered_metrics: dict[str, Any],
) -> dict[str, Any]:
    baseline_mean = baseline_metrics["mean_focal_spot_error_mrad"]
    perturbed_mean = perturbed_metrics["mean_focal_spot_error_mrad"]
    recovered_mean = recovered_metrics["mean_focal_spot_error_mrad"]
    baseline_median = baseline_metrics["median_focal_spot_error_mrad"]
    perturbed_median = perturbed_metrics["median_focal_spot_error_mrad"]
    recovered_median = recovered_metrics["median_focal_spot_error_mrad"]

    mean_degradation = perturbed_mean - baseline_mean
    mean_recovery = perturbed_mean - recovered_mean
    median_degradation = perturbed_median - baseline_median
    median_recovery = perturbed_median - recovered_median

    mean_recovered_fraction = mean_recovery / mean_degradation if mean_degradation > 0 else None
    median_recovered_fraction = median_recovery / median_degradation if median_degradation > 0 else None

    return {
        "baseline_mean_mrad": baseline_mean,
        "perturbed_mean_mrad": perturbed_mean,
        "recovered_mean_mrad": recovered_mean,
        "baseline_median_mrad": baseline_median,
        "perturbed_median_mrad": perturbed_median,
        "recovered_median_mrad": recovered_median,
        "mean_degradation_mrad": mean_degradation,
        "mean_recovery_mrad": mean_recovery,
        "mean_remaining_gap_to_baseline_mrad": recovered_mean - baseline_mean,
        "mean_recovered_fraction": mean_recovered_fraction,
        "median_degradation_mrad": median_degradation,
        "median_recovery_mrad": median_recovery,
        "median_remaining_gap_to_baseline_mrad": recovered_median - baseline_median,
        "median_recovered_fraction": median_recovered_fraction,
    }


def bucket_label(error_mrad: float) -> str:
    if error_mrad < 3.0:
        return "<3 mrad"
    if error_mrad < 5.0:
        return "3-5 mrad"
    if error_mrad < 7.0:
        return "5-7 mrad"
    return ">=7 mrad"


def compute_accuracy_bucket_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    counts = {
        "<3 mrad": 0,
        "3-5 mrad": 0,
        "5-7 mrad": 0,
        ">=7 mrad": 0,
    }
    missing_heliostats: list[str] = []

    for heliostat_id, heliostat_metrics in metrics["per_heliostat"].items():
        error_mrad = heliostat_metrics.get("focal_spot_error_mrad")
        if error_mrad is None or math.isnan(error_mrad):
            missing_heliostats.append(heliostat_id)
            continue
        counts[bucket_label(float(error_mrad))] += 1

    return {
        "counts": counts,
        "num_heliostats_with_finite_error": int(sum(counts.values())),
        "num_missing_heliostats": len(missing_heliostats),
        "missing_heliostat_ids": sorted(missing_heliostats),
    }


def build_per_heliostat_stage_comparison(
    baseline_metrics: dict[str, Any],
    perturbed_metrics: dict[str, Any],
    recovered_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    heliostat_ids = sorted(
        set(baseline_metrics["per_heliostat"])
        | set(perturbed_metrics["per_heliostat"])
        | set(recovered_metrics["per_heliostat"])
    )

    rows: list[dict[str, Any]] = []
    for heliostat_id in heliostat_ids:
        baseline_error = baseline_metrics["per_heliostat"].get(heliostat_id, {}).get("focal_spot_error_mrad")
        perturbed_error = perturbed_metrics["per_heliostat"].get(heliostat_id, {}).get("focal_spot_error_mrad")
        recovered_error = recovered_metrics["per_heliostat"].get(heliostat_id, {}).get("focal_spot_error_mrad")

        rows.append(
            {
                "heliostat_id": heliostat_id,
                "baseline_error_mrad": baseline_error,
                "perturbed_error_mrad": perturbed_error,
                "recovered_error_mrad": recovered_error,
                "baseline_bucket": (
                    bucket_label(float(baseline_error))
                    if baseline_error is not None and not math.isnan(baseline_error)
                    else None
                ),
                "perturbed_bucket": (
                    bucket_label(float(perturbed_error))
                    if perturbed_error is not None and not math.isnan(perturbed_error)
                    else None
                ),
                "recovered_bucket": (
                    bucket_label(float(recovered_error))
                    if recovered_error is not None and not math.isnan(recovered_error)
                    else None
                ),
            }
        )

    return rows


def parameter_group_specs_for_experiment(experiment: RecoveryExperiment) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    perturbation_types = {perturbation.parameter for perturbation in experiment.perturbations}

    if "rotation" in perturbation_types:
        specs.append(
            {
                "key": "rotation_deviation",
                "labels": ["joint1_tilt_n", "joint1_tilt_u", "joint2_tilt_e", "joint2_tilt_n"],
                "unit": "rad",
            }
        )
    if "translation" in perturbation_types:
        specs.append(
            {
                "key": "translation_deviation",
                "labels": [
                    "joint1_trans_e",
                    "joint1_trans_n",
                    "joint1_trans_u",
                    "joint2_trans_e",
                    "joint2_trans_n",
                    "joint2_trans_u",
                    "conc_trans_e",
                    "conc_trans_n",
                    "conc_trans_u",
                ],
                "unit": "m",
            }
        )
    if "actuator_angle" in perturbation_types:
        specs.append(
            {
                "key": "actuator_initial_angle",
                "labels": ["actuator_0", "actuator_1"],
                "unit": "rad",
            }
        )
    if "actuator_offset" in perturbation_types:
        specs.append(
            {
                "key": "actuator_offset",
                "labels": ["actuator_0", "actuator_1"],
                "unit": "m",
            }
        )
    if "base_position" in perturbation_types:
        specs.append(
            {
                "key": "base_position_deviation",
                "labels": ["base_pos_e", "base_pos_n", "base_pos_u"],
                "unit": "m",
            }
        )

    return specs


def extract_heliostat_parameter_groups(
    kinematic,
    heliostat_index: int,
    parameter_group_specs: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    extracted: dict[str, dict[str, float]] = {}

    for spec in parameter_group_specs:
        if spec["key"] == "rotation_deviation":
            values = kinematic.rotation_deviation_parameters[heliostat_index].detach().cpu().tolist()
        elif spec["key"] == "translation_deviation":
            values = kinematic.translation_deviation_parameters[heliostat_index].detach().cpu().tolist()
        elif spec["key"] == "actuator_initial_angle":
            values = (
                kinematic.actuators.optimizable_parameters[
                    heliostat_index, index_mapping.actuator_initial_angle, :
                ]
                .detach()
                .cpu()
                .tolist()
            )
        elif spec["key"] == "actuator_offset":
            values = (
                kinematic.actuators.non_optimizable_parameters[
                    heliostat_index, index_mapping.actuator_offset, :
                ]
                .detach()
                .cpu()
                .tolist()
            )
        elif spec["key"] == "base_position_deviation":
            if hasattr(kinematic, "_base_position_deviation"):
                values = kinematic._base_position_deviation[heliostat_index].detach().cpu().tolist()
            else:
                values = [0.0] * len(spec["labels"])
        else:
            raise ValueError(f"Unsupported parameter group: {spec['key']}")

        extracted[spec["key"]] = {
            label: float(value) for label, value in zip(spec["labels"], values)
        }

    return extracted


def collect_parameter_values_for_all_heliostats(
    scenario: Scenario,
    experiment: RecoveryExperiment,
) -> dict[str, dict[str, dict[str, float]]]:
    parameter_group_specs = parameter_group_specs_for_experiment(experiment)
    parameter_values: dict[str, dict[str, dict[str, float]]] = {}

    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        kinematic = heliostat_group.kinematics
        for heliostat_index, heliostat_id in enumerate(heliostat_group.names):
            parameter_values[heliostat_id] = extract_heliostat_parameter_groups(
                kinematic=kinematic,
                heliostat_index=heliostat_index,
                parameter_group_specs=parameter_group_specs,
            )

    return parameter_values


def select_representative_heliostats_by_bucket(metrics: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    bucket_order = ["<3 mrad", "3-5 mrad", "5-7 mrad", ">=7 mrad"]
    candidates: dict[str, list[tuple[str, float]]] = {bucket: [] for bucket in bucket_order}

    for heliostat_id, heliostat_metrics in metrics["per_heliostat"].items():
        error_mrad = heliostat_metrics.get("focal_spot_error_mrad")
        if error_mrad is None or math.isnan(error_mrad):
            continue
        candidates[bucket_label(float(error_mrad))].append((heliostat_id, float(error_mrad)))

    selected: dict[str, dict[str, Any] | None] = {}
    for bucket in bucket_order:
        bucket_candidates = sorted(candidates[bucket], key=lambda item: item[1])
        if not bucket_candidates:
            selected[bucket] = None
            continue
        representative_id, representative_error = bucket_candidates[len(bucket_candidates) // 2]
        selected[bucket] = {
            "heliostat_id": representative_id,
            "reference_error_mrad": representative_error,
        }

    return selected


def run_recovery_benchmark_experiment(
    *,
    active_experiment: RecoveryExperiment,
    output_dir: pathlib.Path,
    scenario_path: pathlib.Path,
    scenario_num_rays: int,
    number_of_heliostat_groups: int,
    optimization_config: dict[str, Any],
    train_mapping,
    validation_mapping,
    evaluation_mapping,
    train_sample_limit: int,
    eval_sample_limit: int,
    centroid_extraction_method: str,
    bitmap_resolution: torch.Tensor,
    ray_tracing_batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    experiment_dir = output_dir / active_experiment.name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_json(experiment_dir / "experiment_definition.json", active_experiment.to_dict())
    write_json(experiment_dir / "optimization_configuration.json", optimization_config)

    train_parser = PaintCalibrationDataParser(
        sample_limit=train_sample_limit,
        centroid_extraction_method=centroid_extraction_method,
    )
    eval_parser = PaintCalibrationDataParser(
        sample_limit=eval_sample_limit,
        centroid_extraction_method=centroid_extraction_method,
    )

    with setup_distributed_environment(
        number_of_heliostat_groups=number_of_heliostat_groups,
        device=device,
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        scenario = load_fresh_scenario(device=device, scenario_path=scenario_path, scenario_num_rays=scenario_num_rays)
        save_kinematic_parameters(scenario, experiment_dir / "initial_kinematic_parameters.json")

        reference_train_loss_mean, _reference_history = train_reconstructor(
            scenario=scenario,
            reconstructor_cls=active_experiment.reconstructor_cls,
            ddp_setup=ddp_setup,
            device=device,
            train_parser=train_parser,
            train_mapping=train_mapping,
            eval_parser=eval_parser,
            validation_mapping=validation_mapping,
            optimization_config=optimization_config,
            output_dir=experiment_dir,
            artifact_prefix="reference",
        )
        baseline_metrics = evaluate_and_save(
            scenario=scenario,
            heliostat_data_mapping=evaluation_mapping,
            data_parser=eval_parser,
            device=device,
            output_dir=experiment_dir,
            artifact_prefix="baseline",
            title=f"{active_experiment.name} — baseline",
            bitmap_resolution=bitmap_resolution,
            ray_tracing_batch_size=ray_tracing_batch_size,
        )
        reference_state = snapshot_kinematic_state(scenario)
        baseline_parameter_values = collect_parameter_values_for_all_heliostats(
            scenario=scenario,
            experiment=active_experiment,
        )
        save_kinematic_parameters(scenario, experiment_dir / "baseline_kinematic_parameters.json")

        restore_kinematic_state(scenario=scenario, state=reference_state, device=device)
        apply_experiment_perturbations(
            scenario=scenario,
            experiment=active_experiment,
            device=device,
        )
        perturbed_parameter_values = collect_parameter_values_for_all_heliostats(
            scenario=scenario,
            experiment=active_experiment,
        )
        save_kinematic_parameters(scenario, experiment_dir / "perturbed_kinematic_parameters.json")
        perturbed_metrics = evaluate_and_save(
            scenario=scenario,
            heliostat_data_mapping=evaluation_mapping,
            data_parser=eval_parser,
            device=device,
            output_dir=experiment_dir,
            artifact_prefix="perturbed",
            title=f"{active_experiment.name} — perturbed",
            bitmap_resolution=bitmap_resolution,
            ray_tracing_batch_size=ray_tracing_batch_size,
        )

        recovery_train_loss_mean, _recovery_history = train_reconstructor(
            scenario=scenario,
            reconstructor_cls=active_experiment.reconstructor_cls,
            ddp_setup=ddp_setup,
            device=device,
            train_parser=train_parser,
            train_mapping=train_mapping,
            eval_parser=eval_parser,
            validation_mapping=validation_mapping,
            optimization_config=optimization_config,
            output_dir=experiment_dir,
            artifact_prefix="recovery",
        )
        save_kinematic_parameters(scenario, experiment_dir / "recovered_kinematic_parameters.json")
        recovered_metrics = evaluate_and_save(
            scenario=scenario,
            heliostat_data_mapping=evaluation_mapping,
            data_parser=eval_parser,
            device=device,
            output_dir=experiment_dir,
            artifact_prefix="recovered",
            title=f"{active_experiment.name} — recovered",
            bitmap_resolution=bitmap_resolution,
            ray_tracing_batch_size=ray_tracing_batch_size,
        )
        recovered_parameter_values = collect_parameter_values_for_all_heliostats(
            scenario=scenario,
            experiment=active_experiment,
        )

    summary = {
        **active_experiment.to_dict(),
        "reference_train_loss_mean": reference_train_loss_mean,
        "recovery_train_loss_mean": recovery_train_loss_mean,
        **compute_recovery_summary(
            baseline_metrics=baseline_metrics,
            perturbed_metrics=perturbed_metrics,
            recovered_metrics=recovered_metrics,
        ),
    }
    per_heliostat_comparison = build_per_heliostat_stage_comparison(
        baseline_metrics=baseline_metrics,
        perturbed_metrics=perturbed_metrics,
        recovered_metrics=recovered_metrics,
    )
    accuracy_bucket_summary = {
        "Baseline": compute_accuracy_bucket_summary(baseline_metrics),
        "Perturbed": compute_accuracy_bucket_summary(perturbed_metrics),
        "Recovered": compute_accuracy_bucket_summary(recovered_metrics),
    }
    representative_heliostats = select_representative_heliostats_by_bucket(recovered_metrics)
    for selection in representative_heliostats.values():
        if selection is None:
            continue
        heliostat_id = selection["heliostat_id"]
        selection["baseline_error_mrad"] = baseline_metrics["per_heliostat"].get(heliostat_id, {}).get(
            "focal_spot_error_mrad"
        )
        selection["perturbed_error_mrad"] = perturbed_metrics["per_heliostat"].get(heliostat_id, {}).get(
            "focal_spot_error_mrad"
        )
        selection["recovered_error_mrad"] = recovered_metrics["per_heliostat"].get(heliostat_id, {}).get(
            "focal_spot_error_mrad"
        )

    write_json(experiment_dir / "summary.json", summary)
    write_json(experiment_dir / "per_heliostat_stage_comparison.json", per_heliostat_comparison)
    write_json(experiment_dir / "accuracy_bucket_summary.json", accuracy_bucket_summary)
    write_json(experiment_dir / "representative_heliostats_by_recovered_bucket.json", representative_heliostats)

    plot_stage_comparison(
        baseline_metrics=baseline_metrics,
        perturbed_metrics=perturbed_metrics,
        recovered_metrics=recovered_metrics,
        output_path=experiment_dir / "stage_comparison.png",
        title=f"{active_experiment.name} — baseline vs perturbed vs recovered",
    )
    plot_accuracy_bucket_pies(
        stage_bucket_summaries=accuracy_bucket_summary,
        output_path=experiment_dir / "accuracy_bucket_pies.png",
        title=f"{active_experiment.name} — per-heliostat accuracy buckets",
    )
    plot_accuracy_bucket_comparison(
        stage_bucket_summaries=accuracy_bucket_summary,
        output_path=experiment_dir / "accuracy_bucket_comparison.png",
        title=f"{active_experiment.name} — bucket counts by stage",
    )
    plot_representative_heliostat_parameter_comparison(
        representative_selection=representative_heliostats,
        stage_parameter_values={
            "Baseline": baseline_parameter_values,
            "Perturbed": perturbed_parameter_values,
            "Recovered": recovered_parameter_values,
        },
        parameter_group_specs=parameter_group_specs_for_experiment(active_experiment),
        output_dir=experiment_dir,
        title_prefix=f"{active_experiment.name} — parameter comparison",
    )
    write_json(output_dir / "summary.json", [summary])

    return summary
