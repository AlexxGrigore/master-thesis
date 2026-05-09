"""
Training and evaluation for the full-field 200-samples synthetic perturbation experiment.

Three evaluation checkpoints:
  1. pre_perturbation  — clean scenario vs clean synthetic test data (~0 mrad)
  2. post_perturbation — perturbed scenario vs same clean synthetic test data (high mrad)
  3. post_training     — trained (recovered) scenario vs same clean synthetic test data (low mrad)

Loss: PixelLoss (MSE) — blur σ=1 applied by WortbergPixelReconstructor before loss call.
"""
import gc
import json
import logging
import time

import h5py
import torch
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, index_mapping
from artist.util.utils import get_center_of_mass, bitmap_coordinates_to_target_coordinates

from artist.core.loss_functions import FocalSpotLoss, PixelLoss
from artist_extensions.kinematic_reconstructors import (
    WortbergAlignmentReconstructor,
    WortbergKinematicReconstructor,
    WortbergPixelReconstructor,
)
from artist_extensions.loss_functions_ext import AlignmentLoss
from utils.evaluation import evaluate_flux_accuracy, _gaussian_blur_batch

log = logging.getLogger(__name__)

_LOSS_CONFIGS: dict[str, tuple] = {
    "focal_spot": (WortbergKinematicReconstructor, lambda s: FocalSpotLoss(scenario=s)),
    "pixel":      (WortbergPixelReconstructor,     lambda s: PixelLoss(scenario=s)),
    "alignment":  (WortbergAlignmentReconstructor, lambda _: AlignmentLoss()),
}


def _build_reconstructor(loss_type: str, scenario, ddp_setup, data, eval_data, optimization_config, **kwargs):
    if loss_type not in _LOSS_CONFIGS:
        raise ValueError(f"Unknown loss_type {loss_type!r}. Choose from {list(_LOSS_CONFIGS)}.")
    cls, loss_fn_factory = _LOSS_CONFIGS[loss_type]
    extra = {"blur_sigma": 1.0} if cls is WortbergPixelReconstructor else {}
    reconstructor = cls(
        ddp_setup=ddp_setup,
        scenario=scenario,
        data=data,
        optimization_configuration=optimization_config,
        reconstruction_method=config_dictionary.kinematics_reconstruction_raytracing,
        eval_data=eval_data,
        **extra,
        **kwargs,
    )
    return reconstructor, loss_fn_factory(scenario)


def run(
    scenario_path,
    device: torch.device,
    ddp_setup: dict,
    train_mapping: list,
    val_mapping: list,
    test_mapping: list,
    train_parser,
    val_parser,
    test_parser,
    optimization_config: dict,
    output_dir,
    loss_type: str = "pixel",
    dataset_type: str = "synthetic",
    n_surface_pts: int = 25,
    train_rays: int = 10,
    perturbations: dict | None = None,
    heliostat_ids: list | None = None,
) -> dict:
    """
    Train WortbergKinematicReconstructor on clean synthetic data for 63 heliostats.
    Returns metrics dict with all 3 evaluation results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with h5py.File(scenario_path, "r") as f:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=f,
            device=device,
            number_of_surface_points_per_facet=torch.tensor([n_surface_pts, n_surface_pts]),
        )
    scenario.set_number_of_rays(train_rays)
    kinematic = scenario.heliostat_field.heliostat_groups[0].kinematics
    snapshot_clean = snapshot_perturbed = snapshot_trained = None

    if heliostat_ids is not None:
        snapshot_clean = _snapshot_kinematic_state(kinematic, heliostat_ids)

    # ------------------------------------------------------------------
    # Stage 1: pre-perturbation eval
    # ------------------------------------------------------------------
    log.info("Stage 1 — pre-perturbation eval (clean scenario, clean test data)…")
    pre_eval_t0 = time.time()
    pre_eval = evaluate_flux_accuracy(
        scenario=scenario,
        heliostat_data_mapping=test_mapping,
        data_parser=test_parser,
        device=device,
    )
    pre_eval_time_s = time.time() - pre_eval_t0
    log.info(
        f"  pre-perturb : mean={pre_eval['mean_mrad']:.3f} mrad  "
        f"median={pre_eval['median_mrad']:.3f} mrad  "
        f"pixel_loss={pre_eval['mean_pixel_loss']:.4f}  n={pre_eval['num_samples']}"
    )

    # ------------------------------------------------------------------
    # Stage 2: apply perturbations, then eval
    # ------------------------------------------------------------------
    if perturbations is not None:
        from five_heliostats_synth.data import apply_perturbations
        apply_perturbations(kinematic, perturbations, device)
        log.info("Perturbations applied to scenario kinematics in-place.")
        if heliostat_ids is not None:
            snapshot_perturbed = _snapshot_kinematic_state(kinematic, heliostat_ids)

    log.info("Stage 2 — post-perturbation eval (perturbed scenario, clean test data)…")
    post_perturb_eval = evaluate_flux_accuracy(
        scenario=scenario,
        heliostat_data_mapping=test_mapping,
        data_parser=test_parser,
        device=device,
    )
    log.info(
        f"  post-perturb: mean={post_perturb_eval['mean_mrad']:.3f} mrad  "
        f"median={post_perturb_eval['median_mrad']:.3f} mrad  "
        f"pixel_loss={post_perturb_eval['mean_pixel_loss']:.4f}  n={post_perturb_eval['num_samples']}"
    )

    # ------------------------------------------------------------------
    # Stage 3: train
    # ------------------------------------------------------------------
    data = {
        config_dictionary.data_parser:           train_parser,
        config_dictionary.heliostat_data_mapping: train_mapping,
    }
    eval_data = {
        "data_parser":            val_parser,
        "heliostat_data_mapping": val_mapping,
    }

    reconstructor, loss_fn = _build_reconstructor(
        loss_type=loss_type,
        scenario=scenario,
        ddp_setup=ddp_setup,
        data=data,
        eval_data=eval_data,
        optimization_config=optimization_config,
        train_position_deviation=True,
        sample_mini_batch_size=10,
    )

    t0 = time.time()
    reconstructor.reconstruct_kinematics(loss_definition=loss_fn, device=device)
    train_time = time.time() - t0
    log.info(f"Training done in {train_time / 60:.1f} min")

    with open(output_dir / "convergence_history.json", "w") as f:
        json.dump(reconstructor._convergence_history, f, indent=2)

    if heliostat_ids is not None:
        kinematic_history = _build_kinematic_history(
            reconstructor._kinematic_history, heliostat_ids
        )
        with open(output_dir / "kinematic_history.json", "w") as f:
            json.dump(kinematic_history, f, indent=2)

    del reconstructor
    gc.collect()
    torch.cuda.empty_cache()

    if heliostat_ids is not None:
        snapshot_trained = _snapshot_kinematic_state(kinematic, heliostat_ids)

    # ------------------------------------------------------------------
    # Stage 3 eval
    # ------------------------------------------------------------------
    log.info("Stage 3 — post-training eval (trained scenario, clean test data)…")
    post_train_eval_t0 = time.time()
    post_train_eval = evaluate_flux_accuracy(
        scenario=scenario,
        heliostat_data_mapping=test_mapping,
        data_parser=test_parser,
        device=device,
    )
    post_train_eval_time_s = time.time() - post_train_eval_t0
    log.info(
        f"  post-train  : mean={post_train_eval['mean_mrad']:.3f} mrad  "
        f"median={post_train_eval['median_mrad']:.3f} mrad  "
        f"pixel_loss={post_train_eval['mean_pixel_loss']:.4f}  n={post_train_eval['num_samples']}"
    )

    # Save three-stage kinematic snapshot
    if heliostat_ids is not None and snapshot_clean is not None:
        kinematic_stages = {
            "clean":     snapshot_clean,
            "perturbed": snapshot_perturbed or snapshot_clean,
            "trained":   snapshot_trained   or snapshot_clean,
        }
        with open(output_dir / "kinematic_stages.json", "w") as f:
            json.dump(kinematic_stages, f, indent=2)

    _save_flux_comparison_images(
        scenario=scenario,
        test_parser=test_parser,
        test_mapping=test_mapping,
        device=device,
        output_dir=output_dir,
        dataset_type=dataset_type,
    )

    recovery = None
    if perturbations is not None and heliostat_ids is not None:
        recovery = _param_recovery(scenario, perturbations, heliostat_ids, device)

    overall_time_s = time.time() - overall_t0
    timing = {
        "overall_s": round(overall_time_s, 1),
        "overall_min": round(overall_time_s / 60, 2),
        "pre_perturbation_eval_s": round(pre_eval_time_s, 1),
        "training_s": round(train_time, 1),
        "training_min": round(train_time / 60, 2),
        "post_training_eval_s": round(post_train_eval_time_s, 1),
        "peak_gpu_memory_allocated_gb": round(
            torch.cuda.max_memory_allocated() / 1024 ** 3, 3
        ) if torch.cuda.is_available() else None,
        "peak_gpu_memory_reserved_gb": round(
            torch.cuda.max_memory_reserved() / 1024 ** 3, 3
        ) if torch.cuda.is_available() else None,
    }
    with open(output_dir / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    results = {
        "pre_perturbation": {
            "mean_mrad":         pre_eval["mean_mrad"],
            "median_mrad":       pre_eval["median_mrad"],
            "mean_m":            pre_eval["mean_m"],
            "mean_pixel_loss":   pre_eval["mean_pixel_loss"],
            "median_pixel_loss": pre_eval["median_pixel_loss"],
            "num_samples":       pre_eval["num_samples"],
            "num_nan_samples":   pre_eval["num_nan_samples"],
            "nan_heliostat_ids": pre_eval["nan_heliostat_ids"],
            "per_heliostat":     pre_eval["per_heliostat"],
        },
        "post_perturbation": {
            "mean_mrad":         post_perturb_eval["mean_mrad"],
            "median_mrad":       post_perturb_eval["median_mrad"],
            "mean_m":            post_perturb_eval["mean_m"],
            "mean_pixel_loss":   post_perturb_eval["mean_pixel_loss"],
            "median_pixel_loss": post_perturb_eval["median_pixel_loss"],
            "num_samples":       post_perturb_eval["num_samples"],
            "num_nan_samples":   post_perturb_eval["num_nan_samples"],
            "nan_heliostat_ids": post_perturb_eval["nan_heliostat_ids"],
            "per_heliostat":     post_perturb_eval["per_heliostat"],
        },
        "post_training": {
            "mean_mrad":         post_train_eval["mean_mrad"],
            "median_mrad":       post_train_eval["median_mrad"],
            "min_mrad":          post_train_eval["min_mrad"],
            "max_mrad":          post_train_eval["max_mrad"],
            "mean_m":            post_train_eval["mean_m"],
            "mean_pixel_loss":   post_train_eval["mean_pixel_loss"],
            "median_pixel_loss": post_train_eval["median_pixel_loss"],
            "num_samples":       post_train_eval["num_samples"],
            "num_nan_samples":   post_train_eval["num_nan_samples"],
            "nan_heliostat_ids": post_train_eval["nan_heliostat_ids"],
            "per_heliostat":     post_train_eval["per_heliostat"],
        },
        "train_time_min": round(train_time / 60, 2),
        "param_recovery": recovery,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    _save_kinematic_parameters(scenario, output_dir / "kinematic_parameters.json")

    return results


# ---------------------------------------------------------------------------
# Kinematic state snapshot
# ---------------------------------------------------------------------------

def _snapshot_kinematic_state(kinematic, heliostat_ids: list) -> dict:
    result = {}
    for i, hid in enumerate(heliostat_ids):
        base_pos = (
            kinematic._base_position_deviation[i].detach().cpu().tolist()
            if hasattr(kinematic, "_base_position_deviation")
            else [0.0, 0.0, 0.0]
        )
        result[hid] = {
            "rotation_rad":       kinematic.rotation_deviation_parameters[i].detach().cpu().tolist(),
            "actuator_angle_rad": kinematic.actuators.optimizable_parameters[
                i, index_mapping.actuator_initial_angle, :
            ].detach().cpu().tolist(),
            "actuator_stroke_m":  kinematic.actuators.optimizable_parameters[
                i, index_mapping.actuator_initial_stroke_length, :
            ].detach().cpu().tolist(),
            "actuator_offset_m":  kinematic.actuators.non_optimizable_parameters[
                i, index_mapping.actuator_offset, :
            ].detach().cpu().tolist(),
            "translation_m":      kinematic.translation_deviation_parameters[i].detach().cpu().tolist(),
            "base_position_m":    base_pos,
        }
    return result


# ---------------------------------------------------------------------------
# Flux comparison images (saves first test sample per heliostat)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _save_flux_comparison_images(scenario, test_parser, test_mapping, device, output_dir, dataset_type: str = "synthetic") -> None:
    from five_heliostats_synth.reporting import plot_flux_comparison
    output_dir = output_dir / "flux_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)

    bitmap_resolution = torch.tensor([256, 256])

    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        (
            measured_flux,
            focal_spots,
            incident_ray_directions,
            _,
            active_heliostats_mask,
            target_area_mask,
        ) = test_parser.parse_data_for_reconstruction(
            heliostat_data_mapping=test_mapping,
            heliostat_group=heliostat_group,
            scenario=scenario,
            device=device,
        )

        if active_heliostats_mask.sum() == 0:
            continue

        heliostat_group.activate_heliostats(
            active_heliostats_mask=active_heliostats_mask, device=device
        )
        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=scenario.solar_tower.get_centers_of_target_areas(
                target_area_mask, device=device
            ),
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            device=device,
        )

        ray_tracer = HeliostatRayTracer(
            scenario=scenario,
            heliostat_group=heliostat_group,
            blocking_active=False,
            batch_size=min(heliostat_group.number_of_active_heliostats, 32),
            bitmap_resolution=bitmap_resolution.to(device),
        )
        predicted_sampler, _, _, _ = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_heliostats_mask,
            target_area_indices=target_area_mask,
            device=device,
        )

        sample_indices  = ray_tracer.get_sampler_indices()
        inv_perm        = torch.argsort(sample_indices)
        predicted_natural = predicted_sampler[inv_perm]

        bitmap_coords = get_center_of_mass(bitmaps=predicted_sampler, device=device)
        predicted_spots = bitmap_coordinates_to_target_coordinates(
            bitmap_coordinates=bitmap_coords,
            bitmap_resolution=ray_tracer.bitmap_resolution,
            solar_tower=scenario.solar_tower,
            target_area_indices=target_area_mask[sample_indices],
            device=device,
        )
        fse_sampler = torch.norm(
            predicted_spots[:, :3] - focal_spots[sample_indices][:, :3], dim=1
        )
        fse_natural = fse_sampler[inv_perm]

        reference_target = scenario.solar_tower.target_areas[
            index_mapping.planar_target_areas
        ].centers[:, :3].mean(dim=0).to(device)
        active_indices = torch.where(active_heliostats_mask.bool())[0]
        distances = torch.norm(
            heliostat_group.positions[active_indices, :3].to(device) - reference_target, dim=1
        )
        samples_per_hel = active_heliostats_mask[active_indices].long()

        offset = 0
        for j, idx in enumerate(active_indices):
            hid = heliostat_group.names[idx.item()]
            n   = samples_per_hel[j].item()

            meas_raw = measured_flux[offset].cpu()
            pred_raw = predicted_natural[offset].cpu()

            pred_blurred = _gaussian_blur_batch(pred_raw.unsqueeze(0), sigma=1.0).squeeze(0)
            pred_vis = (pred_blurred / pred_blurred.max().clamp(min=1e-12)).numpy()

            if dataset_type == "synthetic":
                meas_blurred = _gaussian_blur_batch(meas_raw.unsqueeze(0), sigma=1.0).squeeze(0)
            else:
                meas_blurred = meas_raw
            meas_vis = (meas_blurred / meas_blurred.max().clamp(min=1e-12)).numpy()

            pixel_loss = float(abs(torch.from_numpy(pred_vis) - torch.from_numpy(meas_vis)).sum())

            fse_val  = fse_natural[offset].item()
            fse_mrad = (fse_val / distances[j].item()) * 1000.0 if not torch.isnan(fse_natural[offset]) else float("nan")

            plot_flux_comparison(
                measured=meas_vis,
                predicted=pred_vis,
                pixel_loss=pixel_loss,
                fse_mrad=fse_mrad,
                heliostat_id=hid,
                output_dir=output_dir,
            )
            log.info(f"  flux_comparison_{hid}: FSE={fse_mrad:.3f} mrad  pixel_loss={pixel_loss:.2f}")

            offset += n


# ---------------------------------------------------------------------------
# Kinematic history
# ---------------------------------------------------------------------------

def _build_kinematic_history(raw_history: list, heliostat_ids: list | None) -> list:
    if not raw_history or heliostat_ids is None:
        return raw_history or []
    result = []
    for entry in raw_history:
        hel_data = {}
        for i, hid in enumerate(heliostat_ids):
            hel_data[hid] = {
                "rotation_rad":               entry["rotation_rad"][i]                 if entry.get("rotation_rad")                else None,
                "actuator_angle_deviation_rad": entry["actuator_angle_deviation_rad"][i] if entry.get("actuator_angle_deviation_rad") else None,
                "actuator_offset_deviation_m":  entry["actuator_offset_deviation_m"][i]  if entry.get("actuator_offset_deviation_m")  else None,
                "base_position_m":              entry["base_position_m"][i]              if entry.get("base_position_m")              else None,
            }
        result.append({"epoch": entry["epoch"], "heliostats": hel_data})
    return result


# ---------------------------------------------------------------------------
# Parameter recovery
# ---------------------------------------------------------------------------

def _param_recovery(scenario, perturbations: dict, heliostat_ids: list, device: torch.device) -> dict:
    kinematic = scenario.heliostat_field.heliostat_groups[0].kinematics
    result = {}

    for i, hid in enumerate(heliostat_ids):
        perturbation_rot = perturbations["rotation"][i].tolist()
        rec_rot = kinematic.rotation_deviation_parameters[i].detach().cpu().tolist()

        perturbation_act = perturbations["actuator_angle"][i].tolist()
        start_ang = (
            kinematic._initial_actuator_initial_angle[i].cpu()
            if hasattr(kinematic, "_initial_actuator_initial_angle")
            else kinematic.actuators.optimizable_parameters[i, index_mapping.actuator_initial_angle, :].detach().cpu()
        )
        final_ang = kinematic.actuators.optimizable_parameters[i, index_mapping.actuator_initial_angle, :].detach().cpu()
        moved_act = (final_ang - start_ang).tolist()
        deviation_act = [p + m for p, m in zip(perturbation_act, moved_act)]

        perturbation_stroke = perturbations["actuator_stroke"][i].tolist()
        rec_stroke = kinematic.actuators.optimizable_parameters[
            i, index_mapping.actuator_initial_stroke_length, :
        ].detach().cpu().tolist()

        perturbation_offset = perturbations["actuator_offset"][i].tolist()
        start_off = (
            kinematic._initial_actuator_offset[i].cpu()
            if hasattr(kinematic, "_initial_actuator_offset")
            else kinematic.actuators.non_optimizable_parameters[i, index_mapping.actuator_offset, :].detach().cpu()
        )
        final_off = kinematic.actuators.non_optimizable_parameters[i, index_mapping.actuator_offset, :].detach().cpu()
        moved_off = (final_off - start_off).tolist()
        deviation_off = [p + m for p, m in zip(perturbation_offset, moved_off)]

        perturbation_trans = perturbations["translation"][i].tolist()
        start_trans = (
            kinematic._initial_translation_deviation[i].cpu()
            if hasattr(kinematic, "_initial_translation_deviation")
            else kinematic.translation_deviation_parameters[i].detach().cpu()
        )
        final_trans = kinematic.translation_deviation_parameters[i].detach().cpu()
        moved_trans = (final_trans - start_trans).tolist()
        deviation_trans = [p + m for p, m in zip(perturbation_trans, moved_trans)]

        perturbation_bp = perturbations["base_position"][i].tolist()
        rec_bp = (
            kinematic._base_position_deviation[i].detach().cpu().tolist()
            if hasattr(kinematic, "_base_position_deviation")
            else [0.0, 0.0, 0.0]
        )

        result[hid] = {
            "rotation":       {"perturbation_rad": perturbation_rot, "recovered_rad": rec_rot,    "abs_residual_rad": [abs(r) for r in rec_rot]},
            "actuator_angle": {"perturbation_rad": perturbation_act, "moved_rad": moved_act,       "deviation_from_clean_rad": deviation_act, "abs_residual_rad": [abs(d) for d in deviation_act]},
            "actuator_stroke":{"perturbation_m":   perturbation_stroke, "final_m": rec_stroke,     "abs_residual_m": [abs(p) for p in perturbation_stroke], "note": "frozen — perturbation is permanent"},
            "actuator_offset":{"perturbation_m":   perturbation_offset, "moved_m": moved_off,      "deviation_from_clean_m": deviation_off, "abs_residual_m": [abs(d) for d in deviation_off]},
            "translation":    {"perturbation_m":   perturbation_trans,  "moved_m": moved_trans,    "deviation_from_clean_m": deviation_trans, "abs_residual_m": [abs(d) for d in deviation_trans]},
            "base_position":  {"perturbation_m":   perturbation_bp,     "recovered_m": rec_bp,     "abs_residual_m": [abs(r) for r in rec_bp]},
        }

    return result


# ---------------------------------------------------------------------------
# Kinematic parameter export
# ---------------------------------------------------------------------------

def _save_kinematic_parameters(scenario, path) -> None:
    import pathlib
    path = pathlib.Path(path)
    heliostat_group = scenario.heliostat_field.heliostat_groups[0]
    kinematic       = heliostat_group.kinematics
    names           = list(heliostat_group.names)
    base_pos = (
        kinematic._base_position_deviation.detach().cpu().tolist()
        if hasattr(kinematic, "_base_position_deviation")
        else [[0.0, 0.0, 0.0]] * len(names)
    )
    payload = {
        "group_0": {
            "heliostat_names":                    names,
            "translation_deviation_parameters":   kinematic.translation_deviation_parameters.detach().cpu().tolist(),
            "rotation_deviation_parameters":      kinematic.rotation_deviation_parameters.detach().cpu().tolist(),
            "actuator_optimizable_parameters":    kinematic.actuators.optimizable_parameters.detach().cpu().tolist(),
            "actuator_nonoptimizable_parameters": kinematic.actuators.non_optimizable_parameters.detach().cpu().tolist(),
            "base_position_deviation_parameters": base_pos,
        }
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    log.info(f"Kinematic parameters saved → {path}")
