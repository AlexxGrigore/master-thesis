"""
Training and evaluation for the one-heliostat train-size sensitivity experiment.

Corrected pipeline: the synthetic dataset was generated from a PERTURBED scenario.
The KR starts from a clean scenario and must discover the perturbation values.

Two evaluation checkpoints:
  1. pre_training  — clean scenario vs perturbed data (high mrad, baseline)
  2. post_training — trained scenario vs perturbed data (low mrad, result)

Two-stage training per run:
  Stage 1 — AlignmentLoss (no ray tracing, fast alignment pre-training)
  Stage 2 — configured loss (focal_spot / pixel) with full ray tracing
"""
import copy
import gc
import json
import logging
import time

try:
    import psutil as _psutil
    def _ram_gb() -> float:
        return _psutil.Process().memory_info().rss / 1024 ** 3
except ImportError:
    _psutil = None
    def _ram_gb() -> float | None:
        return None

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
    loss_type: str = "focal_spot",
    dataset_type: str = "synthetic",
    n_surface_pts: int = 25,
    train_rays: int = 10,
    perturbations_json: dict | None = None,
    heliostat_ids: list | None = None,
    stage1_epochs: int = 50,
    stage2_epochs: int = 100,
) -> dict:
    """
    Train WortbergKinematicReconstructor for a single heliostat (corrected pipeline).

    perturbations_json : dict keyed by heliostat ID loaded from perturbations.json,
                         used only for param_recovery reporting — NOT applied to scenario.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _ram_start = _ram_gb()

    with h5py.File(scenario_path, "r") as f:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=f,
            device=device,
            number_of_surface_points_per_facet=torch.tensor([n_surface_pts, n_surface_pts]),
        )
    scenario.set_number_of_rays(train_rays)

    # ------------------------------------------------------------------
    # Pre-training eval: clean scenario vs perturbed data → high mrad
    # ------------------------------------------------------------------
    log.info("Pre-training eval (clean scenario, perturbed data) …")
    pre_val_eval  = evaluate_flux_accuracy(scenario=scenario, heliostat_data_mapping=val_mapping,  data_parser=val_parser,  device=device)
    pre_test_eval = evaluate_flux_accuracy(scenario=scenario, heliostat_data_mapping=test_mapping, data_parser=test_parser, device=device)
    log.info(
        f"  pre-training: val={pre_val_eval['mean_mrad']:.3f} mrad  "
        f"test={pre_test_eval['mean_mrad']:.3f} mrad"
    )

    # ------------------------------------------------------------------
    # Two-stage training
    # ------------------------------------------------------------------
    data = {
        config_dictionary.data_parser:            train_parser,
        config_dictionary.heliostat_data_mapping: train_mapping,
    }
    eval_data = {
        "data_parser":            val_parser,
        "heliostat_data_mapping": val_mapping,
    }

    stage1_config = copy.deepcopy(optimization_config)
    stage1_config[config_dictionary.max_epoch] = stage1_epochs
    stage2_config = copy.deepcopy(optimization_config)
    stage2_config[config_dictionary.max_epoch] = stage2_epochs

    t0 = time.time()
    _pre_eval_time_s = t0 - overall_t0

    log.info(f"Stage 1 — AlignmentLoss pre-training ({stage1_epochs} epochs) …")
    stage1_reconstructor, stage1_loss_fn = _build_reconstructor(
        loss_type="alignment",
        scenario=scenario,
        ddp_setup=ddp_setup,
        data=data,
        eval_data=eval_data,
        optimization_config=stage1_config,
        train_position_deviation=True,
        sample_mini_batch_size=10,
    )
    stage1_reconstructor.reconstruct_kinematics(loss_definition=stage1_loss_fn, device=device)
    stage1_history = stage1_reconstructor._convergence_history
    stage1_time_s = time.time() - t0
    _ram_after_stage1 = _ram_gb()
    log.info(f"Stage 1 done in {stage1_time_s / 60:.1f} min")

    del stage1_reconstructor
    gc.collect()
    torch.cuda.empty_cache()

    log.info(f"Stage 2 — {loss_type} fine-tuning ({stage2_epochs} epochs) …")
    t1 = time.time()
    stage2_reconstructor, stage2_loss_fn = _build_reconstructor(
        loss_type=loss_type,
        scenario=scenario,
        ddp_setup=ddp_setup,
        data=data,
        eval_data=eval_data,
        optimization_config=stage2_config,
        train_position_deviation=True,
        sample_mini_batch_size=10,
    )
    stage2_reconstructor.reconstruct_kinematics(loss_definition=stage2_loss_fn, device=device)
    stage2_history = stage2_reconstructor._convergence_history
    stage2_time_s = time.time() - t1
    train_time = time.time() - t0
    _ram_after_stage2 = _ram_gb()
    log.info(f"Stage 2 done in {stage2_time_s / 60:.1f} min  (total {train_time / 60:.1f} min)")

    # Merge histories with a continuous epoch counter
    epoch_offset = stage1_history[-1]["epoch"] + 1 if stage1_history else 0
    for entry in stage2_history:
        entry["epoch"] += epoch_offset
    convergence_history = stage1_history + stage2_history

    with open(output_dir / "convergence_history.json", "w") as f:
        json.dump(convergence_history, f, indent=2)
    with open(output_dir / "convergence_history_stage1.json", "w") as f:
        json.dump(stage1_history, f, indent=2)
    with open(output_dir / "convergence_history_stage2.json", "w") as f:
        json.dump(stage2_history, f, indent=2)
    _save_convergence_csv(convergence_history, output_dir / "convergence_history.csv")

    if heliostat_ids is not None:
        kinematic_history = _build_kinematic_history(
            stage2_reconstructor._kinematic_history, heliostat_ids
        )
        with open(output_dir / "kinematic_history.json", "w") as f:
            json.dump(kinematic_history, f, indent=2)

    del stage2_reconstructor
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Post-training eval: trained scenario vs perturbed data → low mrad
    # ------------------------------------------------------------------
    log.info("Post-training eval (trained scenario, perturbed data) …")
    trained_val_eval  = evaluate_flux_accuracy(scenario=scenario, heliostat_data_mapping=val_mapping,  data_parser=val_parser,  device=device)
    trained_test_eval = evaluate_flux_accuracy(scenario=scenario, heliostat_data_mapping=test_mapping, data_parser=test_parser, device=device)
    log.info(
        f"  post-training: val={trained_val_eval['mean_mrad']:.3f} mrad  "
        f"test={trained_test_eval['mean_mrad']:.3f} mrad"
    )

    _save_flux_comparison_images(
        scenario=scenario,
        test_parser=test_parser,
        test_mapping=test_mapping,
        device=device,
        output_dir=output_dir,
        dataset_type=dataset_type,
    )

    recovery = None
    if perturbations_json is not None and heliostat_ids is not None:
        recovery = _param_recovery(scenario, perturbations_json, heliostat_ids, device)

    overall_time_s = time.time() - overall_t0
    _ram_end = _ram_gb()
    _ram_samples = [r for r in [_ram_start, _ram_after_stage1, _ram_after_stage2, _ram_end] if r is not None]
    timing = {
        "overall_s":                    round(overall_time_s, 1),
        "overall_min":                  round(overall_time_s / 60, 2),
        "pre_training_eval_s":          round(_pre_eval_time_s, 1),
        "stage1_training_s":            round(stage1_time_s, 1),
        "stage1_training_min":          round(stage1_time_s / 60, 2),
        "stage2_training_s":            round(stage2_time_s, 1),
        "stage2_training_min":          round(stage2_time_s / 60, 2),
        "total_training_s":             round(train_time, 1),
        "total_training_min":           round(train_time / 60, 2),
        "peak_gpu_memory_allocated_gb": round(
            torch.cuda.max_memory_allocated() / 1024 ** 3, 3
        ) if torch.cuda.is_available() else None,
        "peak_gpu_memory_reserved_gb":  round(
            torch.cuda.max_memory_reserved() / 1024 ** 3, 3
        ) if torch.cuda.is_available() else None,
        "peak_ram_gb":                  round(max(_ram_samples), 3) if _ram_samples else None,
    }
    with open(output_dir / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    def _pack(ev: dict) -> dict:
        return {
            "mean_mrad":         ev["mean_mrad"],
            "median_mrad":       ev["median_mrad"],
            "mean_m":            ev["mean_m"],
            "mean_pixel_loss":   ev["mean_pixel_loss"],
            "median_pixel_loss": ev["median_pixel_loss"],
            "num_samples":       ev["num_samples"],
            "num_nan_samples":   ev["num_nan_samples"],
            "nan_heliostat_ids": ev["nan_heliostat_ids"],
            "per_heliostat":     ev["per_heliostat"],
        }

    results = {
        "pre_training": {
            "val":  _pack(pre_val_eval),
            "test": _pack(pre_test_eval),
        },
        "post_training": {
            "val":  {**_pack(trained_val_eval),  "min_mrad": trained_val_eval["min_mrad"],  "max_mrad": trained_val_eval["max_mrad"]},
            "test": {**_pack(trained_test_eval), "min_mrad": trained_test_eval["min_mrad"], "max_mrad": trained_test_eval["max_mrad"]},
        },
        "train_time_min": round(train_time / 60, 2),
        "loss_type":      loss_type,
        "param_recovery": recovery,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    _write_metrics_table(results, output_dir / "metrics_table.txt")
    _save_kinematic_parameters(scenario, output_dir / "kinematic_parameters.json")

    W = 70
    pre_v  = results["pre_training"]["val"]
    pre_t  = results["pre_training"]["test"]
    post_v = results["post_training"]["val"]
    post_t = results["post_training"]["test"]
    print(f"\n{'=' * W}")
    print(f"  {'Checkpoint':<22} {'Val (mrad)':>10}  {'Test (mrad)':>11}  n")
    print(f"  {'-' * 22} {'-' * 10}  {'-' * 11}  -")
    print(f"  {'Pre-training':<22} {pre_v['mean_mrad']:>10.3f}  {pre_t['mean_mrad']:>11.3f}  {pre_t['num_samples']}")
    print(f"  {'Post-training':<22} {post_v['mean_mrad']:>10.3f}  {post_t['mean_mrad']:>11.3f}  {post_t['num_samples']}")
    print(f"  Min / Max test (post)  : {post_t['min_mrad']:.3f} / {post_t['max_mrad']:.3f} mrad")
    print(f"  Training time          : stage1={stage1_time_s/60:.1f} min  stage2={stage2_time_s/60:.1f} min  total={train_time/60:.1f} min")
    print(f"{'=' * W}\n")

    return results


# ---------------------------------------------------------------------------
# Convergence CSV
# ---------------------------------------------------------------------------

def _save_convergence_csv(history: list, path) -> None:
    import csv, pathlib
    path = pathlib.Path(path)
    if not history:
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for entry in history:
            writer.writerow([
                entry.get("epoch", ""),
                entry.get("loss",  ""),
                entry.get("eval_loss", ""),
            ])


# ---------------------------------------------------------------------------
# Per-run metrics table
# ---------------------------------------------------------------------------

def _write_metrics_table(results: dict, path) -> None:
    import pathlib
    path = pathlib.Path(path)
    lines = [
        f"{'Stage':<22} {'Val mrad':>10} {'Test mrad':>10} {'Val pixel':>11} {'Test pixel':>11}",
        "-" * 66,
    ]
    for stage_key, label in [
        ("pre_training",  "Pre-training"),
        ("post_training", "Post-training"),
    ]:
        v = results[stage_key]["val"]
        t = results[stage_key]["test"]
        lines.append(
            f"{label:<22} {v['mean_mrad']:>10.4f} {t['mean_mrad']:>10.4f} "
            f"{v['mean_pixel_loss']:>11.5f} {t['mean_pixel_loss']:>11.5f}"
        )
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Flux comparison images
# ---------------------------------------------------------------------------

@torch.no_grad()
def _save_flux_comparison_images(scenario, test_parser, test_mapping, device, output_dir, dataset_type: str = "synthetic") -> None:
    from utils.synth_reporting import plot_flux_comparison
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
        kinematic = heliostat_group.kinematics
        if hasattr(kinematic, "_base_position_deviation"):
            base_dev = kinematic._base_position_deviation.repeat_interleave(
                active_heliostats_mask, dim=0
            )
            pad = torch.zeros(base_dev.shape[0], 1, device=device)
            kinematic.active_heliostat_positions = (
                kinematic.active_heliostat_positions + torch.cat([base_dev, pad], dim=1)
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

        sample_indices    = ray_tracer.get_sampler_indices()
        inv_perm          = torch.argsort(sample_indices)
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
        active_indices  = torch.where(active_heliostats_mask.bool())[0]
        distances       = torch.norm(
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
# Parameter recovery — residual = |trained - perturbation|
# ---------------------------------------------------------------------------

def _param_recovery(scenario, perturbations_by_id: dict, heliostat_ids: list, device: torch.device) -> dict:
    """
    Compare trained kinematic parameters against ground-truth perturbations.

    The KR starts from zero and converges toward the perturbation values, so
    residual = |trained - perturbation| (lower is better).

    perturbations_by_id : dict loaded from perturbations.json, keyed by heliostat ID.
    """
    kinematic = scenario.heliostat_field.heliostat_groups[0].kinematics
    result = {}

    for i, hid in enumerate(heliostat_ids):
        if hid not in perturbations_by_id:
            continue
        pert = perturbations_by_id[hid]

        perturbation_rot = pert["rotation_rad"]
        rec_rot = kinematic.rotation_deviation_parameters[i].detach().cpu().tolist()
        residual_rot = [abs(r - p) for r, p in zip(rec_rot, perturbation_rot)]

        perturbation_act = pert["actuator_angle_rad"]
        start_ang = (
            kinematic._initial_actuator_initial_angle[i].cpu()
            if hasattr(kinematic, "_initial_actuator_initial_angle")
            else kinematic.actuators.optimizable_parameters[i, index_mapping.actuator_initial_angle, :].detach().cpu()
        )
        moved_act = (
            kinematic.actuators.optimizable_parameters[i, index_mapping.actuator_initial_angle, :].detach().cpu()
            - start_ang
        ).tolist()
        residual_act = [abs(m - p) for m, p in zip(moved_act, perturbation_act)]

        perturbation_offset = pert["actuator_offset_m"]
        start_off = (
            kinematic._initial_actuator_offset[i].cpu()
            if hasattr(kinematic, "_initial_actuator_offset")
            else kinematic.actuators.non_optimizable_parameters[i, index_mapping.actuator_offset, :].detach().cpu()
        )
        moved_off = (
            kinematic.actuators.non_optimizable_parameters[i, index_mapping.actuator_offset, :].detach().cpu()
            - start_off
        ).tolist()
        residual_off = [abs(m - p) for m, p in zip(moved_off, perturbation_offset)]

        perturbation_trans = pert["translation_m"]
        start_trans = (
            kinematic._initial_translation_deviation[i].cpu()
            if hasattr(kinematic, "_initial_translation_deviation")
            else kinematic.translation_deviation_parameters[i].detach().cpu()
        )
        moved_trans = (
            kinematic.translation_deviation_parameters[i].detach().cpu() - start_trans
        ).tolist()
        residual_trans = [abs(m - p) for m, p in zip(moved_trans, perturbation_trans)]

        perturbation_bp = pert["base_position_m"]
        rec_bp = (
            kinematic._base_position_deviation[i].detach().cpu().tolist()
            if hasattr(kinematic, "_base_position_deviation")
            else [0.0, 0.0, 0.0]
        )
        residual_bp = [abs(r - p) for r, p in zip(rec_bp, perturbation_bp)]

        result[hid] = {
            "rotation":       {"perturbation_rad": perturbation_rot, "recovered_rad": rec_rot,    "abs_residual_rad": residual_rot},
            "actuator_angle": {"perturbation_rad": perturbation_act, "moved_rad": moved_act,       "abs_residual_rad": residual_act},
            "actuator_offset":{"perturbation_m":   perturbation_offset, "moved_m": moved_off,      "abs_residual_m":   residual_off},
            "translation":    {"perturbation_m":   perturbation_trans,  "moved_m": moved_trans,    "abs_residual_m":   residual_trans},
            "base_position":  {"perturbation_m":   perturbation_bp,     "recovered_m": rec_bp,     "abs_residual_m":   residual_bp},
        }

    return result


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
                "rotation_rad":                 entry["rotation_rad"][i]                   if entry.get("rotation_rad")                  else None,
                "actuator_angle_deviation_rad": entry["actuator_angle_deviation_rad"][i]   if entry.get("actuator_angle_deviation_rad")  else None,
                "actuator_offset_deviation_m":  entry["actuator_offset_deviation_m"][i]    if entry.get("actuator_offset_deviation_m")   else None,
                "base_position_m":              entry["base_position_m"][i]                if entry.get("base_position_m")               else None,
            }
        result.append({"epoch": entry["epoch"], "heliostats": hel_data})
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
