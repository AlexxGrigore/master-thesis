"""
Training and evaluation for the full-63-heliostat kinematic reconstruction experiment.

Corrected pipeline vs full_field_200_samples
--------------------------------------------
The synthetic dataset was generated from the PERTURBED scenario, so the KR
starts from a clean scenario and must learn the perturbation values.

Two evaluation checkpoints:
  1. pre_training  — clean scenario vs perturbed test data (high mrad, baseline)
  2. post_training — trained scenario vs perturbed test data (low mrad, result)
"""
import copy
import gc
import json
import logging
import pathlib
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
import numpy as np
from PIL import Image as _PIL_Image
import torch
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
from artist.scenario.scenario import Scenario
from artist.util import constants as config_dictionary, indices as index_mapping
from artist.geometry import bitmap_coordinates_to_target_coordinates
from artist.flux import get_center_of_mass
from artist.optim.loss import FocalSpotLoss, PixelLoss

from artist_extensions.kinematic_reconstructors import (
    WortbergAlignmentReconstructor,
    WortbergContourReconstructor,
    WortbergKinematicReconstructor,
    WortbergPixelReconstructor,
)
from artist_extensions.loss_functions_ext import AlignmentLoss, ContourLoss
from utils.evaluation import evaluate_flux_accuracy, _gaussian_blur_batch
from reporting import (
    plot_gt_flux_grids,
    plot_contour_overlay,
    plot_pipeline_steps,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GT flux filtering + count normalisation
# ---------------------------------------------------------------------------

def _filter_flux_map(
    mapping: list,
    dataset_type: str,
    min_active_pct: float,
    synth_data_dir=None,
    split_name: str = "",
) -> list:
    """Remove mapping entries whose GT flux image is too sparse.

    Rejects samples where fewer than ``min_active_pct`` percent of pixels have
    values > 0.01 (normalised [0,1]).

    For real data the flux path comes directly from the mapping tuple.
    For synthetic data the flux path is constructed from ``synth_data_dir`` and
    ``split_name``; if ``synth_data_dir`` is None the filter is skipped.

    Filtering is especially important for synthetic data: the same fixed perturbation
    is applied to every sample of a heliostat, so perturbed heliostats that are badly
    mis-aimed will miss the target for some sun positions, producing empty flux images
    that must be excluded before training.

    Note: ``SyntheticDatasetParser`` reads samples by sequential index (0000…N-1),
    so removing an entry reduces the loaded sample count rather than skipping a
    specific on-disk index.
    """
    if dataset_type == "synthetic" and synth_data_dir is None:
        return mapping

    synth_base = pathlib.Path(synth_data_dir) if synth_data_dir is not None else None

    kept = []
    removed = 0
    for hid, cal_paths, flux_paths in mapping:
        if not cal_paths:
            kept.append((hid, cal_paths, flux_paths))
            continue
        ok_cal, ok_flux = [], []
        for idx, (cal, flux) in enumerate(zip(cal_paths, flux_paths)):
            if dataset_type == "synthetic":
                flux_to_check = synth_base / split_name / hid / f"{idx:04d}" / "flux_image.png"
            else:
                flux_to_check = flux
            try:
                img = np.array(_PIL_Image.open(flux_to_check)).astype(np.float32) / 255.0
                if float((img > 0.01).mean()) * 100.0 >= min_active_pct:
                    ok_cal.append(cal)
                    ok_flux.append(flux)
                else:
                    removed += 1
            except Exception:
                ok_cal.append(cal)
                ok_flux.append(flux)
        kept.append((hid, ok_cal, ok_flux))

    if removed:
        log.info(f"Flux filter removed {removed} sparse/empty samples from mapping.")
    return kept


def _normalize_mapping(mapping: list) -> list:
    """Drop heliostats with zero samples after flux filtering.

    Heliostats retain their individual sample counts — no padding or repetition.
    The training loop handles heterogeneous per-heliostat counts directly.
    """
    valid = [(hid, cal, flux) for hid, cal, flux in mapping if cal]
    n_dropped = len(mapping) - len(valid)
    if n_dropped:
        log.info(f"Mapping: {n_dropped} heliostats dropped (zero samples after filtering).")
    return valid




# ---------------------------------------------------------------------------
# Centroid trail capture (attached as Stage-2 epoch callback)
# ---------------------------------------------------------------------------

class _CentroidTrailRecorder:
    """Captures predicted centroid positions at regular Stage-2 epoch intervals.

    Preloads up to ``n_disp`` training samples per heliostat once at construction,
    then runs a lightweight forward pass every ``stride`` epochs to record where
    the predicted centroid is at that point in training.

    After training call ``save_trail_plots(trail_dir)`` to write one PNG per
    heliostat into ``trail_dir/``.
    """

    def __init__(
        self,
        train_parser,
        train_mapping: list,
        scenario,
        device: torch.device,
        stride: int,
        n_disp: int,
        bitmap_res: int = 256,
        blur_sigma: float = 0.0,
    ) -> None:
        self.stride      = stride
        self.n_disp      = n_disp
        self.scenario    = scenario
        self.device      = device
        self.bitmap_res  = torch.tensor([bitmap_res, bitmap_res])
        self.blur_sigma  = blur_sigma

        # Populated in _preload: hid → dict of tensors / metadata
        self._hid_data: dict[str, dict] = {}
        # hid → {epoch: [[E,N,U], ...]} per captured epoch
        self.trail_data: dict[str, dict[int, list]] = {}

        self._preload(train_parser, train_mapping)

    @torch.no_grad()
    def _preload(self, parser, mapping: list) -> None:
        """Parse training data once and keep per-heliostat subsets."""
        for heliostat_group in self.scenario.heliostat_field.heliostat_groups:
            try:
                (
                    measured_flux,
                    focal_spots,
                    incident_rays,
                    _,
                    active_mask,
                    target_mask,
                ) = parser.parse_data_for_reconstruction(
                    heliostat_data_mapping=mapping,
                    heliostat_group=heliostat_group,
                    scenario=self.scenario,
                    device=self.device,
                )
            except Exception as exc:
                log.warning(f"CentroidTrailRecorder preload failed: {exc}")
                return

            if active_mask.sum() == 0:
                continue

            active_indices  = torch.where(active_mask.bool())[0]
            samples_per_hel = active_mask[active_indices].long()

            # Target area geometry for ENU → pixel projection.
            planar  = self.scenario.solar_tower.target_areas[index_mapping.planar_target_areas]
            reference_target = planar.centers[:, :3].mean(dim=0).cpu()

            offset = 0
            for j, idx in enumerate(active_indices):
                hid    = heliostat_group.names[idx.item()]
                n      = int(samples_per_hel[j].item())
                n_show = min(n, self.n_disp)   # per-heliostat: counts differ after filtering

                # Flux images (background for the trail plot)
                flux_imgs = []
                for k in range(n_show):
                    img = measured_flux[offset + k]
                    mx  = img.max().item()
                    flux_imgs.append((img / max(mx, 1e-12)).cpu().numpy())

                dist_m = float(torch.norm(heliostat_group.positions[idx, :3].cpu() - reference_target))

                # Per-sample target area geometry (each sample can use a different area).
                sample_area_indices = target_mask[offset: offset + n_show].cpu().tolist()
                sample_centers = [planar.centers[int(a), :3].cpu().tolist() for a in sample_area_indices]
                sample_dims    = [planar.dimensions[int(a)].cpu().tolist()  for a in sample_area_indices]

                self._hid_data[hid] = {
                    "flux_imgs":      flux_imgs,
                    "gt_centroids":   focal_spots[offset: offset + n_show, :3].cpu().tolist(),
                    "rays":           incident_rays[offset: offset + n_show].clone(),
                    "target_mask":    target_mask[offset: offset + n_show].clone(),
                    "area_idx":       int(target_mask[offset].item()),
                    "target_centers": sample_centers,   # per-sample, list of [E,N,U]
                    "target_dims_list": sample_dims,    # per-sample, list of [w,h]
                    "dist_m":         dist_m,
                }
                self.trail_data[hid] = {}
                offset += n

    @torch.no_grad()
    def __call__(self, epoch: int) -> None:
        if epoch % self.stride != 0 or not self._hid_data:
            return

        bm_res = self.bitmap_res.to(self.device)

        for heliostat_group in self.scenario.heliostat_field.heliostat_groups:
            for hid, data in self._hid_data.items():
                if hid not in heliostat_group.names:
                    continue

                hel_idx    = list(heliostat_group.names).index(hid)
                rays       = data["rays"].to(self.device)
                tgt_mask   = data["target_mask"].to(self.device)
                n_show     = len(rays)

                sub_mask = torch.zeros(
                    len(heliostat_group.names), dtype=torch.long, device=self.device
                )
                sub_mask[hel_idx] = n_show

                heliostat_group.activate_heliostats(
                    active_heliostats_mask=sub_mask, device=self.device
                )
                kinematic = heliostat_group.kinematics

                if hasattr(kinematic, "_base_position_deviation"):
                    base_dev = kinematic._base_position_deviation.repeat_interleave(
                        sub_mask, dim=0
                    )
                    pad = torch.zeros(base_dev.shape[0], 1, device=self.device)
                    kinematic.active_heliostat_positions = (
                        kinematic.active_heliostat_positions + torch.cat([base_dev, pad], dim=1)
                    )

                heliostat_group.align_surfaces_with_incident_ray_directions(
                    aim_points=self.scenario.solar_tower.get_centers_of_target_areas(
                        tgt_mask, self.device
                    ),
                    incident_ray_directions=rays,
                    active_heliostats_mask=sub_mask,
                    device=self.device,
                )

                ray_tracer = HeliostatRayTracer(
                    scenario=self.scenario,
                    heliostat_group=heliostat_group,
                    blocking_active=False,
                    batch_size=n_show,
                    random_seed=epoch,
                    bitmap_resolution=bm_res,
                )
                flux, _, _, _ = ray_tracer.trace_rays(
                    incident_ray_directions=rays,
                    active_heliostats_mask=sub_mask,
                    target_area_indices=tgt_mask,
                    device=self.device,
                )
                sample_indices = ray_tracer.get_sampler_indices()
                inv_perm       = torch.argsort(sample_indices)
                flux_nat = flux[sample_indices][inv_perm]
                tgt_nat  = tgt_mask[sample_indices][inv_perm]

                # Apply the same blur+normalize as the training loop so the
                # recorded centroid trajectory matches the optimised loss signal.
                flux_pp  = _gaussian_blur_batch(flux_nat, sigma=self.blur_sigma)
                N_pp     = flux_pp.shape[0]
                peak_pp  = flux_pp.view(N_pp, -1).max(dim=1).values.clamp(min=1e-12)
                flux_pp  = flux_pp / peak_pp.view(N_pp, 1, 1)

                bm_coords = get_center_of_mass(bitmaps=flux_pp, device=self.device)
                centroids  = bitmap_coordinates_to_target_coordinates(
                    bitmap_coordinates=bm_coords,
                    bitmap_resolution=bm_res,
                    solar_tower=self.scenario.solar_tower,
                    target_area_indices=tgt_nat,
                    device=self.device,
                )

                self.trail_data[hid][epoch] = centroids[:, :3].cpu().tolist()

    def has_data(self) -> bool:
        return any(bool(v) for v in self.trail_data.values())

    def compute_mean_mrad_per_epoch(self) -> dict:
        """Return {epoch: mean_mrad} averaged over all heliostats and display samples."""
        epoch_errs: dict[int, list] = {}
        for hid, epoch_map in self.trail_data.items():
            if hid not in self._hid_data:
                continue
            gt_list = self._hid_data[hid]["gt_centroids"]
            dist_m  = self._hid_data[hid]["dist_m"]
            if dist_m <= 0:
                continue
            for ep, pred_list in epoch_map.items():
                if ep not in epoch_errs:
                    epoch_errs[ep] = []
                for gt, pred in zip(gt_list, pred_list):
                    err_m    = float(np.sqrt(sum((p - g) ** 2 for p, g in zip(pred, gt))))
                    epoch_errs[ep].append(err_m / dist_m * 1000.0)
        return {ep: float(np.mean(errs)) for ep, errs in sorted(epoch_errs.items())}

    def save_trail_plots(self, trail_dir) -> None:
        import pathlib
        from reporting import plot_centroid_trail_grids

        trail_dir = pathlib.Path(trail_dir)
        trail_dir.mkdir(parents=True, exist_ok=True)

        n_saved = 0
        for hid, epoch_map in self.trail_data.items():
            if not epoch_map or hid not in self._hid_data:
                continue
            data         = self._hid_data[hid]
            trail_epochs = sorted(epoch_map.keys())
            trail_cents  = {ep: epoch_map[ep] for ep in trail_epochs}

            plot_centroid_trail_grids(
                trail_dir=trail_dir,
                hid=hid,
                flux_images=data["flux_imgs"],
                gt_centroids_enu=data["gt_centroids"],
                trail_epochs=trail_epochs,
                trail_centroids_enu=trail_cents,
                target_centers=data["target_centers"],
                target_dims_list=data["target_dims_list"],
                dist_m=data.get("dist_m", 1.0),
            )
            n_saved += 1

        log.info(f"Centroid trail plots saved for {n_saved} heliostats → {trail_dir}")


# ---------------------------------------------------------------------------

_LOSS_CONFIGS: dict[str, tuple] = {
    "focal_spot": (WortbergKinematicReconstructor, lambda s: FocalSpotLoss(scenario=s)),
    "pixel":      (WortbergPixelReconstructor,     lambda s: PixelLoss(scenario=s)),
    "alignment":  (WortbergAlignmentReconstructor, lambda _: AlignmentLoss()),
    "contour":    (WortbergContourReconstructor,   None),  # loss built from contour_params
}


def _build_reconstructor(loss_type, scenario, ddp_setup, data, eval_data, optimization_config,
                          contour_params=None, blur_sigma: float = 0.0, **kwargs):
    if loss_type not in _LOSS_CONFIGS:
        raise ValueError(f"Unknown loss_type {loss_type!r}. Choose from {list(_LOSS_CONFIGS)}.")
    cls, loss_fn_factory = _LOSS_CONFIGS[loss_type]
    reconstructor = cls(
        ddp_setup=ddp_setup,
        scenario=scenario,
        data=data,
        optimization_configuration=optimization_config,
        reconstruction_method=config_dictionary.kinematics_reconstruction_raytracing,
        eval_data=eval_data,
        blur_sigma=blur_sigma,
        **kwargs,
    )
    if loss_type == "contour":
        loss_fn = ContourLoss(**(contour_params or {}))
    else:
        loss_fn = loss_fn_factory(scenario)
    return reconstructor, loss_fn


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
    perturbations: dict | None = None,
    heliostat_ids: list | None = None,
    stage1_epochs: int = 50,
    stage2_epochs: int = 250,
    contour_params: dict | None = None,
    trail_stride: int = 5,
    trail_n_disp: int = 25,
    min_focal_spot_samples: int = 0,
    blur_sigma: float = 0.0,
) -> dict:
    """
    Train on perturbed synthetic data for 63 heliostats using a two-stage approach:
      Stage 1 — AlignmentLoss (no ray tracing) for stage1_epochs.
      Stage 2 — loss_type for stage2_epochs.

    The scenario starts clean; the KR learns the perturbation values from the data.

    perturbations : dict keyed by heliostat ID (loaded from perturbations.json), used
                    only for param_recovery reporting — NOT applied to the scenario.
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
    # Pre-training eval: clean scenario vs perturbed test data → high mrad
    # ------------------------------------------------------------------
    log.info("Pre-training eval (clean scenario, perturbed test data) …")
    pre_t0 = time.time()
    pre_eval = evaluate_flux_accuracy(
        scenario=scenario,
        heliostat_data_mapping=test_mapping,
        data_parser=test_parser,
        device=device,
        blur_sigma=blur_sigma,
    )
    pre_eval_time_s = time.time() - pre_t0
    log.info(
        f"  pre-training: mean={pre_eval['mean_mrad']:.3f} mrad  "
        f"median={pre_eval['median_mrad']:.3f} mrad  n={pre_eval['num_samples']}"
    )

    # ------------------------------------------------------------------
    # GT flux grids (one PNG per split, independent of training)
    # ------------------------------------------------------------------
    log.info("Saving GT flux grids …")
    for _split_name, _split_parser, _split_mapping in [
        ("train", train_parser, train_mapping),
        ("val",   val_parser,   val_mapping),
        ("test",  test_parser,  test_mapping),
    ]:
        _gt_imgs = _collect_gt_images(_split_parser, _split_mapping, scenario, device)
        plot_gt_flux_grids(_gt_imgs, _split_name, output_dir)
    log.info(f"GT flux grids saved → {output_dir / 'gt_grids'}")

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

    log.info(f"Stage 1 — alignment pre-training ({stage1_epochs} epochs) …")

    trail_recorder_s1 = _CentroidTrailRecorder(
        train_parser=train_parser,
        train_mapping=train_mapping,
        scenario=scenario,
        device=device,
        stride=trail_stride,
        n_disp=trail_n_disp,
        blur_sigma=blur_sigma,
    )
    val_trail_s1 = _CentroidTrailRecorder(
        train_parser=val_parser,
        train_mapping=val_mapping,
        scenario=scenario,
        device=device,
        stride=trail_stride,
        n_disp=trail_n_disp,
        blur_sigma=blur_sigma,
    )

    def _s1_callback(epoch):
        trail_recorder_s1(epoch)
        val_trail_s1(epoch)

    stage1_reconstructor, stage1_loss_fn = _build_reconstructor(
        loss_type="alignment",
        scenario=scenario,
        ddp_setup=ddp_setup,
        data=data,
        eval_data=eval_data,
        optimization_config=stage1_config,
        train_position_deviation=True,
        epoch_callback=_s1_callback,
        blur_sigma=blur_sigma,
    )
    stage1_reconstructor.reconstruct_kinematics(loss_definition=stage1_loss_fn, device=device)
    stage1_history = stage1_reconstructor._convergence_history
    stage1_time_s = time.time() - t0
    _ram_after_stage1 = _ram_gb()
    log.info(f"Stage 1 done in {stage1_time_s / 60:.1f} min")

    log.info("Post-stage1 eval (alignment-trained scenario, perturbed test data) …")
    post_stage1_eval = evaluate_flux_accuracy(
        scenario=scenario,
        heliostat_data_mapping=test_mapping,
        data_parser=test_parser,
        device=device,
        blur_sigma=blur_sigma,
    )
    log.info(
        f"  post-stage1 : mean={post_stage1_eval['mean_mrad']:.3f} mrad  "
        f"median={post_stage1_eval['median_mrad']:.3f} mrad"
    )

    del stage1_reconstructor
    gc.collect()
    torch.cuda.empty_cache()

    # Per-heliostat training: check if this heliostat has enough valid train samples
    # for Stage-2 FocalSpotLoss. If not, return early with the Stage-1 result.
    n_train = len(train_mapping[0][1]) if train_mapping else 0
    if min_focal_spot_samples > 0 and n_train < min_focal_spot_samples:
        log.info(
            f"Only {n_train} valid train samples (< {min_focal_spot_samples}) — skipping Stage 2."
        )
        train_time = time.time() - t0
        early_results = {
            "pre_training":      {
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
            "post_stage1":       {
                "mean_mrad":         post_stage1_eval["mean_mrad"],
                "median_mrad":       post_stage1_eval["median_mrad"],
                "mean_m":            post_stage1_eval["mean_m"],
                "mean_pixel_loss":   post_stage1_eval["mean_pixel_loss"],
                "median_pixel_loss": post_stage1_eval["median_pixel_loss"],
                "num_samples":       post_stage1_eval["num_samples"],
                "num_nan_samples":   post_stage1_eval["num_nan_samples"],
                "nan_heliostat_ids": post_stage1_eval["nan_heliostat_ids"],
                "per_heliostat":     post_stage1_eval["per_heliostat"],
            },
            "post_training":     None,
            "post_training_val": None,
            "train_time_min":    round(train_time / 60, 2),
            "loss_type":         loss_type,
            "stage2_skipped":    True,
        }
        with open(output_dir / "results.json", "w") as f:
            json.dump(early_results, f, indent=2)
        with open(output_dir / "convergence_history_stage1.json", "w") as f:
            json.dump(stage1_history, f, indent=2)
        _save_kinematic_parameters(scenario, output_dir / "kinematic_parameters.json")
        return early_results

    train_map_focal = train_mapping
    val_map_focal   = val_mapping

    log.info(f"Stage 2a — {loss_type} fine-tuning on {len(train_map_focal)} heliostats ({stage2_epochs} epochs) …")
    t1 = time.time()

    trail_recorder = _CentroidTrailRecorder(
        train_parser=train_parser,
        train_mapping=train_map_focal,
        scenario=scenario,
        device=device,
        stride=trail_stride,
        n_disp=trail_n_disp,
        blur_sigma=blur_sigma,
    )
    val_trail_s2 = _CentroidTrailRecorder(
        train_parser=val_parser,
        train_mapping=val_map_focal,
        scenario=scenario,
        device=device,
        stride=trail_stride,
        n_disp=trail_n_disp,
        blur_sigma=blur_sigma,
    )

    def _s2_callback(epoch):
        trail_recorder(epoch)
        val_trail_s2(epoch)

    data_focal    = {config_dictionary.data_parser: train_parser,
                     config_dictionary.heliostat_data_mapping: train_map_focal}
    eval_data_focal = {"data_parser": val_parser,
                       "heliostat_data_mapping": val_map_focal}

    stage2_reconstructor, stage2_loss_fn = _build_reconstructor(
        loss_type=loss_type,
        scenario=scenario,
        ddp_setup=ddp_setup,
        data=data_focal,
        eval_data=eval_data_focal,
        optimization_config=stage2_config,
        contour_params=contour_params,
        train_position_deviation=True,
        epoch_callback=_s2_callback,
        blur_sigma=blur_sigma,
    )
    stage2_reconstructor.reconstruct_kinematics(loss_definition=stage2_loss_fn, device=device)
    stage2_history        = stage2_reconstructor._convergence_history
    stage2_kinematic_hist = stage2_reconstructor._kinematic_history
    stage2_time_s = time.time() - t1
    _ram_after_stage2 = _ram_gb()
    log.info(f"Stage 2a done in {stage2_time_s / 60:.1f} min")

    del stage2_reconstructor
    gc.collect()
    torch.cuda.empty_cache()

    train_time = time.time() - t0
    log.info(f"Total training time: {train_time / 60:.1f} min")

    if trail_recorder.has_data():
        trail_recorder.save_trail_plots(output_dir / "centroid_trails")
    else:
        log.warning("No centroid trail data captured — trail plots skipped.")

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
    if heliostat_ids is not None:
        kinematic_history = _build_kinematic_history(stage2_kinematic_hist, heliostat_ids)
        with open(output_dir / "kinematic_history.json", "w") as f:
            json.dump(kinematic_history, f, indent=2)

    # ------------------------------------------------------------------
    # Post-training eval: trained scenario vs perturbed test data → low mrad
    # ------------------------------------------------------------------
    log.info("Post-training eval (trained scenario, perturbed test data) …")
    post_t0 = time.time()
    post_train_eval = evaluate_flux_accuracy(
        scenario=scenario,
        heliostat_data_mapping=test_mapping,
        data_parser=test_parser,
        device=device,
        blur_sigma=blur_sigma,
    )
    post_train_eval_time_s = time.time() - post_t0
    log.info(
        f"  post-training: mean={post_train_eval['mean_mrad']:.3f} mrad  "
        f"median={post_train_eval['median_mrad']:.3f} mrad  "
        f"pixel_loss={post_train_eval['mean_pixel_loss']:.4f}"
    )

    hel_data = _collect_hel_data(
        scenario=scenario,
        test_parser=test_parser,
        test_mapping=test_mapping,
        device=device,
        dataset_type=dataset_type,
        blur_sigma=blur_sigma,
    )

    if loss_type == "contour" and hel_data:
        _save_contour_diagnostics(
            hel_data=hel_data,
            output_dir=output_dir,
            contour_params=contour_params,
        )

    # Post-training val eval — needed for the summary table.
    log.info("Post-training val eval …")
    post_train_val_eval = evaluate_flux_accuracy(
        scenario=scenario,
        heliostat_data_mapping=val_mapping,
        data_parser=val_parser,
        device=device,
        blur_sigma=blur_sigma,
    )
    log.info(
        f"  post-training val: mean={post_train_val_eval['mean_mrad']:.3f} mrad  "
        f"median={post_train_val_eval['median_mrad']:.3f} mrad"
    )

    _save_field_positions(scenario, output_dir / "field_positions.json")

    # Save per-epoch mrad trajectory (both stages) for the unified convergence plot.
    _save_mrad_trajectory(
        trail_s1=trail_recorder_s1,
        trail_s2=trail_recorder,
        val_trail_s1=val_trail_s1,
        val_trail_s2=val_trail_s2,
        stage1_last_epoch=stage1_history[-1]["epoch"] if stage1_history else 0,
        pre_mrad=pre_eval["mean_mrad"],
        post_s1_mrad=post_stage1_eval["mean_mrad"],
        post_mrad=post_train_eval["mean_mrad"],
        output_dir=output_dir,
    )

    recovery = None
    if perturbations is not None and heliostat_ids is not None:
        recovery = _param_recovery(scenario, perturbations, heliostat_ids, device)

    overall_time_s = time.time() - overall_t0
    _ram_end = _ram_gb()
    _ram_samples = [r for r in [_ram_start, _ram_after_stage1, _ram_after_stage2, _ram_end] if r is not None]
    timing = {
        "overall_s":                    round(overall_time_s, 1),
        "overall_min":                  round(overall_time_s / 60, 2),
        "pre_training_eval_s":          round(pre_eval_time_s, 1),
        "stage1_training_s":            round(stage1_time_s, 1),
        "stage1_training_min":          round(stage1_time_s / 60, 2),
        "stage2_training_s":            round(stage2_time_s, 1),
        "stage2_training_min":          round(stage2_time_s / 60, 2),
        "total_training_s":             round(train_time, 1),
        "total_training_min":           round(train_time / 60, 2),
        "post_training_eval_s":         round(post_train_eval_time_s, 1),
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

    results = {
        "pre_training": {
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
        "post_stage1": {
            "mean_mrad":         post_stage1_eval["mean_mrad"],
            "median_mrad":       post_stage1_eval["median_mrad"],
            "mean_m":            post_stage1_eval["mean_m"],
            "mean_pixel_loss":   post_stage1_eval["mean_pixel_loss"],
            "median_pixel_loss": post_stage1_eval["median_pixel_loss"],
            "num_samples":       post_stage1_eval["num_samples"],
            "num_nan_samples":   post_stage1_eval["num_nan_samples"],
            "nan_heliostat_ids": post_stage1_eval["nan_heliostat_ids"],
            "per_heliostat":     post_stage1_eval["per_heliostat"],
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
        "post_training_val": {
            "mean_mrad":         post_train_val_eval["mean_mrad"],
            "median_mrad":       post_train_val_eval["median_mrad"],
            "mean_m":            post_train_val_eval["mean_m"],
            "mean_pixel_loss":   post_train_val_eval["mean_pixel_loss"],
            "median_pixel_loss": post_train_val_eval["median_pixel_loss"],
            "num_samples":       post_train_val_eval["num_samples"],
            "per_heliostat":     post_train_val_eval["per_heliostat"],
        },
        "train_time_min":           round(train_time / 60, 2),
        "loss_type":                loss_type,
        "stage2_skipped":           False,
        "param_recovery":           recovery,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    _save_kinematic_parameters(scenario, output_dir / "kinematic_parameters.json")

    return results


# ---------------------------------------------------------------------------
# Mrad trajectory  (unified Stage-1 + Stage-2 convergence in mrad)
# ---------------------------------------------------------------------------

def _save_mrad_trajectory(
    trail_s1: "_CentroidTrailRecorder",
    trail_s2: "_CentroidTrailRecorder",
    val_trail_s1: "_CentroidTrailRecorder",
    val_trail_s2: "_CentroidTrailRecorder",
    stage1_last_epoch: int,
    pre_mrad: float,
    post_s1_mrad: float,
    post_mrad: float,
    output_dir,
) -> None:
    """Compute per-epoch mrad from both trail recorders and save to JSON.

    The JSON is consumed by ``reporting.plot_unified_mrad()`` in the reporting step.
    Stage-2 epochs are stored as-is (0-based); the offset is saved separately so
    the plotting function can align them on a combined x-axis.
    """
    import pathlib
    output_dir = pathlib.Path(output_dir)

    s1_mrad     = trail_s1.compute_mean_mrad_per_epoch()     if trail_s1.has_data()     else {}
    s2_mrad     = trail_s2.compute_mean_mrad_per_epoch()     if trail_s2.has_data()     else {}
    s1_val_mrad = val_trail_s1.compute_mean_mrad_per_epoch() if val_trail_s1.has_data() else {}
    s2_val_mrad = val_trail_s2.compute_mean_mrad_per_epoch() if val_trail_s2.has_data() else {}

    payload = {
        "stage1":              {str(k): v for k, v in s1_mrad.items()},
        "stage2":              {str(k): v for k, v in s2_mrad.items()},
        "stage1_val":          {str(k): v for k, v in s1_val_mrad.items()},
        "stage2_val":          {str(k): v for k, v in s2_val_mrad.items()},
        "stage2_epoch_offset": stage1_last_epoch + 1,
        "pre_training_mrad":   pre_mrad,
        "post_stage1_mrad":    post_s1_mrad,
        "post_training_mrad":  post_mrad,
    }
    path = output_dir / "mrad_trajectory.json"
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    log.info(f"mrad trajectory saved → {path}")


# ---------------------------------------------------------------------------
# GT flux image collection  (for gt_grids/ plots)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_gt_images(
    parser,
    mapping: list,
    scenario,
    device: torch.device,
    n_per_hel: int = 10,
) -> "dict[str, list[np.ndarray]]":
    """Return {hid: [H×W float32 normalised array, ...]} for each heliostat."""
    hel_images: dict = {}
    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        try:
            (
                measured_flux, _, _, _,
                active_mask, _,
            ) = parser.parse_data_for_reconstruction(
                heliostat_data_mapping=mapping,
                heliostat_group=heliostat_group,
                scenario=scenario,
                device=device,
            )
        except Exception as exc:
            log.warning(f"_collect_gt_images failed: {exc}")
            continue

        if active_mask.sum() == 0:
            continue

        active_indices  = torch.where(active_mask.bool())[0]
        samples_per_hel = active_mask[active_indices].long()

        offset = 0
        for j, idx in enumerate(active_indices):
            hid   = heliostat_group.names[idx.item()]
            n     = int(samples_per_hel[j].item())
            n_show = min(n, n_per_hel)

            imgs = []
            for k in range(n_show):
                img = measured_flux[offset + k]
                mx  = img.max().item()
                imgs.append((img / max(mx, 1e-12)).cpu().numpy())

            hel_images[hid] = imgs
            offset += n

    return hel_images


# ---------------------------------------------------------------------------
# Per-heliostat data collection (used for contour diagnostics)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_hel_data(scenario, test_parser, test_mapping, device, dataset_type="synthetic",
                      blur_sigma: float = 0.0):
    """
    Identify the best and worst heliostats by post-training FSE and save a
    10-row × 5-pair flux grid (measured | predicted) for each.
    """
    bitmap_resolution = torch.tensor([256, 256])

    # Collect per-heliostat image lists and mean FSE.
    hel_data: dict[str, dict] = {}   # hid -> {measured, predicted, mean_mrad}

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

        sample_indices = ray_tracer.get_sampler_indices()
        inv_perm       = torch.argsort(sample_indices)

        # Subset to sampler-covered instances before preprocessing (mirrors evaluate_flux_accuracy).
        # The ARTIST sampler may drop a few samples (rounding), so N_actual <= N_total.
        predicted_sampler = predicted_sampler[sample_indices]  # [N_actual, H, W]
        measured_natural  = measured_flux[sample_indices][inv_perm]  # [N_actual, H, W], natural order

        # Blur + peak-normalise predicted flux to match the training loss preprocessing.
        pred_blurred = _gaussian_blur_batch(predicted_sampler, sigma=blur_sigma)
        N_pred = pred_blurred.shape[0]
        pred_peak = pred_blurred.view(N_pred, -1).max(dim=1).values.clamp(min=1e-12)
        predicted_sampler_pp = pred_blurred / pred_peak.view(N_pred, 1, 1)
        predicted_natural = predicted_sampler_pp[inv_perm]

        bitmap_coords = get_center_of_mass(bitmaps=predicted_sampler_pp, device=device)
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

        # Compute actual per-heliostat counts from sample_indices (drops from sampler rounding).
        reference_target = scenario.solar_tower.target_areas[
            index_mapping.planar_target_areas
        ].centers[:, :3].mean(dim=0).to(device)
        active_indices  = torch.where(active_heliostats_mask.bool())[0]
        distances       = torch.norm(
            heliostat_group.positions[active_indices, :3].to(device) - reference_target, dim=1
        )
        samples_per_hel = active_heliostats_mask[active_indices].long()

        sorted_natural_indices = sample_indices[inv_perm]  # ascending natural indices

        hel_offsets = [0]
        for cnt in samples_per_hel:
            hel_offsets.append(hel_offsets[-1] + cnt.item())

        actual_offset = 0
        for j, idx in enumerate(active_indices):
            hid = heliostat_group.names[idx.item()]
            dist_m = distances[j].item()

            hel_start = hel_offsets[j]
            hel_end   = hel_offsets[j + 1]
            n_actual  = int(
                ((sorted_natural_indices >= hel_start) & (sorted_natural_indices < hel_end)).sum()
            )

            meas_slice = measured_natural[actual_offset: actual_offset + n_actual].cpu()
            pred_slice = predicted_natural[actual_offset: actual_offset + n_actual].cpu()
            fse_slice  = fse_natural[actual_offset: actual_offset + n_actual]

            # Build peak-normalised image lists.
            # predicted_natural is already blurred+normalised; GT PNG is pre-blurred at
            # generation time, so just peak-normalise when loading.
            meas_imgs, pred_imgs = [], []
            for k in range(n_actual):
                m = meas_slice[k]
                p = pred_slice[k]  # already blurred + peak-normalised
                p_vis = p.numpy()
                m_vis = (m / m.max().clamp(min=1e-12)).numpy()
                meas_imgs.append(m_vis)
                pred_imgs.append(p_vis)

            # Mean FSE in mrad for this heliostat
            fse_vals  = fse_slice.cpu().numpy()
            valid_fse = fse_vals[np.isfinite(fse_vals)]
            if len(valid_fse) > 0 and dist_m > 0:
                mean_mrad = float(np.mean(valid_fse) / dist_m * 1000.0)
            else:
                mean_mrad = float("nan")

            hel_data[hid] = {
                "measured":         meas_imgs,
                "predicted":        pred_imgs,
                "mean_mrad":        mean_mrad,
                # Raw tensors kept for contour overlay / pipeline plots.
                "measured_tensors":  [meas_slice[k].unsqueeze(0) for k in range(n_actual)],
                "predicted_tensors": [pred_slice[k].unsqueeze(0) for k in range(n_actual)],
            }
            actual_offset += n_actual

    if not hel_data:
        log.warning("No heliostat data collected.")
        return {}

    return hel_data


# ---------------------------------------------------------------------------
# Contour-specific diagnostics
# ---------------------------------------------------------------------------

def _save_contour_diagnostics(
    hel_data: dict,
    output_dir,
    contour_params: dict | None = None,
) -> None:
    """Save contour overlay, pipeline steps, and per-heliostat flux grids."""
    valid = {h: d["mean_mrad"] for h, d in hel_data.items() if np.isfinite(d["mean_mrad"])}
    if not valid:
        log.warning("No valid mrad values — skipping contour diagnostics.")
        return

    best_hid  = min(valid, key=valid.get)
    worst_hid = max(valid, key=valid.get)

    for role, hid in [("best", best_hid), ("worst", worst_hid)]:
        d = hel_data[hid]
        m_tensors = d.get("measured_tensors", [])
        p_tensors = d.get("predicted_tensors", [])
        if not m_tensors or not p_tensors:
            continue

        plot_contour_overlay(
            measured_tensors=m_tensors,
            predicted_tensors=p_tensors,
            heliostat_id=hid,
            mean_mrad=d["mean_mrad"],
            role=role,
            output_dir=output_dir,
            contour_params=contour_params,
        )
        log.info(f"Contour overlay saved: {role} heliostat {hid}")

        plot_pipeline_steps(
            measured_tensor=m_tensors[0],
            predicted_tensor=p_tensors[0],
            heliostat_id=hid,
            mean_mrad=d["mean_mrad"],
            role=role,
            output_dir=output_dir,
            contour_params=contour_params,
        )
        log.info(f"Pipeline steps saved: {role} heliostat {hid}")


# ---------------------------------------------------------------------------
# Field positions
# ---------------------------------------------------------------------------

def _save_field_positions(scenario, path) -> None:
    import pathlib
    path = pathlib.Path(path)
    heliostat_group = scenario.heliostat_field.heliostat_groups[0]
    positions_enu   = heliostat_group.positions[:, :3].detach().cpu().tolist()
    names           = list(heliostat_group.names)
    tower_enu       = (
        scenario.solar_tower.target_areas[index_mapping.planar_target_areas]
        .centers[:, :3].mean(dim=0).cpu().tolist()
    )
    payload = {
        "heliostat_ids": names,
        "positions_enu": positions_enu,
        "tower_enu":     tower_enu,
    }
    with open(path, "w") as fh:
        import json as _json
        _json.dump(payload, fh, indent=2)
    log.info(f"Field positions saved → {path}")


# ---------------------------------------------------------------------------
# Parameter recovery — residual = |trained - perturbation|
# ---------------------------------------------------------------------------

def _param_recovery(scenario, perturbations_by_id: dict, heliostat_ids: list, device: torch.device) -> dict:
    """
    Compare trained kinematic parameters against the ground-truth perturbations.

    In the corrected experiment the KR starts from zero and converges toward the
    perturbation values, so residual = |trained - perturbation| (lower is better).
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
        rec_act = kinematic.actuators.optimizable_parameters[
            i, index_mapping.actuator_initial_angle, :
        ].detach().cpu().tolist()
        # Actuator angle: compare change from initial vs perturbation
        start_ang = (
            kinematic._initial_actuator_angle[i].cpu()
            if hasattr(kinematic, "_initial_actuator_angle")
            else kinematic.actuators.optimizable_parameters[i, index_mapping.actuator_initial_angle, :].detach().cpu()
        )
        moved_act = (kinematic.actuators.optimizable_parameters[
            i, index_mapping.actuator_initial_angle, :
        ].detach().cpu() - start_ang).tolist()
        residual_act = [abs(m - p) for m, p in zip(moved_act, perturbation_act)]

        perturbation_offset = pert["actuator_offset_m"]
        start_off = (
            kinematic._initial_actuator_offset[i].cpu()
            if hasattr(kinematic, "_initial_actuator_offset")
            else kinematic.actuators.non_optimizable_parameters[i, index_mapping.actuator_offset, :].detach().cpu()
        )
        moved_off = (kinematic.actuators.non_optimizable_parameters[
            i, index_mapping.actuator_offset, :
        ].detach().cpu() - start_off).tolist()
        residual_off = [abs(m - p) for m, p in zip(moved_off, perturbation_offset)]

        perturbation_trans = pert["translation_m"]
        start_trans = (
            kinematic._initial_translation[i].cpu()
            if hasattr(kinematic, "_initial_translation")
            else kinematic.translation_deviation_parameters[i].detach().cpu()
        )
        moved_trans = (kinematic.translation_deviation_parameters[i].detach().cpu() - start_trans).tolist()
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
# Kinematic history / parameter export
# ---------------------------------------------------------------------------

def _build_kinematic_history(raw_history: list, heliostat_ids: list | None) -> list:
    if not raw_history or heliostat_ids is None:
        return raw_history or []
    result = []
    for entry in raw_history:
        hel_data = {}
        for i, hid in enumerate(heliostat_ids):
            hel_data[hid] = {
                "rotation_rad":                 entry["rotation_rad"][i]                   if entry.get("rotation_rad") else None,
                "actuator_angle_deviation_rad": entry["actuator_angle_deviation_rad"][i]   if entry.get("actuator_angle_deviation_rad") else None,
                "actuator_offset_deviation_m":  entry["actuator_offset_deviation_m"][i]    if entry.get("actuator_offset_deviation_m") else None,
                "base_position_m":              entry["base_position_m"][i]                if entry.get("base_position_m") else None,
            }
        result.append({"epoch": entry["epoch"], "heliostats": hel_data})
    return result


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
