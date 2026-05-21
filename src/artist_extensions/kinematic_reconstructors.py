"""Kinematic reconstructor subclasses for Wortberg (2025) Table 5.3 parameter set.

Class hierarchy
---------------
KinematicsReconstructor (ARTIST)
└── WortbergKinematicReconstructor    — Wortberg params, focal-spot or pixel loss, ray tracing
    ├── WortbergPixelReconstructor    — overrides loss: Gaussian blur + peak-normalised bitmaps
    └── WortbergAlignmentReconstructor — replaces ray tracing with motor-position MSE (stage 1)
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from artist.optim import training, mean_loss_per_heliostat
from artist.optim.kinematics_reconstructor import KinematicsReconstructor
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
from artist.util import constants, indices, get_device

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base Wortberg reconstructor — focal-spot loss via ray tracing
# ---------------------------------------------------------------------------

class WortbergKinematicReconstructor(KinematicsReconstructor):
    """KinematicsReconstructor with the parameter set from Wortberg (2025) Table 5.3.

    Optimised parameters
    --------------------
    - ``translation_deviation_parameters`` (all 9 joints + concentrator), ±0.05 m
    - ``rotation_deviation_parameters`` (all 4 tilts), ±0.005 rad
    - ``actuators.optimizable_parameters[:, actuator_initial_angle]`` (aᵢ), ±0.005 rad
    - ``actuators.non_optimizable_parameters[:, actuator_offset]`` (cᵢ), ±0.005 m
    - ``kinematics._base_position_deviation`` (δe, δn, δu per heliostat), ±0.05 m [optional]

    Frozen
    ------
    - ``actuators.optimizable_parameters[:, actuator_initial_stroke_length]`` (bᵢ)

    Extensions over ARTIST's base KinematicsReconstructor
    ------------------------------------------------------
    - Sample mini-batching: splits calibration samples into sub-batches to avoid OOM
      on large fields (e.g. 63 heliostats × 50 samples)
    - Eval-loss caching: parses validation data once per group, not every epoch
    - Best-validation-loss checkpointing with parameter restore
    - Gradient clipping (max_norm=1.0)
    - ReduceLROnPlateau or CosineAnnealingLR scheduler
    """

    # Deviation bounds from Wortberg (2025) Table 5.3.
    _BOUND_TRANSLATION_M      = 0.05   # joint and concentrator translations
    _BOUND_ROTATION_RAD       = 0.005  # joint tilts
    _BOUND_ACTUATOR_ANGLE_RAD = 0.005  # aᵢ
    _BOUND_ACTUATOR_OFFSET_M  = 0.005  # cᵢ
    _BOUND_BASE_POSITION_M    = 0.05   # heliostat base position (e, n, u)

    def __init__(
        self,
        *args,
        train_position_deviation: bool = True,
        eval_data: dict | None = None,
        sample_mini_batch_size: int | None = None,
        **kwargs,
    ) -> None:
        # Accept a flat optimization config dict (as used in the experiments) and
        # restructure it into the nested format expected by ARTIST's base class.
        if "optimization_configuration" in kwargs:
            flat = kwargs["optimization_configuration"]
            if constants.optimization not in flat:
                sched_params = flat.pop("scheduler_parameters", {})
                kwargs["optimization_configuration"] = {
                    constants.optimization: {
                        k: v for k, v in flat.items() if k != constants.scheduler
                    },
                    constants.scheduler: {
                        constants.scheduler_type: flat.get(
                            constants.scheduler, constants.reduce_on_plateau
                        ),
                        **sched_params,
                    },
                }
        super().__init__(*args, **kwargs)
        self.train_position_deviation = train_position_deviation
        self.eval_data = eval_data
        self.sample_mini_batch_size = sample_mini_batch_size

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def _reconstruct_kinematics_parameters_with_raytracing(
        self,
        loss_definition,
        device=None,
    ):
        device = get_device(device=device)
        rank = self.ddp_setup[constants.rank]
        opt  = self.optimizer_dict

        if rank == 0:
            log.info("Beginning kinematic reconstruction (Wortberg 2025 Table 5.3).")

        final_loss = torch.full(
            (self.scenario.heliostat_field.number_of_heliostats_per_group.sum(),),
            torch.inf,
            device=device,
        )
        group_offsets = torch.cat([
            torch.tensor([0], device=device),
            self.scenario.heliostat_field.number_of_heliostats_per_group.cumsum(
                indices.heliostat_dimension
            ),
        ])

        self._convergence_history = []
        self._kinematic_history   = []

        for group_idx in self.ddp_setup[constants.groups_to_ranks_mapping][rank]:
            group   = self.scenario.heliostat_field.heliostat_groups[group_idx]
            parser  = self.data[constants.data_parser]
            mapping = self.data[constants.heliostat_data_mapping]

            measured_flux, focal_spots, ray_dirs, _, active_mask, target_mask = (
                parser.parse_data_for_reconstruction(
                    heliostat_data_mapping=mapping,
                    heliostat_group=group,
                    scenario=self.scenario,
                    device=device,
                )
            )

            if active_mask.sum() == 0:
                continue

            gt, rdims    = self._get_ground_truth_and_reduction_dims(measured_flux, focal_spots)
            optimizer, init_angle, init_offset = self._setup_optimizer(group, device)
            scheduler    = self._setup_scheduler(optimizer)
            stopper      = self._setup_early_stopper()
            eval_cache   = self._cache_eval_data(group, device)

            # Mini-batch setup.
            n_samples  = int(active_mask[active_mask > 0][0].item())
            n_active   = int((active_mask > 0).sum().item())
            mb_size    = self.sample_mini_batch_size or n_samples
            n_mb       = (n_samples + mb_size - 1) // mb_size
            h_offsets  = torch.arange(n_active, device=device) * n_samples

            if n_mb > 1:
                log.info(f"Mini-batching: {n_samples} samples → {n_mb} batches of ≤{mb_size}.")

            # Loop-scoped variables read after the while-loop.
            sample_indices   = torch.arange(n_active * mb_size, device=device)
            current_mb_size  = mb_size
            loss             = torch.inf
            epoch            = 0
            log_step         = opt[constants.max_epoch] if opt[constants.log_step] == 0 else opt[constants.log_step]
            best_eval, best_snap = float("inf"), None

            pbar = tqdm(total=opt[constants.max_epoch], desc="Training", unit="ep",
                        dynamic_ncols=True, leave=True)

            while loss > float(opt[constants.tolerance]) and epoch <= opt[constants.max_epoch]:
                optimizer.zero_grad()
                kinematic    = group.kinematics
                loss_accum   = None

                for mb in range(n_mb):
                    s, e = mb * mb_size, min((mb + 1) * mb_size, n_samples)
                    current_mb_size = e - s
                    sample_range = torch.arange(s, e, device=device)
                    idx = (h_offsets.unsqueeze(1) + sample_range.unsqueeze(0)).reshape(-1)
                    sub_mask = (active_mask > 0).to(torch.long) * current_mb_size

                    group.activate_heliostats(active_heliostats_mask=sub_mask, device=device)
                    self._inject_base_position(kinematic, sub_mask, device)

                    group.align_surfaces_with_incident_ray_directions(
                        aim_points=self.scenario.solar_tower.get_centers_of_target_areas(
                            target_mask[idx], device
                        ),
                        incident_ray_directions=ray_dirs[idx],
                        active_heliostats_mask=sub_mask,
                        device=device,
                    )

                    ray_tracer = HeliostatRayTracer(
                        scenario=self.scenario,
                        heliostat_group=group,
                        blocking_active=False,
                        world_size=self.ddp_setup[constants.heliostat_group_world_size],
                        rank=self.ddp_setup[constants.heliostat_group_rank],
                        batch_size=opt[constants.batch_size],
                        random_seed=self.ddp_setup[constants.heliostat_group_rank],
                        bitmap_resolution=self.bitmap_resolution,
                    )
                    flux, _, _, _ = ray_tracer.trace_rays(
                        incident_ray_directions=ray_dirs[idx],
                        active_heliostats_mask=sub_mask,
                        target_area_indices=target_mask[idx],
                        device=device,
                    )
                    sample_indices  = ray_tracer.get_sampler_indices()
                    current_mb_size = e - s

                    lph = self._compute_epoch_loss(
                        epoch=epoch,
                        flux=flux,
                        measured_flux=measured_flux[idx],
                        focal_spots=focal_spots[idx],
                        sample_indices=sample_indices,
                        target_indices=target_mask[idx],
                        loss_fn=loss_definition,
                        ground_truth=gt[idx],
                        reduction_dims=rdims,
                        n_samples_per_heliostat=current_mb_size,
                        device=device,
                    )
                    (lph.mean() / n_mb).backward()
                    loss_accum = lph.detach() / n_mb if loss_accum is None else loss_accum + lph.detach() / n_mb

                loss_per_heliostat = loss_accum
                loss = loss_per_heliostat.mean()

                if self.ddp_setup[constants.is_nested]:
                    for pg in optimizer.param_groups:
                        for p in pg["params"]:
                            if p.grad is not None:
                                p.grad = torch.distributed.nn.functional.all_reduce(
                                    p.grad,
                                    op=torch.distributed.ReduceOp.SUM,
                                    group=self.ddp_setup["process_subgroup"],
                                )
                                p.grad /= self.ddp_setup[constants.heliostat_group_world_size]

                self._clip_gradients(kinematic)
                optimizer.step()
                self._apply_deviation_bounds(group, init_angle, init_offset)

                eval_loss = self._eval_loss_from_cache(eval_cache, group, loss_definition, device)
                if eval_loss is not None and eval_loss < best_eval:
                    best_eval, best_snap = eval_loss, self._snapshot(kinematic)

                sched_loss = eval_loss if eval_loss is not None else loss.detach()
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(sched_loss)
                else:
                    scheduler.step()

                if epoch % log_step == 0:
                    self._log_epoch(rank, epoch, loss, eval_loss, optimizer,
                                    kinematic, group_idx, init_angle, init_offset)

                if stopper.step(loss):
                    log.info(f"Rank {rank}: early stopping at epoch {epoch}.")
                    pbar.close()
                    break

                lr = optimizer.param_groups[indices.optimizer_param_group_0]["lr"]
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    eval=f"{eval_loss:.4f}" if eval_loss is not None else "-",
                    lr=f"{lr:.2e}",
                )
                pbar.update(1)
                epoch += 1
            else:
                pbar.close()

            if best_snap is not None:
                self._restore(kinematic, best_snap)
                log.info(f"Rank {rank}: restored best params (eval={best_eval:.6f}).")

            local_idx = sample_indices[::current_mb_size] // current_mb_size
            global_active = torch.nonzero(active_mask != 0, as_tuple=True)[0]
            final_loss[global_active[local_idx] + group_offsets[group_idx]] = loss_per_heliostat
            log.info(f"Rank {rank}: group {group_idx} reconstructed.")

        if self.ddp_setup[constants.is_distributed]:
            self._broadcast_parameters()
            torch.distributed.all_reduce(final_loss, op=torch.distributed.ReduceOp.MIN)
            log.info(f"Rank {rank}: synchronized.")

        return final_loss, self._convergence_history

    # ------------------------------------------------------------------
    # Overridable hooks (used by WortbergPixelReconstructor)
    # ------------------------------------------------------------------

    def _get_ground_truth_and_reduction_dims(self, measured_flux, focal_spots):
        """Return (ground_truth, reduction_dims) for the loss call.
        Default: focal spot centroids (FocalSpotLoss). Override for pixel loss."""
        return focal_spots, (indices.focal_spots,)

    def _preprocess_eval_flux(self, flux, ground_truth):
        """Transform predicted flux and ground truth before eval loss.
        Default: identity. Override in WortbergPixelReconstructor."""
        return flux, ground_truth

    def _compute_epoch_loss(
        self, epoch, flux, measured_flux, focal_spots, sample_indices,
        target_indices, loss_fn, ground_truth, reduction_dims,
        n_samples_per_heliostat, device,
    ) -> torch.Tensor:
        """Compute per-heliostat loss for one forward pass."""
        lps = loss_fn(
            prediction=flux,
            ground_truth=ground_truth[sample_indices],
            target_area_indices=target_indices[sample_indices],
            reduction_dimensions=reduction_dims,
            device=device,
        )
        return mean_loss_per_heliostat(lps, n_samples_per_heliostat)

    # ------------------------------------------------------------------
    # Optimizer / scheduler / early stopping setup
    # ------------------------------------------------------------------

    def _setup_optimizer(self, group, device):
        """Enable gradients for all Wortberg parameters and return a configured Adam."""
        kinematic = group.kinematics
        opt = self.optimizer_dict

        kinematic.translation_deviation_parameters.requires_grad_()
        kinematic.rotation_deviation_parameters.requires_grad_()
        kinematic.actuators.optimizable_parameters.requires_grad_()
        kinematic.actuators.non_optimizable_parameters.requires_grad_()

        # Freeze bᵢ (initial_stroke_length).
        def _freeze_stroke(grad):
            mask = torch.ones_like(grad)
            mask[:, indices.actuator_initial_stroke_length, :] = 0.0
            return grad * mask

        # Restrict non_optimizable gradients to cᵢ (actuator_offset) only.
        def _only_c_i(grad):
            mask = torch.zeros_like(grad)
            mask[:, indices.actuator_offset, :] = 1.0
            return grad * mask

        kinematic.actuators.optimizable_parameters.register_hook(_freeze_stroke)
        kinematic.actuators.non_optimizable_parameters.register_hook(_only_c_i)

        if self.train_position_deviation:
            if hasattr(kinematic, "_base_position_deviation"):
                # Phase 2: re-enable grad on existing values so Phase 1 results carry over.
                kinematic._base_position_deviation = (
                    kinematic._base_position_deviation.detach().requires_grad_(True)
                )
            else:
                # Phase 1 or single-stage: initialise to zero.
                kinematic._base_position_deviation = torch.zeros(
                    kinematic.number_of_heliostats, 3, device=device, requires_grad=True
                )

        # Snapshot initial values once (Phase 1) for deviation-bound clamping.
        if not hasattr(kinematic, "_initial_actuator_angle"):
            kinematic._initial_actuator_angle = (
                kinematic.actuators.optimizable_parameters[
                    :, indices.actuator_initial_angle, :
                ].detach().clone()
            )
            kinematic._initial_actuator_offset = (
                kinematic.actuators.non_optimizable_parameters[
                    :, indices.actuator_offset, :
                ].detach().clone()
            )
        if not hasattr(kinematic, "_initial_translation"):
            kinematic._initial_translation = kinematic.translation_deviation_parameters.detach().clone()

        lr = float(opt[constants.initial_learning_rate])
        # Large-scale params (±0.05 m) get 5× the base LR; small-scale (±0.005) use base LR.
        param_groups = [
            {"params": kinematic.translation_deviation_parameters,          "lr": lr * 5.0},
            {"params": kinematic.rotation_deviation_parameters,             "lr": lr},
            {"params": kinematic.actuators.optimizable_parameters,          "lr": lr},
            {"params": kinematic.actuators.non_optimizable_parameters,      "lr": lr},
        ]
        if self.train_position_deviation:
            param_groups.append(
                {"params": kinematic._base_position_deviation, "lr": lr * 5.0}
            )

        return (
            torch.optim.Adam(param_groups, lr=lr),
            kinematic._initial_actuator_angle,
            kinematic._initial_actuator_offset,
        )

    def _setup_scheduler(self, optimizer):
        """Create an LR scheduler from self.scheduler_dict."""
        stype  = self.scheduler_dict.get(constants.scheduler_type, constants.reduce_on_plateau)
        params = self.scheduler_dict
        if stype == constants.reduce_on_plateau:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min",
                factor=params.get("reduce_factor", 0.5),
                patience=params.get("patience", 10),
                threshold=params.get("threshold", 1e-4),
                cooldown=params.get("cooldown", 5),
                min_lr=params.get("lr_min", 1e-8),
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.optimizer_dict[constants.max_epoch],
            eta_min=params.get("lr_min", 1e-6),
        )

    def _setup_early_stopper(self):
        opt = self.optimizer_dict
        return training.EarlyStopping(
            window_size=opt[constants.early_stopping_window],
            patience=opt[constants.early_stopping_patience],
            min_improvement=opt[constants.early_stopping_delta],
            relative=True,
        )

    # ------------------------------------------------------------------
    # Eval helpers
    # ------------------------------------------------------------------

    def _cache_eval_data(self, group, device):
        """Parse validation data once and cache tensors, or return None."""
        if self.eval_data is None:
            return None
        mf, fs, rd, _, am, tm = self.eval_data["data_parser"].parse_data_for_reconstruction(
            heliostat_data_mapping=self.eval_data["heliostat_data_mapping"],
            heliostat_group=group,
            scenario=self.scenario,
            device=device,
        )
        return (mf, fs, rd, am, tm) if am.sum() > 0 else None

    @torch.no_grad()
    def _eval_loss_from_cache(self, cache, group, loss_fn, device) -> float | None:
        """Compute eval loss from pre-parsed tensors (no PNG re-loading)."""
        if cache is None:
            return None
        measured_flux, focal_spots, ray_dirs, active_mask, target_mask = cache
        opt = self.optimizer_dict

        group.activate_heliostats(active_heliostats_mask=active_mask, device=device)
        kinematic = group.kinematics
        self._inject_base_position(kinematic, active_mask, device)

        group.align_surfaces_with_incident_ray_directions(
            aim_points=self.scenario.solar_tower.get_centers_of_target_areas(target_mask, device),
            incident_ray_directions=ray_dirs,
            active_heliostats_mask=active_mask,
            device=device,
        )
        ray_tracer = HeliostatRayTracer(
            scenario=self.scenario,
            heliostat_group=group,
            blocking_active=False,
            world_size=self.ddp_setup[constants.heliostat_group_world_size],
            rank=self.ddp_setup[constants.heliostat_group_rank],
            batch_size=opt[constants.batch_size],
            random_seed=self.ddp_setup[constants.heliostat_group_rank],
            bitmap_resolution=self.bitmap_resolution,
        )
        flux, _, _, _ = ray_tracer.trace_rays(
            incident_ray_directions=ray_dirs,
            active_heliostats_mask=active_mask,
            target_area_indices=target_mask,
            device=device,
        )
        sample_idx = ray_tracer.get_sampler_indices()
        n = int(active_mask.sum() / (active_mask > 0).sum())

        gt, rdims = self._get_ground_truth_and_reduction_dims(measured_flux, focal_spots)
        flux_pp, gt_pp = self._preprocess_eval_flux(flux, gt)
        lps = loss_fn(
            prediction=flux_pp,
            ground_truth=gt_pp[sample_idx],
            target_area_indices=target_mask[sample_idx],
            reduction_dimensions=rdims,
            device=device,
        )
        return mean_loss_per_heliostat(lps, n).mean().item()

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _inject_base_position(self, kinematic, active_mask, device):
        """Add _base_position_deviation to active_heliostat_positions (autograd-safe)."""
        if self.train_position_deviation and hasattr(kinematic, "_base_position_deviation"):
            dev = kinematic._base_position_deviation.repeat_interleave(active_mask, dim=0)
            pad = torch.zeros(dev.shape[0], 1, device=device)
            kinematic.active_heliostat_positions = (
                kinematic.active_heliostat_positions + torch.cat([dev, pad], dim=1)
            )

    def _clip_gradients(self, kinematic):
        params = [
            kinematic.translation_deviation_parameters,
            kinematic.rotation_deviation_parameters,
            kinematic.actuators.optimizable_parameters,
            kinematic.actuators.non_optimizable_parameters,
        ]
        if self.train_position_deviation and hasattr(kinematic, "_base_position_deviation"):
            params.append(kinematic._base_position_deviation)
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

    def _apply_deviation_bounds(self, group, init_angle, init_offset):
        """Clamp all optimised parameters to Wortberg Table 5.3 bounds."""
        k = group.kinematics
        with torch.no_grad():
            k.translation_deviation_parameters.data.clamp_(
                k._initial_translation - self._BOUND_TRANSLATION_M,
                k._initial_translation + self._BOUND_TRANSLATION_M,
            )
            k.rotation_deviation_parameters.data.clamp_(
                -self._BOUND_ROTATION_RAD, self._BOUND_ROTATION_RAD
            )
            k.actuators.optimizable_parameters.data[:, indices.actuator_initial_angle, :].clamp_(
                init_angle - self._BOUND_ACTUATOR_ANGLE_RAD,
                init_angle + self._BOUND_ACTUATOR_ANGLE_RAD,
            )
            k.actuators.non_optimizable_parameters.data[:, indices.actuator_offset, :].clamp_(
                init_offset - self._BOUND_ACTUATOR_OFFSET_M,
                init_offset + self._BOUND_ACTUATOR_OFFSET_M,
            )
            if self.train_position_deviation and hasattr(k, "_base_position_deviation"):
                k._base_position_deviation.data.clamp_(
                    -self._BOUND_BASE_POSITION_M, self._BOUND_BASE_POSITION_M
                )

    def _snapshot(self, kinematic) -> dict:
        snap = {
            "translation": kinematic.translation_deviation_parameters.detach().clone(),
            "rotation":    kinematic.rotation_deviation_parameters.detach().clone(),
            "opt":         kinematic.actuators.optimizable_parameters.detach().clone(),
            "nonopt":      kinematic.actuators.non_optimizable_parameters.detach().clone(),
        }
        if hasattr(kinematic, "_base_position_deviation"):
            snap["base_pos"] = kinematic._base_position_deviation.detach().clone()
        return snap

    def _restore(self, kinematic, snap: dict) -> None:
        with torch.no_grad():
            kinematic.translation_deviation_parameters.copy_(snap["translation"])
            kinematic.rotation_deviation_parameters.copy_(snap["rotation"])
            kinematic.actuators.optimizable_parameters.copy_(snap["opt"])
            kinematic.actuators.non_optimizable_parameters.copy_(snap["nonopt"])
            if "base_pos" in snap and hasattr(kinematic, "_base_position_deviation"):
                kinematic._base_position_deviation.copy_(snap["base_pos"])

    def _broadcast_parameters(self):
        """Broadcast optimised parameters from the owning rank to all others (DDP)."""
        for idx, group in enumerate(self.scenario.heliostat_field.heliostat_groups):
            src = self.ddp_setup[constants.ranks_to_groups_mapping][idx][
                indices.first_rank_from_group
            ]
            k = group.kinematics
            for tensor in [
                k.translation_deviation_parameters,
                k.rotation_deviation_parameters,
                k.actuators.optimizable_parameters,
                k.actuators.non_optimizable_parameters,
            ]:
                torch.distributed.broadcast(tensor, src=src)
            if self.train_position_deviation and hasattr(k, "_base_position_deviation"):
                torch.distributed.broadcast(k._base_position_deviation, src=src)

    def _log_epoch(self, rank, epoch, loss, eval_loss, optimizer, kinematic,
                   group_idx, init_angle, init_offset):
        mem_gb  = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        lr      = optimizer.param_groups[indices.optimizer_param_group_0]["lr"]
        log.info(
            f"Rank {rank} | epoch {epoch:4d} | loss {loss:.6f}"
            + (f" | eval {eval_loss:.6f}" if eval_loss is not None else "")
            + f" | lr {lr:.2e} | GPU {mem_gb:.2f}/{peak_gb:.2f} GB"
        )
        entry = {
            "epoch": epoch,
            "group": group_idx,
            "loss":  loss.item(),
            "translation_dev_abs_mean": kinematic.translation_deviation_parameters.abs().mean().item(),
            "rotation_dev_abs_mean":    kinematic.rotation_deviation_parameters.abs().mean().item(),
            "actuator_angle_dev_abs_mean": (
                kinematic.actuators.optimizable_parameters[:, indices.actuator_initial_angle, :]
                - init_angle
            ).abs().mean().item(),
            "actuator_offset_dev_abs_mean": (
                kinematic.actuators.non_optimizable_parameters[:, indices.actuator_offset, :]
                - init_offset
            ).abs().mean().item(),
        }
        if eval_loss is not None:
            entry["eval_loss"] = eval_loss
        if self.train_position_deviation and hasattr(kinematic, "_base_position_deviation"):
            entry["base_pos_abs_mean"] = kinematic._base_position_deviation.abs().mean().item()
        self._convergence_history.append(entry)

        self._kinematic_history.append({
            "epoch":    epoch,
            "rotation": kinematic.rotation_deviation_parameters.detach().cpu().tolist(),
            "actuator_angle_dev": (
                kinematic.actuators.optimizable_parameters[:, indices.actuator_initial_angle, :]
                - init_angle
            ).detach().cpu().tolist(),
            "actuator_offset_dev": (
                kinematic.actuators.non_optimizable_parameters[:, indices.actuator_offset, :]
                - init_offset
            ).detach().cpu().tolist(),
            "base_position": (
                kinematic._base_position_deviation.detach().cpu().tolist()
                if self.train_position_deviation and hasattr(kinematic, "_base_position_deviation")
                else None
            ),
        })


# ---------------------------------------------------------------------------
# Pixel-loss variant
# ---------------------------------------------------------------------------

class WortbergPixelReconstructor(WortbergKinematicReconstructor):
    """Pixel-loss variant: Gaussian-blurred, peak-normalised flux bitmaps as ground truth.

    Before computing the loss each epoch the predicted flux is blurred
    (to smooth ray-tracing sparsity) and both predicted and measured images
    are peak-normalised to [0, 1] per image, making the loss scale-invariant.
    """

    def __init__(self, *args, blur_sigma: float = 5.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.blur_sigma = blur_sigma

    def _get_ground_truth_and_reduction_dims(self, measured_flux, focal_spots):
        return measured_flux, (indices.batched_bitmap_e, indices.batched_bitmap_u)

    def _preprocess_eval_flux(self, flux, ground_truth):
        return self._peak_normalize(self._blur(flux)), self._peak_normalize(ground_truth)

    def _compute_epoch_loss(
        self, epoch, flux, measured_flux, focal_spots, sample_indices,
        target_indices, loss_fn, ground_truth, reduction_dims,
        n_samples_per_heliostat, device,
    ) -> torch.Tensor:
        pred = self._peak_normalize(self._blur(flux))
        gt   = self._peak_normalize(ground_truth[sample_indices])
        lps  = loss_fn(
            prediction=pred,
            ground_truth=gt,
            target_area_indices=target_indices[sample_indices],
            reduction_dimensions=reduction_dims,
            device=device,
        )
        return mean_loss_per_heliostat(lps, n_samples_per_heliostat)

    def _blur(self, flux: torch.Tensor) -> torch.Tensor:
        """Differentiable separable Gaussian blur over a batch of flux images [N, H, W]."""
        if self.blur_sigma <= 0:
            return flux
        ks = int(4 * self.blur_sigma + 0.5) * 2 + 1
        coords = torch.arange(ks, device=flux.device, dtype=flux.dtype) - ks // 2
        g = torch.exp(-0.5 * (coords / self.blur_sigma) ** 2)
        g = g / g.sum()
        kernel = (g[:, None] * g[None, :]).view(1, 1, ks, ks)
        return F.conv2d(flux.unsqueeze(1), kernel, padding=ks // 2).squeeze(1)

    @staticmethod
    def _peak_normalize(flux: torch.Tensor) -> torch.Tensor:
        """Peak-normalise each image in a batch to [0, 1] independently."""
        N    = flux.shape[0]
        peak = flux.view(N, -1).max(dim=1).values.clamp(min=1e-12)
        return flux / peak.view(N, 1, 1)


# ---------------------------------------------------------------------------
# Alignment-loss variant (stage 1 — no ray tracing)
# ---------------------------------------------------------------------------

class WortbergAlignmentReconstructor(WortbergKinematicReconstructor):
    """Alignment-loss variant: motor-position MSE without ray tracing.

    Replaces the ray-tracing loop with a pure kinematics forward pass:
    the kinematic inverse-kinematics solver predicts motor positions which are
    compared directly to measured positions from PAINT calibration files.
    Each epoch is therefore much faster and requires no GPU memory for ray tracing.

    Expects ``loss_definition`` to be an ``AlignmentLoss`` instance.
    """

    def _reconstruct_kinematics_parameters_with_raytracing(
        self,
        loss_definition,
        device=None,
    ):
        device = get_device(device=device)
        rank   = self.ddp_setup[constants.rank]
        opt    = self.optimizer_dict

        if rank == 0:
            log.info("Beginning kinematic reconstruction with alignment loss (motor-position MSE).")

        final_loss = torch.full(
            (self.scenario.heliostat_field.number_of_heliostats_per_group.sum(),),
            torch.inf,
            device=device,
        )
        group_offsets = torch.cat([
            torch.tensor([0], device=device),
            self.scenario.heliostat_field.number_of_heliostats_per_group.cumsum(
                indices.heliostat_dimension
            ),
        ])

        self._convergence_history = []

        for group_idx in self.ddp_setup[constants.groups_to_ranks_mapping][rank]:
            group   = self.scenario.heliostat_field.heliostat_groups[group_idx]
            parser  = self.data[constants.data_parser]
            mapping = self.data[constants.heliostat_data_mapping]

            _, _, ray_dirs, motor_positions, active_mask, target_mask = (
                parser.parse_data_for_reconstruction(
                    heliostat_data_mapping=mapping,
                    heliostat_group=group,
                    scenario=self.scenario,
                    device=device,
                )
            )

            if active_mask.sum() == 0:
                continue

            optimizer, init_angle, init_offset = self._setup_optimizer(group, device)
            scheduler = self._setup_scheduler(optimizer)
            stopper   = self._setup_early_stopper()

            # Per-heliostat sample counts (may vary if some heliostats have fewer samples).
            nonzero_counts = active_mask[active_mask > 0].tolist()

            loss     = torch.inf
            epoch    = 0
            log_step = opt[constants.max_epoch] if opt[constants.log_step] == 0 else opt[constants.log_step]
            best_eval, best_snap = float("inf"), None
            # Cache eval data once (motor positions + kinematics only, no PNG loading needed).
            eval_cache = self._cache_eval_alignment_data(group, device)

            while loss > float(opt[constants.tolerance]) and epoch <= opt[constants.max_epoch]:
                optimizer.zero_grad()

                group.activate_heliostats(active_heliostats_mask=active_mask, device=device)
                kinematic = group.kinematics
                self._inject_base_position(kinematic, active_mask, device)

                group.align_surfaces_with_incident_ray_directions(
                    aim_points=self.scenario.solar_tower.get_centers_of_target_areas(
                        target_mask, device
                    ),
                    incident_ray_directions=ray_dirs,
                    active_heliostats_mask=active_mask,
                    device=device,
                )

                pred_mp = kinematic.active_motor_positions  # [N_active, 2]
                lps     = loss_definition(
                    predicted_motor_positions=pred_mp,
                    measured_motor_positions=motor_positions,
                    actuators=kinematic.actuators,
                    device=device,
                )
                # Split by heliostat (may be non-uniform) and average within each.
                loss_per_heliostat = torch.stack(
                    [c.mean() for c in torch.split(lps, nonzero_counts)]
                )
                loss = loss_per_heliostat.mean()
                loss.backward()

                optimizer.step()
                self._apply_deviation_bounds(group, init_angle, init_offset)

                eval_loss = self._eval_alignment_loss_from_cache(
                    eval_cache, group, loss_definition, device
                )
                if eval_loss is not None and eval_loss < best_eval:
                    best_eval, best_snap = eval_loss, self._snapshot(kinematic)

                sched_val = eval_loss if eval_loss is not None else loss.detach()
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(sched_val)
                else:
                    scheduler.step()

                if epoch % log_step == 0:
                    log.info(
                        f"Rank {rank} | epoch {epoch:4d} | loss {loss:.6f}"
                        + (f" | eval {eval_loss:.6f}" if eval_loss is not None else "")
                    )
                    self._convergence_history.append({
                        "epoch": epoch,
                        "group": group_idx,
                        "loss":  loss.item(),
                        **({"eval_loss": eval_loss} if eval_loss is not None else {}),
                    })

                if stopper.step(loss):
                    log.info(f"Rank {rank}: early stopping at epoch {epoch}.")
                    break
                epoch += 1

            if best_snap is not None:
                self._restore(kinematic, best_snap)
                log.info(f"Rank {rank}: restored best params (eval={best_eval:.6f}).")

            global_active = torch.nonzero(active_mask != 0, as_tuple=True)[0]
            final_loss[global_active + group_offsets[group_idx]] = loss_per_heliostat.detach()
            log.info(f"Rank {rank}: group {group_idx} reconstructed.")

        if self.ddp_setup[constants.is_distributed]:
            self._broadcast_parameters()
            torch.distributed.all_reduce(final_loss, op=torch.distributed.ReduceOp.MIN)
            log.info(f"Rank {rank}: synchronized.")

        return final_loss, self._convergence_history

    def _cache_eval_alignment_data(self, group, device):
        """Parse validation motor positions + ray directions once and cache them."""
        if self.eval_data is None:
            return None
        _, _, rd, mp, am, tm = self.eval_data["data_parser"].parse_data_for_reconstruction(
            heliostat_data_mapping=self.eval_data["heliostat_data_mapping"],
            heliostat_group=group,
            scenario=self.scenario,
            device=device,
        )
        return (rd, mp, am, tm) if am.sum() > 0 else None

    @torch.no_grad()
    def _eval_alignment_loss_from_cache(self, cache, group, loss_fn, device) -> float | None:
        """Compute alignment eval loss from cached tensors (no ray tracing)."""
        if cache is None:
            return None
        ray_dirs, motor_positions, active_mask, target_mask = cache

        group.activate_heliostats(active_heliostats_mask=active_mask, device=device)
        kinematic = group.kinematics
        self._inject_base_position(kinematic, active_mask, device)

        group.align_surfaces_with_incident_ray_directions(
            aim_points=self.scenario.solar_tower.get_centers_of_target_areas(target_mask, device),
            incident_ray_directions=ray_dirs,
            active_heliostats_mask=active_mask,
            device=device,
        )

        lps    = loss_fn(
            predicted_motor_positions=kinematic.active_motor_positions,
            measured_motor_positions=motor_positions,
            actuators=kinematic.actuators,
            device=device,
        )
        counts = active_mask[active_mask > 0].tolist()
        return torch.stack([c.mean() for c in torch.split(lps, counts)]).mean().item()


# ---------------------------------------------------------------------------
# Contour-loss variant
# ---------------------------------------------------------------------------

class WortbergContourReconstructor(WortbergKinematicReconstructor):
    """Contour-loss variant: upper-edge contour matching instead of COM regression.

    Uses ``ContourLoss`` (three-term: coarse distance field + DICE + gravity) on
    the raw flux images rather than collapsing each image to a single COM point.
    The full preprocessing pipeline (normalise → smooth → soft-threshold →
    soft-erosion → vertical Sobel) is applied internally by ``ContourLoss``.

    This subclass only needs to redirect the ground-truth tensor from focal-spot
    coordinates to flux images; all other training logic is inherited unchanged.
    """

    def _get_ground_truth_and_reduction_dims(self, measured_flux, focal_spots):
        """Use raw flux images as ground truth (same as WortbergPixelReconstructor)."""
        return measured_flux, (indices.batched_bitmap_e, indices.batched_bitmap_u)
