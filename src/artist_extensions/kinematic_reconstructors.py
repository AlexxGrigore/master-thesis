"""
Custom KinematicReconstructor subclasses.

Each class in this module represents a distinct experiment configuration that
overrides the default ARTIST parameter selection and/or deviation bounds.
Adding a new experiment means adding a new subclass here — the training scripts
stay thin and only deal with data loading and configuration.
"""

import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from artist.core import core_utils, learning_rate_schedulers
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.kinematics_reconstructor import KinematicsReconstructor
from artist.util import config_dictionary, index_mapping
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)


class WortbergKinematicReconstructor(KinematicsReconstructor):
    """
    KinematicReconstructor following the parameter setup of Wortberg (2025) Table 5.3.

    Compared to the default ARTIST reconstructor, this variant:

    Adds to the optimised set
    -------------------------
    - ``translation_deviation_parameters`` (all 9: joints + concentrator), ±0.05 m
    - ``actuators.non_optimizable_parameters[:, actuator_offset]`` (c_i), ±0.005 m
    - ``_base_position_deviation`` (3 translations: e, n, u), ±0.05 m — injected into
      ``kinematic.active_heliostat_positions`` each epoch without modifying ARTIST

    Keeps from the default
    ----------------------
    - ``rotation_deviation_parameters`` (all 4 tilts), ±0.005 rad
    - ``actuators.optimizable_parameters[:, actuator_initial_angle]`` (a_i), ±0.005 rad

    Freezes (deviation bound ±0.0)
    --------------------------------
    - ``actuators.optimizable_parameters[:, actuator_initial_stroke_length]`` (b_i)
    - ``actuators.non_optimizable_parameters[:, actuator_pivot_radius]`` (d_i) — already
      non-optimizable in ARTIST by design

    Parameters not yet implemented in ARTIST (excluded for now)
    -----------------------------------------------------------
    - Concentrator tilts (2 rotations)
    """

    # Deviation bounds from Wortberg (2025) Table 5.3.
    _BOUND_TRANSLATION_M = 0.05        # joint and concentrator translations
    _BOUND_ROTATION_RAD = 0.005        # joint tilts
    _BOUND_ACTUATOR_ANGLE_RAD = 0.005  # a_i — offset radius shift
    _BOUND_ACTUATOR_OFFSET_M = 0.005   # c_i — joint's distance offset
    _BOUND_BASE_POSITION_M = 0.05      # heliostat base position (e, n, u)

    def __init__(self, *args, train_position_deviation: bool = True, eval_data: dict | None = None, sample_mini_batch_size: int | None = None, **kwargs):
        if "optimization_configuration" in kwargs:
            flat = kwargs["optimization_configuration"]
            if config_dictionary.optimization not in flat:
                _SCHED_PARAMS = "scheduler_parameters"
                scheduler_params = flat.get(_SCHED_PARAMS, {})
                kwargs["optimization_configuration"] = {
                    config_dictionary.optimization: {
                        k: v for k, v in flat.items()
                        if k not in (config_dictionary.scheduler, _SCHED_PARAMS)
                    },
                    config_dictionary.scheduler: {
                        config_dictionary.scheduler_type: flat.get(
                            config_dictionary.scheduler, config_dictionary.reduce_on_plateau
                        ),
                        **scheduler_params,
                    },
                }

        super().__init__(*args, **kwargs)
        self.train_position_deviation = train_position_deviation
        self.eval_data = eval_data
        self.sample_mini_batch_size = sample_mini_batch_size

        self.optimization_configuration = {
            **self.optimizer_dict,
            config_dictionary.scheduler: self.scheduler_dict.get(
                config_dictionary.scheduler_type, config_dictionary.reduce_on_plateau
            ),
        }
        self._scheduler_params = {
            k: v for k, v in self.scheduler_dict.items()
            if k != config_dictionary.scheduler_type
        }

    def _reconstruct_kinematics_parameters_with_raytracing(
        self,
        loss_definition,
        device=None,
    ):
        device = get_device(device=device)
        rank = self.ddp_setup[config_dictionary.rank]

        if rank == 0:
            log.info(
                "Beginning kinematic reconstruction with ray tracing "
                "(Wortberg 2025 Table 5.3 parameter set)."
            )

        final_loss_per_heliostat = torch.full(
            (self.scenario.heliostat_field.number_of_heliostats_per_group.sum(),),
            torch.inf,
            device=device,
        )
        final_loss_start_indices = torch.cat(
            [
                torch.tensor([0], device=device),
                self.scenario.heliostat_field.number_of_heliostats_per_group.cumsum(
                    index_mapping.heliostat_dimension
                ),
            ]
        )

        self._convergence_history = []
        self._kinematic_history = []

        for heliostat_group_index in self.ddp_setup[config_dictionary.groups_to_ranks_mapping][rank]:
            heliostat_group = self.scenario.heliostat_field.heliostat_groups[heliostat_group_index]
            parser = self.data[config_dictionary.data_parser]
            heliostat_mapping = self.data[config_dictionary.heliostat_data_mapping]

            (
                measured_flux,
                focal_spots_measured,
                incident_ray_directions,
                _,
                active_heliostats_mask,
                target_area_mask,
            ) = parser.parse_data_for_reconstruction(
                heliostat_data_mapping=heliostat_mapping,
                heliostat_group=heliostat_group,
                scenario=self.scenario,
                device=device,
            )

            if active_heliostats_mask.sum() > 0:
                ground_truth, reduction_dims = self._get_ground_truth_and_reduction_dims(
                    measured_flux=measured_flux,
                    focal_spots_measured=focal_spots_measured,
                )

                optimizer, initial_actuator_initial_angle, initial_actuator_offset = (
                    self._setup_optimizer(heliostat_group, device)
                )
                scheduler = self._setup_scheduler(optimizer)
                early_stopper = self._setup_early_stopper()

                loss = torch.inf
                epoch = 0
                log_step = (
                    self.optimization_configuration[config_dictionary.max_epoch]
                    if self.optimization_configuration[config_dictionary.log_step] == 0
                    else self.optimization_configuration[config_dictionary.log_step]
                )

                # Cache parsed eval data once so we avoid re-loading PNG files every epoch.
                _eval_cache = self._parse_eval_data_for_group(heliostat_group, device)
                best_eval_loss = float("inf")
                best_params: dict | None = None

                # Sample mini-batch setup.
                # incident_ray_directions layout: [h0_s0..h0_s(N-1), h1_s0..., ...]
                # where N = n_samples_full. With mini-batching we run K forward passes
                # per epoch (each with mb_size samples), accumulating gradients before
                # the optimizer step. This avoids OOM on large fields (e.g. 1315 × 50).
                n_samples_full = int(active_heliostats_mask[active_heliostats_mask > 0][0].item())
                n_active_heliostats = int((active_heliostats_mask > 0).sum().item())
                mb_size = (
                    self.sample_mini_batch_size
                    if self.sample_mini_batch_size is not None
                    else n_samples_full
                )
                n_mini_batches = (n_samples_full + mb_size - 1) // mb_size
                # Precompute per-heliostat start offsets into the flat tensor.
                h_offsets = torch.arange(n_active_heliostats, device=device) * n_samples_full

                if n_mini_batches > 1:
                    log.info(
                        f"Sample mini-batching enabled: {n_samples_full} samples split into "
                        f"{n_mini_batches} mini-batches of up to {mb_size} samples each."
                    )

                # ground_truth_full is the full-dataset reference tensor (measured_flux or
                # focal_spots_measured) that gets sliced to sub_ground_truth each mini-batch.
                ground_truth_full = ground_truth

                # Initialise loop-scoped variables that are read after the while loop.
                sample_indices_for_local_rank = torch.arange(
                    n_active_heliostats * mb_size, device=device
                )
                number_of_samples_per_heliostat = mb_size

                max_epoch = self.optimization_configuration[config_dictionary.max_epoch]
                pbar = tqdm(
                    total=max_epoch,
                    desc="Training",
                    unit="ep",
                    dynamic_ncols=True,
                    leave=True,
                )

                while (
                    loss > float(self.optimization_configuration[config_dictionary.tolerance])
                    and epoch <= max_epoch
                ):
                    optimizer.zero_grad()
                    kinematic = heliostat_group.kinematics
                    loss_per_heliostat_accum: torch.Tensor | None = None

                    for mb in range(n_mini_batches):
                        s_start = mb * mb_size
                        s_end = min(s_start + mb_size, n_samples_full)
                        current_mb_size = s_end - s_start

                        sample_range = torch.arange(s_start, s_end, device=device)
                        indices = (h_offsets.unsqueeze(1) + sample_range.unsqueeze(0)).reshape(-1)

                        sub_rays = incident_ray_directions[indices]
                        sub_ground_truth = ground_truth_full[indices]
                        sub_target_mask = target_area_mask[indices]
                        sub_measured_flux = measured_flux[indices]
                        sub_focal_spots = focal_spots_measured[indices]
                        sub_mask = (active_heliostats_mask > 0).to(torch.long) * current_mb_size

                        heliostat_group.activate_heliostats(
                            active_heliostats_mask=sub_mask, device=device
                        )

                        if self.train_position_deviation and hasattr(kinematic, "_base_position_deviation"):
                            # Inject base position deviation into the active positions that
                            # ARTIST just set.  We overwrite with a new tensor (no in-place op)
                            # so autograd traces through repeat_interleave → _base_position_deviation.
                            active_base_dev = kinematic._base_position_deviation.repeat_interleave(
                                sub_mask, dim=0
                            )  # [N_active, 3]
                            pad = torch.zeros(active_base_dev.shape[0], 1, device=device)
                            kinematic.active_heliostat_positions = (
                                kinematic.active_heliostat_positions
                                + torch.cat([active_base_dev, pad], dim=1)
                            )

                        heliostat_group.align_surfaces_with_incident_ray_directions(
                            aim_points=self.scenario.solar_tower.get_centers_of_target_areas(
                                sub_target_mask, device=device
                            ),
                            incident_ray_directions=sub_rays,
                            active_heliostats_mask=sub_mask,
                            device=device,
                        )

                        ray_tracer = HeliostatRayTracer(
                            scenario=self.scenario,
                            heliostat_group=heliostat_group,
                            blocking_active=False,
                            world_size=self.ddp_setup[config_dictionary.heliostat_group_world_size],
                            rank=self.ddp_setup[config_dictionary.heliostat_group_rank],
                            batch_size=self.optimization_configuration[config_dictionary.batch_size],
                            random_seed=self.ddp_setup[config_dictionary.heliostat_group_rank],
                        )

                        flux_distributions, _, _, _ = ray_tracer.trace_rays(
                            incident_ray_directions=sub_rays,
                            active_heliostats_mask=sub_mask,
                            target_area_indices=sub_target_mask,
                            device=device,
                        )

                        sample_indices_for_local_rank = ray_tracer.get_sampler_indices()
                        number_of_samples_per_heliostat = current_mb_size

                        loss_per_heliostat = self._compute_epoch_loss(
                            epoch=epoch,
                            flux_distributions=flux_distributions,
                            measured_flux=sub_measured_flux,
                            focal_spots_measured=sub_focal_spots,
                            sample_indices=sample_indices_for_local_rank,
                            target_area_indices=sub_target_mask,
                            loss_definition=loss_definition,
                            ground_truth=sub_ground_truth,
                            reduction_dims=reduction_dims,
                            number_of_samples_per_heliostat=number_of_samples_per_heliostat,
                            device=device,
                        )

                        # Divide by n_mini_batches before backward so gradients accumulate
                        # to the correct epoch-level mean across all mini-batches.
                        (loss_per_heliostat.mean() / n_mini_batches).backward()

                        if loss_per_heliostat_accum is None:
                            loss_per_heliostat_accum = loss_per_heliostat.detach() / n_mini_batches
                        else:
                            loss_per_heliostat_accum = (
                                loss_per_heliostat_accum + loss_per_heliostat.detach() / n_mini_batches
                            )

                    loss_per_heliostat = loss_per_heliostat_accum
                    loss = loss_per_heliostat.mean()

                    # DDP nested: reduce gradients across ranks within the heliostat group,
                    # then divide by group world size (mirrors ARTIST KinematicsReconstructor).
                    # Done once after all mini-batch backward passes.
                    if self.ddp_setup[config_dictionary.is_nested]:
                        for param_group in optimizer.param_groups:
                            for param in param_group["params"]:
                                if param.grad is not None:
                                    param.grad = (
                                        torch.distributed.nn.functional.all_reduce(
                                            param.grad,
                                            op=torch.distributed.ReduceOp.SUM,
                                            group=self.ddp_setup[
                                                config_dictionary.process_subgroup
                                            ],
                                        )
                                    )
                                    param.grad /= self.ddp_setup[
                                        config_dictionary.heliostat_group_world_size
                                    ]

                    # Gradient clipping — keeps large-scale parameters (translations,
                    # base position) from taking destabilizing steps, matching the
                    # max_norm=1.0 used in ARTIST's base KinematicsReconstructor.
                    _clip_params = [
                        kinematic.translation_deviation_parameters,
                        kinematic.rotation_deviation_parameters,
                        kinematic.actuators.optimizable_parameters,
                        kinematic.actuators.non_optimizable_parameters,
                    ]
                    if self.train_position_deviation and hasattr(kinematic, "_base_position_deviation"):
                        _clip_params.append(kinematic._base_position_deviation)
                    torch.nn.utils.clip_grad_norm_(_clip_params, max_norm=1.0)

                    # Capture gradient magnitudes after clipping, before optimizer clears them.
                    def _grad_mean(t: torch.Tensor) -> float:
                        return t.grad.abs().mean().item() if t.grad is not None else 0.0

                    grad_rotation = _grad_mean(kinematic.rotation_deviation_parameters)
                    grad_act_angle = _grad_mean(kinematic.actuators.optimizable_parameters)
                    grad_act_offset = _grad_mean(kinematic.actuators.non_optimizable_parameters)
                    grad_base_pos = (
                        _grad_mean(kinematic._base_position_deviation)
                        if self.train_position_deviation and hasattr(kinematic, "_base_position_deviation")
                        else 0.0
                    )

                    optimizer.step()

                    self._apply_deviation_bounds(
                        heliostat_group,
                        initial_actuator_initial_angle,
                        initial_actuator_offset,
                    )

                    # Eval every epoch from cached parsed data (no PNG re-loading).
                    eval_loss = self._compute_eval_loss_from_cache(
                        _eval_cache, heliostat_group, loss_definition, device
                    )
                    if eval_loss is not None and eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_params = self._snapshot_kinematic_params(heliostat_group.kinematics)

                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(eval_loss if eval_loss is not None else loss.detach())
                    else:
                        scheduler.step()

                    if epoch % log_step == 0:
                        mem_alloc = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
                        mem_peak = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
                        log.info(
                            f"Rank: {rank}, Epoch: {epoch}, Loss: {loss:.6f}, "
                            f"LR: {optimizer.param_groups[index_mapping.optimizer_param_group_0]['lr']}, "
                            f"GPU mem: {mem_alloc:.2f} GB (peak: {mem_peak:.2f} GB)"
                        )
                        if eval_loss is not None:
                            log.info(f"Rank: {rank}, Epoch: {epoch}, Eval Loss: {eval_loss:.6f}")
                        entry = {
                            "epoch": epoch,
                            "group": heliostat_group_index,
                            "loss": loss.item(),
                            "translation_deviation_mean_abs": kinematic.translation_deviation_parameters.abs().mean().item(),
                            "rotation_deviation_mean_abs": kinematic.rotation_deviation_parameters.abs().mean().item(),
                            "actuator_angle_dev_mean_abs": (
                                kinematic.actuators.optimizable_parameters[:, index_mapping.actuator_initial_angle, :]
                                - initial_actuator_initial_angle
                            ).abs().mean().item(),
                            "actuator_offset_dev_mean_abs": (
                                kinematic.actuators.non_optimizable_parameters[:, index_mapping.actuator_offset, :]
                                - initial_actuator_offset
                            ).abs().mean().item(),
                            "grad_rotation": grad_rotation,
                            "grad_act_angle": grad_act_angle,
                            "grad_act_offset": grad_act_offset,
                            "grad_base_pos": grad_base_pos,
                        }
                        if self.train_position_deviation and hasattr(kinematic, "_base_position_deviation"):
                            entry["base_pos_dev_e_mean_abs"] = kinematic._base_position_deviation[:, 0].abs().mean().item()
                            entry["base_pos_dev_n_mean_abs"] = kinematic._base_position_deviation[:, 1].abs().mean().item()
                            entry["base_pos_dev_u_mean_abs"] = kinematic._base_position_deviation[:, 2].abs().mean().item()
                        if eval_loss is not None:
                            entry["eval_loss"] = eval_loss
                        self._convergence_history.append(entry)

                        # Kinematic parameter snapshot — per heliostat, in physical units.
                        k_entry = {
                            "epoch": epoch,
                            "rotation_rad": kinematic.rotation_deviation_parameters.detach().cpu().tolist(),
                            "actuator_angle_deviation_rad": (
                                kinematic.actuators.optimizable_parameters[
                                    :, index_mapping.actuator_initial_angle, :
                                ]
                                - initial_actuator_initial_angle
                            ).detach().cpu().tolist(),
                            "actuator_offset_deviation_m": (
                                kinematic.actuators.non_optimizable_parameters[
                                    :, index_mapping.actuator_offset, :
                                ]
                                - initial_actuator_offset
                            ).detach().cpu().tolist(),
                        }
                        if self.train_position_deviation and hasattr(kinematic, "_base_position_deviation"):
                            k_entry["base_position_m"] = kinematic._base_position_deviation.detach().cpu().tolist()
                        else:
                            k_entry["base_position_m"] = None
                        self._kinematic_history.append(k_entry)

                    if early_stopper.step(loss):
                        log.info(f"Early stopping at epoch {epoch}.")
                        pbar.close()
                        break

                    lr = optimizer.param_groups[index_mapping.optimizer_param_group_0]["lr"]
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        eval=f"{eval_loss:.4f}" if eval_loss is not None else "-",
                        lr=f"{lr:.2e}",
                    )
                    pbar.update(1)
                    epoch += 1
                else:
                    pbar.close()

                if best_params is not None:
                    self._restore_kinematic_params(heliostat_group.kinematics, best_params)
                    log.info(f"Restored best kinematic parameters (val_loss={best_eval_loss:.6f}).")

                local_indices = (
                    sample_indices_for_local_rank[::number_of_samples_per_heliostat]
                    // number_of_samples_per_heliostat
                )
                global_active_indices = torch.nonzero(active_heliostats_mask != 0, as_tuple=True)[0]
                rank_active_indices_global = global_active_indices[local_indices]
                final_indices = (
                    rank_active_indices_global + final_loss_start_indices[heliostat_group_index]
                )
                final_loss_per_heliostat[final_indices] = loss_per_heliostat
                log.info(f"Rank: {rank}, Kinematic reconstructed.")

        if self.ddp_setup[config_dictionary.is_distributed]:
            for index, heliostat_group in enumerate(self.scenario.heliostat_field.heliostat_groups):
                src = self.ddp_setup[config_dictionary.ranks_to_groups_mapping][index][
                    index_mapping.first_rank_from_group
                ]
                torch.distributed.broadcast(
                    heliostat_group.kinematics.translation_deviation_parameters, src=src
                )
                torch.distributed.broadcast(
                    heliostat_group.kinematics.rotation_deviation_parameters, src=src
                )
                torch.distributed.broadcast(
                    heliostat_group.kinematics.actuators.optimizable_parameters, src=src
                )
                torch.distributed.broadcast(
                    heliostat_group.kinematics.actuators.non_optimizable_parameters, src=src
                )
                if self.train_position_deviation and hasattr(heliostat_group.kinematics, "_base_position_deviation"):
                    torch.distributed.broadcast(
                        heliostat_group.kinematics._base_position_deviation, src=src
                    )
            torch.distributed.all_reduce(
                final_loss_per_heliostat, op=torch.distributed.ReduceOp.MIN
            )
            log.info(f"Rank: {rank}, synchronized after kinematic reconstruction.")

        return final_loss_per_heliostat, self._convergence_history

    # ------------------------------------------------------------------
    # Private helpers — each responsible for one setup concern
    # ------------------------------------------------------------------

    def _setup_optimizer(self, heliostat_group, device):
        """Enable gradients, register freeze hooks, and return a configured Adam optimizer."""
        kinematic = heliostat_group.kinematics

        kinematic.translation_deviation_parameters.requires_grad_()
        kinematic.rotation_deviation_parameters.requires_grad_()
        kinematic.actuators.optimizable_parameters.requires_grad_()
        kinematic.actuators.non_optimizable_parameters.requires_grad_()

        # Freeze b_i (initial_stroke_length) — zero its gradient on every backward pass.
        def _freeze_stroke_length(grad: torch.Tensor) -> torch.Tensor:
            mask = torch.ones_like(grad)
            mask[:, index_mapping.actuator_initial_stroke_length, :] = 0.0
            return grad * mask

        kinematic.actuators.optimizable_parameters.register_hook(_freeze_stroke_length)

        # Restrict non_optimizable gradients to c_i (actuator_offset) only.
        def _only_actuator_offset(grad: torch.Tensor) -> torch.Tensor:
            mask = torch.zeros_like(grad)
            mask[:, index_mapping.actuator_offset, :] = 1.0
            return grad * mask

        kinematic.actuators.non_optimizable_parameters.register_hook(_only_actuator_offset)

        if self.train_position_deviation:
            if hasattr(kinematic, "_base_position_deviation"):
                # Phase 2: re-enable grad on existing values so Phase 1 alignment is preserved.
                kinematic._base_position_deviation = (
                    kinematic._base_position_deviation.detach().requires_grad_(True)
                )
            else:
                # Phase 1: initialise to zero.
                kinematic._base_position_deviation = torch.zeros(
                    kinematic.number_of_heliostats, 3, device=device, requires_grad=True
                )

        # Snapshot non-zero nominal values for bound clamping.
        # Only take the snapshot once (Phase 1) so deviation bounds stay relative
        # to the original scenario values across both phases.
        if not hasattr(kinematic, "_initial_actuator_initial_angle"):
            initial_actuator_initial_angle = (
                kinematic.actuators.optimizable_parameters[
                    :, index_mapping.actuator_initial_angle, :
                ]
                .detach()
                .clone()
            )
            initial_actuator_offset = (
                kinematic.actuators.non_optimizable_parameters[
                    :, index_mapping.actuator_offset, :
                ]
                .detach()
                .clone()
            )
            kinematic._initial_actuator_initial_angle = initial_actuator_initial_angle
            kinematic._initial_actuator_offset = initial_actuator_offset
        else:
            initial_actuator_initial_angle = kinematic._initial_actuator_initial_angle
            initial_actuator_offset = kinematic._initial_actuator_offset

        # Snapshot translation_deviation_parameters nominal values.
        # These may be non-zero in the scenario (e.g. concentrator_translation_n ≈ 0.175 m),
        # so bounds must be relative to the loaded values, not absolute ±0.05 m.
        if not hasattr(kinematic, "_initial_translation_deviation"):
            kinematic._initial_translation_deviation = (
                kinematic.translation_deviation_parameters.detach().clone()
            )

        base_lr = float(self.optimization_configuration[config_dictionary.initial_learning_rate])
        # Large-scale parameters (±0.05 m bounds) get 5× the base LR;
        # small-scale parameters (±0.005 rad/m bounds) use the base LR directly.
        lr_large = base_lr * 5.0
        lr_small = base_lr

        optimizer_params = [
            {"params": kinematic.translation_deviation_parameters, "lr": lr_large},
            {"params": kinematic.rotation_deviation_parameters, "lr": lr_small},
            {"params": kinematic.actuators.optimizable_parameters, "lr": lr_small},
            {"params": kinematic.actuators.non_optimizable_parameters, "lr": lr_small},
        ]
        if self.train_position_deviation:
            optimizer_params.append(
                {"params": kinematic._base_position_deviation, "lr": lr_large}
            )

        optimizer = torch.optim.Adam(optimizer_params, lr=base_lr)

        return optimizer, initial_actuator_initial_angle, initial_actuator_offset

    def _setup_scheduler(self, optimizer):
        """Create an LR scheduler from self.optimization_configuration."""
        scheduler_type = self.optimization_configuration.get(
            config_dictionary.scheduler, config_dictionary.reduce_on_plateau
        )
        params = self._scheduler_params

        if scheduler_type == config_dictionary.reduce_on_plateau:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=params.get(config_dictionary.reduce_factor, 0.5),
                patience=params.get(config_dictionary.patience, 10),
                threshold=params.get(config_dictionary.threshold, 1e-4),
                cooldown=params.get(config_dictionary.cooldown, 5),
                min_lr=params.get(config_dictionary.lr_min, 1e-8),
            )

        # Fallback: cosine annealing (original behaviour)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.optimization_configuration[config_dictionary.max_epoch],
            eta_min=params.get(config_dictionary.lr_min, 1e-6),
        )

    def _setup_early_stopper(self):
        """Build and return the early stopping instance."""
        return learning_rate_schedulers.EarlyStopping(
            window_size=self.optimization_configuration["early_stopping_window"],
            patience=self.optimization_configuration[config_dictionary.early_stopping_patience],
            min_improvement=self.optimization_configuration[config_dictionary.early_stopping_delta],
            relative=True,
        )

    def _compute_epoch_loss(
        self,
        epoch: int,
        flux_distributions: torch.Tensor,
        measured_flux: torch.Tensor,
        focal_spots_measured: torch.Tensor,
        sample_indices: torch.Tensor,
        target_area_indices: torch.Tensor,
        loss_definition,
        ground_truth: torch.Tensor,
        reduction_dims: tuple,
        number_of_samples_per_heliostat: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute per-heliostat loss for one epoch.

        Override in subclasses to change the loss combination logic
        (e.g. annealed multi-loss).  The default calls ``loss_definition``
        with the pre-selected ``ground_truth`` and ``reduction_dims``.
        """
        loss_per_sample = loss_definition(
            prediction=flux_distributions,
            ground_truth=ground_truth[sample_indices],
            target_area_indices=target_area_indices[sample_indices],
            reduction_dimensions=reduction_dims,
            device=device,
        )
        return core_utils.mean_loss_per_heliostat(
            loss_per_sample=loss_per_sample,
            number_of_samples_per_heliostat=number_of_samples_per_heliostat,
        )

    def _get_ground_truth_and_reduction_dims(
        self,
        measured_flux: torch.Tensor,
        focal_spots_measured: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple]:
        """Return (ground_truth, reduction_dimensions) for the loss call.

        Default: focal spot centroids, compatible with FocalSpotLoss.
        Override in subclasses to use a different ground truth type.
        """
        return focal_spots_measured, (index_mapping.focal_spots,)  # measured_flux unused by default

    def _preprocess_eval_flux(
        self,
        predicted_flux: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocess predicted flux and ground truth before eval loss.

        Default: identity (no transformation). Override in subclasses that need
        blur or normalization (e.g. WortbergPixelReconstructor).
        """
        return predicted_flux, ground_truth

    @torch.no_grad()
    def _compute_eval_loss_no_grad(self, heliostat_group, loss_definition, device) -> float | None:
        """Compute eval loss on the validation set without affecting the training graph.

        Returns the mean loss across all validation heliostats, or None if no
        eval_data was provided or the heliostat group has no active heliostats.
        Uses _get_ground_truth_and_reduction_dims and _preprocess_eval_flux so
        subclasses (e.g. WortbergPixelReconstructor) work correctly.
        """
        if self.eval_data is None:
            return None
        eval_parser = self.eval_data["data_parser"]
        eval_mapping = self.eval_data["heliostat_data_mapping"]

        (
            measured_flux,
            focal_spots_measured,
            incident_ray_directions,
            _,
            active_mask,
            target_mask,
        ) = eval_parser.parse_data_for_reconstruction(
            heliostat_data_mapping=eval_mapping,
            heliostat_group=heliostat_group,
            scenario=self.scenario,
            device=device,
        )

        if active_mask.sum() == 0:
            return None

        heliostat_group.activate_heliostats(active_heliostats_mask=active_mask, device=device)
        kinematic = heliostat_group.kinematics

        if self.train_position_deviation and hasattr(kinematic, "_base_position_deviation"):
            active_base_dev = kinematic._base_position_deviation.repeat_interleave(active_mask, dim=0)
            pad = torch.zeros(active_base_dev.shape[0], 1, device=device)
            kinematic.active_heliostat_positions = (
                kinematic.active_heliostat_positions + torch.cat([active_base_dev, pad], dim=1)
            )

        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=self.scenario.solar_tower.get_centers_of_target_areas(
                target_mask, device=device
            ),
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_mask,
            device=device,
        )

        ray_tracer = HeliostatRayTracer(
            scenario=self.scenario,
            heliostat_group=heliostat_group,
            blocking_active=False,
            world_size=self.ddp_setup[config_dictionary.heliostat_group_world_size],
            rank=self.ddp_setup[config_dictionary.heliostat_group_rank],
            batch_size=self.optimization_configuration[config_dictionary.batch_size],
            random_seed=self.ddp_setup[config_dictionary.heliostat_group_rank],
        )
        flux, _, _, _ = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_mask,
            target_area_indices=target_mask,
            device=device,
        )
        sample_indices = ray_tracer.get_sampler_indices()
        n_samples = int(active_mask.sum() / (active_mask > 0).sum())

        ground_truth, reduction_dims = self._get_ground_truth_and_reduction_dims(
            measured_flux, focal_spots_measured
        )
        flux_eval, gt_eval = self._preprocess_eval_flux(flux, ground_truth)

        loss_per_sample = loss_definition(
            prediction=flux_eval,
            ground_truth=gt_eval[sample_indices],
            target_area_indices=target_mask[sample_indices],
            reduction_dimensions=reduction_dims,
            device=device,
        )
        loss_per_heliostat = core_utils.mean_loss_per_heliostat(
            loss_per_sample=loss_per_sample,
            number_of_samples_per_heliostat=n_samples,
        )
        return loss_per_heliostat.mean().item()

    def _parse_eval_data_for_group(self, heliostat_group, device):
        """Load and return eval tensors once — avoids re-reading PNG files every epoch.

        Returns a tuple (measured_flux, focal_spots_measured, incident_ray_directions,
        active_mask, target_mask), or None if no eval_data was configured.
        """
        if self.eval_data is None:
            return None
        eval_parser = self.eval_data["data_parser"]
        eval_mapping = self.eval_data["heliostat_data_mapping"]
        (
            measured_flux,
            focal_spots_measured,
            incident_ray_directions,
            _,
            active_mask,
            target_mask,
        ) = eval_parser.parse_data_for_reconstruction(
            heliostat_data_mapping=eval_mapping,
            heliostat_group=heliostat_group,
            scenario=self.scenario,
            device=device,
        )
        if active_mask.sum() == 0:
            return None
        return measured_flux, focal_spots_measured, incident_ray_directions, active_mask, target_mask

    def _compute_eval_loss_from_cache(self, eval_cache, heliostat_group, loss_definition, device) -> float | None:
        """Compute eval loss using pre-parsed tensors (no PNG re-loading).

        Parameters are read from the current state of heliostat_group, so this
        correctly reflects the latest optimised values each epoch.
        """
        if eval_cache is None:
            return None
        measured_flux, focal_spots_measured, incident_ray_directions, active_mask, target_mask = eval_cache

        with torch.no_grad():
            heliostat_group.activate_heliostats(active_heliostats_mask=active_mask, device=device)
            kinematic = heliostat_group.kinematics

            if self.train_position_deviation and hasattr(kinematic, "_base_position_deviation"):
                active_base_dev = kinematic._base_position_deviation.repeat_interleave(active_mask, dim=0)
                pad = torch.zeros(active_base_dev.shape[0], 1, device=device)
                kinematic.active_heliostat_positions = (
                    kinematic.active_heliostat_positions + torch.cat([active_base_dev, pad], dim=1)
                )

            heliostat_group.align_surfaces_with_incident_ray_directions(
                aim_points=self.scenario.solar_tower.get_centers_of_target_areas(
                    target_mask, device=device
                ),
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_mask,
                device=device,
            )

            ray_tracer = HeliostatRayTracer(
                scenario=self.scenario,
                heliostat_group=heliostat_group,
                blocking_active=False,
                world_size=self.ddp_setup[config_dictionary.heliostat_group_world_size],
                rank=self.ddp_setup[config_dictionary.heliostat_group_rank],
                batch_size=self.optimization_configuration[config_dictionary.batch_size],
                random_seed=self.ddp_setup[config_dictionary.heliostat_group_rank],
            )
            flux, _, _, _ = ray_tracer.trace_rays(
                incident_ray_directions=incident_ray_directions,
                active_heliostats_mask=active_mask,
                target_area_indices=target_mask,
                device=device,
            )
            sample_indices = ray_tracer.get_sampler_indices()
            n_samples = int(active_mask.sum() / (active_mask > 0).sum())

            ground_truth, reduction_dims = self._get_ground_truth_and_reduction_dims(
                measured_flux, focal_spots_measured
            )
            flux_eval, gt_eval = self._preprocess_eval_flux(flux, ground_truth)

            loss_per_sample = loss_definition(
                prediction=flux_eval,
                ground_truth=gt_eval[sample_indices],
                target_area_indices=target_mask[sample_indices],
                reduction_dimensions=reduction_dims,
                device=device,
            )
            loss_per_heliostat = core_utils.mean_loss_per_heliostat(
                loss_per_sample=loss_per_sample,
                number_of_samples_per_heliostat=n_samples,
            )
            return loss_per_heliostat.mean().item()

    @staticmethod
    def _snapshot_kinematic_params(kinematic) -> dict:
        """Copy all optimisable kinematic tensors — used for best-val-loss checkpointing."""
        snap = {
            "translation": kinematic.translation_deviation_parameters.detach().clone(),
            "rotation": kinematic.rotation_deviation_parameters.detach().clone(),
            "opt_params": kinematic.actuators.optimizable_parameters.detach().clone(),
            "nonopt_params": kinematic.actuators.non_optimizable_parameters.detach().clone(),
        }
        if hasattr(kinematic, "_base_position_deviation"):
            snap["base_pos"] = kinematic._base_position_deviation.detach().clone()
        return snap

    @staticmethod
    def _restore_kinematic_params(kinematic, snap: dict) -> None:
        """Restore kinematic tensors from a snapshot produced by _snapshot_kinematic_params."""
        with torch.no_grad():
            kinematic.translation_deviation_parameters.copy_(snap["translation"])
            kinematic.rotation_deviation_parameters.copy_(snap["rotation"])
            kinematic.actuators.optimizable_parameters.copy_(snap["opt_params"])
            kinematic.actuators.non_optimizable_parameters.copy_(snap["nonopt_params"])
            if "base_pos" in snap and hasattr(kinematic, "_base_position_deviation"):
                kinematic._base_position_deviation.copy_(snap["base_pos"])

    def _apply_deviation_bounds(self, heliostat_group, initial_actuator_initial_angle, initial_actuator_offset):
        """
        Clamp all optimised parameters to their Table 5.3 deviation bounds.

        Both translation_deviation and actuator parameters may have non-zero nominal
        values loaded from the scenario (e.g. concentrator_translation_n ≈ 0.175 m),
        so all clamps are computed relative to the snapshotted initial values.
        """
        kinematic = heliostat_group.kinematics
        with torch.no_grad():
            kinematic.translation_deviation_parameters.data.clamp_(
                kinematic._initial_translation_deviation - self._BOUND_TRANSLATION_M,
                kinematic._initial_translation_deviation + self._BOUND_TRANSLATION_M,
            )
            kinematic.rotation_deviation_parameters.data.clamp_(
                -self._BOUND_ROTATION_RAD, self._BOUND_ROTATION_RAD
            )
            kinematic.actuators.optimizable_parameters.data[
                :, index_mapping.actuator_initial_angle, :
            ].clamp_(
                initial_actuator_initial_angle - self._BOUND_ACTUATOR_ANGLE_RAD,
                initial_actuator_initial_angle + self._BOUND_ACTUATOR_ANGLE_RAD,
            )
            kinematic.actuators.non_optimizable_parameters.data[
                :, index_mapping.actuator_offset, :
            ].clamp_(
                initial_actuator_offset - self._BOUND_ACTUATOR_OFFSET_M,
                initial_actuator_offset + self._BOUND_ACTUATOR_OFFSET_M,
            )
            if self.train_position_deviation and hasattr(kinematic, "_base_position_deviation"):
                kinematic._base_position_deviation.data.clamp_(
                    -self._BOUND_BASE_POSITION_M, self._BOUND_BASE_POSITION_M
                )


# ---------------------------------------------------------------------------
# Optimizer setup mixins
#
# Each mixin encodes the _setup_optimizer logic for one parameter subset.
# Combining a mixin with WortbergKinematicReconstructor gives the focal-spot
# variant; combining it with WortbergPixelReconstructor gives the pixel-loss
# variant.  This avoids duplicating the optimizer setup code across the two
# reconstructor families.
# ---------------------------------------------------------------------------

class _SingleAxisRotationMixin:
    """Optimizer mixin: only a subset of rotation_deviation_parameters are optimised.

    Subclasses set ``_ROTATION_INDICES`` to select which of the 4 rotation
    deviations receive gradients.  All other parameters (translations,
    actuators, base position) are frozen.
    """

    _ROTATION_INDICES: tuple[int, ...] = ()  # override in subclasses

    def _setup_optimizer(self, heliostat_group, device):
        kinematic = heliostat_group.kinematics
        kinematic.rotation_deviation_parameters.requires_grad_()

        # Mask gradients to the selected rotation indices only.
        active = set(self._ROTATION_INDICES)

        def _mask_rotation_grad(grad: torch.Tensor) -> torch.Tensor:
            mask = torch.zeros_like(grad)
            for idx in active:
                mask[:, idx] = 1.0
            return grad * mask

        kinematic.rotation_deviation_parameters.register_hook(_mask_rotation_grad)

        # Snapshot nominals — required by parent _apply_deviation_bounds.
        if not hasattr(kinematic, "_initial_actuator_initial_angle"):
            kinematic._initial_actuator_initial_angle = (
                kinematic.actuators.optimizable_parameters[
                    :, index_mapping.actuator_initial_angle, :
                ].detach().clone()
            )
            kinematic._initial_actuator_offset = (
                kinematic.actuators.non_optimizable_parameters[
                    :, index_mapping.actuator_offset, :
                ].detach().clone()
            )
        if not hasattr(kinematic, "_initial_translation_deviation"):
            kinematic._initial_translation_deviation = (
                kinematic.translation_deviation_parameters.detach().clone()
            )
        initial_actuator_initial_angle = kinematic._initial_actuator_initial_angle
        initial_actuator_offset = kinematic._initial_actuator_offset

        base_lr = float(self.optimization_configuration[config_dictionary.initial_learning_rate])
        optimizer = torch.optim.Adam(
            [{"params": kinematic.rotation_deviation_parameters, "lr": base_lr}],
            lr=base_lr,
        )
        return optimizer, initial_actuator_initial_angle, initial_actuator_offset


class _FirstJointRotationsOnlyMixin(_SingleAxisRotationMixin):
    """Optimizer mixin: only first joint tilts (elevation axis) — indices 0, 1."""
    _ROTATION_INDICES = (0, 1)


class _SecondJointRotationsOnlyMixin(_SingleAxisRotationMixin):
    """Optimizer mixin: only second joint tilts (azimuth axis) — indices 2, 3."""
    _ROTATION_INDICES = (2, 3)


class _RotationsOnlyMixin:
    """Optimizer mixin: only rotation_deviation_parameters are optimised."""

    def _setup_optimizer(self, heliostat_group, device):
        kinematic = heliostat_group.kinematics
        kinematic.rotation_deviation_parameters.requires_grad_()

        # Snapshot nominals — required by parent _apply_deviation_bounds.
        if not hasattr(kinematic, "_initial_actuator_initial_angle"):
            kinematic._initial_actuator_initial_angle = (
                kinematic.actuators.optimizable_parameters[
                    :, index_mapping.actuator_initial_angle, :
                ].detach().clone()
            )
            kinematic._initial_actuator_offset = (
                kinematic.actuators.non_optimizable_parameters[
                    :, index_mapping.actuator_offset, :
                ].detach().clone()
            )
        if not hasattr(kinematic, "_initial_translation_deviation"):
            kinematic._initial_translation_deviation = (
                kinematic.translation_deviation_parameters.detach().clone()
            )
        initial_actuator_initial_angle = kinematic._initial_actuator_initial_angle
        initial_actuator_offset = kinematic._initial_actuator_offset

        base_lr = float(self.optimization_configuration[config_dictionary.initial_learning_rate])
        optimizer = torch.optim.Adam(
            [{"params": kinematic.rotation_deviation_parameters, "lr": base_lr}],
            lr=base_lr,
        )
        return optimizer, initial_actuator_initial_angle, initial_actuator_offset


class _RotationsActuatorsMixin:
    """Optimizer mixin: rotation_deviation_parameters + actuator params (aᵢ, cᵢ)."""

    def _setup_optimizer(self, heliostat_group, device):
        kinematic = heliostat_group.kinematics
        kinematic.rotation_deviation_parameters.requires_grad_()
        kinematic.actuators.optimizable_parameters.requires_grad_()
        kinematic.actuators.non_optimizable_parameters.requires_grad_()

        # Freeze bᵢ (initial_stroke_length).
        def _freeze_stroke_length(grad: torch.Tensor) -> torch.Tensor:
            mask = torch.ones_like(grad)
            mask[:, index_mapping.actuator_initial_stroke_length, :] = 0.0
            return grad * mask

        kinematic.actuators.optimizable_parameters.register_hook(_freeze_stroke_length)

        # Restrict non_optimizable gradients to cᵢ only.
        def _only_actuator_offset(grad: torch.Tensor) -> torch.Tensor:
            mask = torch.zeros_like(grad)
            mask[:, index_mapping.actuator_offset, :] = 1.0
            return grad * mask

        kinematic.actuators.non_optimizable_parameters.register_hook(_only_actuator_offset)

        if not hasattr(kinematic, "_initial_actuator_initial_angle"):
            kinematic._initial_actuator_initial_angle = (
                kinematic.actuators.optimizable_parameters[
                    :, index_mapping.actuator_initial_angle, :
                ].detach().clone()
            )
            kinematic._initial_actuator_offset = (
                kinematic.actuators.non_optimizable_parameters[
                    :, index_mapping.actuator_offset, :
                ].detach().clone()
            )
        if not hasattr(kinematic, "_initial_translation_deviation"):
            kinematic._initial_translation_deviation = (
                kinematic.translation_deviation_parameters.detach().clone()
            )
        initial_actuator_initial_angle = kinematic._initial_actuator_initial_angle
        initial_actuator_offset = kinematic._initial_actuator_offset

        base_lr = float(self.optimization_configuration[config_dictionary.initial_learning_rate])
        optimizer = torch.optim.Adam(
            [
                {"params": kinematic.rotation_deviation_parameters, "lr": base_lr},
                {"params": kinematic.actuators.optimizable_parameters, "lr": base_lr},
                {"params": kinematic.actuators.non_optimizable_parameters, "lr": base_lr},
            ],
            lr=base_lr,
        )
        return optimizer, initial_actuator_initial_angle, initial_actuator_offset


class _RotationsTranslationsMixin:
    """Optimizer mixin: rotation_deviation_parameters + translation_deviation_parameters."""

    def _setup_optimizer(self, heliostat_group, device):
        kinematic = heliostat_group.kinematics
        kinematic.rotation_deviation_parameters.requires_grad_()
        kinematic.translation_deviation_parameters.requires_grad_()

        if not hasattr(kinematic, "_initial_actuator_initial_angle"):
            kinematic._initial_actuator_initial_angle = (
                kinematic.actuators.optimizable_parameters[
                    :, index_mapping.actuator_initial_angle, :
                ].detach().clone()
            )
            kinematic._initial_actuator_offset = (
                kinematic.actuators.non_optimizable_parameters[
                    :, index_mapping.actuator_offset, :
                ].detach().clone()
            )
        if not hasattr(kinematic, "_initial_translation_deviation"):
            kinematic._initial_translation_deviation = (
                kinematic.translation_deviation_parameters.detach().clone()
            )
        initial_actuator_initial_angle = kinematic._initial_actuator_initial_angle
        initial_actuator_offset = kinematic._initial_actuator_offset

        base_lr = float(self.optimization_configuration[config_dictionary.initial_learning_rate])
        optimizer = torch.optim.Adam(
            [
                {"params": kinematic.translation_deviation_parameters, "lr": base_lr * 5.0},
                {"params": kinematic.rotation_deviation_parameters, "lr": base_lr},
            ],
            lr=base_lr,
        )
        return optimizer, initial_actuator_initial_angle, initial_actuator_offset


class NoParametersReconstructor(WortbergKinematicReconstructor):
    """
    Sanity-check reconstructor: run the full training loop with all kinematic parameters frozen.

    A single dummy scalar is optimized so backward, scheduler, and optimizer logic still execute.
    If any kinematic parameter changes in this configuration, the change comes from leakage
    elsewhere in the pipeline rather than from the intended optimization set.
    """

    def __init__(self, *args, **kwargs):
        kwargs["train_position_deviation"] = False
        super().__init__(*args, **kwargs)
        self._dummy_train_anchor: torch.nn.Parameter | None = None

    def _setup_optimizer(self, heliostat_group, device):
        kinematic = heliostat_group.kinematics

        kinematic.translation_deviation_parameters.requires_grad_(False)
        kinematic.rotation_deviation_parameters.requires_grad_(False)
        kinematic.actuators.optimizable_parameters.requires_grad_(False)
        kinematic.actuators.non_optimizable_parameters.requires_grad_(False)

        if hasattr(kinematic, "_base_position_deviation"):
            kinematic._base_position_deviation = kinematic._base_position_deviation.detach()

        if not hasattr(kinematic, "_initial_actuator_initial_angle"):
            kinematic._initial_actuator_initial_angle = (
                kinematic.actuators.optimizable_parameters[
                    :, index_mapping.actuator_initial_angle, :
                ]
                .detach()
                .clone()
            )
            kinematic._initial_actuator_offset = (
                kinematic.actuators.non_optimizable_parameters[
                    :, index_mapping.actuator_offset, :
                ]
                .detach()
                .clone()
            )
        if not hasattr(kinematic, "_initial_translation_deviation"):
            kinematic._initial_translation_deviation = (
                kinematic.translation_deviation_parameters.detach().clone()
            )

        self._dummy_train_anchor = torch.nn.Parameter(torch.zeros(1, device=device))
        base_lr = float(self.optimization_configuration[config_dictionary.initial_learning_rate])
        optimizer = torch.optim.Adam([self._dummy_train_anchor], lr=base_lr)
        return (
            optimizer,
            kinematic._initial_actuator_initial_angle,
            kinematic._initial_actuator_offset,
        )

    def _compute_epoch_loss(
        self,
        epoch: int,
        flux_distributions: torch.Tensor,
        measured_flux: torch.Tensor,
        focal_spots_measured: torch.Tensor,
        sample_indices: torch.Tensor,
        target_area_indices: torch.Tensor,
        loss_definition,
        ground_truth: torch.Tensor,
        reduction_dims: tuple,
        number_of_samples_per_heliostat: int,
        device: torch.device,
    ) -> torch.Tensor:
        loss_per_heliostat = super()._compute_epoch_loss(
            epoch=epoch,
            flux_distributions=flux_distributions,
            measured_flux=measured_flux,
            focal_spots_measured=focal_spots_measured,
            sample_indices=sample_indices,
            target_area_indices=target_area_indices,
            loss_definition=loss_definition,
            ground_truth=ground_truth,
            reduction_dims=reduction_dims,
            number_of_samples_per_heliostat=number_of_samples_per_heliostat,
            device=device,
        )
        if self._dummy_train_anchor is None:
            return loss_per_heliostat
        return loss_per_heliostat + (0.0 * self._dummy_train_anchor.sum())


# ---------------------------------------------------------------------------
# Focal-spot ablation variants (A–E)
# ---------------------------------------------------------------------------

class RotationsOnlyReconstructor(_RotationsOnlyMixin, WortbergKinematicReconstructor):
    """
    Config A: only ``rotation_deviation_parameters`` (4 main-axis tilts) are optimised.

    All other parameters (translations, actuators, base position) are frozen.
    Use this as the minimal structural baseline.
    """

    def __init__(self, *args, **kwargs):
        kwargs["train_position_deviation"] = False
        super().__init__(*args, **kwargs)


class FirstJointRotationsReconstructor(_FirstJointRotationsOnlyMixin, WortbergKinematicReconstructor):
    """
    Config 0a: only first joint tilts (elevation axis, 2 params) are optimised.

    Even more minimal than Config A — tests whether the elevation axis alone
    can explain the tracking error.
    """

    def __init__(self, *args, **kwargs):
        kwargs["train_position_deviation"] = False
        super().__init__(*args, **kwargs)


class SecondJointRotationsReconstructor(_SecondJointRotationsOnlyMixin, WortbergKinematicReconstructor):
    """
    Config 0b: only second joint tilts (azimuth axis, 2 params) are optimised.

    Counterpart to 0a — tests whether the azimuth axis alone is sufficient.
    """

    def __init__(self, *args, **kwargs):
        kwargs["train_position_deviation"] = False
        super().__init__(*args, **kwargs)


class RotationsActuatorsReconstructor(_RotationsActuatorsMixin, WortbergKinematicReconstructor):
    """
    Config B: ``rotation_deviation_parameters`` + actuator params (aᵢ, cᵢ).

    Translations and base position are frozen.
    """

    def __init__(self, *args, **kwargs):
        kwargs["train_position_deviation"] = False
        super().__init__(*args, **kwargs)


class RotationsTranslationsReconstructor(_RotationsTranslationsMixin, WortbergKinematicReconstructor):
    """
    Config C: ``rotation_deviation_parameters`` + ``translation_deviation_parameters``.

    Actuators and base position are frozen.
    Translations use 5× the base LR (large-scale params, same as Wortberg).
    """

    def __init__(self, *args, **kwargs):
        kwargs["train_position_deviation"] = False
        super().__init__(*args, **kwargs)


class FullStructuralReconstructor(WortbergKinematicReconstructor):
    """
    Config D: rotations + translations + actuators (aᵢ, cᵢ), no base position.

    Identical to WortbergKinematicReconstructor(train_position_deviation=False).
    Named explicitly for clarity in the parameter ablation study.
    """

    def __init__(self, *args, **kwargs):
        kwargs["train_position_deviation"] = False
        super().__init__(*args, **kwargs)


class _RotationsActuatorsBasePosMixin(_RotationsActuatorsMixin):
    """Optimizer mixin: rotations + actuators (aᵢ, cᵢ) + base position.

    Translations are frozen. Base position deviation (δe, δn, δu) is trained.
    This is the largest parameter set that excludes the poorly-identifiable
    translation_deviation_parameters.
    """

    def _setup_optimizer(self, heliostat_group, device):
        # Delegate the rotation + actuator setup to the parent mixin.
        optimizer, initial_actuator_initial_angle, initial_actuator_offset = (
            super()._setup_optimizer(heliostat_group, device)
        )
        kinematic = heliostat_group.kinematics

        # Initialise (or re-enable) base position deviation.
        if hasattr(kinematic, "_base_position_deviation"):
            kinematic._base_position_deviation = (
                kinematic._base_position_deviation.detach().requires_grad_(True)
            )
        else:
            kinematic._base_position_deviation = torch.zeros(
                kinematic.number_of_heliostats, 3, device=device, requires_grad=True
            )

        base_lr = float(self.optimization_configuration[config_dictionary.initial_learning_rate])
        optimizer.add_param_group(
            {"params": kinematic._base_position_deviation, "lr": base_lr * 5.0}
        )
        return optimizer, initial_actuator_initial_angle, initial_actuator_offset


class RotationsActuatorsBasePosReconstructor(_RotationsActuatorsBasePosMixin, WortbergKinematicReconstructor):
    """
    Config C (no-translation variant): rotations + actuators (aᵢ, cᵢ) + base position (11 params).

    Translations are excluded. This is the recommended full config when translations
    are found to be non-identifiable from focal-spot data.
    """

    def __init__(self, *args, **kwargs):
        kwargs["train_position_deviation"] = True
        super().__init__(*args, **kwargs)


class WortbergPixelReconstructor(WortbergKinematicReconstructor):
    """
    Variant of WortbergKinematicReconstructor that uses measured flux bitmaps
    as ground truth instead of focal spot centroids.

    Before computing the pixel loss each epoch, the predicted flux is Gaussian-
    blurred (to smooth ray-tracing sparsity at low surface-point counts) and
    both predicted and measured images are peak-normalized to [0, 1] per image.

    This makes it compatible with pixel-based loss functions such as
    PixelLoss and KLDivergenceLoss, which compare full flux images rather
    than centroid positions.
    """

    # Default Gaussian blur σ (pixels). Override via constructor blur_sigma argument.
    BLUR_SIGMA: float = 5.0

    def __init__(self, *args, blur_sigma: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.BLUR_SIGMA = blur_sigma

    def _get_ground_truth_and_reduction_dims(
        self,
        measured_flux: torch.Tensor,
        focal_spots_measured: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple]:
        """Return measured flux bitmaps and spatial reduction dims."""
        return measured_flux, (index_mapping.batched_bitmap_e, index_mapping.batched_bitmap_u)  # focal_spots_measured unused

    def _preprocess_eval_flux(
        self,
        predicted_flux: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Blur and peak-normalize predicted flux; peak-normalize ground truth."""
        blurred = self._gaussian_blur(predicted_flux, self.BLUR_SIGMA)
        return self._peak_normalize(blurred), self._peak_normalize(ground_truth)

    @staticmethod
    def _gaussian_blur(flux: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply a differentiable Gaussian blur to a batch of flux images [N, H, W].

        Uses separable 2-D Gaussian convolution so gradients flow through
        the blur into the kinematic parameters.
        """
        if sigma <= 0:
            return flux
        kernel_size = int(4 * sigma + 0.5) * 2 + 1  # smallest odd integer covering ±2σ
        coords = torch.arange(kernel_size, device=flux.device, dtype=flux.dtype) - kernel_size // 2
        gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel = (gauss_1d[:, None] * gauss_1d[None, :]).view(1, 1, kernel_size, kernel_size)
        return F.conv2d(flux.unsqueeze(1), kernel, padding=kernel_size // 2).squeeze(1)

    @staticmethod
    def _peak_normalize(flux: torch.Tensor) -> torch.Tensor:
        """Peak-normalize each image in a batch to [0, 1] independently."""
        N = flux.shape[0]
        max_vals = flux.view(N, -1).max(dim=1).values.clamp(min=1e-12)
        return flux / max_vals.view(N, 1, 1)

    def _compute_epoch_loss(
        self,
        epoch: int,
        flux_distributions: torch.Tensor,
        measured_flux: torch.Tensor,
        focal_spots_measured: torch.Tensor,
        sample_indices: torch.Tensor,
        target_area_indices: torch.Tensor,
        loss_definition,
        ground_truth: torch.Tensor,
        reduction_dims: tuple,
        number_of_samples_per_heliostat: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Blur and peak-normalize predicted flux, peak-normalize ground truth, then compute pixel loss."""
        blurred = self._gaussian_blur(flux_distributions, self.BLUR_SIGMA)
        loss_per_sample = loss_definition(
            prediction=self._peak_normalize(blurred),
            ground_truth=self._peak_normalize(ground_truth[sample_indices]),
            target_area_indices=target_area_indices[sample_indices],
            reduction_dimensions=reduction_dims,
            device=device,
        )
        return core_utils.mean_loss_per_heliostat(
            loss_per_sample=loss_per_sample,
            number_of_samples_per_heliostat=number_of_samples_per_heliostat,
        )


# ---------------------------------------------------------------------------
# Pixel-loss ablation variants (A–E)
#
# Each class combines one of the optimizer-setup mixins with
# WortbergPixelReconstructor so that the same parameter subset is active
# during pixel-loss fine-tuning as during focal-spot pretraining.
# ---------------------------------------------------------------------------

class RotationsOnlyPixelReconstructor(_RotationsOnlyMixin, WortbergPixelReconstructor):
    """
    Config A (pixel): only ``rotation_deviation_parameters`` optimised, pixel loss.
    """


class FirstJointRotationsPixelReconstructor(_FirstJointRotationsOnlyMixin, WortbergPixelReconstructor):
    """
    Config 0a (pixel): only first joint tilts (elevation axis, 2 params), pixel loss.
    """


class SecondJointRotationsPixelReconstructor(_SecondJointRotationsOnlyMixin, WortbergPixelReconstructor):
    """
    Config 0b (pixel): only second joint tilts (azimuth axis, 2 params), pixel loss.
    """


class RotationsActuatorsPixelReconstructor(_RotationsActuatorsMixin, WortbergPixelReconstructor):
    """
    Config B (pixel): rotation_deviation_parameters + actuators (aᵢ, cᵢ), pixel loss.
    """


class RotationsTranslationsPixelReconstructor(_RotationsTranslationsMixin, WortbergPixelReconstructor):
    """
    Config C (pixel): rotation_deviation_parameters + translations, pixel loss.
    Translations use 5× base LR.
    """


class FullStructuralPixelReconstructor(WortbergPixelReconstructor):
    """
    Config D (pixel): rotations + translations + actuators, no base position, pixel loss.
    """

    def __init__(self, *args, **kwargs):
        kwargs["train_position_deviation"] = False
        super().__init__(*args, **kwargs)


# Config E (pixel): WortbergPixelReconstructor directly (all params, pixel loss).


class WortbergAnnealingReconstructor(WortbergKinematicReconstructor):
    """
    Variant of WortbergKinematicReconstructor that linearly anneals between
    FocalSpotLoss and PixelLoss over the course of training.

    At epoch 0 the combined loss is purely FocalSpotLoss (alpha=1, beta=0).
    At max_epoch it is purely PixelLoss (alpha=0, beta=1).  In between,
    both losses are normalised by their value at epoch 0 so that they
    contribute equally when alpha == beta == 0.5.

    Parameters
    ----------
    focal_loss : callable
        A FocalSpotLoss instance.
    pixel_loss : callable
        A PixelLoss instance.
    """

    def __init__(self, *args, focal_loss, pixel_loss, **kwargs):
        super().__init__(*args, **kwargs)
        self._focal_loss_fn = focal_loss
        self._pixel_loss_fn = pixel_loss
        self._focal_loss_0: torch.Tensor | None = None
        self._pixel_loss_0: torch.Tensor | None = None

    def _compute_epoch_loss(
        self,
        epoch: int,
        flux_distributions: torch.Tensor,
        measured_flux: torch.Tensor,
        focal_spots_measured: torch.Tensor,
        sample_indices: torch.Tensor,
        target_area_indices: torch.Tensor,
        loss_definition,           # unused — uses self._focal_loss_fn / _pixel_loss_fn
        ground_truth: torch.Tensor,  # unused
        reduction_dims: tuple,       # unused
        number_of_samples_per_heliostat: int,
        device: torch.device,
    ) -> torch.Tensor:
        max_epoch = self.optimization_configuration[config_dictionary.max_epoch]
        alpha = 1.0 - epoch / max_epoch
        beta = epoch / max_epoch

        # ---- Focal spot loss ----
        focal_per_sample = self._focal_loss_fn(
            prediction=flux_distributions,
            ground_truth=focal_spots_measured[sample_indices],
            target_area_indices=target_area_indices[sample_indices],
            reduction_dimensions=(index_mapping.focal_spots,),
            device=device,
        )
        focal_per_heliostat = core_utils.mean_loss_per_heliostat(
            loss_per_sample=focal_per_sample,
            number_of_samples_per_heliostat=number_of_samples_per_heliostat,
        )

        # ---- Pixel loss ----
        pixel_per_sample = self._pixel_loss_fn(
            prediction=flux_distributions,
            ground_truth=measured_flux[sample_indices],
            target_area_indices=target_area_indices[sample_indices],
            reduction_dimensions=(index_mapping.batched_bitmap_e, index_mapping.batched_bitmap_u),
            device=device,
        )
        pixel_per_heliostat = core_utils.mean_loss_per_heliostat(
            loss_per_sample=pixel_per_sample,
            number_of_samples_per_heliostat=number_of_samples_per_heliostat,
        )

        # ---- Normalise by initial values (set on first call) ----
        if self._focal_loss_0 is None:
            self._focal_loss_0 = focal_per_heliostat.mean().detach().clamp(min=1e-8)
            self._pixel_loss_0 = pixel_per_heliostat.mean().detach().clamp(min=1e-8)

        focal_norm = focal_per_heliostat / self._focal_loss_0
        pixel_norm = pixel_per_heliostat / self._pixel_loss_0

        return alpha * focal_norm + beta * pixel_norm


class WortbergAlignmentReconstructor(WortbergKinematicReconstructor):
    """
    Variant of WortbergKinematicReconstructor that uses motor position (alignment) loss.

    Instead of ray-tracing to produce flux images, this reconstructor compares the motor
    positions predicted by the kinematic inverse-kinematics solver against the motor positions
    measured during calibration (from PAINT calibration-properties JSON files).

    No ray tracing is performed during training or validation, making each epoch significantly
    faster and requiring much less GPU memory.

    The loss is computed via ``AlignmentLoss`` (passed as ``loss_definition``), which converts
    both predicted and measured motor positions to joint angles (radians) before computing MSE.
    """

    def _reconstruct_kinematics_parameters_with_raytracing(
        self,
        loss_definition,
        device=None,
    ) -> torch.Tensor:
        """Training loop without ray tracing — uses motor position MSE instead."""
        device = get_device(device=device)
        rank = self.ddp_setup[config_dictionary.rank]

        if rank == 0:
            log.info(
                "Beginning kinematic reconstruction with alignment loss "
                "(motor position MSE, no ray tracing)."
            )

        final_loss_per_heliostat = torch.full(
            (self.scenario.heliostat_field.number_of_heliostats_per_group.sum(),),
            torch.inf,
            device=device,
        )
        final_loss_start_indices = torch.cat(
            [
                torch.tensor([0], device=device),
                self.scenario.heliostat_field.number_of_heliostats_per_group.cumsum(
                    index_mapping.heliostat_dimension
                ),
            ]
        )

        self._convergence_history = []

        for heliostat_group_index in self.ddp_setup[config_dictionary.groups_to_ranks_mapping][rank]:
            heliostat_group = self.scenario.heliostat_field.heliostat_groups[heliostat_group_index]
            parser = self.data[config_dictionary.data_parser]
            heliostat_mapping = self.data[config_dictionary.heliostat_data_mapping]

            (
                _measured_flux,
                _focal_spots_measured,
                incident_ray_directions,
                motor_positions_measured,   # [N_active, 2] — kept, not discarded
                active_heliostats_mask,
                target_area_mask,
            ) = parser.parse_data_for_reconstruction(
                heliostat_data_mapping=heliostat_mapping,
                heliostat_group=heliostat_group,
                scenario=self.scenario,
                device=device,
            )

            if active_heliostats_mask.sum() == 0:
                continue

            optimizer, initial_actuator_initial_angle, initial_actuator_offset = (
                self._setup_optimizer(heliostat_group, device)
            )
            scheduler = self._setup_scheduler(optimizer)
            early_stopper = self._setup_early_stopper()

            # Per-heliostat sample counts from the mask (may be non-uniform when
            # some heliostats have fewer than sample_limit calibration files).
            nonzero_sample_counts = active_heliostats_mask[active_heliostats_mask > 0].tolist()

            loss = torch.inf
            epoch = 0
            log_step = (
                self.optimization_configuration[config_dictionary.max_epoch]
                if self.optimization_configuration[config_dictionary.log_step] == 0
                else self.optimization_configuration[config_dictionary.log_step]
            )

            best_eval_loss = float("inf")
            best_params: dict | None = None

            while (
                loss > float(self.optimization_configuration[config_dictionary.tolerance])
                and epoch <= self.optimization_configuration[config_dictionary.max_epoch]
            ):
                optimizer.zero_grad()

                heliostat_group.activate_heliostats(
                    active_heliostats_mask=active_heliostats_mask, device=device
                )
                kinematic = heliostat_group.kinematics

                if self.train_position_deviation and hasattr(kinematic, "_base_position_deviation"):
                    active_base_dev = kinematic._base_position_deviation.repeat_interleave(
                        active_heliostats_mask, dim=0
                    )
                    pad = torch.zeros(active_base_dev.shape[0], 1, device=device)
                    kinematic.active_heliostat_positions = (
                        kinematic.active_heliostat_positions
                        + torch.cat([active_base_dev, pad], dim=1)
                    )

                # Forward pass — sets kinematic.active_motor_positions as a byproduct
                heliostat_group.align_surfaces_with_incident_ray_directions(
                    aim_points=self.scenario.solar_tower.get_centers_of_target_areas(
                        target_area_mask, device=device
                    ),
                    incident_ray_directions=incident_ray_directions,
                    active_heliostats_mask=active_heliostats_mask,
                    device=device,
                )

                # Alignment loss — no ray tracing needed
                predicted_mp = kinematic.active_motor_positions  # [N_active, 2]
                loss_per_sample = loss_definition(
                    predicted_motor_positions=predicted_mp,
                    measured_motor_positions=motor_positions_measured,
                    actuators=kinematic.actuators,
                    device=device,
                )
                # Split per-heliostat using actual sample counts from the mask.
                # This handles variable sample counts (some heliostats may have
                # fewer than sample_limit calibration files available).
                split_losses = torch.split(loss_per_sample, nonzero_sample_counts)
                loss_per_heliostat = torch.stack([c.mean() for c in split_losses])

                loss = loss_per_heliostat.mean()
                loss.backward()
                optimizer.step()

                self._apply_deviation_bounds(
                    heliostat_group,
                    initial_actuator_initial_angle,
                    initial_actuator_offset,
                )

                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss.detach())
                else:
                    scheduler.step()

                if epoch % log_step == 0:
                    mem_alloc = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
                    mem_peak = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
                    log.info(
                        f"Rank: {rank}, Epoch: {epoch}, Loss: {loss:.6f}, "
                        f"LR: {optimizer.param_groups[index_mapping.optimizer_param_group_0]['lr']}, "
                        f"GPU mem: {mem_alloc:.2f} GB (peak: {mem_peak:.2f} GB)"
                    )
                    entry = {
                        "epoch": epoch,
                        "group": heliostat_group_index,
                        "loss": loss.item(),
                        "translation_deviation_mean_abs": kinematic.translation_deviation_parameters.abs().mean().item(),
                        "rotation_deviation_mean_abs": kinematic.rotation_deviation_parameters.abs().mean().item(),
                        "actuator_angle_dev_mean_abs": (
                            kinematic.actuators.optimizable_parameters[:, index_mapping.actuator_initial_angle, :]
                            - initial_actuator_initial_angle
                        ).abs().mean().item(),
                        "actuator_offset_dev_mean_abs": (
                            kinematic.actuators.non_optimizable_parameters[:, index_mapping.actuator_offset, :]
                            - initial_actuator_offset
                        ).abs().mean().item(),
                    }
                    if self.train_position_deviation and hasattr(kinematic, "_base_position_deviation"):
                        entry["base_pos_dev_e_mean_abs"] = kinematic._base_position_deviation[:, 0].abs().mean().item()
                        entry["base_pos_dev_n_mean_abs"] = kinematic._base_position_deviation[:, 1].abs().mean().item()
                        entry["base_pos_dev_u_mean_abs"] = kinematic._base_position_deviation[:, 2].abs().mean().item()
                    eval_loss = self._compute_eval_loss_no_grad(heliostat_group, loss_definition, device)
                    if eval_loss is not None:
                        entry["eval_loss"] = eval_loss
                        log.info(f"Rank: {rank}, Epoch: {epoch}, Eval Loss: {eval_loss:.6f}")
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            best_params = self._snapshot_kinematic_params(heliostat_group.kinematics)
                    self._convergence_history.append(entry)

                if early_stopper.step(loss):
                    log.info(f"Early stopping at epoch {epoch}.")
                    break

                epoch += 1

            if best_params is not None:
                self._restore_kinematic_params(heliostat_group.kinematics, best_params)
                log.info(f"Restored best kinematic parameters (val_loss={best_eval_loss:.6f}).")

            # Map per-heliostat loss back into the global result tensor
            global_active_indices = torch.nonzero(active_heliostats_mask != 0, as_tuple=True)[0]
            final_indices = global_active_indices + final_loss_start_indices[heliostat_group_index]
            final_loss_per_heliostat[final_indices] = loss_per_heliostat.detach()
            log.info(f"Rank: {rank}, Kinematic reconstructed.")

        if self.ddp_setup[config_dictionary.is_distributed]:
            for index, heliostat_group in enumerate(self.scenario.heliostat_field.heliostat_groups):
                src = self.ddp_setup[config_dictionary.ranks_to_groups_mapping][index][
                    index_mapping.first_rank_from_group
                ]
                torch.distributed.broadcast(
                    heliostat_group.kinematics.translation_deviation_parameters, src=src
                )
                torch.distributed.broadcast(
                    heliostat_group.kinematics.rotation_deviation_parameters, src=src
                )
                torch.distributed.broadcast(
                    heliostat_group.kinematics.actuators.optimizable_parameters, src=src
                )
                torch.distributed.broadcast(
                    heliostat_group.kinematics.actuators.non_optimizable_parameters, src=src
                )
                if self.train_position_deviation and hasattr(heliostat_group.kinematics, "_base_position_deviation"):
                    torch.distributed.broadcast(
                        heliostat_group.kinematics._base_position_deviation, src=src
                    )
            torch.distributed.all_reduce(
                final_loss_per_heliostat, op=torch.distributed.ReduceOp.MIN
            )
            log.info(f"Rank: {rank}, synchronized after kinematic reconstruction.")

        return final_loss_per_heliostat, self._convergence_history

    @torch.no_grad()
    def _compute_eval_loss_no_grad(self, heliostat_group, loss_definition, device) -> float | None:
        """Compute validation alignment loss without ray tracing."""
        if self.eval_data is None:
            return None
        eval_parser = self.eval_data["data_parser"]
        eval_mapping = self.eval_data["heliostat_data_mapping"]

        (
            _measured_flux,
            _focal_spots_measured,
            incident_ray_directions,
            motor_positions_measured,
            active_mask,
            target_mask,
        ) = eval_parser.parse_data_for_reconstruction(
            heliostat_data_mapping=eval_mapping,
            heliostat_group=heliostat_group,
            scenario=self.scenario,
            device=device,
        )

        if active_mask.sum() == 0:
            return None

        heliostat_group.activate_heliostats(active_heliostats_mask=active_mask, device=device)
        kinematic = heliostat_group.kinematics

        if self.train_position_deviation and hasattr(kinematic, "_base_position_deviation"):
            active_base_dev = kinematic._base_position_deviation.repeat_interleave(active_mask, dim=0)
            pad = torch.zeros(active_base_dev.shape[0], 1, device=device)
            kinematic.active_heliostat_positions = (
                kinematic.active_heliostat_positions + torch.cat([active_base_dev, pad], dim=1)
            )

        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=self.scenario.solar_tower.get_centers_of_target_areas(
                target_mask, device=device
            ),
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_mask,
            device=device,
        )

        loss_per_sample = loss_definition(
            predicted_motor_positions=kinematic.active_motor_positions,
            measured_motor_positions=motor_positions_measured,
            actuators=kinematic.actuators,
            device=device,
        )
        nonzero_counts = active_mask[active_mask > 0].tolist()
        split_losses = torch.split(loss_per_sample, nonzero_counts)
        loss_per_heliostat = torch.stack([c.mean() for c in split_losses])
        return loss_per_heliostat.mean().item()
