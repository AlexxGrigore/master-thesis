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

from artist.core import core_utils, learning_rate_schedulers
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.kinematic_reconstructor import KinematicReconstructor
from artist.util import config_dictionary, index_mapping
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)


class WortbergKinematicReconstructor(KinematicReconstructor):
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

    def __init__(self, *args, train_position_deviation: bool = True, eval_data: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_position_deviation = train_position_deviation
        self.eval_data = eval_data

    def _reconstruct_kinematic_parameters_with_raytracing(
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

                while (
                    loss > float(self.optimization_configuration[config_dictionary.tolerance])
                    and epoch <= self.optimization_configuration[config_dictionary.max_epoch]
                ):
                    optimizer.zero_grad()

                    heliostat_group.activate_heliostats(
                        active_heliostats_mask=active_heliostats_mask, device=device
                    )

                    kinematic = heliostat_group.kinematic

                    if self.train_position_deviation:
                        # Inject base position deviation into the active positions that
                        # ARTIST just set.  We overwrite with a new tensor (no in-place op)
                        # so autograd traces through repeat_interleave → _base_position_deviation.
                        active_base_dev = kinematic._base_position_deviation.repeat_interleave(
                            active_heliostats_mask, dim=0
                        )  # [N_active, 3]
                        pad = torch.zeros(active_base_dev.shape[0], 1, device=device)
                        kinematic.active_heliostat_positions = (
                            kinematic.active_heliostat_positions
                            + torch.cat([active_base_dev, pad], dim=1)
                        )

                    heliostat_group.align_surfaces_with_incident_ray_directions(
                        aim_points=self.scenario.target_areas.centers[target_area_mask],
                        incident_ray_directions=incident_ray_directions,
                        active_heliostats_mask=active_heliostats_mask,
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

                    flux_distributions = ray_tracer.trace_rays(
                        incident_ray_directions=incident_ray_directions,
                        active_heliostats_mask=active_heliostats_mask,
                        target_area_mask=target_area_mask,
                        device=device,
                    )

                    sample_indices_for_local_rank = ray_tracer.get_sampler_indices()

                    number_of_samples_per_heliostat = int(
                        heliostat_group.active_heliostats_mask.sum()
                        / (heliostat_group.active_heliostats_mask > 0).sum()
                    )

                    loss_per_heliostat = self._compute_epoch_loss(
                        epoch=epoch,
                        flux_distributions=flux_distributions,
                        measured_flux=measured_flux,
                        focal_spots_measured=focal_spots_measured,
                        sample_indices=sample_indices_for_local_rank,
                        target_area_mask=target_area_mask,
                        loss_definition=loss_definition,
                        ground_truth=ground_truth,
                        reduction_dims=reduction_dims,
                        number_of_samples_per_heliostat=number_of_samples_per_heliostat,
                        device=device,
                    )

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
                        if self.train_position_deviation:
                            entry["base_pos_dev_e_mean_abs"] = kinematic._base_position_deviation[:, 0].abs().mean().item()
                            entry["base_pos_dev_n_mean_abs"] = kinematic._base_position_deviation[:, 1].abs().mean().item()
                            entry["base_pos_dev_u_mean_abs"] = kinematic._base_position_deviation[:, 2].abs().mean().item()
                        eval_loss = self._compute_eval_loss_no_grad(heliostat_group, loss_definition, device)
                        if eval_loss is not None:
                            entry["eval_loss"] = eval_loss
                            log.info(f"Rank: {rank}, Epoch: {epoch}, Eval Loss: {eval_loss:.6f}")
                        self._convergence_history.append(entry)

                    if early_stopper.step(loss):
                        log.info(f"Early stopping at epoch {epoch}.")
                        break

                    epoch += 1

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
                    heliostat_group.kinematic.translation_deviation_parameters, src=src
                )
                torch.distributed.broadcast(
                    heliostat_group.kinematic.rotation_deviation_parameters, src=src
                )
                torch.distributed.broadcast(
                    heliostat_group.kinematic.actuators.optimizable_parameters, src=src
                )
                torch.distributed.broadcast(
                    heliostat_group.kinematic.actuators.non_optimizable_parameters, src=src
                )
                if self.train_position_deviation:
                    torch.distributed.broadcast(
                        heliostat_group.kinematic._base_position_deviation, src=src
                    )
            torch.distributed.all_reduce(
                final_loss_per_heliostat, op=torch.distributed.ReduceOp.MIN
            )
            log.info(f"Rank: {rank}, synchronized after kinematic reconstruction.")

        return final_loss_per_heliostat

    # ------------------------------------------------------------------
    # Private helpers — each responsible for one setup concern
    # ------------------------------------------------------------------

    def _setup_optimizer(self, heliostat_group, device):
        """Enable gradients, register freeze hooks, and return a configured Adam optimizer."""
        kinematic = heliostat_group.kinematic

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
        params = self.optimization_configuration.get(config_dictionary.scheduler_parameters, {})

        if scheduler_type == config_dictionary.reduce_on_plateau:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=params.get(config_dictionary.reduce_factor, 0.5),
                patience=params.get(config_dictionary.patience, 10),
                threshold=params.get(config_dictionary.threshold, 1e-4),
                cooldown=params.get(config_dictionary.cooldown, 5),
                min_lr=params.get(config_dictionary.min, 1e-8),
            )

        # Fallback: cosine annealing (original behaviour)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.optimization_configuration[config_dictionary.max_epoch],
            eta_min=params.get(config_dictionary.min, 1e-6),
        )

    def _setup_early_stopper(self):
        """Build and return the early stopping instance."""
        return learning_rate_schedulers.EarlyStopping(
            window_size=self.optimization_configuration[config_dictionary.early_stopping_window],
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
        target_area_mask: torch.Tensor,
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
            target_area_mask=target_area_mask[sample_indices],
            reduction_dimensions=reduction_dims,
            device=device,
        )
        return core_utils.mean_loss_per_heliostat(
            loss_per_sample=loss_per_sample,
            number_of_samples_per_heliostat=number_of_samples_per_heliostat,
            device=device,
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
        kinematic = heliostat_group.kinematic

        if self.train_position_deviation and hasattr(kinematic, "_base_position_deviation"):
            active_base_dev = kinematic._base_position_deviation.repeat_interleave(active_mask, dim=0)
            pad = torch.zeros(active_base_dev.shape[0], 1, device=device)
            kinematic.active_heliostat_positions = (
                kinematic.active_heliostat_positions + torch.cat([active_base_dev, pad], dim=1)
            )

        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=self.scenario.target_areas.centers[target_mask],
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
        flux = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_directions,
            active_heliostats_mask=active_mask,
            target_area_mask=target_mask,
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
            target_area_mask=target_mask[sample_indices],
            reduction_dimensions=reduction_dims,
            device=device,
        )
        loss_per_heliostat = core_utils.mean_loss_per_heliostat(
            loss_per_sample=loss_per_sample,
            number_of_samples_per_heliostat=n_samples,
            device=device,
        )
        return loss_per_heliostat.mean().item()

    def _apply_deviation_bounds(self, heliostat_group, initial_actuator_initial_angle, initial_actuator_offset):
        """
        Clamp all optimised parameters to their Table 5.3 deviation bounds.

        Both translation_deviation and actuator parameters may have non-zero nominal
        values loaded from the scenario (e.g. concentrator_translation_n ≈ 0.175 m),
        so all clamps are computed relative to the snapshotted initial values.
        """
        kinematic = heliostat_group.kinematic
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
            if self.train_position_deviation:
                kinematic._base_position_deviation.data.clamp_(
                    -self._BOUND_BASE_POSITION_M, self._BOUND_BASE_POSITION_M
                )


class RotationsOnlyReconstructor(WortbergKinematicReconstructor):
    """
    Config A: only ``rotation_deviation_parameters`` (4 main-axis tilts) are optimised.

    All other parameters (translations, actuators, base position) are frozen.
    Use this as the minimal structural baseline.
    """

    def _setup_optimizer(self, heliostat_group, device):
        kinematic = heliostat_group.kinematic
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


class RotationsActuatorsReconstructor(WortbergKinematicReconstructor):
    """
    Config B: ``rotation_deviation_parameters`` + actuator params (aᵢ, cᵢ).

    Translations and base position are frozen.
    """

    def _setup_optimizer(self, heliostat_group, device):
        kinematic = heliostat_group.kinematic
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


class RotationsTranslationsReconstructor(WortbergKinematicReconstructor):
    """
    Config C: ``rotation_deviation_parameters`` + ``translation_deviation_parameters``.

    Actuators and base position are frozen.
    Translations use 5× the base LR (large-scale params, same as Wortberg).
    """

    def _setup_optimizer(self, heliostat_group, device):
        kinematic = heliostat_group.kinematic
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


class FullStructuralReconstructor(WortbergKinematicReconstructor):
    """
    Config D: rotations + translations + actuators (aᵢ, cᵢ), no base position.

    Identical to WortbergKinematicReconstructor(train_position_deviation=False).
    Named explicitly for clarity in the parameter ablation study.
    """

    def __init__(self, *args, **kwargs):
        kwargs["train_position_deviation"] = False
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
        """Apply blur to predicted flux (normalization is handled inside PixelLoss)."""
        blurred = self._gaussian_blur(predicted_flux, self.BLUR_SIGMA)
        return blurred, ground_truth

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
        target_area_mask: torch.Tensor,
        loss_definition,
        ground_truth: torch.Tensor,
        reduction_dims: tuple,
        number_of_samples_per_heliostat: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Blur predicted flux then compute pixel loss (normalization is inside PixelLoss)."""
        blurred = self._gaussian_blur(flux_distributions, self.BLUR_SIGMA)
        loss_per_sample = loss_definition(
            prediction=blurred,
            ground_truth=ground_truth[sample_indices],
            target_area_mask=target_area_mask[sample_indices],
            reduction_dimensions=reduction_dims,
            device=device,
        )
        return core_utils.mean_loss_per_heliostat(
            loss_per_sample=loss_per_sample,
            number_of_samples_per_heliostat=number_of_samples_per_heliostat,
            device=device,
        )


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
        target_area_mask: torch.Tensor,
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
            target_area_mask=target_area_mask[sample_indices],
            reduction_dimensions=(index_mapping.focal_spots,),
            device=device,
        )
        focal_per_heliostat = core_utils.mean_loss_per_heliostat(
            loss_per_sample=focal_per_sample,
            number_of_samples_per_heliostat=number_of_samples_per_heliostat,
            device=device,
        )

        # ---- Pixel loss ----
        pixel_per_sample = self._pixel_loss_fn(
            prediction=flux_distributions,
            ground_truth=measured_flux[sample_indices],
            target_area_mask=target_area_mask[sample_indices],
            reduction_dimensions=(index_mapping.batched_bitmap_e, index_mapping.batched_bitmap_u),
            device=device,
        )
        pixel_per_heliostat = core_utils.mean_loss_per_heliostat(
            loss_per_sample=pixel_per_sample,
            number_of_samples_per_heliostat=number_of_samples_per_heliostat,
            device=device,
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

    def _reconstruct_kinematic_parameters_with_raytracing(
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
                loss_per_heliostat_all.append(torch.tensor(float("inf"), device=device))
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

            while (
                loss > float(self.optimization_configuration[config_dictionary.tolerance])
                and epoch <= self.optimization_configuration[config_dictionary.max_epoch]
            ):
                optimizer.zero_grad()

                heliostat_group.activate_heliostats(
                    active_heliostats_mask=active_heliostats_mask, device=device
                )
                kinematic = heliostat_group.kinematic

                if self.train_position_deviation:
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
                    aim_points=self.scenario.target_areas.centers[target_area_mask],
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
                    if self.train_position_deviation:
                        entry["base_pos_dev_e_mean_abs"] = kinematic._base_position_deviation[:, 0].abs().mean().item()
                        entry["base_pos_dev_n_mean_abs"] = kinematic._base_position_deviation[:, 1].abs().mean().item()
                        entry["base_pos_dev_u_mean_abs"] = kinematic._base_position_deviation[:, 2].abs().mean().item()
                    eval_loss = self._compute_eval_loss_no_grad(heliostat_group, loss_definition, device)
                    if eval_loss is not None:
                        entry["eval_loss"] = eval_loss
                        log.info(f"Rank: {rank}, Epoch: {epoch}, Eval Loss: {eval_loss:.6f}")
                    self._convergence_history.append(entry)

                if early_stopper.step(loss):
                    log.info(f"Early stopping at epoch {epoch}.")
                    break

                epoch += 1

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
                    heliostat_group.kinematic.translation_deviation_parameters, src=src
                )
                torch.distributed.broadcast(
                    heliostat_group.kinematic.rotation_deviation_parameters, src=src
                )
                torch.distributed.broadcast(
                    heliostat_group.kinematic.actuators.optimizable_parameters, src=src
                )
                torch.distributed.broadcast(
                    heliostat_group.kinematic.actuators.non_optimizable_parameters, src=src
                )
                if self.train_position_deviation:
                    torch.distributed.broadcast(
                        heliostat_group.kinematic._base_position_deviation, src=src
                    )
            torch.distributed.all_reduce(
                final_loss_per_heliostat, op=torch.distributed.ReduceOp.MIN
            )
            log.info(f"Rank: {rank}, synchronized after kinematic reconstruction.")

        return final_loss_per_heliostat

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
        kinematic = heliostat_group.kinematic

        if self.train_position_deviation and hasattr(kinematic, "_base_position_deviation"):
            active_base_dev = kinematic._base_position_deviation.repeat_interleave(active_mask, dim=0)
            pad = torch.zeros(active_base_dev.shape[0], 1, device=device)
            kinematic.active_heliostat_positions = (
                kinematic.active_heliostat_positions + torch.cat([active_base_dev, pad], dim=1)
            )

        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=self.scenario.target_areas.centers[target_mask],
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
