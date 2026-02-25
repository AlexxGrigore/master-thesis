"""
Custom KinematicReconstructor subclasses.

Each class in this module represents a distinct experiment configuration that
overrides the default ARTIST parameter selection and/or deviation bounds.
Adding a new experiment means adding a new subclass here — the training scripts
stay thin and only deal with data loading and configuration.
"""

import logging
from collections import deque
from typing import Deque

import torch

from artist.core import core_utils, learning_rate_schedulers
from artist.core.heliostat_ray_tracer import HeliostatRayTracer
from artist.core.kinematic_reconstructor import KinematicReconstructor
from artist.util import config_dictionary, index_mapping
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)


class EarlyStopping:
    """Local EarlyStopping for compatibility with older ARTIST installs."""

    def __init__(self, window_size=10, patience=20, min_improvement=1e-4, relative=True, eps=1e-8):
        self.window_size = window_size
        self.patience = patience
        self.min_improvement = min_improvement
        self.relative = relative
        self.eps = eps
        self.loss_history: Deque[float] = deque(maxlen=window_size)
        self.counter = 0

    def step(self, loss: float) -> bool:
        self.loss_history.append(loss)
        if len(self.loss_history) < self.window_size:
            return False
        improvement = self.loss_history[0] - self.loss_history[-1]
        if self.relative:
            improvement /= max(abs(self.loss_history[0]), self.eps)
        if improvement > self.min_improvement:
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class WortbergKinematicReconstructor(KinematicReconstructor):
    """
    KinematicReconstructor following the parameter setup of Wortberg (2025) Table 5.3.

    Compared to the default ARTIST reconstructor, this variant:

    Adds to the optimised set
    -------------------------
    - ``translation_deviation_parameters`` (all 9: joints + concentrator), ±0.05 m
    - ``actuators.non_optimizable_parameters[:, actuator_offset]`` (c_i), ±0.005 m

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
    - Heliostat base position deviations (3 translations)
    - Concentrator tilts (2 rotations)
    """

    # Deviation bounds from Wortberg (2025) Table 5.3.
    _BOUND_TRANSLATION_M = 0.05       # joint and concentrator translations
    _BOUND_ROTATION_RAD = 0.005       # joint tilts
    _BOUND_ACTUATOR_ANGLE_RAD = 0.005  # a_i — offset radius shift
    _BOUND_ACTUATOR_OFFSET_M = 0.005   # c_i — joint's distance offset

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

        for heliostat_group_index in self.ddp_setup[config_dictionary.groups_to_ranks_mapping][rank]:
            heliostat_group = self.scenario.heliostat_field.heliostat_groups[heliostat_group_index]
            parser = self.data[config_dictionary.data_parser]
            heliostat_mapping = self.data[config_dictionary.heliostat_data_mapping]

            (
                _,
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

                    loss_per_sample = loss_definition(
                        prediction=flux_distributions,
                        ground_truth=focal_spots_measured[sample_indices_for_local_rank],
                        target_area_mask=target_area_mask[sample_indices_for_local_rank],
                        reduction_dimensions=(index_mapping.focal_spots,),
                        device=device,
                    )

                    number_of_samples_per_heliostat = int(
                        heliostat_group.active_heliostats_mask.sum()
                        / (heliostat_group.active_heliostats_mask > 0).sum()
                    )

                    loss_per_heliostat = core_utils.mean_loss_per_heliostat(
                        loss_per_sample=loss_per_sample,
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
                        log.info(
                            f"Rank: {rank}, Epoch: {epoch}, Loss: {loss:.6f}, "
                            f"LR: {optimizer.param_groups[index_mapping.optimizer_param_group_0]['lr']}"
                        )

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

        # Snapshot non-zero nominal values for bound clamping.
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

        optimizer = torch.optim.Adam(
            [
                kinematic.translation_deviation_parameters,
                kinematic.rotation_deviation_parameters,
                kinematic.actuators.optimizable_parameters,
                kinematic.actuators.non_optimizable_parameters,
            ],
            lr=float(self.optimization_configuration[config_dictionary.initial_learning_rate]),
        )

        return optimizer, initial_actuator_initial_angle, initial_actuator_offset

    def _setup_scheduler(self, optimizer):
        """Build and return the learning rate scheduler."""
        scheduler_fn = getattr(
            learning_rate_schedulers,
            self.optimization_configuration[config_dictionary.scheduler],
        )
        return scheduler_fn(
            optimizer=optimizer,
            parameters=self.optimization_configuration[config_dictionary.scheduler_parameters],
        )

    def _setup_early_stopper(self):
        """Build and return the early stopping instance."""
        return EarlyStopping(
            window_size=self.optimization_configuration[config_dictionary.early_stopping_window],
            patience=self.optimization_configuration[config_dictionary.early_stopping_patience],
            min_improvement=self.optimization_configuration[config_dictionary.early_stopping_delta],
            relative=True,
        )

    def _apply_deviation_bounds(self, heliostat_group, initial_actuator_initial_angle, initial_actuator_offset):
        """
        Clamp all optimised parameters to their Table 5.3 deviation bounds.

        translation_deviation and rotation_deviation start at zero, so the absolute
        value equals the deviation.  Actuator parameters start from a non-zero nominal
        loaded from the scenario, so the clamp is computed relative to that snapshot.
        """
        kinematic = heliostat_group.kinematic
        with torch.no_grad():
            kinematic.translation_deviation_parameters.data.clamp_(
                -self._BOUND_TRANSLATION_M, self._BOUND_TRANSLATION_M
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
