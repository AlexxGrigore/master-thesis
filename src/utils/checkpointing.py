import json
import pathlib

import torch
from artist.scenario.scenario import Scenario


def save_kinematic_parameters(scenario: Scenario, output_path: pathlib.Path) -> None:
    """Save all optimized kinematic parameters for every heliostat group to a JSON file.

    Saves the full set of parameters that WortbergKinematicReconstructor optimizes:
      - translation_deviation_parameters  [N, 9]  — joint/concentrator position offsets
      - rotation_deviation_parameters     [N, 4]  — tilt deviations
      - actuator_optimizable_parameters   [N, 2, 2] — initial angles (a_i) and stroke lengths
      - actuator_nonoptimizable_parameters[N, 7, 2] — includes actuator offset c_i at index 5
      - base_position_deviation_parameters[N, 3]  — only present when train_position_deviation=True

    The file can be reloaded with load_kinematic_parameters() to restore the exact state.
    """
    all_params = {}
    for group_index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
        kinematic = heliostat_group.kinematic
        group_entry = {
            "heliostat_names": heliostat_group.names,
            "translation_deviation_parameters": kinematic.translation_deviation_parameters.detach().cpu().tolist(),
            "rotation_deviation_parameters": kinematic.rotation_deviation_parameters.detach().cpu().tolist(),
            "actuator_optimizable_parameters": kinematic.actuators.optimizable_parameters.detach().cpu().tolist(),
            "actuator_nonoptimizable_parameters": kinematic.actuators.non_optimizable_parameters.detach().cpu().tolist(),
        }
        if hasattr(kinematic, "_base_position_deviation"):
            group_entry["base_position_deviation_parameters"] = kinematic._base_position_deviation.detach().cpu().tolist()
        all_params[f"group_{group_index}"] = group_entry

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_params, f, indent=2)


def load_kinematic_parameters(scenario: Scenario, checkpoint_path: pathlib.Path, device: torch.device) -> None:
    """Load kinematic parameters from a JSON checkpoint and apply them to the scenario in-place.

    Restores all four parameter tensors saved by save_kinematic_parameters().
    Groups in the checkpoint are matched to scenario groups by their index (group_0, group_1, …).
    """
    with open(checkpoint_path, "r") as f:
        all_params = json.load(f)

    for group_index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
        key = f"group_{group_index}"
        if key not in all_params:
            raise KeyError(f"Checkpoint missing '{key}' — does it match this scenario?")
        saved = all_params[key]
        kinematic = heliostat_group.kinematic

        kinematic.translation_deviation_parameters.data = torch.tensor(
            saved["translation_deviation_parameters"], dtype=torch.float32, device=device
        )
        kinematic.rotation_deviation_parameters.data = torch.tensor(
            saved["rotation_deviation_parameters"], dtype=torch.float32, device=device
        )
        kinematic.actuators.optimizable_parameters.data = torch.tensor(
            saved["actuator_optimizable_parameters"], dtype=torch.float32, device=device
        )
        kinematic.actuators.non_optimizable_parameters.data = torch.tensor(
            saved["actuator_nonoptimizable_parameters"], dtype=torch.float32, device=device
        )
        if "base_position_deviation_parameters" in saved and hasattr(kinematic, "_base_position_deviation"):
            kinematic._base_position_deviation.data = torch.tensor(
                saved["base_position_deviation_parameters"], dtype=torch.float32, device=device
            )
