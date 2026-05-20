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
        kinematic = heliostat_group.kinematics
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

    Heliostats are matched by name, so this works correctly when the scenario is a subset
    of the heliostats that were present when the checkpoint was saved (e.g. 18 out of 376).
    """
    with open(checkpoint_path, "r") as f:
        all_params = json.load(f)

    # Build a flat name → params lookup across all checkpoint groups.
    name_to_params: dict[str, dict] = {}
    for saved in all_params.values():
        for i, name in enumerate(saved["heliostat_names"]):
            name_to_params[name] = {
                "translation": saved["translation_deviation_parameters"][i],
                "rotation":    saved["rotation_deviation_parameters"][i],
                "act_opt":     saved["actuator_optimizable_parameters"][i],
                "act_nonopt":  saved["actuator_nonoptimizable_parameters"][i],
                "base_pos":    saved.get("base_position_deviation_parameters", [None] * len(saved["heliostat_names"]))[i],
            }

    for heliostat_group in scenario.heliostat_field.heliostat_groups:
        kinematic = heliostat_group.kinematics

        missing = [n for n in heliostat_group.names if n not in name_to_params]
        if missing:
            raise KeyError(f"Checkpoint missing parameters for heliostats: {missing}")

        translation = torch.tensor(
            [name_to_params[n]["translation"] for n in heliostat_group.names],
            dtype=torch.float32, device=device,
        )
        rotation = torch.tensor(
            [name_to_params[n]["rotation"] for n in heliostat_group.names],
            dtype=torch.float32, device=device,
        )
        act_opt = torch.tensor(
            [name_to_params[n]["act_opt"] for n in heliostat_group.names],
            dtype=torch.float32, device=device,
        )
        act_nonopt = torch.tensor(
            [name_to_params[n]["act_nonopt"] for n in heliostat_group.names],
            dtype=torch.float32, device=device,
        )

        kinematic.translation_deviation_parameters.data = translation
        kinematic.rotation_deviation_parameters.data = rotation
        kinematic.actuators.optimizable_parameters.data = act_opt
        kinematic.actuators.non_optimizable_parameters.data = act_nonopt

        if hasattr(kinematic, "_base_position_deviation"):
            base_pos_rows = [name_to_params[n]["base_pos"] for n in heliostat_group.names]
            if base_pos_rows[0] is not None:
                kinematic._base_position_deviation.data = torch.tensor(
                    base_pos_rows, dtype=torch.float32, device=device,
                )
