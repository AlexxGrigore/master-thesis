"""Gate for the blocking study: is there any blocking to see, before anything is trained?

For every calibration sample of one heliostat and every neighbour-state hypothesis, pose
the blockers, orient the trained heliostat from its RECORDED motor positions, ray trace
once, and read the `blocking_factor` that ARTIST already returns (fraction of rays that
reach the target unblocked).

If no hypothesis loses a meaningful fraction of rays on any sample, this heliostat has no
observable blocking and there is nothing for a blocking-aware Stage 2 to learn. That is a
result, and it costs one forward pass per sample instead of a training run.

Two structural notes, both verified in DESIGN.md section 4:

- `MINI_BATCH_SIZE` must be 1. ARTIST's blocking code assumes one active instance per
  heliostat row, while this pipeline encodes N samples as N repeats of one row.
- The blocker poses are injected through the two public attributes that `trace_rays`
  actually reads, so no ARTIST method is overridden or copied.

Usage
-----
    python gate.py BE25 --max-samples 30
    python gate.py BE25 AA27 --daic
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

import h5py
import torch

_here = pathlib.Path(__file__).resolve().parent
_src = _here.parents[1]
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_here))

from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer  # noqa: E402
from artist.scenario.scenario import Scenario  # noqa: E402
from artist.util import set_logger_config  # noqa: E402

from brute_blocking import exact_blocking  # noqa: E402

log = logging.getLogger(__name__)

# Neighbour-state hypotheses. "stow" is the unrotated pose, which for the PAINT facet
# layout is the horizontal mirror (verified: surface points lie in the e-n plane at
# u ~ 0.04 m). The rest aim the blockers at one of the four PAINT targets.
HYPOTHESES = ("stow", "upper", "lower", "mft", "receiver")

_TARGET_OF = {
    "upper": "solar_tower_juelich_upper",
    "lower": "solar_tower_juelich_lower",
    "mft": "multi_focus_tower",
    "receiver": "receiver",
}

_DAIC_ROOT = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
_DAIC_PAINT = pathlib.Path(
    "/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint"
)
BENCHMARK = "benchmark_split-balanced_train-50_validation-20"


def _paths(daic: bool) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Return (paint_dir, scenario_root, output_root)."""
    if daic:
        return (
            _DAIC_PAINT,
            _DAIC_ROOT / "scenarios" / "neighbourhoods",
            _DAIC_ROOT / "outputs" / "new_mapping_function" / "blocking_study",
        )
    root = _here.parents[2]
    return (
        root / "datasets" / "paint",
        root / "scenarios" / "neighbourhoods",
        root / "outputs" / "new_mapping_function" / "blocking_study",
    )


def one_hot_mask(index: int, count: int, size: int, device: torch.device) -> torch.Tensor:
    """Active mask of length `size` holding `count` at `index`, zero elsewhere."""
    mask = torch.zeros(size, dtype=torch.int32, device=device)
    mask[index] = count
    return mask


def blocker_surfaces(
    heliostat_group,
    hypothesis: str,
    scenario,
    incident_ray_direction: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    World-coordinate surface points of every heliostat in the group under one hypothesis.

    Parameters
    ----------
    heliostat_group : HeliostatGroup
        The group holding the trained heliostat and its blockers.
    hypothesis : str
        One of `HYPOTHESES`.
    scenario : Scenario
        The scenario, used to look up target area centres.
    incident_ray_direction : torch.Tensor
        The sun direction for this sample, shared by every heliostat.
        Shape is ``[4]``.
    device : torch.device
        The device on which to perform computations.

    Returns
    -------
    torch.Tensor
        Surface points in world coordinates.
        Shape is ``[number_of_heliostats, number_of_surface_points, 4]``.
    """
    number_of_heliostats = heliostat_group.number_of_heliostats
    if hypothesis == "stow":
        # Unrotated mirror translated to its position: the horizontal pose, which is also
        # ARTIST's own fallback for heliostats that were never aligned.
        return (
            heliostat_group.surface_points
            + heliostat_group.positions.unsqueeze(1)
        ).detach()

    target_index = scenario.solar_tower.target_name_to_index[_TARGET_OF[hypothesis]]
    aim_point = scenario.solar_tower.get_centers_of_target_areas(
        target_area_indices=torch.tensor([target_index], device=device), device=device
    )[0]

    mask = torch.ones(number_of_heliostats, dtype=torch.int32, device=device)
    with torch.no_grad():
        heliostat_group.activate_heliostats(active_heliostats_mask=mask, device=device)
        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=aim_point.expand(number_of_heliostats, -1),
            incident_ray_directions=incident_ray_direction.expand(
                number_of_heliostats, -1
            ),
            active_heliostats_mask=mask,
            device=device,
        )
        surfaces = heliostat_group.active_surface_points.detach().clone()
    return surfaces


def trace_one_sample(
    scenario,
    heliostat_group,
    heliostat_index: int,
    motor_position: torch.Tensor,
    incident_ray_direction: torch.Tensor,
    target_area_index: torch.Tensor,
    blockers: torch.Tensor | None,
    device: torch.device,
    centroid: torch.Tensor | None = None,
):
    """
    Ray trace a single calibration sample, optionally with blocking.

    `blockers` of None disables blocking entirely (the current baseline). Otherwise it is
    the ``[number_of_heliostats, number_of_surface_points, 4]`` tensor from
    `blocker_surfaces`, injected through the two public attributes `trace_rays` reads.

    When `centroid` is given the trained heliostat is aimed at the MEASURED focal spot
    instead of being oriented from its recorded motor positions. That is the right pose
    for a pure geometry screen: with nominal kinematics the field median pointing error is
    around 40 mrad, which at 200 m throws the beam ~8 m off a target a few metres wide, so
    the beam never reaches the target and no ray can ever be reported as blocked. Aiming
    at the measured spot puts the beam where the measurement says it actually went.
    """
    mask = one_hot_mask(
        heliostat_index, 1, heliostat_group.number_of_heliostats, device
    )
    heliostat_group.activate_heliostats(active_heliostats_mask=mask, device=device)
    if centroid is None:
        heliostat_group.align_surfaces_with_motor_positions(
            motor_positions=motor_position.unsqueeze(0),
            active_heliostats_mask=mask,
            device=device,
        )
    else:
        heliostat_group.align_surfaces_with_incident_ray_directions(
            aim_points=centroid.unsqueeze(0),
            incident_ray_directions=incident_ray_direction.unsqueeze(0),
            active_heliostats_mask=mask,
            device=device,
        )

    ray_tracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=heliostat_group,
        blocking_active=False,
        world_size=1,
        rank=0,
        batch_size=1,
        random_seed=7,
    )
    if blockers is not None:
        ray_tracer.blocking_active = True
        ray_tracer.blocking_heliostat_surfaces_active = blockers

    # ARTIST's LBVH blocking filter silently under-reports (see brute_blocking.py), so
    # every blocked trace goes through the exact filter instead.
    with exact_blocking():
        flux, intercept_factor, on_target_factor, blocking_factor = ray_tracer.trace_rays(
            incident_ray_directions=incident_ray_direction.unsqueeze(0),
            active_heliostats_mask=mask,
            target_area_indices=target_area_index.unsqueeze(0),
            device=device,
        )
    return flux, intercept_factor, on_target_factor, blocking_factor


def run_gate(
    heliostat_id: str,
    paint_dir: pathlib.Path,
    scenario_root: pathlib.Path,
    output_root: pathlib.Path,
    max_samples: int,
    surface_points_per_facet: int,
    number_of_rays: int,
    device: torch.device,
    pose: str = "centroid",
) -> dict:
    """Run the gate for one heliostat and return the summary dictionary."""
    from artist.io.paint_calibration_parser import PaintCalibrationDataParser

    from utils.evaluation import build_heliostat_data_mapping

    scenario_dir = scenario_root / heliostat_id
    scenario_path = scenario_dir / "scenario.h5"
    if not scenario_path.exists():
        scenario_path = scenario_dir / "scenario_ideal.h5"
    if not scenario_path.exists():
        raise FileNotFoundError(
            f"No neighbourhood scenario for {heliostat_id}. "
            f"Run build_neighbourhood_scenario.py {heliostat_id} first."
        )

    with h5py.File(scenario_path) as file_handle:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=file_handle,
            device=device,
            number_of_surface_points_per_facet=torch.tensor(
                [surface_points_per_facet, surface_points_per_facet]
            ),
        )
    scenario.set_number_of_rays(number_of_rays)
    heliostat_group = scenario.heliostat_field.heliostat_groups[0]
    heliostat_index = heliostat_group.names.index(heliostat_id)
    log.info(
        f"{heliostat_id}: scenario {scenario_path.name}, group members "
        f"{heliostat_group.names}, row index {heliostat_index}"
    )

    # Real PAINT calibration samples of the 50-20-20 benchmark, test split.
    mapping = build_heliostat_data_mapping(
        paint_dir / "splits" / f"{BENCHMARK}.csv",
        paint_dir / BENCHMARK / "calibration_properties",
        paint_dir / BENCHMARK / "flux_image",
        "test",
    )
    mapping = [entry for entry in mapping if entry[0] == heliostat_id]
    if not mapping:
        raise ValueError(f"{heliostat_id} is not in the {BENCHMARK} test split")

    _, centroids, rays, motor_positions, _, target_mask = PaintCalibrationDataParser(
        centroid_extraction_method="UTIS"
    ).parse_data_for_reconstruction(
        heliostat_data_mapping=mapping,
        heliostat_group=heliostat_group,
        scenario=scenario,
        device=device,
    )
    number_of_samples = min(max_samples, rays.shape[0])
    log.info(f"  {rays.shape[0]} test samples, using {number_of_samples}")

    results: dict[str, list[float]] = {}
    for hypothesis in HYPOTHESES:
        losses = []
        for sample in range(number_of_samples):
            surfaces = blocker_surfaces(
                heliostat_group,
                hypothesis,
                scenario,
                rays[sample],
                device,
            )
            _, _, on_target, blocking_factor = trace_one_sample(
                scenario,
                heliostat_group,
                heliostat_index,
                motor_positions[sample],
                rays[sample],
                target_mask[sample],
                surfaces,
                device,
                centroid=centroids[sample] if pose == "centroid" else None,
            )
            if on_target.item() < 1e-6:
                # The beam never reached the target, so "nothing was blocked" is
                # meaningless rather than informative. Do not let it count as evidence.
                raise RuntimeError(
                    f"{heliostat_id} sample {sample}: beam misses the target entirely "
                    f"(on_target=0). A blocking screen on a beam that never arrives is "
                    f"vacuous. Use --pose centroid, or gate after Stage 1."
                )
            losses.append(float(1.0 - blocking_factor.item()))
        blocked = torch.tensor(losses)
        results[hypothesis] = losses
        log.info(
            f"  {hypothesis:9s} rays blocked: "
            f"mean {blocked.mean() * 100:6.2f}%  median {blocked.median() * 100:6.2f}%  "
            f"max {blocked.max() * 100:6.2f}%  "
            f"samples above 1%: {int((blocked > 0.01).sum())}/{number_of_samples}"
        )

    summary = {
        "heliostat_id": heliostat_id,
        "scenario": str(scenario_path),
        "group_members": heliostat_group.names,
        "heliostat_index": heliostat_index,
        "number_of_samples": number_of_samples,
        "surface_points_per_facet": surface_points_per_facet,
        "number_of_rays": number_of_rays,
        "pose": pose,
        "blocked_fraction": results,
        "max_blocked_over_hypotheses": max(max(v) for v in results.values()),
    }
    out_dir = output_root / heliostat_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate.json").write_text(json.dumps(summary, indent=2))
    log.info(f"  written {out_dir / 'gate.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Blocking gate.")
    parser.add_argument("heliostat_ids", nargs="+")
    parser.add_argument("--daic", action="store_true")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--surface-points", type=int, default=25)
    parser.add_argument("--rays", type=int, default=10)
    parser.add_argument(
        "--pose",
        choices=("centroid", "motors"),
        default="centroid",
        help="How to orient the trained heliostat: aimed at the measured focal spot "
        "(default, pure geometry) or from its recorded motor positions (needs a "
        "calibrated model, otherwise the beam misses the target).",
    )
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("artist", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    torch.manual_seed(7)

    paint_dir, scenario_root, output_root = _paths(args.daic)
    device = torch.device("cpu")

    verdicts = []
    for hid in args.heliostat_ids:
        summary = run_gate(
            heliostat_id=hid,
            paint_dir=paint_dir,
            scenario_root=scenario_root,
            output_root=output_root,
            max_samples=args.max_samples,
            surface_points_per_facet=args.surface_points,
            number_of_rays=args.rays,
            device=device,
            pose=args.pose,
        )
        verdicts.append((hid, summary["max_blocked_over_hypotheses"]))

    print()
    print("=" * 62)
    for hid, worst in verdicts:
        state = "BLOCKING PRESENT" if worst > 0.01 else "no observable blocking"
        print(f"  {hid:8s} max blocked over hypotheses {worst * 100:6.2f}%   {state}")
    print("=" * 62)


if __name__ == "__main__":
    main()
