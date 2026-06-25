"""Phase A of the upscaling experiment: build one scenario per field size and time it.

For each N in ``--sizes`` (default 1 10 20 50) this fits NURBS surfaces from
deflectometry for the first N eligible heliostats and writes a scenario:

    scenarios/profiling/N{n}/scenario.h5

recording, per N, the wall-clock time and peak GPU VRAM of scenario creation. The exact
same N heliostats are later trained by ``run_training.py`` (both read the order from
``selection.py``), so these scenarios feed directly into Phase B.

Written against the CURRENT ARTIST API in ``artist-local.sif`` — the ``artist.io`` /
``constants`` / ``artist.util.env`` layout (after the "rename data_parser subpackage to
io" refactor). The repo's ``create_all_scenarios.py`` is stale: it still imports the
pre-refactor ``data_parser`` / ``config_dictionary`` / ``environment_setup`` names, so we
deliberately do not depend on it here.

Run on DAIC (see run_profiling_experiment.sh):
    cd .../master-thesis/src
    apptainer exec --nv --bind /tudelft.net:/tudelft.net <sif> \
        python profiling_experiment/create_scenarios.py --daic
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

_SRC = pathlib.Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import paths  # noqa: E402
import selection  # noqa: E402
from profiling import Profiler, free_cuda  # noqa: E402

from artist.io import paint_scenario_parser  # noqa: E402
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator  # noqa: E402
from artist.util import constants, set_logger_config  # noqa: E402
from artist.util.config import LightSourceConfig, LightSourceListConfig  # noqa: E402
from artist.util.env import get_device  # noqa: E402

set_logger_config()


def _make_nurbs_optimizer_scheduler() -> tuple:
    """Fresh (optimizer, scheduler) for NURBS fitting (matches the thesis defaults)."""
    optimizer = torch.optim.Adam([torch.empty(1, requires_grad=True)], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=50,
        threshold=1e-7, threshold_mode="abs",
    )
    return optimizer, scheduler


def _make_light_source_list_config() -> LightSourceListConfig:
    return LightSourceListConfig(
        light_source_list=[
            LightSourceConfig(
                light_source_key="sun_1",
                light_source_type=constants.sun_key,
                number_of_rays=10,
                distribution_type=constants.light_source_distribution_is_normal,
                mean=0.0,
                covariance=4.3681e-06,
            )
        ]
    )


def build_subset_scenario(
    fitting_list: list[selection.FittingEntry],
    tower_file: pathlib.Path,
    output: pathlib.Path,
    device: torch.device,
) -> None:
    """Fit NURBS surfaces for the given heliostats and write a scenario .h5."""
    # Old API: tower extraction returns THREE configs (planar + cylindrical split out).
    power_plant_config, target_area_planar, target_area_cylindrical = (
        paint_scenario_parser.extract_paint_tower_measurements(
            tower_measurements_path=tower_file, device=device
        )
    )

    optimizer, scheduler = _make_nurbs_optimizer_scheduler()
    heliostat_list_config, prototype_config = (
        paint_scenario_parser.extract_paint_heliostats_fitted_surface(
            paths=fitting_list,
            power_plant_position=power_plant_config.power_plant_position,
            number_of_nurbs_control_points=torch.tensor([20, 20], device=device),
            deflectometry_step_size=100,
            nurbs_fit_method=constants.fit_nurbs_from_normals,
            nurbs_fit_tolerance=1e-10,
            nurbs_fit_max_epoch=400,
            nurbs_fit_optimizer=optimizer,
            nurbs_fit_scheduler=scheduler,
            device=device,
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    H5ScenarioGenerator(
        file_path=output,
        power_plant_config=power_plant_config,
        target_area_list_planar_config=target_area_planar,
        target_area_list_cylindrical_config=target_area_cylindrical,
        light_source_list_config=_make_light_source_list_config(),
        heliostat_list_config=heliostat_list_config,
        prototype_config=prototype_config,
    ).generate_scenario()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile ARTIST scenario creation vs field size.")
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 10, 20, 50])
    p.add_argument("--min-train-samples", type=int, default=10)
    p.add_argument("--daic", action="store_true", help="Use DAIC (umbrella) dataset paths.")
    p.add_argument("--force", action="store_true", help="Rebuild scenarios that already exist.")
    p.add_argument("--results", type=pathlib.Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    scenario_dir = paths.scenario_dir()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = args.results or (paths.output_dir() / f"creation_{timestamp}.json")
    tower_file = paths.tower_file(args.daic)

    device = get_device()
    print(f"device={device}  sizes={args.sizes}  scenario_dir={scenario_dir}")

    prof = Profiler()
    for n in args.sizes:
        fitting_list = selection.select_fitting_list(
            n, daic=args.daic, min_train_samples=args.min_train_samples
        )
        output = scenario_dir / f"N{n}" / "scenario.h5"
        phase = f"create_N{n}"

        if output.exists() and not args.force:
            print(f"[N={n}] scenario exists, skipping build: {output}")
            prof.annotate(phase, n_heliostats=n, scenario_path=str(output), skipped=True)
            continue

        print(f"[N={n}] fitting {n} heliostats -> {output}")
        with prof.measure(phase):
            build_subset_scenario(fitting_list, tower_file, output, device)
        prof.annotate(
            phase, n_heliostats=n, scenario_path=str(output),
            heliostats=[e[0] for e in fitting_list],
        )
        rec = prof.records[phase]
        print(f"[N={n}] created in {rec.get('seconds')}s  "
              f"peak_vram={rec.get('peak_vram_alloc_gb')}GB")

        # Free NURBS-fit tensors before the next N so each VRAM figure is independent.
        del fitting_list
        free_cuda()

    prof.save(results_path, meta={"phase": "scenario_creation", "sizes": args.sizes,
                                  "min_train_samples": args.min_train_samples})
    print(f"\nSaved creation profile -> {results_path}")


if __name__ == "__main__":
    main()
