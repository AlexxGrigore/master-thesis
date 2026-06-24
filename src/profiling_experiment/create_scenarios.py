"""Phase A of the upscaling experiment: build one scenario per field size and time it.

For each N in ``--sizes`` (default 1 10 20 50) this fits NURBS surfaces from
deflectometry for the first N eligible heliostats and writes a scenario:

    scenarios/profiling/N{n}/scenario.h5

recording, per N, the wall-clock time and peak GPU VRAM of scenario creation. The
exact same N heliostats are later trained by ``run_training.py`` (both read the order
from ``selection.py``), so these scenarios feed directly into Phase B.

The NURBS-fit configuration (20x20 control points, 400 epochs, fit-from-normals, etc.)
is reused verbatim from ``create_all_scenarios.build_deflectometry_scenario`` so the
per-heliostat cost matches the thesis's real scenarios.

Run on DAIC (see run_profiling.sbatch):
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

import create_all_scenarios as cas  # noqa: E402
import selection  # noqa: E402  (profiling_experiment/selection.py)
from profiling import Profiler, free_cuda  # noqa: E402  (profiling_experiment/profiling.py)

from artist.data_parser import paint_scenario_parser  # noqa: E402
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator  # noqa: E402
from artist.util import config_dictionary, set_logger_config  # noqa: E402
from artist.util.environment_setup import get_device  # noqa: E402

set_logger_config()


def build_subset_scenario(
    fitting_list: list[selection.FittingEntry],
    paint_dir: pathlib.Path,
    output: pathlib.Path,
    device: torch.device,
) -> None:
    """Fit NURBS surfaces for the given heliostats and write a scenario .h5."""
    tower_file = paint_dir / "WRI1030197-tower-measurements.json"
    power_plant_config, target_area_list_config = (
        paint_scenario_parser.extract_paint_tower_measurements(
            tower_measurements_path=tower_file, device=device
        )
    )

    optimizer, scheduler = cas._make_nurbs_optimizer_scheduler()
    heliostat_list_config, prototype_config = (
        paint_scenario_parser.extract_paint_heliostats_fitted_surface(
            paths=fitting_list,
            power_plant_position=power_plant_config.power_plant_position,
            number_of_nurbs_control_points=torch.tensor([20, 20], device=device),
            deflectometry_step_size=100,
            nurbs_fit_method=config_dictionary.fit_nurbs_from_normals,
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
        target_area_list_config=target_area_list_config,
        light_source_list_config=cas._make_light_source_list_config(),
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    ).generate_scenario()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile ARTIST scenario creation vs field size.")
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 10, 20, 50],
                   help="Field sizes (number of heliostats) to build (default: 1 10 20 50).")
    p.add_argument("--min-train-samples", type=int, default=10,
                   help="Min train-split calibration samples a heliostat must have to be eligible.")
    p.add_argument("--daic", action="store_true", help="Use DAIC dataset paths.")
    p.add_argument("--force", action="store_true", help="Rebuild scenarios that already exist.")
    p.add_argument("--scenario-dir", type=pathlib.Path, default=None,
                   help="Where to write scenarios (default: <repo>/scenarios/profiling).")
    p.add_argument("--results", type=pathlib.Path, default=None,
                   help="Where to write the creation timing JSON.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    base_dir = cas.DAIC_BASE_DIR if args.daic else cas.LOCAL_BASE_DIR
    paint_dir = cas.DAIC_PAINT_DIR if args.daic else cas.LOCAL_PAINT_DIR

    scenario_dir = args.scenario_dir or (base_dir / "scenarios" / "profiling")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = args.results or (
        base_dir / "outputs" / "new_mapping_function" / "profiling_experiment"
        / f"creation_{timestamp}.json"
    )

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
            build_subset_scenario(fitting_list, paint_dir, output, device)
        prof.annotate(
            phase,
            n_heliostats=n,
            scenario_path=str(output),
            heliostats=[e[0] for e in fitting_list],
        )
        rec = prof.records[phase]
        print(f"[N={n}] created in {rec.get('seconds')}s  "
              f"peak_vram={rec.get('peak_vram_alloc_gb')}GB")

        # Free the NURBS-fit tensors/optimizer before the next N so each field
        # size's peak VRAM is measured independently.
        del fitting_list
        free_cuda()

    prof.save(results_path, meta={"phase": "scenario_creation", "sizes": args.sizes,
                                  "min_train_samples": args.min_train_samples})
    print(f"\nSaved creation profile -> {results_path}")


if __name__ == "__main__":
    main()
