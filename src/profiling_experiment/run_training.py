"""Phase B of the upscaling experiment: profile joint kinematics training vs field size.

For each N in ``--sizes`` (default 1 10 20 50) this loads the scenario built by
``create_scenarios.py`` (scenarios/profiling/N{n}/scenario.h5), then runs ARTIST's own
``KinematicsReconstructor`` + ``FocalSpotLoss`` over all N heliostats *jointly* in one
group — the parallel, batched ray-tracing path that actually exercises the GPU (NOT the
one-heliostat-at-a-time loop used elsewhere in the thesis).

Recorded per N:
  * total training wall-clock time and peak GPU VRAM (the binding resource),
  * the share of that time spent inside ``HeliostatRayTracer.trace_rays`` (forward
    ray tracing), via a runtime monkeypatch that leaves ARTIST source untouched.

A fixed ``--max-epoch`` (early stopping effectively off) keeps the work per heliostat
constant across N, so time/VRAM-vs-N is a clean scaling curve.

Optimizer / scheduler / loss configuration is copied verbatim from the canonical
reference (``canonical_artist_reconstruction/run_canonical.py``).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import h5py
import torch

_HERE = pathlib.Path(__file__).resolve().parent
_SRC = _HERE.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import paint.util.paint_mappings as paint_mappings  # noqa: E402
import paths  # noqa: E402  (sibling — dataset / output path resolution)
import selection  # noqa: E402
from profiling import Profiler, accumulate_raytrace_time, free_cuda  # noqa: E402

from artist.io import PaintCalibrationDataParser  # noqa: E402
from artist.optim import KinematicsReconstructor  # noqa: E402
from artist.optim.loss import FocalSpotLoss  # noqa: E402
from artist.raytracing import HeliostatRayTracer  # noqa: E402
from artist.scenario import Scenario  # noqa: E402
from artist.util import constants, set_logger_config  # noqa: E402
from artist.util.env import get_device, setup_distributed_environment  # noqa: E402

from utils.evaluation import build_heliostat_data_mapping  # noqa: E402


def _optimization_configuration(max_epoch: int, batch_size: int) -> dict:
    """Canonical ARTIST optimizer/scheduler config; early stopping effectively off."""
    optimizer_dict = {
        constants.initial_learning_rate_rotation_deviation: 1e-4,
        constants.initial_learning_rate_initial_angles: 1e-3,
        constants.initial_learning_rate_initial_stroke_length: 1e-2,
        constants.tolerance: 0.0,
        constants.max_epoch: max_epoch,
        constants.batch_size: batch_size,
        constants.log_step: 0,
        constants.early_stopping_delta: 1e-8,
        constants.early_stopping_patience: 10**6,
        constants.early_stopping_window: 10**6,
    }
    scheduler_dict = {
        constants.scheduler_type: constants.reduce_on_plateau,
        constants.gamma: 0.9, constants.lr_min: 1e-6, constants.lr_max: 1e-3,
        constants.step_size_up: 500, constants.reduce_factor: 0.0001,
        constants.patience: 50, constants.threshold: 1e-3, constants.cooldown: 10,
    }
    return {constants.optimization: optimizer_dict, constants.scheduler: scheduler_dict}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile ARTIST joint kinematics training vs field size.")
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 10, 20, 50])
    p.add_argument("--max-epoch", type=int, default=150,
                   help="Fixed epochs per run (keeps work-per-heliostat constant across N).")
    p.add_argument("--train-samples", type=int, default=10,
                   help="Uniform calibration samples per heliostat (PaintCalibrationDataParser cap).")
    p.add_argument("--surface-points", type=int, default=25, help="Surface points per facet side.")
    p.add_argument("--resolution", type=int, default=256, help="Flux bitmap side length.")
    p.add_argument("--dni", type=float, default=1000.0, help="Direct normal irradiance (ray magnitude).")
    p.add_argument("--batch-size", type=int, default=50, help="Ray-tracer dataloader batch size.")
    p.add_argument("--min-train-samples", type=int, default=10)
    p.add_argument("--daic", action="store_true")
    p.add_argument("--scenario-dir", type=pathlib.Path, default=None)
    p.add_argument("--results", type=pathlib.Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    set_logger_config()
    import logging
    logging.getLogger().setLevel(logging.WARNING)  # quiet ARTIST chatter

    scenario_dir = args.scenario_dir or paths.scenario_dir()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = args.results or (paths.output_dir() / f"training_{timestamp}.json")

    device = get_device()
    resolution = torch.tensor([args.resolution, args.resolution], device=device)
    print(f"device={device}  sizes={args.sizes}  scenario_dir={scenario_dir}")

    # Build the full train mapping once; filter to the chosen heliostats per N.
    train_map_all = build_heliostat_data_mapping(
        paths.benchmark_csv(args.daic),
        paths.calibration_dir(args.daic),
        paths.flux_dir(args.daic),
        "train",
    )

    prof = Profiler()
    for n in args.sizes:
        names = set(selection.select_names(n, daic=args.daic,
                                           min_train_samples=args.min_train_samples))
        scen_path = scenario_dir / f"N{n}" / "scenario.h5"
        phase = f"train_N{n}"

        if not scen_path.exists():
            print(f"[N={n}] scenario missing, run create_scenarios.py first: {scen_path}")
            prof.annotate(phase, n_heliostats=n, error="scenario_missing")
            continue

        train_map = [(h, c, f) for (h, c, f) in train_map_all if h in names]
        if len(train_map) != n:
            print(f"[N={n}] WARNING: train mapping has {len(train_map)} of {n} heliostats")

        n_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(scenario_path=scen_path)
        with setup_distributed_environment(number_of_heliostat_groups=n_groups, device=device) as ddp:
            dev = ddp[constants.device]
            with h5py.File(scen_path, "r") as fh:
                scenario = Scenario.load_scenario_from_hdf5(
                    scenario_file=fh, device=dev,
                    number_of_surface_points_per_facet=torch.tensor(
                        [args.surface_points, args.surface_points]),
                )

            recon_parser = PaintCalibrationDataParser(
                sample_limit=args.train_samples,
                centroid_extraction_method=paint_mappings.UTIS_KEY,
            )
            data = {
                constants.data_parser: recon_parser,
                constants.heliostat_data_mapping: train_map,
            }
            reconstructor = KinematicsReconstructor(
                ddp_setup=ddp, scenario=scenario, data=data, dni=args.dni,
                optimization_configuration=_optimization_configuration(
                    args.max_epoch, args.batch_size),
                reconstruction_method=constants.kinematics_reconstruction_raytracing,
                bitmap_resolution=resolution,
            )
            loss_def = FocalSpotLoss(scenario=scenario)

            print(f"[N={n}] training {len(train_map)} heliostats jointly, "
                  f"{args.max_epoch} epochs ...")
            with prof.measure(phase), accumulate_raytrace_time(HeliostatRayTracer) as rt:
                reconstructor.reconstruct_kinematics(loss_definition=loss_def, device=dev)

        rec = prof.records[phase]
        rt_share = (rt["seconds"] / rec["seconds"] * 100) if rec.get("seconds") else 0.0
        prof.annotate(
            phase,
            n_heliostats=n,
            n_trained=len(train_map),
            max_epoch=args.max_epoch,
            raytrace_seconds=round(rt["seconds"], 4),
            raytrace_calls=rt["calls"],
            raytrace_share_pct=round(rt_share, 1),
        )
        print(f"[N={n}] total={rec.get('seconds')}s  "
              f"raytrace={round(rt['seconds'], 1)}s ({round(rt_share, 1)}%)  "
              f"peak_vram={rec.get('peak_vram_alloc_gb')}GB")

        # Drop every GPU-resident object so the next N starts from a clean device:
        # otherwise this N's scenario / reconstructor / optimizer state would still
        # be live and inflate the next N's peak-VRAM reading (and risk an OOM).
        del scenario, reconstructor, loss_def, data, recon_parser, train_map
        free_cuda()

    prof.save(results_path, meta={
        "phase": "training", "sizes": args.sizes, "max_epoch": args.max_epoch,
        "train_samples": args.train_samples, "surface_points": args.surface_points,
        "resolution": args.resolution, "batch_size": args.batch_size,
        "reconstructor": "ARTIST KinematicsReconstructor", "loss": "FocalSpotLoss",
    })
    print(f"\nSaved training profile -> {results_path}")


if __name__ == "__main__":
    main()
