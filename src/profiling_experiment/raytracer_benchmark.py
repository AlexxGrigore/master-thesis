"""Ray-tracer cost benchmark: VRAM + time vs surface resolution and field size.

Answers "how much does 25x25 -> 50x50 surface points cost?" directly, by running ONLY
the ray tracer (no training, no scenario creation) on an existing scenario.

For each (surface resolution) x (number of heliostats) x (mode) it measures peak GPU VRAM
and wall-time of a single ray-tracing pass:

  * mode "forward" — pure inference (torch.no_grad): the operational ray-trace cost.
  * mode "grad"    — forward + backward with the autograd graph retained through the
    surface points: the *training-like* memory (the graph is what fills VRAM in training).

Surface resolution is set at scenario-load time, so 50x50 = 4x the surface points of 25x25.
Out-of-memory at a given config is caught and recorded as "OOM" rather than crashing the
sweep — so the A40 ceiling shows up as data.

Inputs (sun direction, aim points) are fabricated: VRAM and time depend on tensor shapes,
not on the data values, so no calibration data is needed. ``--samples`` replicates each
heliostat (mask value = samples); training uses ~100, so multiply by samples for
training-scale numbers. Default 1 = the unit ray-trace.

Run (≈ minutes, see run_raytracer_benchmark.sh):
    apptainer exec --nv --bind /tudelft.net:/tudelft.net <sif> \
        python profiling_experiment/raytracer_benchmark.py --daic
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import h5py
import torch

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import paths  # noqa: E402
from profiling import Profiler, free_cuda  # noqa: E402

from artist.raytracing import HeliostatRayTracer  # noqa: E402
from artist.scenario import Scenario  # noqa: E402
from artist.util.env import get_device  # noqa: E402


def _is_oom(exc: Exception) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def _setup_trace(scenario, hg, n_hel: int, samples: int, device):
    """Activate ``n_hel`` heliostats (x samples), align them, and fabricate trace inputs."""
    mask = torch.zeros(hg.number_of_heliostats, dtype=torch.int64, device=device)
    mask[:n_hel] = samples  # mask value = replication count (samples) per heliostat
    n_active = int(mask.sum())
    target_indices = torch.zeros(n_active, dtype=torch.int64, device=device)
    # Fixed sun direction (value irrelevant to cost), one per active instance.
    rays = torch.tensor([0.0, -1.0, 0.0, 0.0], device=device).repeat(n_active, 1)
    aim = scenario.solar_tower.get_centers_of_target_areas(target_indices, device=device)

    hg.activate_heliostats(active_heliostats_mask=mask, device=device)
    hg.align_surfaces_with_incident_ray_directions(
        aim_points=aim, incident_ray_directions=rays,
        active_heliostats_mask=mask, device=device,
    )
    ray_tracer = HeliostatRayTracer(
        scenario=scenario, heliostat_group=hg, blocking_active=False,
        batch_size=max(n_active, 1),
    )
    return ray_tracer, mask, rays, target_indices


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ray-tracer VRAM/time vs surface resolution & field size.")
    p.add_argument("--resolutions", type=int, nargs="+", default=[25, 50],
                   help="Surface points per facet side to compare (default: 25 50).")
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 10, 30, 63],
                   help="Heliostat counts to activate (default: 1 10 30 63).")
    p.add_argument("--samples", type=int, default=1,
                   help="Replication per heliostat (mask value). Training uses ~100. Default 1.")
    p.add_argument("--modes", nargs="+", default=["forward", "grad"],
                   choices=["forward", "grad"])
    p.add_argument("--scenario", type=pathlib.Path, default=None,
                   help="Scenario .h5 (default: full_63_heli_kin_reconstruct/scenario.h5).")
    p.add_argument("--daic", action="store_true")
    p.add_argument("--results", type=pathlib.Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    scenario_path = args.scenario or (
        paths.REPO / "scenarios" / "full_63_heli_kin_reconstruct" / "scenario.h5"
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = args.results or (paths.output_dir() / f"raytracer_benchmark_{timestamp}.json")

    device = get_device()
    print(f"device={device}  scenario={scenario_path}")
    print(f"resolutions={args.resolutions}  sizes={args.sizes}  samples={args.samples}  modes={args.modes}")

    prof = Profiler()
    for res in args.resolutions:
        # Load the scenario once per resolution (surface points set here).
        with h5py.File(scenario_path, "r") as fh:
            scenario = Scenario.load_scenario_from_hdf5(
                scenario_file=fh, device=device,
                number_of_surface_points_per_facet=torch.tensor([res, res]),
            )
        hg = scenario.heliostat_field.heliostat_groups[0]
        max_hel = hg.number_of_heliostats

        for n_hel in args.sizes:
            if n_hel > max_hel:
                print(f"[res {res}x{res}] N={n_hel} > {max_hel} in scenario, skipping")
                continue
            for mode in args.modes:
                key = f"res{res}_N{n_hel}_{mode}"
                try:
                    ray_tracer, mask, rays, tidx = _setup_trace(
                        scenario, hg, n_hel, args.samples, device)
                    if mode == "grad":
                        hg.active_surface_points.requires_grad_(True)
                    with prof.measure(key):
                        if mode == "grad":
                            with torch.enable_grad():
                                flux, *_ = ray_tracer.trace_rays(
                                    incident_ray_directions=rays, active_heliostats_mask=mask,
                                    target_area_indices=tidx, device=device)
                                flux.sum().backward()
                        else:
                            with torch.no_grad():
                                ray_tracer.trace_rays(
                                    incident_ray_directions=rays, active_heliostats_mask=mask,
                                    target_area_indices=tidx, device=device)
                    prof.annotate(key, resolution=res, n_heliostats=n_hel,
                                  samples=args.samples, mode=mode)
                    rec = prof.records[key]
                    print(f"[{key}] {rec.get('seconds')}s  "
                          f"peak_vram={rec.get('peak_vram_alloc_gb')}GB")
                except Exception as exc:  # noqa: BLE001
                    status = "OOM" if _is_oom(exc) else f"ERROR: {type(exc).__name__}"
                    prof.annotate(key, resolution=res, n_heliostats=n_hel,
                                  samples=args.samples, mode=mode, status=status,
                                  detail=str(exc)[:200])
                    print(f"[{key}] {status}")
                finally:
                    # Reset grad state (re-activation overwrites active_surface_points
                    # next iteration, but reset defensively) and free GPU memory so each
                    # config's peak VRAM is measured independently.
                    if hg.active_surface_points is not None:
                        hg.active_surface_points.requires_grad_(False)
                    free_cuda()

        del scenario, hg
        free_cuda()

    prof.save(results_path, meta={
        "experiment": "raytracer_benchmark", "scenario": str(scenario_path),
        "resolutions": args.resolutions, "sizes": args.sizes,
        "samples": args.samples, "rays_per_point": "from scenario light source (10)",
    })
    print(f"\nSaved -> {results_path}")


if __name__ == "__main__":
    main()
