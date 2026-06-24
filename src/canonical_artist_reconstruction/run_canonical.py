"""
Canonical ARTIST kinematics reconstruction — reference baseline.

This subproject runs ARTIST's OWN ``KinematicsReconstructor`` exactly as the official
tutorial (ARTIST/tutorials/04_kinematics_reconstruction_tutorial.py) prescribes, with NO
project-specific modifications:

  * single stage (no Stage-1 alignment loss),
  * loss = FocalSpotLoss (ray-traced focal-spot centroid),
  * heliostat aligned by aiming at the TARGET CENTRE (get_centers_of_target_areas),
  * optimizes rotation_deviation, initial_angles (a_i), initial_stroke_length (b_i).

It is deliberately separate from src/one_heliostat_demo (which uses a custom two-stage,
centre-free formulation). Purpose: a trustworthy reference number to compare against.

Runs on a handful of real PAINT heliostats, each with its own one-heliostat scenario.
Reports focal-spot error (mrad) on a held-out TEST split, before vs after reconstruction.

Usage:
    python run_canonical.py                       # default 3 heliostats
    python run_canonical.py --heliostats AC33 BE35 AB33
    python run_canonical.py --max-epoch 100 --train-samples 10 --test-samples 20
"""

import argparse
import json
import logging
import pathlib
import sys
import time

import h5py
import numpy as np
import torch

_here = pathlib.Path(__file__).resolve().parent
_src = _here.parent  # src/
sys.path.insert(0, str(_src))
# reuse the one_heliostat_demo config for paths (scenarios, PAINT benchmark)
sys.path.insert(0, str(_src / "one_heliostat_demo" / "single_heliostat"))

import config as cfg  # noqa: E402
import paint.util.paint_mappings as paint_mappings  # noqa: E402

from artist.flux import bitmap  # noqa: E402
from artist.io import PaintCalibrationDataParser  # noqa: E402
from artist.optim import KinematicsReconstructor  # noqa: E402
from artist.optim.loss import FocalSpotLoss  # noqa: E402
from artist.raytracing import HeliostatRayTracer  # noqa: E402
from artist.scenario import Scenario  # noqa: E402
from artist.geometry import bitmap_coordinates_to_target_coordinates  # noqa: E402
from artist.util import constants, indices, set_logger_config  # noqa: E402
from artist.util.env import get_device, setup_distributed_environment  # noqa: E402

from utils.evaluation import build_heliostat_data_mapping  # noqa: E402

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Canonical ARTIST kinematics reconstruction.")
    p.add_argument("--heliostats", nargs="+", default=["AC33", "AB33", "BE35"],
                   help="Heliostat IDs to reconstruct (default: AC33 AB33 BE35).")
    p.add_argument("--max-epoch", type=int, default=150)
    p.add_argument("--train-samples", type=int, default=10,
                   help="Calibration measurements per heliostat used for reconstruction.")
    p.add_argument("--test-samples", type=int, default=20,
                   help="Held-out measurements per heliostat for before/after evaluation.")
    p.add_argument("--rays", type=int, default=100, help="DNI / rays per surface point.")
    p.add_argument("--resolution", type=int, default=128, help="Flux bitmap side length.")
    p.add_argument("--surface-points", type=int, default=25, help="Surface points per facet side.")
    p.add_argument("--output-dir", type=pathlib.Path, default=None)
    return p.parse_args()


def _focal_error_mrad(scenario, hg, mapping_one_heliostat, parser, resolution, dist_m, device):
    """Aim-at-centre focal-spot error in mrad for one heliostat's samples (ARTIST convention).

    Aligns by aiming at the target centre, ray-traces, and compares the predicted flux
    centre to the measured centroid c_gt — the same forward map the reconstructor's loss uses.
    """
    (measured_flux, measured_cents, rays, _motor, active_mask, target_mask) = (
        parser.parse_data_for_reconstruction(
            heliostat_data_mapping=mapping_one_heliostat,
            heliostat_group=hg, scenario=scenario,
            bitmap_resolution=resolution, device=device,
        )
    )
    if active_mask.sum() == 0:
        return None

    hg.activate_heliostats(active_heliostats_mask=active_mask, device=device)
    hg.align_surfaces_with_incident_ray_directions(
        aim_points=scenario.solar_tower.get_centers_of_target_areas(target_mask, device=device),
        incident_ray_directions=rays,
        active_heliostats_mask=active_mask, device=device,
    )
    ray_tracer = HeliostatRayTracer(
        scenario=scenario, heliostat_group=hg, blocking_active=False,
        batch_size=int(active_mask.sum().item()), bitmap_resolution=resolution,
    )
    flux, _, _, _ = ray_tracer.trace_rays(
        incident_ray_directions=rays, active_heliostats_mask=active_mask,
        target_area_indices=target_mask, device=device,
    )
    sidx = ray_tracer.get_sampler_indices()
    pred_bitmap_coords = bitmap.get_center_of_mass(bitmaps=flux, device=device)
    pred_cents = bitmap_coordinates_to_target_coordinates(
        bitmap_coordinates=pred_bitmap_coords, bitmap_resolution=ray_tracer.bitmap_resolution,
        solar_tower=scenario.solar_tower, target_area_indices=target_mask[sidx], device=device,
    )
    inv = torch.argsort(sidx)
    pred_cents = pred_cents[inv]
    errs = (torch.norm(pred_cents[:, :3] - measured_cents[:, :3], dim=1) / dist_m * 1000)
    return errs.detach().cpu().numpy()


def main() -> None:
    args = _parse_args()
    set_logger_config()
    logging.getLogger().setLevel(logging.WARNING)  # quiet ARTIST chatter

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out = args.output_dir or (
        cfg.BASE_DIR / "outputs" / "new_mapping_function"
        / f"canonical_artist_reconstruction_{timestamp}"
    )
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"device={device}  heliostats={args.heliostats}  out={out}")

    # ARTIST tutorial optimizer / scheduler configuration (verbatim values).
    optimizer_dict = {
        constants.initial_learning_rate_rotation_deviation: 1e-4,
        constants.initial_learning_rate_initial_angles: 1e-3,
        constants.initial_learning_rate_initial_stroke_length: 1e-2,
        constants.tolerance: 0.0,
        constants.max_epoch: args.max_epoch,
        constants.batch_size: 50,
        constants.log_step: 0,
        constants.early_stopping_delta: 1e-8,
        constants.early_stopping_patience: 1000,
        constants.early_stopping_window: 2000,
    }
    scheduler_dict = {
        constants.scheduler_type: constants.reduce_on_plateau,
        constants.gamma: 0.9, constants.lr_min: 1e-6, constants.lr_max: 1e-3,
        constants.step_size_up: 500, constants.reduce_factor: 0.0001,
        constants.patience: 50, constants.threshold: 1e-3, constants.cooldown: 10,
    }
    optimization_configuration = {
        constants.optimization: optimizer_dict, constants.scheduler: scheduler_dict,
    }

    resolution = torch.tensor([args.resolution, args.resolution], device=device)
    results = {}

    # Build the train/test mappings once (filtered per-heliostat below).
    train_map_all = build_heliostat_data_mapping(
        pathlib.Path(cfg.BENCHMARK_CSV), pathlib.Path(cfg.CALIBRATION_DIR),
        pathlib.Path(cfg.REAL_FLUX_DIR), "train")
    test_map_all = build_heliostat_data_mapping(
        pathlib.Path(cfg.BENCHMARK_CSV), pathlib.Path(cfg.CALIBRATION_DIR),
        pathlib.Path(cfg.REAL_FLUX_DIR), "test")

    for hid in args.heliostats:
        t0 = time.time()
        scen_path = pathlib.Path(cfg.SCENARIO_PATH_TEMPLATE.format(heliostat_id=hid))
        if not scen_path.exists():
            print(f"[{hid}] no scenario — skip"); results[hid] = {"skip": "no scenario"}; continue
        train_map = [(h, c, f) for h, c, f in train_map_all if h == hid]
        test_map = [(h, c, f) for h, c, f in test_map_all if h == hid]
        if not train_map or not test_map:
            print(f"[{hid}] missing split data — skip"); results[hid] = {"skip": "no data"}; continue

        n_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(scenario_path=scen_path)
        with setup_distributed_environment(number_of_heliostat_groups=n_groups, device=device) as ddp:
            dev = ddp[constants.device]
            with h5py.File(scen_path, "r") as fh:
                scenario = Scenario.load_scenario_from_hdf5(
                    scenario_file=fh, device=dev,
                    number_of_surface_points_per_facet=torch.tensor(
                        [args.surface_points, args.surface_points]),
                )
            hg = scenario.heliostat_field.heliostat_groups[0]
            hel_pos = hg.positions[0, :3].float()
            tower_ref = scenario.solar_tower.target_areas[indices.planar_target_areas].centers[:, :3].float().mean(0)
            dist_m = torch.norm(hel_pos - tower_ref.to(dev)).item()

            eval_parser = PaintCalibrationDataParser(
                sample_limit=args.test_samples, centroid_extraction_method=paint_mappings.UTIS_KEY)

            err_before = _focal_error_mrad(scenario, hg, test_map, eval_parser, resolution, dist_m, dev)

            # ---- canonical reconstruction (ARTIST KinematicsReconstructor + FocalSpotLoss) ----
            recon_parser = PaintCalibrationDataParser(
                sample_limit=args.train_samples, centroid_extraction_method=paint_mappings.UTIS_KEY)
            data = {constants.data_parser: recon_parser, constants.heliostat_data_mapping: train_map}
            reconstructor = KinematicsReconstructor(
                ddp_setup=ddp, scenario=scenario, data=data, dni=args.rays,
                optimization_configuration=optimization_configuration,
                reconstruction_method=constants.kinematics_reconstruction_raytracing,
                bitmap_resolution=resolution,
            )
            loss_def = FocalSpotLoss(scenario=scenario)
            final_loss = reconstructor.reconstruct_kinematics(loss_definition=loss_def, device=dev)

            err_after = _focal_error_mrad(scenario, hg, test_map, eval_parser, resolution, dist_m, dev)

        def _stat(e):
            return None if e is None else {"mean": float(e.mean()), "median": float(np.median(e)), "n": int(len(e))}
        results[hid] = {
            "dist_m": dist_m, "before": _stat(err_before), "after": _stat(err_after),
            "train_samples": args.train_samples, "test_samples": args.test_samples,
            "minutes": round((time.time() - t0) / 60, 2),
        }
        b = results[hid]["before"]; a = results[hid]["after"]
        print(f"[{hid}] dist={dist_m:5.0f}m  BEFORE mean={b['mean']:6.2f} med={b['median']:6.2f}  "
              f"AFTER mean={a['mean']:6.2f} med={a['median']:6.2f}  ({results[hid]['minutes']}min)")

    with open(out / "results.json", "w") as fh:
        json.dump({"config": vars(args), "results": results}, fh, indent=2, default=str)
    print(f"\nSaved → {out / 'results.json'}")


if __name__ == "__main__":
    main()
