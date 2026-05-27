"""
One-heliostat multi-seed sensitivity experiment (in-memory data generation).

Generates synthetic calibration data entirely in memory — no pre-generated
dataset on disk required.  For a given perturbation seed the experiment:

  1. Samples perturbations for all 5 heliostats at once (reproducible from seed).
  2. For each heliostat:
     a. Loads the single-heliostat scenario.
     b. Temporarily applies perturbations, ray-traces train/val/test splits in
        memory, then resets the scenario to its clean state.
     c. Sweeps all configured training sample sizes, calling the standard
        one_heliostat_train_sizes training loop for each.
  3. Saves per-heliostat and aggregate results under outputs/.../seed_{seed}/.

Designed as a SLURM array job — each task handles one seed in parallel.

Usage (local)
-------------
    cd src
    python one_hel_multi_seed/main.py --seed-index 0
    python one_hel_multi_seed/main.py --seed-index 0 --no-deflectometry

Usage (DAIC array job)
----------------------
    sbatch sbatch_files/run_one_hel_multi_seed.sh
"""
import gc
import json
import logging
import pathlib
import sys
import argparse

import matplotlib
matplotlib.use("Agg")

import h5py
import torch
from artist.raytracing.heliostat_ray_tracer import HeliostatRayTracer
from artist.io.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import constants as config_dictionary, set_logger_config
from artist.util.env import get_device, setup_distributed_environment
from artist.flux import get_center_of_mass
from artist.geometry import bitmap_coordinates_to_target_coordinates

_here = pathlib.Path(__file__).resolve().parent
_src  = _here.parent
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_src / "one_heliostat_train_sizes"))

import config as cfg
import train as _train_mod
from utils.evaluation import build_heliostat_data_mapping
from utils.synth_data import (
    _equalize_mapping,
    apply_perturbations,
    reset_perturbations,
    sample_perturbations,
    perturbations_to_json,
)

run = _train_mod.run


# ---------------------------------------------------------------------------
# In-memory parser
# ---------------------------------------------------------------------------

class InMemorySyntheticParser:
    """
    Drop-in replacement for SyntheticDatasetParser backed by pre-generated
    in-memory tensors.

    Subsets to however many samples the heliostat_data_mapping requests, so
    the same parser instance serves all entries in the train-size sweep without
    regenerating data.
    """

    def __init__(
        self,
        heliostat_id: str,
        flux: torch.Tensor,            # [N, H, W]  CPU
        focal_spots: torch.Tensor,     # [N, 4]     CPU
        incident_rays: torch.Tensor,   # [N, 4]     CPU
        motor_pos: torch.Tensor,       # [N, 2]     CPU
        target_mask: torch.Tensor,     # [N]        CPU  (long)
        heliostat_group_names: list,
    ) -> None:
        self._hid           = heliostat_id
        self._flux          = flux
        self._focal_spots   = focal_spots
        self._incident_rays = incident_rays
        self._motor_pos     = motor_pos
        self._target_mask   = target_mask
        self._names         = heliostat_group_names

    def parse_data_for_reconstruction(
        self, heliostat_data_mapping, heliostat_group, scenario, device=None, **_
    ):
        mapping_dict = {hid: len(cal) for hid, cal, _ in heliostat_data_mapping if cal}
        n = mapping_dict.get(self._hid, 0)

        active_mask = torch.zeros(len(self._names), dtype=torch.long)
        if self._hid in self._names:
            active_mask[self._names.index(self._hid)] = n

        return (
            self._flux[:n].to(device),
            self._focal_spots[:n].to(device),
            self._incident_rays[:n].to(device),
            self._motor_pos[:n].to(device),
            active_mask.to(device),
            self._target_mask[:n].to(device),
        )


# ---------------------------------------------------------------------------
# In-memory generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_split(
    scenario,
    heliostat_group,
    mapping: list,
    perturbations: dict,
    n_samples: int,
    n_gen_rays: int,
    device: torch.device,
) -> InMemorySyntheticParser:
    """
    Temporarily apply perturbations, ray-trace n_samples calibration images,
    reset, and return an InMemorySyntheticParser holding the result on CPU.

    Incident ray directions come from real PAINT calibration measurements
    (the geometry is real; only the output flux/centroid is synthetic).
    Motor positions are captured from the perturbed kinematic model so that
    Stage-1 AlignmentLoss pre-training uses the correct ground-truth target.
    """
    real_parser = PaintCalibrationDataParser(
        sample_limit=n_samples,
        centroid_extraction_method=cfg.CENTROID_METHOD,
    )
    mapping_eq = _equalize_mapping(mapping, n_samples)

    _, _, incident_rays, _, active_mask, target_mask = (
        real_parser.parse_data_for_reconstruction(
            heliostat_data_mapping=mapping_eq,
            heliostat_group=heliostat_group,
            scenario=scenario,
            device=device,
        )
    )

    if active_mask.sum() == 0:
        raise RuntimeError("No active heliostats found in mapping.")

    old_rays = scenario.light_sources.light_source_list[0].number_of_rays
    scenario.set_number_of_rays(n_gen_rays)

    original = apply_perturbations(heliostat_group.kinematics, perturbations, device)

    heliostat_group.activate_heliostats(active_heliostats_mask=active_mask, device=device)
    kinematic = heliostat_group.kinematics

    base_dev = perturbations["base_position"].to(device).repeat_interleave(active_mask, dim=0)
    pad = torch.zeros(base_dev.shape[0], 1, device=device)
    kinematic.active_heliostat_positions = (
        kinematic.active_heliostat_positions + torch.cat([base_dev, pad], dim=1)
    )

    heliostat_group.align_surfaces_with_incident_ray_directions(
        aim_points=scenario.solar_tower.get_centers_of_target_areas(target_mask, device=device),
        incident_ray_directions=incident_rays,
        active_heliostats_mask=active_mask,
        device=device,
    )

    perturbed_motor_pos = kinematic.active_motor_positions.detach().clone()

    ray_tracer = HeliostatRayTracer(
        scenario=scenario,
        heliostat_group=heliostat_group,
        blocking_active=False,
        world_size=1,
        rank=0,
        batch_size=max(8, int(active_mask.sum().item())),
        random_seed=42,
    )
    flux_sampler, _, _, _ = ray_tracer.trace_rays(
        incident_ray_directions=incident_rays,
        active_heliostats_mask=active_mask,
        target_area_indices=target_mask,
        device=device,
    )
    sample_indices = ray_tracer.get_sampler_indices()

    bitmap_coords = get_center_of_mass(bitmaps=flux_sampler, device=device)
    centroids_sampler = bitmap_coordinates_to_target_coordinates(
        bitmap_coordinates=bitmap_coords,
        bitmap_resolution=ray_tracer.bitmap_resolution,
        solar_tower=scenario.solar_tower,
        target_area_indices=target_mask[sample_indices],
        device=device,
    )

    inv_perm    = torch.argsort(sample_indices)
    focal_spots = centroids_sampler[inv_perm]
    flux        = flux_sampler[inv_perm]

    reset_perturbations(heliostat_group.kinematics, original)
    scenario.set_number_of_rays(old_rays)

    hid = heliostat_group.names[torch.where(active_mask > 0)[0][0].item()]

    return InMemorySyntheticParser(
        heliostat_id=hid,
        flux=flux.cpu(),
        focal_spots=focal_spots.cpu(),
        incident_rays=incident_rays.cpu(),
        motor_pos=perturbed_motor_pos.cpu(),
        target_mask=target_mask.cpu(),
        heliostat_group_names=list(heliostat_group.names),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-heliostat multi-seed sensitivity sweep (in-memory generation)."
    )
    parser.add_argument(
        "--seed-index", type=int, default=0,
        help="Index into cfg.SEEDS (0–9). Pass $SLURM_ARRAY_TASK_ID for array jobs.",
    )
    parser.add_argument("--daic", action="store_true", help="Use DAIC cluster paths.")
    parser.add_argument(
        "--no-deflectometry", dest="ideal_scenario", action="store_true",
        help="Train using ideal (flat) scenarios instead of deflectometry-fitted ones.",
    )
    parser.add_argument("--output-parent", type=pathlib.Path, default=None)
    args = parser.parse_args()

    if args.daic:
        cfg.IS_ON_DAIC = True
        cfg.BASE_DIR   = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
        cfg.PAINT_DIR  = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")
        cfg.ONE_HELIOSTAT_SCENARIOS_DIR = cfg.BASE_DIR / "scenarios" / "one_heliostat_scenarios"
        cfg.BENCHMARK_CSV              = cfg.PAINT_DIR / "splits" / f"{cfg.BENCHMARK_NAME}.csv"
        cfg.CALIBRATION_PROPERTIES_DIR = cfg.PAINT_DIR / cfg.BENCHMARK_NAME / "calibration_properties"
        cfg.FLUX_IMAGE_DIR             = cfg.PAINT_DIR / cfg.BENCHMARK_NAME / "flux_image"

    seed           = cfg.SEEDS[args.seed_index]
    scenario_label = "ideal" if args.ideal_scenario else "deflectometry"
    scenarios_root = (
        cfg.ONE_HELIOSTAT_SCENARIOS_DIR / "ideal"
        if args.ideal_scenario
        else cfg.ONE_HELIOSTAT_SCENARIOS_DIR
    )

    if args.output_parent is None:
        base = cfg.BASE_DIR / "outputs"
        if not cfg.IS_ON_DAIC:
            base = base / "local_runs"
        args.output_parent = base / f"one_hel_multi_seed_{scenario_label}"

    seed_dir = args.output_parent / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    log = logging.getLogger(__name__)

    fh = logging.FileHandler(seed_dir / "run.log")
    fh.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info(f"Seed index : {args.seed_index}  →  seed = {seed}")
    log.info(f"Scenario   : {scenario_label}")
    log.info(f"Output dir : {seed_dir}")

    torch.manual_seed(0)
    device = get_device()
    log.info(f"Device: {device}")

    # Sample perturbations for ALL heliostats from this seed in one call so that
    # each heliostat gets a distinct but reproducible perturbation.
    all_perturbations = sample_perturbations(
        n_heliostats=len(cfg.HELIOSTATS),
        ranges=cfg.PERTURBATION_RANGES,
        seed=seed,
    )

    # Build PAINT data mappings once — incident ray directions are from real
    # calibration measurements and are the same across all seeds.
    def _build_mapping(split: str) -> list:
        return build_heliostat_data_mapping(
            benchmark_csv=cfg.BENCHMARK_CSV,
            calibration_properties_dir=cfg.CALIBRATION_PROPERTIES_DIR,
            flux_image_dir=cfg.FLUX_IMAGE_DIR,
            split=split,
        )

    max_train = max(cfg.TRAIN_SIZES)
    raw_train = _build_mapping("train")
    raw_val   = _build_mapping("validation")
    raw_test  = _build_mapping("test")

    all_results: dict[str, dict] = {}

    with setup_distributed_environment(number_of_heliostat_groups=1, device=device) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        for hel_idx, hid in enumerate(cfg.HELIOSTATS):
            log.info(f"\n{'='*60}\nHeliostat: {hid}  ({hel_idx+1}/{len(cfg.HELIOSTATS)})\n{'='*60}")

            scenario_path = scenarios_root / hid / "scenario.h5"
            if not scenario_path.exists():
                log.error(f"Scenario not found: {scenario_path} — skipping {hid}.")
                continue

            # Per-heliostat perturbations: row hel_idx from the joint draw.
            perturbations = {k: v[hel_idx:hel_idx + 1] for k, v in all_perturbations.items()}
            pert_json = perturbations_to_json(perturbations, [hid])

            # PAINT mappings filtered to this heliostat.
            full_train_map = [e for e in _equalize_mapping(raw_train, max_train)       if e[0] == hid]
            val_map        = [e for e in _equalize_mapping(raw_val,   cfg.VAL_SAMPLES) if e[0] == hid]
            test_map       = [e for e in _equalize_mapping(raw_test,  cfg.TEST_SAMPLES) if e[0] == hid]

            if not full_train_map or not val_map or not test_map:
                log.warning(f"{hid}: missing PAINT mapping entries — skipping.")
                continue

            # Load scenario for generation (will be discarded after data generation).
            with h5py.File(scenario_path, "r") as f:
                gen_scenario = Scenario.load_scenario_from_hdf5(
                    scenario_file=f,
                    device=device,
                    number_of_surface_points_per_facet=torch.tensor(
                        [cfg.SURFACE_POINTS_PER_FACET, cfg.SURFACE_POINTS_PER_FACET]
                    ),
                )
            gen_group = gen_scenario.heliostat_field.heliostat_groups[0]

            log.info(f"  Generating in-memory splits (gen_rays={cfg.SYNTH_GEN_RAYS}) …")
            train_ds = _generate_split(gen_scenario, gen_group, full_train_map, perturbations, max_train,        cfg.SYNTH_GEN_RAYS, device)
            val_ds   = _generate_split(gen_scenario, gen_group, val_map,        perturbations, cfg.VAL_SAMPLES,  cfg.SYNTH_GEN_RAYS, device)
            test_ds  = _generate_split(gen_scenario, gen_group, test_map,       perturbations, cfg.TEST_SAMPLES, cfg.SYNTH_GEN_RAYS, device)
            log.info(
                f"  Generated: train={len(train_ds._flux)}  "
                f"val={len(val_ds._flux)}  test={len(test_ds._flux)} samples"
            )

            del gen_scenario, gen_group
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Train-size sweep.
            hel_results: dict[int, dict] = {}
            hel_dir = seed_dir / hid
            hel_dir.mkdir(parents=True, exist_ok=True)

            for train_size in cfg.TRAIN_SIZES:
                log.info(f"  Train size: {train_size}")

                train_map = [
                    (h, cal[:train_size], flux[:train_size])
                    for h, cal, flux in full_train_map
                ]
                subdir = hel_dir / f"train_size_{train_size}"

                results = run(
                    scenario_path=scenario_path,
                    device=device,
                    ddp_setup=ddp_setup,
                    train_mapping=train_map,
                    val_mapping=val_map,
                    test_mapping=test_map,
                    train_parser=train_ds,
                    val_parser=val_ds,
                    test_parser=test_ds,
                    optimization_config=cfg.OPTIMIZATION_CONFIG,
                    output_dir=subdir,
                    loss_type=cfg.LOSS_TYPE,
                    dataset_type="synthetic",
                    n_surface_pts=cfg.SURFACE_POINTS_PER_FACET,
                    train_rays=cfg.TRAIN_RAYS,
                    perturbations_json=pert_json,
                    heliostat_ids=[hid],
                    stage1_epochs=cfg.STAGE1_EPOCHS,
                    stage2_epochs=cfg.STAGE2_EPOCHS,
                )
                hel_results[train_size] = results
                log.info(
                    f"  [{hid} n={train_size:3d}]  "
                    f"pre: val={results['pre_training']['val']['mean_mrad']:.3f}  "
                    f"test={results['pre_training']['test']['mean_mrad']:.3f}  |  "
                    f"post: val={results['post_training']['val']['mean_mrad']:.3f}  "
                    f"test={results['post_training']['test']['mean_mrad']:.3f} mrad"
                )

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            with open(hel_dir / "summary.json", "w") as fh:
                json.dump({
                    "heliostat_id":  hid,
                    "seed":          seed,
                    "scenario":      scenario_label,
                    "train_sizes":   cfg.TRAIN_SIZES,
                    "perturbations": pert_json,
                    "results":       {str(n): r for n, r in hel_results.items()},
                }, fh, indent=2)

            all_results[hid] = hel_results
            del train_ds, val_ds, test_ds
            gc.collect()

    with open(seed_dir / "summary.json", "w") as fh:
        json.dump({
            "seed":       seed,
            "seed_index": args.seed_index,
            "scenario":   scenario_label,
            "heliostats": cfg.HELIOSTATS,
            "train_sizes": cfg.TRAIN_SIZES,
            "results": {
                hid: {str(n): r for n, r in hel.items()}
                for hid, hel in all_results.items()
            },
        }, fh, indent=2)

    log.info("\n" + "="*60)
    log.info(f"Seed {seed} complete. Post-training test mrad:")
    for hid, hel_results in all_results.items():
        vals = "  ".join(
            f"n={n}: {r['post_training']['test']['mean_mrad']:.3f}"
            for n, r in sorted(hel_results.items())
        )
        log.info(f"  {hid}: {vals}")
    log.info(f"Results → {seed_dir}")


if __name__ == "__main__":
    main()
