"""
Prepare the held-out generalization test set for fine_error_learning.

Twelve heliostats with deflectometry data that are NOT in the 63-heliostat
balanced benchmark, spread evenly by distance from the tower (25–273 m):

    AA36 AA29 AF32 AG29 AJ46 AP32 AW39 AZ33 BB28 BE40 BA70 BH65

For each of them this script:
  1. Creates a per-heliostat scenario with a deflectometry-fitted NURBS surface
     (same builder as scenarios/one_heliostat_scenarios/, output kept separate
     in scenarios/heldout_heliostat_scenarios/<ID>/scenario.h5).
  2. Generates a synthetic calibration dataset with FRESH random perturbations
     (RANDOM_SEED 4242, not the balanced dataset's 42) using sun positions from
     the field-wide 50/20 PAINT benchmark → datasets/synthetic/heldout_dataset/
     dataset/{train,val,test}/<ID>/... plus an aggregated perturbations.json.
  3. Runs stage-1-only kinematic reconstruction (fixed on-disk split, no
     re-pooling — the 50/20/20 pool is too small for the 100/50/50 splitter) →
     outputs/fine_error_learning/heldout_stage1_synth/<ID>/stage1_checkpoint.pt.

After this, FEL trained on the 62 benchmark heliostats can be evaluated on
unseen heliostats:

    python fine_error_learning/main.py --evaluate <run_dir> --eval-split test \
        --heliostats AA36 AA29 ... --checkpoint-dir <heldout ckpts> \
        --data-dir <heldout dataset>

Usage
-----
    cd src && python fine_error_learning/prepare_heldout.py
    python fine_error_learning/prepare_heldout.py --skip-scenarios --skip-generation
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time

import torch

_HERE = pathlib.Path(__file__).resolve().parent          # fine_error_learning/
_SRC = _HERE.parent                                       # src/
_SH = _SRC / "one_heliostat_demo" / "single_heliostat"    # live KR pipeline
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SH))

from artist.io import paint_scenario_parser
from artist.util.config import (
    LightSourceConfig,
    LightSourceListConfig,
)
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator
from artist.util import constants as config_dictionary, get_device, set_logger_config

import config as kr_cfg  # single_heliostat/config.py (mutated below)
import generate_dataset as gd
import train as tr

log = logging.getLogger(__name__)

BASE_DIR = _SRC.parent  # master-thesis/

HELDOUT_IDS = [
    "AA36", "AA29", "AF32", "AG29", "AJ46", "AP32",
    "AW39", "AZ33", "BB28", "BE40", "BA70", "BH65",
]

SCENARIO_DIR = BASE_DIR / "scenarios" / "heldout_heliostat_scenarios"
DATASET_ROOT = BASE_DIR / "datasets" / "synthetic" / "heldout_dataset"  # + /dataset
STAGE1_DIR = BASE_DIR / "outputs" / "fine_error_learning" / "heldout_stage1_synth"

# Field-wide benchmark: the 12 held-out heliostats are not in the 63-heliostat
# train-100 benchmark, but they are in this 1277-heliostat 50/20/20 split.
FIELD_BENCHMARK = "benchmark_split-balanced_train-50_validation-20"

# Different from the balanced dataset's RANDOM_SEED = 42 → fresh perturbation draws.
HELDOUT_RANDOM_SEED = 4242

# NURBS fit settings — identical to too_old/one_heliostat_train_sizes/create_scenarios.py
NUMBER_OF_NURBS_CONTROL_POINTS = torch.tensor([20, 20])
NURBS_FIT_METHOD = config_dictionary.fit_nurbs_from_normals
NURBS_DEFLECTOMETRY_STEP_SIZE = 100
NURBS_FIT_TOLERANCE = 1e-10
NURBS_FIT_MAX_EPOCH = 400


# ---------------------------------------------------------------------------
# Step 1 — scenarios
# ---------------------------------------------------------------------------

def create_scenario(hid: str, device: torch.device) -> None:
    """Build one per-heliostat scenario with a deflectometry-fitted surface."""
    heliostats_dir = kr_cfg.PAINT_DIR / "heliostats"
    tower_file = BASE_DIR.parent / "ARTIST" / "tutorials" / "data" / "paint" / "tower-measurements.json"
    props = heliostats_dir / hid / "Properties" / f"{hid}-heliostat-properties.json"
    defl_dir = heliostats_dir / hid / "Deflectometry"
    filled = sorted(defl_dir.glob(f"{hid}-filled-*-deflectometry.h5"))
    if not props.exists() or not filled:
        raise FileNotFoundError(f"PAINT properties/deflectometry missing for {hid}")

    out_path = SCENARIO_DIR / hid / "scenario.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    power_plant_config, planar_cfg, cylindrical_cfg = (
        paint_scenario_parser.extract_paint_tower_measurements(
            tower_measurements_path=tower_file, device=device
        )
    )
    light_source_list_config = LightSourceListConfig(
        light_source_list=[
            LightSourceConfig(
                light_source_key="sun_1",
                light_source_type=config_dictionary.sun_key,
                number_of_rays=10,
                distribution_type=config_dictionary.light_source_distribution_is_normal,
                mean=0.0,
                covariance=4.3681e-06,
            )
        ]
    )
    nurbs_fit_optimizer = torch.optim.Adam([torch.empty(1, requires_grad=True)], lr=1e-3)
    nurbs_fit_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        nurbs_fit_optimizer, mode="min", factor=0.2, patience=50,
        threshold=1e-7, threshold_mode="abs",
    )
    heliostat_list_config, prototype_config = (
        paint_scenario_parser.extract_paint_heliostats_fitted_surface(
            paths=[(hid, props, filled[-1])],
            power_plant_position=power_plant_config.power_plant_position,
            number_of_nurbs_control_points=NUMBER_OF_NURBS_CONTROL_POINTS,
            deflectometry_step_size=NURBS_DEFLECTOMETRY_STEP_SIZE,
            nurbs_fit_method=NURBS_FIT_METHOD,
            nurbs_fit_tolerance=NURBS_FIT_TOLERANCE,
            nurbs_fit_max_epoch=NURBS_FIT_MAX_EPOCH,
            nurbs_fit_optimizer=nurbs_fit_optimizer,
            nurbs_fit_scheduler=nurbs_fit_scheduler,
            device=device,
        )
    )
    H5ScenarioGenerator(
        file_path=out_path,
        power_plant_config=power_plant_config,
        target_area_list_planar_config=planar_cfg,
        target_area_list_cylindrical_config=cylindrical_cfg,
        light_source_list_config=light_source_list_config,
        prototype_config=prototype_config,
        heliostat_list_config=heliostat_list_config,
    ).generate_scenario()
    log.info(f"  scenario → {out_path}")


# ---------------------------------------------------------------------------
# Step 2 — synthetic data generation
# ---------------------------------------------------------------------------

def configure_generation() -> None:
    """Mutate the live KR config for held-out dataset generation."""
    kr_cfg.SCENARIO_PATH_TEMPLATE = str(SCENARIO_DIR / "{heliostat_id}" / "scenario.h5")
    kr_cfg.DATA_MODE = "random_synthetic"
    kr_cfg.RANDOM_SEED = HELDOUT_RANDOM_SEED
    paint = kr_cfg.PAINT_DIR
    kr_cfg.BENCHMARK_CSV = paint / "splits" / f"{FIELD_BENCHMARK}.csv"
    kr_cfg.CALIBRATION_DIR = paint / FIELD_BENCHMARK / "calibration_properties"
    kr_cfg.REAL_FLUX_DIR = paint / FIELD_BENCHMARK / "flux_image"
    # The 50/20/20 pool is smaller than the balanced 100/50/50 one — relax the
    # acceptance thresholds (active-pixel filter drops a few samples).
    kr_cfg.MIN_TRAIN_SAMPLES = 40
    kr_cfg.MIN_VAL_SAMPLES = 15
    kr_cfg.MIN_TEST_SAMPLES = 15
    kr_cfg.MAX_RESAMPLE_ATTEMPTS = 25


def generate_heldout_data(device: torch.device) -> dict:
    """Generate the perturbed dataset per heliostat; aggregate perturbations.json."""
    configure_generation()
    perturbations: dict = {}
    dataset_dir = DATASET_ROOT / "dataset"
    for i, hid in enumerate(HELDOUT_IDS):
        log.info(f"[{i + 1}/{len(HELDOUT_IDS)}] Generating held-out data for {hid} ...")
        t0 = time.time()
        result = gd.generate(
            heliostat_id=hid,
            output_dir=DATASET_ROOT,
            cfg=kr_cfg,
            device=device,
            seed_offset=i,
        )
        pert = result["pert_tensors"]
        # Same JSON schema as generate_dataset.py:322-330 (which would overwrite
        # the shared file per call — aggregate here instead).
        perturbations[hid] = {
            "rotation_rad": pert["rotation"][0].cpu().tolist(),
            "actuator_angle_rad": pert["actuator_angle"][0].cpu().tolist(),
            "actuator_stroke_m": pert["actuator_stroke"][0].cpu().tolist(),
            "actuator_offset_m": pert["actuator_offset"][0].cpu().tolist(),
            "translation_m": pert["translation"][0].cpu().tolist(),
            "base_position_m": pert["base_position"][0].cpu().tolist(),
        }
        with open(dataset_dir / "perturbations.json", "w") as f:
            json.dump(perturbations, f, indent=2)
        log.info(f"  done in {(time.time() - t0) / 60:.1f} min")
    return perturbations


# ---------------------------------------------------------------------------
# Step 3 — stage-1 warm starts
# ---------------------------------------------------------------------------

def run_stage1(device: torch.device) -> None:
    """Stage-1-only KR per held-out heliostat → stage1_checkpoint.pt."""
    kr_cfg.SCENARIO_PATH_TEMPLATE = str(SCENARIO_DIR / "{heliostat_id}" / "scenario.h5")
    kr_cfg.DATA_MODE = "synthetic"
    # Honor the generated on-disk 50/20/20 split verbatim (the pool is too
    # small for the 100/50/50 DatasetSplitter re-split).
    kr_cfg.USE_FIXED_SPLIT = True
    dataset_dir = DATASET_ROOT / "dataset"
    for i, hid in enumerate(HELDOUT_IDS):
        hid_dir = STAGE1_DIR / hid
        hid_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"[{i + 1}/{len(HELDOUT_IDS)}] Stage 1 for {hid} ...")
        tr.run(
            heliostat_id=hid,
            dataset_dir=dataset_dir,
            output_dir=hid_dir,
            cfg=kr_cfg,
            device=device,
            skip_stage1=False,
            skip_stage2=True,
            stage1_checkpoint=None,
            make_plots=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the FEL held-out test set.")
    parser.add_argument("--skip-scenarios", action="store_true",
                        help="Skip scenario creation (already built).")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip synthetic data generation.")
    parser.add_argument("--skip-stage1", action="store_true",
                        help="Skip stage-1 warm-start training.")
    args = parser.parse_args()

    set_logger_config()
    device = get_device()
    log.info(f"Device: {device}  |  held-out heliostats: {len(HELDOUT_IDS)}")

    if not args.skip_scenarios:
        for hid in HELDOUT_IDS:
            if (SCENARIO_DIR / hid / "scenario.h5").exists():
                log.info(f"  [skip] scenario for {hid} exists")
                continue
            create_scenario(hid, device)

    if not args.skip_generation:
        generate_heldout_data(device)

    if not args.skip_stage1:
        run_stage1(device)

    summary = {
        "heliostat_ids": HELDOUT_IDS,
        "scenario_dir": str(SCENARIO_DIR),
        "dataset_dir": str(DATASET_ROOT / "dataset"),
        "stage1_dir": str(STAGE1_DIR),
        "random_seed": HELDOUT_RANDOM_SEED,
        "field_benchmark": FIELD_BENCHMARK,
    }
    STAGE1_DIR.mkdir(parents=True, exist_ok=True)
    with open(STAGE1_DIR / "heldout_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Held-out preparation complete → {STAGE1_DIR / 'heldout_summary.json'}")


if __name__ == "__main__":
    main()
