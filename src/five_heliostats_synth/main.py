"""
5-heliostat synthetic perturbation experiment.

Loops over training data sizes (1 / 5 / 10 / 15 / 25 / 50 samples/heliostat) using
WortbergKinematicReconstructor (full parameter set) for each run.

Three evaluation checkpoints per run:
  1. pre_perturbation  — clean scenario vs clean synthetic test  (~0 mrad)
  2. post_perturbation — perturbed scenario vs same clean test  (high mrad)
  3. post_training     — trained (recovered) scenario vs same clean test (low mrad)

Usage
-----
    python main.py
    python main.py --output-dir outputs/my_run
"""
import argparse
import datetime
import gc
import json
import logging
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")

import torch
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config
from artist.util.environment_setup import get_device, setup_distributed_environment

_here = pathlib.Path(__file__).resolve().parent
_src  = _here.parent
sys.path.insert(0, str(_src))

import config as cfg
from data import perturbations_to_json, sample_perturbations, _equalize_mapping, SyntheticDatasetParser
from generate_dataset import synthetic_data_dir, split_dir, _split_complete
from train import run
from reporting import (
    plot_convergence,
    plot_param_recovery,
    plot_kinematic_evolution,
    plot_kinematic_stages,
    write_summary,
    write_ablation_summary,
    plot_ablation_comparison,
    plot_combined_convergence,
)
from utils.evaluation import build_heliostat_data_mapping

from artist_extensions.kinematic_reconstructors import WortbergKinematicReconstructor


def _build_mapping(split: str) -> list:
    return build_heliostat_data_mapping(
        benchmark_csv=cfg.BENCHMARK_CSV,
        calibration_properties_dir=cfg.CALIBRATION_PROPERTIES_DIR,
        flux_image_dir=cfg.FLUX_IMAGE_DIR,
        split=split,
    )


def _filter_heliostats(mapping: list, heliostat_ids: list) -> list:
    ids = set(heliostat_ids)
    filtered = [(hid, cal, flux) for hid, cal, flux in mapping if hid in ids]
    if not filtered:
        raise RuntimeError(
            f"None of {heliostat_ids} found in mapping. "
            f"Available (first 10): {[hid for hid, *_ in mapping[:10]]}"
        )
    return filtered


def _run_reporting(results: dict, perturbations_json: dict, sub_dir: pathlib.Path) -> None:
    """Generate all plots and summary for a single train-size run."""
    history_file = sub_dir / "convergence_history.json"
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
        plot_convergence(
            history, sub_dir,
            pre_perturbation_m=results.get("pre_perturbation",  {}).get("mean_m"),
            post_perturbation_m=results.get("post_perturbation", {}).get("mean_m"),
            post_training_m=results.get("post_training",     {}).get("mean_m"),
            pre_perturbation_mrad=results.get("pre_perturbation",  {}).get("mean_mrad"),
            post_perturbation_mrad=results.get("post_perturbation", {}).get("mean_mrad"),
            post_training_mrad=results.get("post_training",     {}).get("mean_mrad"),
        )

    if results.get("param_recovery"):
        plot_param_recovery(results["param_recovery"], sub_dir)

    kinematic_history_file = sub_dir / "kinematic_history.json"
    if kinematic_history_file.exists():
        with open(kinematic_history_file) as f:
            kinematic_history = json.load(f)
        plot_kinematic_evolution(
            kinematic_history, perturbations_json, cfg.HELIOSTAT_IDS, sub_dir
        )

    kinematic_stages_file = sub_dir / "kinematic_stages.json"
    if kinematic_stages_file.exists():
        with open(kinematic_stages_file) as f:
            kinematic_stages = json.load(f)
        plot_kinematic_stages(kinematic_stages, cfg.HELIOSTAT_IDS, sub_dir)

    write_summary(results, perturbations_json, sub_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="5-heliostat synthetic perturbation experiment.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        run_dir = args.output_dir
    elif cfg.IS_ON_DAIC:
        run_dir = cfg.BASE_DIR / "outputs" / f"five_hel_synth_{timestamp}"
    else:
        run_dir = cfg.BASE_DIR / "outputs" / "local_runs" / f"five_hel_synth_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    log = logging.getLogger(__name__)

    fh = logging.FileHandler(run_dir / "run.log")
    fh.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info(f"Heliostats : {cfg.HELIOSTAT_IDS}")
    log.info(f"Benchmark  : {cfg.BENCHMARK_NAME}")
    log.info(f"Scenario   : {cfg.SCENARIO_PATH}")
    log.info(f"Output dir : {run_dir}")
    log.info(f"Train sample counts : {cfg.TRAIN_SAMPLE_COUNTS}")
    log.info(f"Reconstructor       : WortbergKinematicReconstructor")

    torch.manual_seed(0)
    device = get_device()
    log.info(f"Device: {device}")

    # ------------------------------------------------------------------ perturbations
    perturbations = sample_perturbations(
        n_heliostats=len(cfg.HELIOSTAT_IDS),
        ranges=cfg.PERTURBATION_RANGES,
        seed=cfg.PERTURBATION_SEED,
    )
    perturbations_json = perturbations_to_json(perturbations, cfg.HELIOSTAT_IDS)
    with open(run_dir / "perturbations.json", "w") as f:
        json.dump(perturbations_json, f, indent=2)
    log.info("Perturbations sampled and saved.")
    for hid, p in perturbations_json.items():
        log.info(
            f"  {hid}  rot={[f'{v*1000:.2f}' for v in p['rotation_rad']]} mrad  "
            f"act={[f'{v*1000:.2f}' for v in p['actuator_angle_rad']]} mrad  "
            f"base={[f'{v*1000:.1f}' for v in p['base_position_m']]} mm"
        )

    # ------------------------------------------------------------------ data mappings
    log.info("Building data mappings …")
    val_map  = _equalize_mapping(_filter_heliostats(_build_mapping("validation"), cfg.HELIOSTAT_IDS), cfg.VAL_SAMPLES)
    test_map = _equalize_mapping(_filter_heliostats(_build_mapping("test"),       cfg.HELIOSTAT_IDS), cfg.TEST_SAMPLES)
    # train_map_n is built per-iteration inside the loop (different n_train per run)

    # ------------------------------------------------------------------ DDP
    n_groups = Scenario.get_number_of_heliostat_groups_from_hdf5(cfg.SCENARIO_PATH)

    with setup_distributed_environment(
        number_of_heliostat_groups=n_groups, device=device
    ) as ddp_setup:
        device = ddp_setup[config_dictionary.device]

        # ------------------------------------------------------------ check pre-generated data exists
        _incomplete = [
            s for s, n in [("val", cfg.VAL_SAMPLES), ("test", cfg.TEST_SAMPLES), ("train", cfg.TRAIN_SAMPLES)]
            if not _split_complete(s, cfg.HELIOSTAT_IDS, n)
        ]
        if _incomplete:
            raise FileNotFoundError(
                f"Synthetic dataset incomplete for splits: {_incomplete}\n"
                "Run:  python generate_dataset.py"
            )

        synth_val   = SyntheticDatasetParser(split_dir("val"))
        synth_test  = SyntheticDatasetParser(split_dir("test"))
        synth_train = SyntheticDatasetParser(split_dir("train"))
        log.info("Synthetic parsers ready (file-based).")

        # ------------------------------------------------------------ train-size loop
        results_by_trainsize = {}   # {train_key: results}

        for n_train in cfg.TRAIN_SAMPLE_COUNTS:
            train_key = f"train_{n_train}"
            sub_dir = run_dir / train_key
            log.info(f"\n{'=' * 70}")
            log.info(f"  Train size: {n_train} samples/heliostat  →  {train_key}")
            log.info(f"{'=' * 70}")

            # Build a mapping trimmed to n_train samples — the parser uses this to
            # decide how many files to load per heliostat.
            train_map_n = _equalize_mapping(_filter_heliostats(_build_mapping("train"), cfg.HELIOSTAT_IDS), n_train)

            results = run(
                scenario_path=cfg.SCENARIO_PATH,
                device=device,
                ddp_setup=ddp_setup,
                train_mapping=train_map_n,
                val_mapping=val_map,
                test_mapping=test_map,
                train_parser=synth_train,
                val_parser=synth_val,
                test_parser=synth_test,
                optimization_config=cfg.OPTIMIZATION_CONFIG,
                output_dir=sub_dir,
                n_surface_pts=cfg.SURFACE_POINTS_PER_FACET,
                train_rays=cfg.TRAIN_RAYS,
                perturbations=perturbations,
                heliostat_ids=cfg.HELIOSTAT_IDS,
                reconstructor_class=WortbergKinematicReconstructor,
            )
            results_by_trainsize[train_key] = results

            gc.collect()
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ per-run reporting
    log.info("Generating per-run reports …")
    for n_train in cfg.TRAIN_SAMPLE_COUNTS:
        train_key = f"train_{n_train}"
        sub_dir = run_dir / train_key
        _run_reporting(results_by_trainsize[train_key], perturbations_json, sub_dir)

    # ------------------------------------------------------------------ cross-train-size reporting
    log.info("Generating train-size comparison reports …")
    write_ablation_summary(results_by_trainsize, run_dir)
    plot_ablation_comparison(results_by_trainsize, run_dir)

    histories = {}
    for n_train in cfg.TRAIN_SAMPLE_COUNTS:
        hf = run_dir / f"train_{n_train}" / "convergence_history.json"
        if hf.exists():
            with open(hf) as f:
                histories[f"train_{n_train}"] = json.load(f)
    if histories:
        plot_combined_convergence(histories, results_by_trainsize, run_dir)

    log.info(f"Done. Results in: {run_dir}")


if __name__ == "__main__":
    main()
