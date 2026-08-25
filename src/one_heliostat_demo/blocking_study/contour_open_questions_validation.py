"""Validation runs for the contour-loss artifact's "What's still open" items 3-6:

  3: retest real-data BA72/BE25 with the FULL tuned config (tau=0.70,
     beta=0.10/gamma=0.40 -- these were only ever tested SEPARATELY before,
     never combined)
  4: band-widening (CONTOUR_BAND_SIGMA), on top of the tuned config, at
     heavy occlusion where the noisy-gradient issue was diagnosed
  5: HybridFocalContourLoss (STAGE2_LOSS="hybrid"), both occlusion scenarios
  6: guardrail deliberate-divergence validation (artificially high LR)

Usage
-----
    python contour_open_questions_validation.py --mode real_ba72
    python contour_open_questions_validation.py --mode real_be25
    python contour_open_questions_validation.py --mode heavy_tuned
    python contour_open_questions_validation.py --mode heavy_band
    python contour_open_questions_validation.py --mode heavy_hybrid
    python contour_open_questions_validation.py --mode mild_hybrid
    python contour_open_questions_validation.py --mode guardrail_divergence
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

_here = pathlib.Path(__file__).resolve().parent          # blocking_study/
_src = _here.parents[1]                                   # src/
_sh = _src / "one_heliostat_demo" / "single_heliostat"    # single_heliostat/
for _p in (str(_src), str(_sh), str(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from artist.util import get_device, set_logger_config  # noqa: E402

import config as cfg  # noqa: E402
import train as tr  # noqa: E402

log = logging.getLogger(__name__)

_ROOT = _here.parents[2]  # master-thesis/
PAINT_DIR = _ROOT / "datasets" / "paint"
BENCHMARK_NAME = "benchmark_split-balanced_train-50_validation-20"

# The FULL tuned config -- never tested combined before this script.
TAU_TUNED = 0.70
ETA_TUNED = 70.0
BETA_TUNED = 0.10
GAMMA_TUNED = 0.40
COARSE_SCALE = 1200.0
GRAVITY_SCALE = 0.1
BA72_SUNSHAPE_STD_MRAD = 3.36

BA72 = dict(
    heliostat_id="BA72",
    dataset_dir=_ROOT / "datasets" / "synthetic" / "ba72_occlusion" / "tilt_000" / "dataset",
    scenario_path=_ROOT / "scenarios" / "neighbourhoods" / "BA72" / "scenario.h5",
    stage1_checkpoint=_ROOT / "outputs" / "new_mapping_function" / "ba72_occlusion"
    / "stage1_only" / "stage1_checkpoint.pt",
    blocking_fixed_tilt=(["AZ70", "AZ71"], 0.0),
    blocker_target_name=None,
)
BE25 = dict(
    heliostat_id="BE25",
    dataset_dir=_ROOT / "datasets" / "synthetic" / "be25_receiver_occlusion" / "dataset",
    scenario_path=_ROOT / "scenarios" / "neighbourhoods" / "BE25" / "scenario.h5",
    stage1_checkpoint=_ROOT / "outputs" / "new_mapping_function" / "be25_receiver_occlusion"
    / "stage1_only" / "stage1_checkpoint.pt",
    blocking_fixed_tilt=None,
    blocker_target_name="receiver",
)

OUT_ROOT = _ROOT / "outputs" / "new_mapping_function" / "contour_open_questions"


def _print_result(label: str, results: dict) -> None:
    s2 = results["after_stage2"]
    print(f"{label}: dir mean={s2['direction_mrad_mean']:.3f} median={s2['direction_mrad_median']:.3f}  "
          f"cen mean={s2['centroid_mrad_mean']:.3f} median={s2['centroid_mrad_median']:.3f}")


def run_real(heliostat_id: str) -> dict:
    """#3: real data, calibrated sunshape, FULL tuned config, blocking off."""
    output_dir = OUT_ROOT / f"real_{heliostat_id.lower()}_tuned"
    scenario_path = _ROOT / "scenarios" / "neighbourhoods" / heliostat_id / "scenario.h5"
    stage1_checkpoint = (
        _ROOT / "outputs" / "new_mapping_function" / f"{heliostat_id.lower()}_real_data"
        / "stage1_only" / "stage1_checkpoint.pt"
    )
    cfg.DATA_MODE = "real"
    cfg.USE_FIXED_SPLIT = True
    cfg.BENCHMARK_NAME = BENCHMARK_NAME
    cfg.BENCHMARK_CSV = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
    cfg.CALIBRATION_DIR = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
    cfg.REAL_FLUX_DIR = PAINT_DIR / BENCHMARK_NAME / "flux_image"
    cfg.SURFACE_POINTS_PER_FACET = 25
    cfg.TRAIN_RAYS = 10
    cfg.STAGE2_EPOCHS = 500
    cfg.STAGE2_LOSS = "contour"
    cfg.BASE_LR = 2e-5
    cfg.CONTOUR_TAU = TAU_TUNED
    cfg.CONTOUR_ETA = ETA_TUNED
    cfg.CONTOUR_BETA = BETA_TUNED
    cfg.CONTOUR_GAMMA = GAMMA_TUNED
    cfg.CONTOUR_COARSE_SCALE = COARSE_SCALE
    cfg.CONTOUR_GRAVITY_SCALE = GRAVITY_SCALE

    device = get_device()
    dummy_dataset_dir = _ROOT / "datasets" / "synthetic" / "unused_real_mode"
    results = tr.run(
        heliostat_id=heliostat_id, dataset_dir=dummy_dataset_dir, output_dir=output_dir,
        cfg=cfg, device=device, skip_stage2=False, stage1_checkpoint=stage1_checkpoint,
        scenario_path=scenario_path, blocking=False,
        sunshape_std_mrad=BA72_SUNSHAPE_STD_MRAD,
    )
    (output_dir / "arm_results.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def run_synthetic(sc: dict, label: str, output_dir: pathlib.Path, *,
                   band_sigma: float = 0.0, hybrid: bool = False,
                   focal_weight: float = 0.5, focal_scale: float = 0.02,
                   lr_override: float | None = None) -> dict:
    cfg.DATA_MODE = "synthetic"
    cfg.USE_FIXED_SPLIT = True
    cfg.SURFACE_POINTS_PER_FACET = 25
    cfg.TRAIN_RAYS = 10
    cfg.STAGE2_EPOCHS = 500
    cfg.STAGE2_LOSS = "hybrid" if hybrid else "contour"
    cfg.BASE_LR = lr_override if lr_override is not None else 2e-5
    cfg.MINI_BATCH_SIZE = 1
    cfg.CONTOUR_TAU = TAU_TUNED
    cfg.CONTOUR_ETA = ETA_TUNED
    cfg.CONTOUR_BETA = BETA_TUNED
    cfg.CONTOUR_GAMMA = GAMMA_TUNED
    cfg.CONTOUR_COARSE_SCALE = COARSE_SCALE
    cfg.CONTOUR_GRAVITY_SCALE = GRAVITY_SCALE
    cfg.CONTOUR_BAND_SIGMA = band_sigma
    cfg.HYBRID_FOCAL_WEIGHT = focal_weight
    cfg.HYBRID_FOCAL_SCALE = focal_scale
    cfg.BLOCKER_TARGET_NAME = sc["blocker_target_name"]

    device = get_device()
    log.info(f"=== {label} ===")
    results = tr.run(
        heliostat_id=sc["heliostat_id"], dataset_dir=sc["dataset_dir"], output_dir=output_dir,
        cfg=cfg, device=device, skip_stage2=False, stage1_checkpoint=sc["stage1_checkpoint"],
        scenario_path=sc["scenario_path"], blocking=True,
        blocking_fixed_tilt=sc["blocking_fixed_tilt"],
    )
    (output_dir / "arm_results.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=[
        "real_ba72", "real_be25", "heavy_tuned", "heavy_band", "heavy_hybrid",
        "mild_hybrid", "guardrail_divergence",
    ])
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)

    if args.mode == "real_ba72":
        results = run_real("BA72")
    elif args.mode == "real_be25":
        results = run_real("BE25")
    elif args.mode == "heavy_tuned":
        results = run_synthetic(BA72, "BA72 heavy, tuned (tau=0.70, beta=0.10/gamma=0.40)",
                                 OUT_ROOT / "heavy_tuned")
    elif args.mode == "heavy_band":
        results = run_synthetic(BA72, "BA72 heavy, tuned + band_sigma=2.0",
                                 OUT_ROOT / "heavy_band", band_sigma=2.0)
    elif args.mode == "heavy_hybrid":
        results = run_synthetic(BA72, "BA72 heavy, hybrid (focal_weight=0.5)",
                                 OUT_ROOT / "heavy_hybrid", hybrid=True)
    elif args.mode == "mild_hybrid":
        results = run_synthetic(BE25, "BE25 mild, hybrid (focal_weight=0.5)",
                                 OUT_ROOT / "mild_hybrid", hybrid=True)
    elif args.mode == "guardrail_divergence":
        # Deliberately unstable LR (50x the normal 2e-5) on BE25, which never
        # trips at the normal LR -- confirms the guardrail catches AND
        # recovers a genuine forced divergence, not just BA72's chronic
        # near-threshold oscillation.
        results = run_synthetic(BE25, "BE25, deliberate divergence (lr=1e-3)",
                                 OUT_ROOT / "guardrail_divergence", lr_override=1e-3)

    _print_result(args.mode, results)


if __name__ == "__main__":
    main()
