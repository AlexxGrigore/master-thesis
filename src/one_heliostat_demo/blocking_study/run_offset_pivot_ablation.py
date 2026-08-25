"""Ablation: does unfreezing actuator offset (c_i) / pivot radius (r_i) in Stage 2
actually help, on real data, starting from the existing full-field ideal-surface
Stage-1 checkpoints (outputs/full_field_1277/stage1_only_ideal_surfaces/).

Stage 1 (GEOMETRIC_INIT_ORIENTATION_ONLY=True, the project default used to build
those checkpoints) hardcodes the WHOLE non_optimizable_parameters group to LR=0,
so offset/pivot_radius are frozen there regardless of the OPTIMIZE_* flags -- the
flags only have real effect in Stage 2 (STAGE2_PARAM_SET="all", ray-traced).
Hence this ablation runs Stage 2 twice per heliostat from the SAME existing
Stage-1 checkpoint (no Stage-1 retraining):

  frozen   : OPTIMIZE_ACTUATOR_OFFSET=False, OPTIMIZE_PIVOT_RADIUS=False
  unfrozen : both True (== current project default, unmodified)

Real data, blocking off, focal_spot loss, 200 epochs, default lr -- everything
else at project defaults.

Usage
-----
    python run_offset_pivot_ablation.py AH33 --arm frozen
    python run_offset_pivot_ablation.py AH33 --arm unfrozen
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
BENCHMARK_NAME = "benchmark_split-balanced_train-50_validation-20"
PAINT_DIR = _ROOT / "datasets" / "paint"
CKPT_ROOT = _ROOT / "outputs" / "full_field_1277" / "stage1_only_ideal_surfaces"
SCEN_ROOT = _ROOT / "scenarios" / "full_field_one_heliostat_scenarios" / "ideal"

ARMS = {
    "frozen":   dict(offset=False, pivot=False),
    "unfrozen": dict(offset=True,  pivot=True),
}
STAGE2_EPOCHS = 200
SURFACE_POINTS_PER_FACET = 25
TRAIN_RAYS = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("heliostat_id")
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    args = parser.parse_args()
    heliostat_id = args.heliostat_id
    spec = ARMS[args.arm]

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)

    scenario_path = SCEN_ROOT / heliostat_id / "scenario_ideal.h5"
    stage1_checkpoint = CKPT_ROOT / heliostat_id / "stage1_checkpoint.pt"
    output_dir = (
        _ROOT / "outputs" / "new_mapping_function" / "offset_pivot_ablation"
        / heliostat_id / args.arm
    )
    if not scenario_path.exists():
        raise FileNotFoundError(f"{scenario_path} missing.")
    if not stage1_checkpoint.exists():
        raise FileNotFoundError(f"{stage1_checkpoint} missing.")

    cfg.DATA_MODE = "real"
    cfg.USE_FIXED_SPLIT = True
    cfg.BENCHMARK_NAME = BENCHMARK_NAME
    cfg.BENCHMARK_CSV = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
    cfg.CALIBRATION_DIR = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
    cfg.REAL_FLUX_DIR = PAINT_DIR / BENCHMARK_NAME / "flux_image"
    cfg.SURFACE_POINTS_PER_FACET = SURFACE_POINTS_PER_FACET
    cfg.TRAIN_RAYS = TRAIN_RAYS
    cfg.STAGE2_EPOCHS = STAGE2_EPOCHS
    cfg.STAGE2_LOSS = "focal_spot"
    cfg.OPTIMIZE_ACTUATOR_OFFSET = spec["offset"]
    cfg.OPTIMIZE_PIVOT_RADIUS = spec["pivot"]

    device = get_device()
    dummy_dataset_dir = _ROOT / "datasets" / "synthetic" / "unused_real_mode"
    log.info(f"=== {heliostat_id} arm '{args.arm}': OPTIMIZE_ACTUATOR_OFFSET={spec['offset']} "
             f"OPTIMIZE_PIVOT_RADIUS={spec['pivot']} ===")
    results = tr.run(
        heliostat_id=heliostat_id,
        dataset_dir=dummy_dataset_dir,
        output_dir=output_dir,
        cfg=cfg,
        device=device,
        skip_stage2=False,
        stage1_checkpoint=stage1_checkpoint,
        scenario_path=scenario_path,
        blocking=False,
    )
    (output_dir / "arm_results.json").write_text(json.dumps(results, indent=2, default=str))

    def _line(label: str, ev: dict) -> str:
        return (f"  {label:<20} centroid mrad: mean={ev['centroid_mrad_mean']:8.3f}  "
                f"median={ev['centroid_mrad_median']:8.3f}  |  "
                f"direction mrad: mean={ev['direction_mrad_mean']:8.3f}  "
                f"median={ev['direction_mrad_median']:8.3f}")

    print()
    print("=" * 90)
    print(f"{heliostat_id} arm '{args.arm}' (offset={spec['offset']}, pivot={spec['pivot']})")
    print("-" * 90)
    print(_line("After Stage 1", results["after_stage1"]))
    print(_line("After Stage 2", results["after_stage2"]))
    print("=" * 90)


if __name__ == "__main__":
    main()
