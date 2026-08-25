"""Stage 2 (focal-spot loss) on REAL PAINT data, blocking OFF vs ON, for BA72.

Both arms start from the real-data Stage 1 checkpoint (run_ba72_real_data_stage1.py)
and train on the real 50/20/20 benchmark split (real recorded flux images and
calibration properties -- no synthetic data at all).

  Arm A ("blocking off"): Stage 2's forward pass ignores AZ70/AZ71 entirely.
  Arm B ("blocking on"):  Stage 2's forward pass injects AZ70/AZ71 as live
      blocking primitives, aimed at the RECEIVER target (index 3, world centre
      ~(-0.003, -0.003, 54.99) in the scenario's tower-local frame -- the
      highest of the 4 named target areas, distinct from BA72's own recorded
      per-sample target) via ``cfg.BLOCKER_TARGET_NAME = "receiver"``. This is
      Experiment F's existing override plumbing in train.py
      (`_blocker_target_override` -> `target_index_override`, threaded through
      both `aimed_neighbour_surfaces` and `forward_pass_blocking`) -- no new
      code needed, just the config flag.

Unlike the synthetic occlusion experiments, there is no fixed-tilt override
here: the neighbours are aimed normally (at the receiver) under each sample's
real sun direction, so blocking reflects whatever real occlusion geometry
that produces, not a controlled/fixed level.

Usage
-----
    python run_ba72_real_data_stage2_comparison.py
"""

from __future__ import annotations

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
HELIOSTAT_ID = "BA72"
BENCHMARK_NAME = "benchmark_split-balanced_train-50_validation-20"
PAINT_DIR = _ROOT / "datasets" / "paint"
SCENARIO_PATH = _ROOT / "scenarios" / "neighbourhoods" / HELIOSTAT_ID / "scenario.h5"
STAGE1_CHECKPOINT = (
    _ROOT / "outputs" / "new_mapping_function" / "ba72_real_data" / "stage1_only"
    / "stage1_checkpoint.pt"
)
OUTPUT_ROOT = _ROOT / "outputs" / "new_mapping_function" / "ba72_real_data" / "stage2_blocking_comparison"
BLOCKER_TARGET_NAME = "receiver"

SURFACE_POINTS_PER_FACET = 25
TRAIN_RAYS = 10
STAGE2_EPOCHS = 200


def run_arm(blocking: bool) -> dict:
    label = "blocking_on" if blocking else "blocking_off"
    output_dir = OUTPUT_ROOT / label
    log.info(f"=== Stage 2 arm (REAL DATA): {label} ===")

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
    cfg.MINI_BATCH_SIZE = 1  # forced for blocking=True anyway; matched here for arm A too
    cfg.BLOCKER_TARGET_NAME = BLOCKER_TARGET_NAME if blocking else None

    device = get_device()
    dummy_dataset_dir = _ROOT / "datasets" / "synthetic" / "unused_real_mode"
    results = tr.run(
        heliostat_id=HELIOSTAT_ID,
        dataset_dir=dummy_dataset_dir,
        output_dir=output_dir,
        cfg=cfg,
        device=device,
        skip_stage2=False,
        stage1_checkpoint=STAGE1_CHECKPOINT,
        scenario_path=SCENARIO_PATH,
        blocking=blocking,
    )
    (output_dir / "arm_results.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def main() -> None:
    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)

    if not STAGE1_CHECKPOINT.exists():
        raise FileNotFoundError(f"{STAGE1_CHECKPOINT} missing -- run run_ba72_real_data_stage1.py first.")

    results_off = run_arm(blocking=False)
    results_on = run_arm(blocking=True)

    def _line(label: str, ev: dict) -> str:
        return (f"  {label:<28} centroid mrad: mean={ev['centroid_mrad_mean']:8.3f}  "
                f"median={ev['centroid_mrad_median']:8.3f}  |  "
                f"direction mrad: mean={ev['direction_mrad_mean']:8.3f}  "
                f"median={ev['direction_mrad_median']:8.3f}")

    print()
    print("=" * 100)
    print(f"BA72 Stage 2 (focal_spot), REAL DATA, benchmark={BENCHMARK_NAME}")
    print(f"{SURFACE_POINTS_PER_FACET}x{SURFACE_POINTS_PER_FACET} pts/facet, {TRAIN_RAYS} rays/pt, "
          f"{STAGE2_EPOCHS} epochs, both arms from the same real-data Stage-1 checkpoint")
    print(f"Blocking-ON neighbours (AZ70/AZ71) aimed at target '{BLOCKER_TARGET_NAME}'")
    print("-" * 100)
    print(_line("Pre-training (shared)", results_off["pre_training"]))
    print(_line("After Stage 1 (shared)", results_off["after_stage1"]))
    print(_line("After Stage 2, blocking OFF", results_off["after_stage2"]))
    print(_line("After Stage 2, blocking ON", results_on["after_stage2"]))
    print("=" * 100)


if __name__ == "__main__":
    main()
