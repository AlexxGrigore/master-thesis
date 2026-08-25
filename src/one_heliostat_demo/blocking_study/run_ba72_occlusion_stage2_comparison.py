"""Stage 2 (focal-spot loss), blocking OFF vs ON, on one BA72 occlusion dataset.

Both arms start from the SAME Stage 1 checkpoint (run_ba72_occlusion_stage1.py)
and train on the SAME --tilt-level dataset -- the training targets
(focal_spot_enu) were ray-traced WITH AZ70/AZ71 held at that dataset's own
fixed tilt (0.0/0.3333/0.6667/1.0 for tilt_000/033/067/100 respectively).

  Arm A ("blocking off"): Stage 2's own forward pass ignores blocking entirely
      -- it matches a blocked ground-truth centroid with an unblocked model, a
      structural mismatch the optimizer can only partially absorb into other
      parameters.
  Arm B ("blocking on"): Stage 2's forward pass re-derives the SAME occlusion
      geometry live (AZ70/AZ71 held at the SAME fixed tilt, via the
      `blocking_fixed_tilt` param) -- the model matches the process that
      generated the targets.

Question: does matching the occlusion model to reality (arm B) recover better
true kinematic parameters than ignoring it (arm A) -- and does the answer
change as the occlusion level itself drops toward zero?

MINI_BATCH_SIZE is forced to 1 for BOTH arms (blocking=True requires it; arm A
is set to match, so batch-size isn't a confound in the comparison).

Usage
-----
    python run_ba72_occlusion_stage2_comparison.py --tilt-level 000
    python run_ba72_occlusion_stage2_comparison.py --tilt-level 033
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
HELIOSTAT_ID = "BA72"
BLOCKER_NAMES = ["AZ70", "AZ71"]
TILT_LEVELS = {"000": 0.0, "033": 1.0 / 3.0, "067": 2.0 / 3.0, "100": 1.0}
OCCLUSION_ROOT = _ROOT / "datasets" / "synthetic" / "ba72_occlusion"
SCENARIO_PATH = _ROOT / "scenarios" / "neighbourhoods" / HELIOSTAT_ID / "scenario.h5"
STAGE1_CHECKPOINT = (
    _ROOT / "outputs" / "new_mapping_function" / "ba72_occlusion" / "stage1_only"
    / "stage1_checkpoint.pt"
)

SURFACE_POINTS_PER_FACET = 50
TRAIN_RAYS = 10
STAGE2_EPOCHS = 200
# Higher-resolution rerun (50x50 pts/facet, vs the original 25x25) -- kept in its
# own output tree so the original 25x25 comparison results are not overwritten.
COMPARISON_DIR_NAME = "stage2_blocking_comparison_sp50"


def run_arm(tilt_level: str, blocking: bool) -> dict:
    generation_tilt = TILT_LEVELS[tilt_level]
    dataset_dir = OCCLUSION_ROOT / f"tilt_{tilt_level}" / "dataset"
    output_root = (
        _ROOT / "outputs" / "new_mapping_function" / "ba72_occlusion"
        / COMPARISON_DIR_NAME / f"tilt_{tilt_level}"
    )
    label = "blocking_on" if blocking else "blocking_off"
    output_dir = output_root / label
    log.info(f"=== Stage 2 arm: tilt_{tilt_level} / {label} ===")

    cfg.DATA_MODE = "synthetic"
    cfg.USE_FIXED_SPLIT = True
    cfg.SURFACE_POINTS_PER_FACET = SURFACE_POINTS_PER_FACET
    cfg.TRAIN_RAYS = TRAIN_RAYS
    cfg.STAGE2_EPOCHS = STAGE2_EPOCHS
    cfg.STAGE2_LOSS = "focal_spot"
    cfg.MINI_BATCH_SIZE = 1  # forced for blocking=True anyway; matched here for arm A too

    device = get_device()
    results = tr.run(
        heliostat_id=HELIOSTAT_ID,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        cfg=cfg,
        device=device,
        skip_stage2=False,
        stage1_checkpoint=STAGE1_CHECKPOINT,
        scenario_path=SCENARIO_PATH,
        blocking=blocking,
        blocking_fixed_tilt=(BLOCKER_NAMES, generation_tilt) if blocking else None,
    )
    (output_dir / "arm_results.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tilt-level", choices=sorted(TILT_LEVELS), required=True)
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)

    dataset_dir = OCCLUSION_ROOT / f"tilt_{args.tilt_level}" / "dataset"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"{dataset_dir} missing -- run generate_occlusion_dataset.py first.")
    if not STAGE1_CHECKPOINT.exists():
        raise FileNotFoundError(f"{STAGE1_CHECKPOINT} missing -- run run_ba72_occlusion_stage1.py first.")

    results_off = run_arm(args.tilt_level, blocking=False)
    results_on = run_arm(args.tilt_level, blocking=True)

    def _line(label: str, ev: dict) -> str:
        return (f"  {label:<28} centroid mrad: mean={ev['centroid_mrad_mean']:8.3f}  "
                f"median={ev['centroid_mrad_median']:8.3f}")

    print()
    print("=" * 90)
    print(f"BA72 Stage 2 (focal_spot), tilt_{args.tilt_level} dataset")
    print(f"{SURFACE_POINTS_PER_FACET}x{SURFACE_POINTS_PER_FACET} pts/facet, {TRAIN_RAYS} rays/pt, "
          f"{STAGE2_EPOCHS} epochs, both arms from the same Stage-1 checkpoint")
    print("-" * 90)
    print(_line("Pre-training (shared)", results_off["pre_training"]))
    print(_line("After Stage 1 (shared)", results_off["after_stage1"]))
    print(_line("After Stage 2, blocking OFF", results_off["after_stage2"]))
    print(_line("After Stage 2, blocking ON", results_on["after_stage2"]))
    print("=" * 90)


if __name__ == "__main__":
    main()
