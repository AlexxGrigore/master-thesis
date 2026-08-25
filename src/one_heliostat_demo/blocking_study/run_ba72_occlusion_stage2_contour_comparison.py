"""Stage 2 (CONTOUR loss), blocking OFF vs ON, on the BA72 occlusion datasets.

Sibling of run_ba72_occlusion_stage2_comparison.py (which does the same
comparison with focal_spot loss) -- reuses the exact same datasets, Stage-1
checkpoint, and fixed-tilt blocking mechanism, but with STAGE2_LOSS="contour"
and the established contour training recipe (500 epochs, lr=2e-5, since
contour gradients are much hotter than focal_spot's).

Purpose: the earlier synthetic BE25 receiver-occlusion experiment (mild
occlusion, ~9.6% mean blocked) found contour loss converges cleanly but still
lands well behind focal_spot on both metrics. This tests whether that gap
shrinks or reverses at BA72's tilt_000 dataset, which has much heavier
occlusion (~18.4% mean blocked, up to 41.7% max) -- i.e. whether contour's
"discard the corrupted lower region" trade-off actually pays off once there
is a lot more to be robust to. 25x25 pts/facet, 10 rays/pt (NOT the 50x50
rerun) to match the ORIGINAL focal_spot tilt_000 comparison
(stage2_blocking_comparison/, not the _sp50 variant) for a true apples-to-
apples comparison at the same mesh resolution.

Usage
-----
    python run_ba72_occlusion_stage2_contour_comparison.py --tilt-level 000
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

SURFACE_POINTS_PER_FACET = 25
TRAIN_RAYS = 10
STAGE2_EPOCHS = 500
BASE_LR = 2e-5
COMPARISON_DIR_NAME = "stage2_contour_comparison"


def run_arm(tilt_level: str, blocking: bool) -> dict:
    generation_tilt = TILT_LEVELS[tilt_level]
    dataset_dir = OCCLUSION_ROOT / f"tilt_{tilt_level}" / "dataset"
    output_root = (
        _ROOT / "outputs" / "new_mapping_function" / "ba72_occlusion"
        / COMPARISON_DIR_NAME / f"tilt_{tilt_level}"
    )
    label = "blocking_on" if blocking else "blocking_off"
    output_dir = output_root / label
    log.info(f"=== Stage 2 (contour) arm: tilt_{tilt_level} / {label} ===")

    cfg.DATA_MODE = "synthetic"
    cfg.USE_FIXED_SPLIT = True
    cfg.SURFACE_POINTS_PER_FACET = SURFACE_POINTS_PER_FACET
    cfg.TRAIN_RAYS = TRAIN_RAYS
    cfg.STAGE2_EPOCHS = STAGE2_EPOCHS
    cfg.STAGE2_LOSS = "contour"
    cfg.BASE_LR = BASE_LR
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
                f"median={ev['centroid_mrad_median']:8.3f}  |  "
                f"direction mrad: mean={ev['direction_mrad_mean']:8.3f}  "
                f"median={ev['direction_mrad_median']:8.3f}")

    print()
    print("=" * 100)
    print(f"BA72 Stage 2 (contour), tilt_{args.tilt_level} dataset")
    print(f"{SURFACE_POINTS_PER_FACET}x{SURFACE_POINTS_PER_FACET} pts/facet, {TRAIN_RAYS} rays/pt, "
          f"{STAGE2_EPOCHS} epochs, lr={BASE_LR:.1e}, both arms from the same Stage-1 checkpoint")
    print("-" * 100)
    print(_line("Pre-training (shared)", results_off["pre_training"]))
    print(_line("After Stage 1 (shared)", results_off["after_stage1"]))
    print(_line("After Stage 2, blocking OFF", results_off["after_stage2"]))
    print(_line("After Stage 2, blocking ON", results_on["after_stage2"]))
    print("=" * 100)


if __name__ == "__main__":
    main()
