"""Stage 2 (contour loss), blocking OFF, calibrated sunshape -- for any heliostat.

Starts from that heliostat's own real-data Stage-1 checkpoint
(run_real_data_stage1.py), trains Stage 2 with WortbergContourLoss
(`--stage2-loss contour`, i.e. cfg.STAGE2_LOSS="contour") on the real 50/20/20
benchmark split, blocking disabled, and the sunshape overridden to a
calibrated std (default 3.36 mrad, from ba72_sunshape_ideal_aim_calibration.py's
bbox-threshold method on BA72 -- reused as-is for other heliostats unless a
per-heliostat value is given).

Learning rate: contour-loss gradients are much hotter than focal_spot's
(memory: one epoch at the focal_spot-tuned BASE_LR=1e-4 kicked val mrad up
5-8 mrad in earlier AY36/37/39 runs), so BASE_LR is cut to 2e-5 (5x lower)
here. The guardrail (trip to ForwardAimLoss + optimizer-state reset, already
validated) is the safety net under that regardless.

Usage
-----
    python run_stage2_contour_calibrated.py BA72
    python run_stage2_contour_calibrated.py BE25 --epochs 500 --lr 2e-5 --sunshape-std 3.36
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

SURFACE_POINTS_PER_FACET = 25
TRAIN_RAYS = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("heliostat_id")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--sunshape-std", type=float, default=3.36,
                         help="sunshape std in mrad (default: BA72's calibrated value)")
    args = parser.parse_args()
    heliostat_id = args.heliostat_id

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)

    scenario_path = _ROOT / "scenarios" / "neighbourhoods" / heliostat_id / "scenario.h5"
    stage1_checkpoint = (
        _ROOT / "outputs" / "new_mapping_function" / f"{heliostat_id.lower()}_real_data"
        / "stage1_only" / "stage1_checkpoint.pt"
    )
    output_dir = (
        _ROOT / "outputs" / "new_mapping_function" / f"{heliostat_id.lower()}_real_data"
        / "stage2_contour_calibrated_sunshape"
    )
    if not scenario_path.exists():
        raise FileNotFoundError(f"{scenario_path} missing.")
    if not stage1_checkpoint.exists():
        raise FileNotFoundError(f"{stage1_checkpoint} missing -- run run_real_data_stage1.py {heliostat_id} first.")

    cfg.DATA_MODE = "real"
    cfg.USE_FIXED_SPLIT = True
    cfg.BENCHMARK_NAME = BENCHMARK_NAME
    cfg.BENCHMARK_CSV = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
    cfg.CALIBRATION_DIR = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
    cfg.REAL_FLUX_DIR = PAINT_DIR / BENCHMARK_NAME / "flux_image"
    cfg.SURFACE_POINTS_PER_FACET = SURFACE_POINTS_PER_FACET
    cfg.TRAIN_RAYS = TRAIN_RAYS
    cfg.STAGE2_EPOCHS = args.epochs
    cfg.STAGE2_LOSS = "contour"
    cfg.BASE_LR = args.lr

    device = get_device()
    dummy_dataset_dir = _ROOT / "datasets" / "synthetic" / "unused_real_mode"
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
        sunshape_std_mrad=args.sunshape_std,
    )
    (output_dir / "run_results.json").write_text(json.dumps(results, indent=2, default=str))

    def _line(label: str, ev: dict) -> str:
        return (f"  {label:<24} centroid mrad: mean={ev['centroid_mrad_mean']:8.3f}  "
                f"median={ev['centroid_mrad_median']:8.3f}  |  "
                f"direction mrad: mean={ev['direction_mrad_mean']:8.3f}  "
                f"median={ev['direction_mrad_median']:8.3f}")

    print()
    print("=" * 100)
    print(f"{heliostat_id} Stage 2 (contour loss), REAL DATA, benchmark={BENCHMARK_NAME}")
    print(f"{SURFACE_POINTS_PER_FACET}x{SURFACE_POINTS_PER_FACET} pts/facet, {TRAIN_RAYS} rays/pt, "
          f"{args.epochs} epochs, lr={args.lr:.1e}, sunshape std={args.sunshape_std} mrad, blocking=False")
    print("-" * 100)
    print(_line("Pre-training", results["pre_training"]))
    print(_line("After Stage 1", results["after_stage1"]))
    print(_line("After Stage 2 (contour)", results["after_stage2"]))
    print("=" * 100)


if __name__ == "__main__":
    main()
