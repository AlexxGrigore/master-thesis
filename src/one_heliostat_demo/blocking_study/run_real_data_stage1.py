"""Stage 1 (forward-aim loss) on REAL PAINT data, for any heliostat -- generalizes
run_ba72_real_data_stage1.py so the same recipe can be reused (e.g. for BE25 as a
second heliostat for the calibrated-sunshape contour-loss comparison).

Real recorded flux images and calibration properties from the field-wide 50/20/20
benchmark (`benchmark_split-balanced_train-50_validation-20`), honored verbatim via
train.py's `_load_fixed_split_real` (DATA_MODE="real" + USE_FIXED_SPLIT=True).

Uses the heliostat's own neighbourhood scenario (scenarios/neighbourhoods/<ID>/scenario.h5)
so a later blocking-aware Stage 2 run can reuse this checkpoint directly, even though
the immediate use is blocking=False.

Usage
-----
    python run_real_data_stage1.py BE25
    python run_real_data_stage1.py BA72
"""

from __future__ import annotations

import argparse
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

_ROOT = _here.parents[2]  # master-thesis/
BENCHMARK_NAME = "benchmark_split-balanced_train-50_validation-20"
PAINT_DIR = _ROOT / "datasets" / "paint"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("heliostat_id")
    args = parser.parse_args()
    heliostat_id = args.heliostat_id

    set_logger_config()
    import logging
    logging.getLogger().setLevel(logging.INFO)

    scenario_path = _ROOT / "scenarios" / "neighbourhoods" / heliostat_id / "scenario.h5"
    output_dir = _ROOT / "outputs" / "new_mapping_function" / f"{heliostat_id.lower()}_real_data" / "stage1_only"
    if not scenario_path.exists():
        raise FileNotFoundError(f"{scenario_path} missing -- run build_neighbourhood_scenario.py {heliostat_id} first.")

    cfg.DATA_MODE = "real"
    cfg.USE_FIXED_SPLIT = True
    cfg.BENCHMARK_NAME = BENCHMARK_NAME
    cfg.BENCHMARK_CSV = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
    cfg.CALIBRATION_DIR = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
    cfg.REAL_FLUX_DIR = PAINT_DIR / BENCHMARK_NAME / "flux_image"
    cfg.BASE_LR = cfg.BASE_LR / 2.0  # halved -- validated on BA72 as genuine noise floor, not an artifact
    cfg.STAGE1_EPOCHS = cfg.STAGE1_EPOCHS * 2  # doubled, same validated recipe

    device = get_device()
    dummy_dataset_dir = _ROOT / "datasets" / "synthetic" / "unused_real_mode"
    results = tr.run(
        heliostat_id=heliostat_id,
        dataset_dir=dummy_dataset_dir,
        output_dir=output_dir,
        cfg=cfg,
        device=device,
        skip_stage2=True,
        scenario_path=scenario_path,
    )

    print()
    print("=" * 70)
    print(f"Stage 1 (REAL DATA) done for {heliostat_id}, benchmark={BENCHMARK_NAME}")
    print(f"  BASE_LR={cfg.BASE_LR:.2e}  STAGE1_EPOCHS={cfg.STAGE1_EPOCHS}")
    print(f"  pre-training  centroid mrad: mean={results['pre_training']['centroid_mrad_mean']:.3f}"
          f"  median={results['pre_training']['centroid_mrad_median']:.3f}")
    print(f"  after Stage 1 centroid mrad: mean={results['after_stage1']['centroid_mrad_mean']:.3f}"
          f"  median={results['after_stage1']['centroid_mrad_median']:.3f}")
    print(f"  n_train={results['n_train']}  n_val={results['n_val']}  n_test={results['n_test']}"
          f"  total_time_min={results['total_time_min']:.1f}")
    print(f"  checkpoint -> {output_dir / 'stage1_checkpoint.pt'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
