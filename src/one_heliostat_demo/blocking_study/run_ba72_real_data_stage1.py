"""Stage 1 (forward-aim loss) for BA72 on REAL PAINT data -- run ONCE, the base
for a real-data blocking-on/off Stage 2 comparison (to follow).

Unlike the synthetic-occlusion experiments (generate_occlusion_dataset.py),
this uses the REAL recorded flux images and REAL calibration properties from
the field-wide 50/20/20 benchmark (`benchmark_split-balanced_train-50_validation-20`,
the same split BA72's synthetic sun positions were pooled from), honored
verbatim via train.py's existing `_load_fixed_split_real` (DATA_MODE="real" +
USE_FIXED_SPLIT=True) -- no synthetic dataset, no injected perturbation, no
`generate_occlusion_dataset.py` involved at all.

Stage 1 never ray-traces or touches blocking (ForwardAimLoss is a purely
kinematic/geometric comparison against the recorded motors + real centroid),
so this checkpoint is the shared starting point for whatever real-data
Stage-2 blocking-on/off comparison comes next.

Uses the SAME neighbourhood scenario as the synthetic experiment
(scenarios/neighbourhoods/BA72/scenario.h5, AZ70/AZ71 present) so a later
blocking-aware Stage 2 run can reuse this checkpoint directly.

Usage
-----
    python run_ba72_real_data_stage1.py
"""

from __future__ import annotations

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
HELIOSTAT_ID = "BA72"
BENCHMARK_NAME = "benchmark_split-balanced_train-50_validation-20"
PAINT_DIR = _ROOT / "datasets" / "paint"
SCENARIO_PATH = _ROOT / "scenarios" / "neighbourhoods" / HELIOSTAT_ID / "scenario.h5"
OUTPUT_DIR = _ROOT / "outputs" / "new_mapping_function" / "ba72_real_data" / "stage1_only"


def main() -> None:
    set_logger_config()
    import logging
    logging.getLogger().setLevel(logging.INFO)

    if not SCENARIO_PATH.exists():
        raise FileNotFoundError(f"{SCENARIO_PATH} missing -- run build_neighbourhood_scenario.py {HELIOSTAT_ID} first.")

    cfg.DATA_MODE = "real"
    cfg.USE_FIXED_SPLIT = True
    cfg.BENCHMARK_NAME = BENCHMARK_NAME
    cfg.BENCHMARK_CSV = PAINT_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
    cfg.CALIBRATION_DIR = PAINT_DIR / BENCHMARK_NAME / "calibration_properties"
    cfg.REAL_FLUX_DIR = PAINT_DIR / BENCHMARK_NAME / "flux_image"
    cfg.BASE_LR = cfg.BASE_LR / 2.0  # halved, per request (was 1e-4)
    cfg.STAGE1_EPOCHS = cfg.STAGE1_EPOCHS * 2  # doubled, per request (was 100)

    device = get_device()
    # dataset_dir is unused in DATA_MODE="real" (train.py reads CALIBRATION_DIR/
    # REAL_FLUX_DIR instead) -- pass a placeholder, matching main.py's own handling.
    dummy_dataset_dir = _ROOT / "datasets" / "synthetic" / "unused_real_mode"
    results = tr.run(
        heliostat_id=HELIOSTAT_ID,
        dataset_dir=dummy_dataset_dir,
        output_dir=OUTPUT_DIR,
        cfg=cfg,
        device=device,
        skip_stage2=True,
        scenario_path=SCENARIO_PATH,
    )

    print()
    print("=" * 70)
    print(f"Stage 1 (REAL DATA) done for {HELIOSTAT_ID}, benchmark={BENCHMARK_NAME}")
    print(f"  BASE_LR={cfg.BASE_LR:.2e}  STAGE1_EPOCHS={cfg.STAGE1_EPOCHS}")
    print(f"  pre-training  centroid mrad: mean={results['pre_training']['centroid_mrad_mean']:.3f}"
          f"  median={results['pre_training']['centroid_mrad_median']:.3f}")
    print(f"  after Stage 1 centroid mrad: mean={results['after_stage1']['centroid_mrad_mean']:.3f}"
          f"  median={results['after_stage1']['centroid_mrad_median']:.3f}")
    print(f"  n_train={results['n_train']}  n_val={results['n_val']}  n_test={results['n_test']}"
          f"  total_time_min={results['total_time_min']:.1f}")
    print(f"  checkpoint -> {OUTPUT_DIR / 'stage1_checkpoint.pt'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
