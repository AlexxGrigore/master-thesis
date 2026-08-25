"""Stage 1 (forward-aim loss) for the receiver-occlusion synthetic experiment -- run ONCE.

Stage 1 never ray-traces or touches blocking, so one run on the receiver-occlusion
dataset (generate_receiver_occlusion_dataset.py) is shared as the starting point
for all 5 Stage-2 arms (focal_spot/contour x blocking off/on-receiver/on-horizontal).

Usage
-----
    python run_receiver_occlusion_stage1.py BE25
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("heliostat_id")
    args = parser.parse_args()
    heliostat_id = args.heliostat_id

    set_logger_config()
    import logging
    logging.getLogger().setLevel(logging.INFO)

    occlusion_root = _ROOT / "datasets" / "synthetic" / f"{heliostat_id.lower()}_receiver_occlusion"
    dataset_dir = occlusion_root / "dataset"
    scenario_path = _ROOT / "scenarios" / "neighbourhoods" / heliostat_id / "scenario.h5"
    output_dir = (
        _ROOT / "outputs" / "new_mapping_function" / f"{heliostat_id.lower()}_receiver_occlusion"
        / "stage1_only"
    )

    if not dataset_dir.exists():
        raise FileNotFoundError(f"{dataset_dir} missing -- run generate_receiver_occlusion_dataset.py {heliostat_id} first.")
    if not scenario_path.exists():
        raise FileNotFoundError(f"{scenario_path} missing -- run build_neighbourhood_scenario.py {heliostat_id} first.")

    cfg.DATA_MODE = "synthetic"
    cfg.USE_FIXED_SPLIT = True

    device = get_device()
    results = tr.run(
        heliostat_id=heliostat_id,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        cfg=cfg,
        device=device,
        skip_stage2=True,
        scenario_path=scenario_path,
    )

    print()
    print("=" * 70)
    print(f"Stage 1 done for {heliostat_id} (receiver-occlusion synthetic dataset)")
    print(f"  pre-training  centroid mrad: mean={results['pre_training']['centroid_mrad_mean']:.3f}"
          f"  median={results['pre_training']['centroid_mrad_median']:.3f}")
    print(f"  after Stage 1 centroid mrad: mean={results['after_stage1']['centroid_mrad_mean']:.3f}"
          f"  median={results['after_stage1']['centroid_mrad_median']:.3f}")
    print(f"  n_test={results['n_test']}  total_time_min={results['total_time_min']:.1f}")
    print(f"  checkpoint -> {output_dir / 'stage1_checkpoint.pt'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
