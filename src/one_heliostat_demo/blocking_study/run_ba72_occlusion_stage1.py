"""Stage 1 (forward-aim loss) for the BA72 occlusion experiment -- run ONCE.

Stage 1 (`ForwardAimLoss`) never ray-traces or touches blocking; its loss is a
purely kinematic/geometric comparison between the forward mirror normal at the
recorded motors and the sun-centroid bisector normal. Since the 4 occlusion
datasets (`generate_occlusion_dataset.py`, tilt 0/33/67/100 %) share identical
`incident_ray_direction` / `motor_position` for every sample (only the flux and
its centroid shift with blocking), one Stage-1 run is representative of all 4;
its checkpoint is reused as the Stage-2 starting point for each tilt level.

Uses the UNBLOCKED (tilt_100) dataset as the Stage-1 source: its focal-spot
centroid is the least biased by blocking, so it's the cleanest ground truth to
align against.

Honors the dataset's own fixed 50/20/20 split verbatim (no DatasetSplitter
re-pooling) via the new `USE_FIXED_SPLIT` + non-real `DATA_MODE` path in
train.py (`_load_fixed_split_synthetic`).

Usage
-----
    python run_ba72_occlusion_stage1.py
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
OCCLUSION_ROOT = _ROOT / "datasets" / "synthetic" / "ba72_occlusion"
STAGE1_SOURCE_TILT_DIR = OCCLUSION_ROOT / "tilt_100" / "dataset"
SCENARIO_PATH = _ROOT / "scenarios" / "neighbourhoods" / HELIOSTAT_ID / "scenario.h5"
OUTPUT_DIR = (
    _ROOT / "outputs" / "new_mapping_function" / "ba72_occlusion" / "stage1_only"
)


def main() -> None:
    set_logger_config()
    import logging
    logging.getLogger().setLevel(logging.INFO)

    if not STAGE1_SOURCE_TILT_DIR.exists():
        raise FileNotFoundError(
            f"{STAGE1_SOURCE_TILT_DIR} does not exist -- run generate_occlusion_dataset.py first."
        )
    if not SCENARIO_PATH.exists():
        raise FileNotFoundError(
            f"{SCENARIO_PATH} does not exist -- run build_neighbourhood_scenario.py {HELIOSTAT_ID} first."
        )

    cfg.DATA_MODE = "synthetic"
    cfg.USE_FIXED_SPLIT = True

    device = get_device()
    results = tr.run(
        heliostat_id=HELIOSTAT_ID,
        dataset_dir=STAGE1_SOURCE_TILT_DIR,
        output_dir=OUTPUT_DIR,
        cfg=cfg,
        device=device,
        skip_stage2=True,
        scenario_path=SCENARIO_PATH,
    )

    print()
    print("=" * 70)
    print(f"Stage 1 done for {HELIOSTAT_ID} (source: {STAGE1_SOURCE_TILT_DIR.parent.name})")
    print(f"  pre-training  centroid mrad: mean={results['pre_training']['centroid_mrad_mean']:.3f}"
          f"  median={results['pre_training']['centroid_mrad_median']:.3f}")
    print(f"  after Stage 1 centroid mrad: mean={results['after_stage1']['centroid_mrad_mean']:.3f}"
          f"  median={results['after_stage1']['centroid_mrad_median']:.3f}")
    print(f"  n_test={results['n_test']}  total_time_min={results['total_time_min']:.1f}")
    print(f"  checkpoint -> {OUTPUT_DIR / 'stage1_checkpoint.pt'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
