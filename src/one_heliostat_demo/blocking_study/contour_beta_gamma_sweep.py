"""Stage 2 of the contour-loss retuning plan: sweep (beta, gamma) with terms
normalized (Step 2.0's "strongly preferred" option), on 2 already-available
occlusion scenarios spanning severity, scored on the DIRECTION metric
(test split) -- this project's established primary comparison measure.

Scenarios (both already generated this session, blocking ON with matched
pose, reusing existing Stage-1 checkpoints -- no new data, no Stage-1
retraining):
  BA72 tilt_000        -- heavy occlusion, ~18.4% mean blocked
  BE25 receiver-occlusion -- mild occlusion, ~9.6% mean blocked

Normalization: coarse_scale=1200, gravity_scale=0.1, representative
magnitudes read off this project's own prior contour runs (raw coarse
~800-1700, raw gravity ~0.03-0.4 m). With these, beta/gamma become genuine
0-1 mixing weights instead of magnitudes ~1e4 apart.

The CURRENT unnormalized default (beta=1e-4, gamma=0.3, scale=1.0, i.e. a
no-op) is the baseline arm for both scenarios -- already computed earlier
this session (run_ba72_occlusion_stage2_contour_comparison.py's blocking_on
arm, and run_receiver_occlusion_stage2_arms.py's contour_receiver arm), so
this script only trains the NEW normalized (beta, gamma) points:
  A: (0.15, 0.15) -- mild coarse+gravity assist, fine still dominant (0.70)
  B: (0.30, 0.30) -- balanced three-way mix (fine=0.40)
  C: (0.10, 0.40) -- gravity-leaning (tests whether leaning into Gravity,
     which is structurally a focal-spot-style centroid term computed on the
     contour pixels, helps -- the parallel the user pointed out directly)

Usage
-----
    python contour_beta_gamma_sweep.py --scenario ba72
    python contour_beta_gamma_sweep.py --scenario be25
    python contour_beta_gamma_sweep.py --scenario ba72 --point A
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

COARSE_SCALE = 1200.0
GRAVITY_SCALE = 0.1

GRID_POINTS = {
    "A": (0.15, 0.15),
    "B": (0.30, 0.30),
    "C": (0.10, 0.40),
}

SCENARIOS = {
    "ba72": dict(
        heliostat_id="BA72",
        dataset_dir=_ROOT / "datasets" / "synthetic" / "ba72_occlusion" / "tilt_000" / "dataset",
        scenario_path=_ROOT / "scenarios" / "neighbourhoods" / "BA72" / "scenario.h5",
        stage1_checkpoint=_ROOT / "outputs" / "new_mapping_function" / "ba72_occlusion"
        / "stage1_only" / "stage1_checkpoint.pt",
        blocking_fixed_tilt=(["AZ70", "AZ71"], 0.0),
        blocker_target_name=None,
    ),
    "be25": dict(
        heliostat_id="BE25",
        dataset_dir=_ROOT / "datasets" / "synthetic" / "be25_receiver_occlusion" / "dataset",
        scenario_path=_ROOT / "scenarios" / "neighbourhoods" / "BE25" / "scenario.h5",
        stage1_checkpoint=_ROOT / "outputs" / "new_mapping_function" / "be25_receiver_occlusion"
        / "stage1_only" / "stage1_checkpoint.pt",
        blocking_fixed_tilt=None,
        blocker_target_name="receiver",
    ),
}

SURFACE_POINTS_PER_FACET = 25
TRAIN_RAYS = 10
STAGE2_EPOCHS = 500
BASE_LR = 2e-5


def run_point(scenario_key: str, point_key: str) -> dict:
    sc = SCENARIOS[scenario_key]
    beta, gamma = GRID_POINTS[point_key]
    output_dir = (
        _ROOT / "outputs" / "new_mapping_function" / "contour_hp_sweep" / "stage2_beta_gamma"
        / scenario_key / point_key
    )
    log.info(f"=== {scenario_key} / point {point_key}: beta={beta} gamma={gamma} "
             f"(normalized, coarse_scale={COARSE_SCALE}, gravity_scale={GRAVITY_SCALE}) ===")

    cfg.DATA_MODE = "synthetic"
    cfg.USE_FIXED_SPLIT = True
    cfg.SURFACE_POINTS_PER_FACET = SURFACE_POINTS_PER_FACET
    cfg.TRAIN_RAYS = TRAIN_RAYS
    cfg.STAGE2_EPOCHS = STAGE2_EPOCHS
    cfg.STAGE2_LOSS = "contour"
    cfg.BASE_LR = BASE_LR
    cfg.MINI_BATCH_SIZE = 1
    cfg.CONTOUR_BETA = beta
    cfg.CONTOUR_GAMMA = gamma
    cfg.CONTOUR_COARSE_SCALE = COARSE_SCALE
    cfg.CONTOUR_GRAVITY_SCALE = GRAVITY_SCALE
    cfg.BLOCKER_TARGET_NAME = sc["blocker_target_name"]

    device = get_device()
    results = tr.run(
        heliostat_id=sc["heliostat_id"],
        dataset_dir=sc["dataset_dir"],
        output_dir=output_dir,
        cfg=cfg,
        device=device,
        skip_stage2=False,
        stage1_checkpoint=sc["stage1_checkpoint"],
        scenario_path=sc["scenario_path"],
        blocking=True,
        blocking_fixed_tilt=sc["blocking_fixed_tilt"],
    )
    (output_dir / "arm_results.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    parser.add_argument("--point", choices=sorted(GRID_POINTS), default=None,
                         help="single grid point; omit to run all 3 (A, B, C) sequentially")
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)

    sc = SCENARIOS[args.scenario]
    if not sc["dataset_dir"].exists():
        raise FileNotFoundError(f"{sc['dataset_dir']} missing.")
    if not sc["stage1_checkpoint"].exists():
        raise FileNotFoundError(f"{sc['stage1_checkpoint']} missing.")

    points = [args.point] if args.point else sorted(GRID_POINTS)
    for pk in points:
        results = run_point(args.scenario, pk)
        s2 = results["after_stage2"]
        print(f"{args.scenario}/{pk} (beta={GRID_POINTS[pk][0]}, gamma={GRID_POINTS[pk][1]}): "
              f"dir mean={s2['direction_mrad_mean']:.3f} median={s2['direction_mrad_median']:.3f}  "
              f"cen mean={s2['centroid_mrad_mean']:.3f} median={s2['centroid_mrad_median']:.3f}")


if __name__ == "__main__":
    main()
