"""Stage 2, 5 arms, on the receiver-occlusion synthetic dataset.

All 5 arms train from the SAME Stage-1 checkpoint
(run_receiver_occlusion_stage1.py) and the SAME dataset
(generate_receiver_occlusion_dataset.py, blocking on, neighbours aimed at the
receiver during generation):

  focal_off        : focal_spot loss, blocking OFF
  focal_receiver    : focal_spot loss, blocking ON, neighbours aimed at receiver
                       (matches generation exactly)
  focal_horizontal  : focal_spot loss, blocking ON, neighbours held horizontal
                       (mismatched pose -- blocking mechanism engaged but with
                       an assumption that produces ~no real occlusion)
  contour_off       : contour loss, blocking OFF
  contour_receiver  : contour loss, blocking ON, neighbours aimed at receiver
                       (matches generation exactly)

Neighbour list for the fixed-horizontal override and the receiver aim (via
cfg.BLOCKER_TARGET_NAME) comes from the scenario's own blockers.json.

Usage
-----
    python run_receiver_occlusion_stage2_arms.py BE25 --arm focal_off
    python run_receiver_occlusion_stage2_arms.py BE25 --arm contour_receiver --epochs 500 --lr 2e-5
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

ARMS = {
    "focal_off":       dict(loss="focal_spot", blocking=False, mode="none"),
    "focal_receiver":  dict(loss="focal_spot", blocking=True,  mode="receiver"),
    "focal_horizontal":dict(loss="focal_spot", blocking=True,  mode="horizontal"),
    "contour_off":     dict(loss="contour",    blocking=False, mode="none"),
    "contour_receiver":dict(loss="contour",    blocking=True,  mode="receiver"),
}

DEFAULT_EPOCHS = {"focal_spot": 200, "contour": 500}
DEFAULT_LR = {"focal_spot": None, "contour": 2e-5}  # None -> cfg default (1e-4)

SURFACE_POINTS_PER_FACET = 25
TRAIN_RAYS = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("heliostat_id")
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()
    heliostat_id = args.heliostat_id
    spec = ARMS[args.arm]

    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)

    occlusion_root = _ROOT / "datasets" / "synthetic" / f"{heliostat_id.lower()}_receiver_occlusion"
    dataset_dir = occlusion_root / "dataset"
    scenario_path = _ROOT / "scenarios" / "neighbourhoods" / heliostat_id / "scenario.h5"
    stage1_checkpoint = (
        _ROOT / "outputs" / "new_mapping_function" / f"{heliostat_id.lower()}_receiver_occlusion"
        / "stage1_only" / "stage1_checkpoint.pt"
    )
    output_dir = (
        _ROOT / "outputs" / "new_mapping_function" / f"{heliostat_id.lower()}_receiver_occlusion"
        / "stage2_arms" / args.arm
    )
    blockers_file = scenario_path.parent / "blockers.json"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"{dataset_dir} missing -- run generate_receiver_occlusion_dataset.py {heliostat_id} first.")
    if not stage1_checkpoint.exists():
        raise FileNotFoundError(f"{stage1_checkpoint} missing -- run run_receiver_occlusion_stage1.py {heliostat_id} first.")

    blocker_names = json.loads(blockers_file.read_text())["blockers"] if blockers_file.exists() else []

    epochs = args.epochs or DEFAULT_EPOCHS[spec["loss"]]
    lr = args.lr if args.lr is not None else DEFAULT_LR[spec["loss"]]

    cfg.DATA_MODE = "synthetic"
    cfg.USE_FIXED_SPLIT = True
    cfg.SURFACE_POINTS_PER_FACET = SURFACE_POINTS_PER_FACET
    cfg.TRAIN_RAYS = TRAIN_RAYS
    cfg.STAGE2_EPOCHS = epochs
    cfg.STAGE2_LOSS = spec["loss"]
    if lr is not None:
        cfg.BASE_LR = lr
    cfg.BLOCKER_TARGET_NAME = "receiver" if spec["mode"] == "receiver" else None

    blocking_fixed_tilt = (blocker_names, 1.0) if spec["mode"] == "horizontal" else None

    device = get_device()
    log.info(f"=== {heliostat_id} arm '{args.arm}': loss={spec['loss']} blocking={spec['blocking']} "
             f"mode={spec['mode']} epochs={epochs} lr={cfg.BASE_LR:.1e} "
             f"blockers={blocker_names if spec['mode'] != 'none' else '(n/a)'} ===")

    results = tr.run(
        heliostat_id=heliostat_id,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        cfg=cfg,
        device=device,
        skip_stage2=False,
        stage1_checkpoint=stage1_checkpoint,
        scenario_path=scenario_path,
        blocking=spec["blocking"],
        blocking_fixed_tilt=blocking_fixed_tilt,
    )
    (output_dir / "arm_results.json").write_text(json.dumps(results, indent=2, default=str))

    def _line(label: str, ev: dict) -> str:
        return (f"  {label:<24} centroid mrad: mean={ev['centroid_mrad_mean']:8.3f}  "
                f"median={ev['centroid_mrad_median']:8.3f}  |  "
                f"direction mrad: mean={ev['direction_mrad_mean']:8.3f}  "
                f"median={ev['direction_mrad_median']:8.3f}")

    print()
    print("=" * 100)
    print(f"{heliostat_id} arm '{args.arm}' ({spec['loss']}, blocking={spec['blocking']}, mode={spec['mode']})")
    print(f"{SURFACE_POINTS_PER_FACET}x{SURFACE_POINTS_PER_FACET} pts/facet, {TRAIN_RAYS} rays/pt, "
          f"{epochs} epochs, lr={cfg.BASE_LR:.1e}")
    print("-" * 100)
    print(_line("Pre-training", results["pre_training"]))
    print(_line("After Stage 1", results["after_stage1"]))
    print(_line(f"After Stage 2 ({args.arm})", results["after_stage2"]))
    print("=" * 100)


if __name__ == "__main__":
    main()
