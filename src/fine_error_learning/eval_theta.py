"""
Evaluate a saved per-heliostat 24-D parameter vector on a data split.

Companion to evaluate.py: that module evaluates θ_KR + Δθ from a trained FEL
model; this one evaluates a plain parameter vector (no model) — used for the
stage-2 (FocalSpotLoss per-heliostat optimization) baseline, whose final
parameters are stored as kinematic_parameters.json in a run_all output dir.

    python fine_error_learning/eval_theta.py \
        --params-dir ../outputs/fine_error_learning/all63_stage2_synth \
        --split test

Writes <params-dir>/eval_theta_<split>.json and prints the field summary.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

import torch

_here = pathlib.Path(__file__).resolve().parent   # fine_error_learning/
_src = _here.parent                               # src/
sys.path.insert(0, str(_src))

from artist.util import get_device, set_logger_config

from fine_error_learning import config as cfg
from fine_error_learning import data as fel_data
from fine_error_learning import warm_start as fel_warm_start
from fine_error_learning.evaluate import _mrad_under_theta

log = logging.getLogger(__name__)


def theta_from_kinematic_parameters(kp: dict) -> torch.Tensor:
    """kinematic_parameters.json dict → 24-D vector in the pipeline ordering."""
    return torch.tensor(
        kp["rotation_dev_rad"]          # 4
        + kp["translation_dev_m"]       # 9
        + kp["actuator_angle_dev_rad"]  # 2
        + kp["actuator_stroke_dev_m"]   # 2
        + kp["actuator_offset_dev_m"]   # 2
        + kp["pivot_radius_dev_m"]      # 2
        + kp["base_position_dev_m"],    # 3
        dtype=torch.float32,
    )


def reconstruct_absolute_theta(
    kp: dict, kp_stage1: dict, theta_kr: torch.Tensor
) -> torch.Tensor:
    """24-D absolute parameter vector from a run's kinematic_parameters.json.

    rotation/translation/base entries are deviation tensors — the JSON matches
    the checkpoint convention directly. The actuator groups (angle, stroke,
    offset, pivot) are stored in the JSON as deviations from nominal while the
    24-D pipeline vector holds ABSOLUTE values, so the nominal is reconstructed
    from the stage-1 run: nominal = θ_KR (stage-1 checkpoint) − stage-1 dev.
    """
    dev = theta_from_kinematic_parameters(kp)
    dev_s1 = theta_from_kinematic_parameters(kp_stage1)
    theta = dev.clone()
    actuator = slice(13, 21)  # angle 2 + stroke 2 + offset 2 + pivot 2
    theta[actuator] = theta_kr[actuator].cpu() - dev_s1[actuator] + dev[actuator]
    return theta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--params-dir", type=pathlib.Path, required=True,
                   help="run_all output dir with <HID>/kinematic_parameters.json")
    p.add_argument("--stage1-params-dir", type=pathlib.Path, default=None,
                   help="Stage-1 run dir (nominal actuator reconstruction; "
                        "default: cfg.STAGE1_CHECKPOINT_DIR)")
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--data-dir", type=pathlib.Path, default=None,
                   help="Override cfg.SYNTHETIC_DATA_DIR")
    args = p.parse_args()

    if args.data_dir is not None:
        cfg.SYNTHETIC_DATA_DIR = args.data_dir
    stage1_dir = pathlib.Path(args.stage1_params_dir or cfg.STAGE1_CHECKPOINT_DIR)

    set_logger_config()
    device = get_device()

    rows = []
    for hel_dir in sorted(args.params_dir.iterdir()):
        kp_path = hel_dir / "kinematic_parameters.json"
        kp_s1_path = stage1_dir / hel_dir.name / "kinematic_parameters.json"
        if not kp_path.exists() or not kp_s1_path.exists():
            continue
        hid = hel_dir.name
        state = fel_warm_start.load_warm_start_state(hid, cfg, device)
        measurements = fel_data.load_measurements(
            pathlib.Path(cfg.SYNTHETIC_DATA_DIR) / args.split, hid,
            state.heliostat_group, state.scenario, device,
        )
        if measurements is None:
            log.warning(f"  {hid}: no {args.split} data — skipped")
            continue
        theta = reconstruct_absolute_theta(
            json.load(open(kp_path)),
            json.load(open(kp_s1_path)),
            state.theta_kr,
        ).to(device)
        mean, median = _mrad_under_theta(state, theta, measurements, 25, device)
        rows.append({"heliostat_id": hid, "mrad_mean": mean, "mrad_median": median})
        log.info(f"  {hid}: {mean:.3f} mrad (mean, {args.split})")

    if not rows:
        raise RuntimeError(f"No heliostats evaluated under {args.params_dir}")

    aggregate = {
        "mrad_mean": sum(r["mrad_mean"] for r in rows) / len(rows),
        "mrad_median": sorted(r["mrad_median"] for r in rows)[len(rows) // 2],
        "n_heliostats": len(rows),
        "split": args.split,
    }
    out = args.params_dir / f"eval_theta_{args.split}.json"
    with open(out, "w") as f:
        json.dump({"aggregate": aggregate, "per_heliostat": rows}, f, indent=2)
    print(
        f"eval_theta ({args.split}): mean {aggregate['mrad_mean']:.3f} mrad, "
        f"median {aggregate['mrad_median']:.3f} mrad over {len(rows)} heliostats"
    )


if __name__ == "__main__":
    main()
