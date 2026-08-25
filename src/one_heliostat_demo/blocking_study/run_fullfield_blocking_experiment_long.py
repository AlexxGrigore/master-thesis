"""Long-training focal-spot variant of Experiment F.

Same dataset/scenario as run_fullfield_blocking_experiment.py, but:
  - Stage 1: 500 epochs
  - Stage 2: 600 epochs for BOTH arms (raised from 300; B1 needed the extra
    epochs to converge — MINI_BATCH_SIZE=1 under blocking carries far less
    signal per epoch than B0's batch-of-25 — and B0 is rerun at the same
    budget so the comparison isn't confounded by unequal training epochs).
  - make_plots = False (no GIFs/trail snapshots/flux animations)
  - outputs go to experiment_full_field_long/{hid}/B0_blocking_off_long etc.

Usage
-----
    python run_fullfield_blocking_experiment_long.py AY36
"""
from __future__ import annotations

import logging
import pathlib
import sys
import time

_here = pathlib.Path(__file__).resolve().parent
_src = _here.parents[1]
_sh = _src / "one_heliostat_demo" / "single_heliostat"
for _p in (str(_src), str(_sh), str(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")
import torch  # noqa: E402

import config as cfg  # noqa: E402
import train as tr  # noqa: E402
from artist.util import set_logger_config  # noqa: E402

log = logging.getLogger(__name__)

_ROOT = _here.parents[2]
DATASET_DIR = _ROOT / "datasets" / "synthetic" / "fullfield_blocking_dataset" / "dataset"
SCENARIO_PATH = _ROOT / "scenarios" / "neighbourhoods_fullfield" / "AY36" / "scenario.h5"
OUT_ROOT = _ROOT / "outputs" / "new_mapping_function" / "blocking_study" / "experiment_full_field_long"
BLOCKER_TARGET_NAME = "solar_tower_juelich_lower"

ARMS = {"B0_blocking_off_long": False, "B1_blocking_on_long": True}
STAGE2_EPOCHS = {"B0_blocking_off_long": 600, "B1_blocking_on_long": 600}


def run_arms(hid: str, device: torch.device) -> None:
    if not SCENARIO_PATH.exists():
        raise FileNotFoundError(f"Missing scenario: {SCENARIO_PATH}")
    if not (DATASET_DIR / "train" / hid).exists():
        raise FileNotFoundError(f"Missing dataset for {hid} under {DATASET_DIR}")

    cfg.DATA_MODE = "synthetic"
    cfg.STAGE1_EPOCHS = 500
    cfg.STAGE1_TRAIL_PLOTS = False
    cfg.GEOMETRIC_INIT = True
    cfg.STAGE1_REDUCTION = "soft_l1"
    cfg.STAGE2_REDUCTION = "soft_l1"
    cfg.STAGE2_LOSS = "focal_spot"
    cfg.DISPLAY_RAYS = 10
    cfg.BLOCKER_TARGET_NAME = BLOCKER_TARGET_NAME

    for arm_name, blocking in ARMS.items():
        out_dir = OUT_ROOT / hid / arm_name
        if (out_dir / "results.json").exists():
            log.info(f"[SKIP] {hid} {arm_name} — results.json exists")
            continue
        cfg.MINI_BATCH_SIZE = 1 if blocking else 25
        cfg.PLOT_EVERY = 5 if blocking else 1
        cfg.STAGE2_EPOCHS = STAGE2_EPOCHS[arm_name]
        log.info(f"=== {hid} — {arm_name} (Stage1 500 / Stage2 {cfg.STAGE2_EPOCHS}, no plots) ===")
        t0 = time.time()
        tr.run(
            hid,
            DATASET_DIR,
            out_dir,
            cfg,
            device,
            scenario_path=SCENARIO_PATH,
            blocking=blocking,
            make_plots=False,
        )
        log.info(f"    done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


def main() -> None:
    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("artist", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    device = torch.device("cpu")
    run_arms("AY36", device)


if __name__ == "__main__":
    main()
