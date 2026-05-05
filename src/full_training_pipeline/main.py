"""
Entry point for the full training pipeline.

Usage
-----
    python main.py
    python main.py --dataset-type synthetic --loss-type focal_spot
    python main.py --on-daic --epochs 50
    python main.py --smoke-test
"""
from __future__ import annotations

import argparse
import pathlib
import sys

_pkg = pathlib.Path(__file__).parent
_src = _pkg.parent
sys.path.insert(0, str(_src))

from artist.util import set_logger_config

from full_training_pipeline.config import build_default_config
from full_training_pipeline.train import run

set_logger_config()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fine-error-learning residual pipeline.")
    parser.add_argument("--on-daic", action="store_true", help="Use DAIC path defaults.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a minimal local smoke configuration.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the default epoch count.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=None, help="Optional output directory.")
    parser.add_argument(
        "--dataset-type",
        choices=["real", "synthetic"],
        default="real",
        help="'real' uses PAINT calibration images; 'synthetic' uses pre-generated data.",
    )
    parser.add_argument(
        "--loss-type",
        choices=["focal_spot", "pixel", "alignment"],
        default="focal_spot",
        help="Loss function to use during training.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = build_default_config(
        is_on_daic=args.on_daic,
        smoke_test=args.smoke_test,
        max_epochs=args.epochs,
        output_dir=args.output_dir,
        dataset_type=args.dataset_type,
        loss_type=args.loss_type,
    )
    run(config)


if __name__ == "__main__":
    main()
