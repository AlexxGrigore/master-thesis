"""
Entry point for the full training pipeline.

Edit config.py to set parameters, then run:
    python main.py
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

from full_training_pipeline.config import build_config
from full_training_pipeline.train import run

set_logger_config()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fine-error-learning residual pipeline.")
    parser.add_argument("--smoke-test", action="store_true", help="Run 3 epochs as a quick sanity check.")
    parser.add_argument("--dataset-type", choices=["real", "synthetic"], default=None,
                        help="Override the DATASET_TYPE setting in config.py.")
    parser.add_argument("--daic", action="store_true",
                        help="Use DAIC cluster paths (overrides IS_ON_DAIC in config.py).")
    args = parser.parse_args()

    config = build_config(smoke_test=args.smoke_test, dataset_type=args.dataset_type, is_on_daic=args.daic)
    run(config)


if __name__ == "__main__":
    main()
