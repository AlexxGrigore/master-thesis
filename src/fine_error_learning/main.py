"""
Fine error learning — main entry point.

Trains one shared transformer residual model that predicts a 24-D kinematic
correction Δθ per heliostat on top of the stage-1 warm start θ_KR, end-to-end
through the ARTIST ray tracer.

Usage
-----
    python fine_error_learning/main.py                  # full run (config defaults)
    python fine_error_learning/main.py --smoke-test     # tiny local sanity check
    python fine_error_learning/main.py --heliostats AB43 AB33 --epochs 10
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from datetime import datetime

_here = pathlib.Path(__file__).resolve().parent   # fine_error_learning/
_src = _here.parent                               # src/
sys.path.insert(0, str(_src))

from artist.util import get_device, set_logger_config

from fine_error_learning import config as cfg
from fine_error_learning import train as fel_train

# Smoke-test heliostats: AB43 and AB33 — both on-target under the synthetic
# stage-1 warm start (measured by the warm-start on-target diagnostic).
SMOKE_HELIOSTATS = ["AB43", "AB33"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine error learning: shared Δθ transformer.")
    p.add_argument("--run-name", default=None,
                   help="Run name (default: fel_<timestamp>). Outputs go to "
                        "outputs/fine_error_learning/<run_name>/")
    p.add_argument("--heliostats", nargs="+", default=None,
                   help="Heliostat IDs (default: all with a stage-1 checkpoint)")
    p.add_argument("--epochs", type=int, default=None, help="Override cfg.EPOCHS")
    p.add_argument("--warm-start", choices=["stage1", "nominal"], default=None,
                   help="Warm-start mode (overrides cfg.WARM_START)")
    p.add_argument("--no-flux", dest="use_flux", action="store_false", default=None,
                   help="Images-off ablation: skip the CNN, zero image features")
    p.add_argument("--bounded-head", action="store_true", default=None,
                   help="tanh × bounds output head instead of the unbounded zero-init head")
    p.add_argument("--pixel-loss", type=float, default=None, metavar="RATIO",
                   help="Experiment A: auxiliary pixelwise flux-distribution loss; "
                        "RATIO is the target L_pixel/L_centroid fraction at the warm "
                        "start (e.g. 0.1). Auto-calibrated on the first mini-batch.")
    p.add_argument("--query-decoder", action="store_true",
                   help="Experiment B: query-conditioned decoder — per-measurement "
                        "Δθ(sun position) via cross-attention instead of one static Δθ")
    p.add_argument("--output-gain", type=float, default=None,
                   help="Override cfg.OUTPUT_GAIN (head output scaling)")
    p.add_argument("--daic", action="store_true",
                   help="Use DAIC cluster paths: repo at /home/nfs/agrigore/..., "
                        "synthetic dataset under /tudelft.net/... (mirrors "
                        "run_all.py --daic)")
    p.add_argument("--smoke-test", action="store_true",
                   help="Tiny run: 2 heliostats, 2 epochs, 4 rays, K=8, small model")
    p.add_argument("--evaluate", type=pathlib.Path, default=None, metavar="RUN_DIR",
                   help="Skip training; evaluate an existing run directory "
                        "(fel_model_best.pt + scaler_stats.json) before/after Δθ.")
    p.add_argument("--eval-split", choices=["train", "val", "test"], default="test",
                   help="Split for --evaluate (default: test)")
    p.add_argument("--checkpoint-dir", type=pathlib.Path, default=None,
                   help="Override cfg.STAGE1_CHECKPOINT_DIR (e.g. held-out stage-1 "
                        "checkpoints for generalization evaluation)")
    p.add_argument("--data-dir", type=pathlib.Path, default=None,
                   help="Override cfg.SYNTHETIC_DATA_DIR (e.g. the held-out dataset)")
    p.add_argument("--scenario-template", default=None, metavar="PATH_TEMPLATE",
                   help="Override cfg.SCENARIO_PATH_TEMPLATE, a str.format template "
                        "with a {heliostat_id} placeholder (e.g. held-out scenarios)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.daic:
        cfg.BASE_DIR = pathlib.Path(
            "/home/nfs/agrigore/projects/githubProjects/master-thesis"
        )
        paint = pathlib.Path(
            "/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint"
        )
        cfg.SCENARIO_PATH_TEMPLATE = str(
            cfg.BASE_DIR / "scenarios" / "one_heliostat_scenarios"
            / "{heliostat_id}" / "scenario.h5"
        )
        cfg.SYNTHETIC_DATA_DIR = paint / "synthetic" / "balanced_dataset" / "dataset"
        cfg.STAGE1_CHECKPOINT_DIR = (
            cfg.BASE_DIR / "outputs" / "fine_error_learning" / "all63_stage1_synth"
        )
        cfg.OUTPUT_ROOT = cfg.BASE_DIR / "outputs" / "fine_error_learning"

    # Evaluation-only mode: no training, no new run directory.
    if args.evaluate is not None:
        from fine_error_learning import evaluate as fel_eval

        if args.checkpoint_dir is not None:
            cfg.STAGE1_CHECKPOINT_DIR = args.checkpoint_dir
        if args.data_dir is not None:
            cfg.SYNTHETIC_DATA_DIR = args.data_dir
        if args.scenario_template is not None:
            cfg.SCENARIO_PATH_TEMPLATE = args.scenario_template
        set_logger_config()
        device = get_device()
        results = fel_eval.evaluate_run(
            cfg=cfg, device=device, run_dir=args.evaluate, split=args.eval_split,
            heliostat_ids=args.heliostats,
        )
        agg = results["aggregate"]
        print(
            f"Eval ({agg['split']}): {agg['mrad_before_mean']:.3f} → "
            f"{agg['mrad_after_mean']:.3f} mrad "
            f"(mean over {agg['n_heliostats']} heliostats)"
        )
        return

    if args.heliostats is not None:
        cfg.HELIOSTAT_IDS = args.heliostats
    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    if args.warm_start is not None:
        cfg.WARM_START = args.warm_start
    if args.use_flux is not None:
        cfg.USE_FLUX = args.use_flux
    if args.bounded_head is not None and args.bounded_head:
        cfg.BOUNDED_HEAD = True
    if args.pixel_loss is not None:
        cfg.PIXEL_LOSS_RATIO = args.pixel_loss
    if args.query_decoder:
        cfg.QUERY_DECODER = True
    if args.output_gain is not None:
        cfg.OUTPUT_GAIN = args.output_gain

    if args.smoke_test:
        cfg.HELIOSTAT_IDS = SMOKE_HELIOSTATS
        if args.epochs is None:
            cfg.EPOCHS = 2
        cfg.TRAIN_RAYS = 4
        cfg.VAL_RAYS = 4
        cfg.ON_TARGET_RAYS = 4
        cfg.K_TOKENS = 8
        cfg.MAX_MEASUREMENTS = 8
        cfg.D_MODEL = 32
        cfg.N_HEADS = 2
        cfg.D_FF = 64
        cfg.D_IMG = 16
        cfg.SURFACE_POINTS_PER_FACET = 10
        cfg.ON_TARGET_MAX_MEASUREMENTS = 4
        cfg.MINI_BATCH_SIZE = 4
        cfg.HELI_BATCH_SIZE = 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"fel_{timestamp}"
    if args.smoke_test and args.run_name is None:
        run_name = f"fel_smoke_{timestamp}"
    output_dir = cfg.OUTPUT_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #
    set_logger_config()
    log = logging.getLogger(__name__)
    logging.getLogger().setLevel(logging.INFO)

    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"))
    logging.getLogger().addHandler(fh)

    device = get_device()
    log.info(f"Run name  : {run_name}")
    log.info(f"Output    : {output_dir}")
    log.info(f"Device    : {device}")
    log.info(f"Smoke test: {args.smoke_test}")

    config_snap = {
        k: str(v) if isinstance(v, pathlib.Path) else v
        for k, v in vars(cfg).items()
        if not k.startswith("__")
    }
    config_snap["smoke_test"] = args.smoke_test
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_snap, f, indent=2, default=str)

    results = fel_train.run(cfg=cfg, device=device, output_dir=output_dir)

    print()
    print("=" * 78)
    print(f"  FEL RESULTS  —  {results['n_heliostats']} heliostats  |  "
          f"{results['total_time_min']:.1f} min")
    print("=" * 78)
    print(f"  train loss : {results['initial_train_loss']:.4f} → "
          f"{results['final_train_loss']:.4f} m²")
    print(f"  val mrad   : {results['initial_val_mrad']:.3f} → "
          f"best {results['best_val_mrad']:.3f}")
    print(f"  |Δθ|/epoch : "
          + ", ".join(f"{d:.4f}" for d in results["delta_norm_history"]))
    print("=" * 78)
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
