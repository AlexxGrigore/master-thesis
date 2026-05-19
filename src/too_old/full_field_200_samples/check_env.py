"""
Sanity check for the full_field_200_samples experiment.

Run before submitting to DAIC:
    python check_env.py --dataset-type synthetic
    python check_env.py --dataset-type real
    python check_env.py --dataset-type synthetic --daic
"""
from __future__ import annotations

import argparse
import pathlib
import sys

_here = pathlib.Path(__file__).resolve().parent
_src  = _here.parent
sys.path.insert(0, str(_src))


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _section(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def check_cuda() -> None:
    _section("CUDA / device")
    import torch
    _ok(f"torch version: {torch.__version__}")
    if torch.cuda.is_available():
        _ok(f"CUDA available — {torch.cuda.device_count()} device(s)")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024 ** 3
            _ok(f"  GPU {i}: {props.name}  ({mem_gb:.1f} GB)")
    else:
        print("  [WARN] CUDA not available — will run on CPU")

    from artist.util.environment_setup import get_device
    device = get_device()
    _ok(f"get_device() → {device}")


def check_imports() -> bool:
    _section("Key imports")
    imports = [
        ("torch",              "torch"),
        ("artist",             "artist"),
        ("paint",              "paint"),
        ("h5py",               "h5py"),
        ("tqdm",               "tqdm"),
        ("full_field_200_samples.config",   "config"),
        ("full_field_200_samples.train",    "train"),
        ("five_heliostats_synth.data",      "five_heliostats_synth.data"),
        ("artist_extensions.kinematic_reconstructors",
         "artist_extensions.kinematic_reconstructors"),
    ]
    all_ok = True
    for label, module in imports:
        try:
            __import__(module)
            _ok(label)
        except ImportError as e:
            _fail(f"{label} — {e}")
            all_ok = False
    return all_ok


def _apply_daic_paths(cfg) -> None:
    cfg.IS_ON_DAIC = True
    cfg.BASE_DIR  = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
    cfg.PAINT_DIR = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")
    cfg.SCENARIO_PATH              = cfg.BASE_DIR / "scenarios" / "full_field_200_samples_scenario" / "scenario.h5"
    cfg.BENCHMARK_CSV              = cfg.PAINT_DIR / "splits" / f"{cfg.BENCHMARK_NAME}.csv"
    cfg.CALIBRATION_PROPERTIES_DIR = cfg.PAINT_DIR / cfg.BENCHMARK_NAME / "calibration_properties"
    cfg.FLUX_IMAGE_DIR             = cfg.PAINT_DIR / cfg.BENCHMARK_NAME / "flux_image"


def check_paths(dataset_type: str, is_on_daic: bool) -> bool:
    _section("Paths")
    import config as cfg
    if is_on_daic:
        _apply_daic_paths(cfg)

    synth_base = cfg.BASE_DIR / "scenarios" / "full_field_200_samples_scenario" / "synthetic_data"

    required_files = [
        ("Scenario file",   cfg.SCENARIO_PATH),
        ("Benchmark CSV",   cfg.BENCHMARK_CSV),
    ]
    required_dirs = [
        ("Calibration props dir", cfg.CALIBRATION_PROPERTIES_DIR),
        ("Flux image dir",        cfg.FLUX_IMAGE_DIR),
    ]
    if dataset_type == "synthetic":
        required_dirs += [
            ("Synthetic train dir", synth_base / "train"),
            ("Synthetic val dir",   synth_base / "val"),
            ("Synthetic test dir",  synth_base / "test"),
        ]

    all_ok = True
    for label, path in required_files:
        if path.exists():
            _ok(f"{label}: {path}")
        else:
            _fail(f"{label} MISSING: {path}")
            all_ok = False

    for label, path in required_dirs:
        if path.is_dir():
            n = sum(1 for _ in path.rglob("*") if _.is_file())
            _ok(f"{label}: {path}  ({n} files)")
        else:
            _fail(f"{label} MISSING: {path}")
            all_ok = False

    _section("Resolved config")
    print(f"  dataset_type : {dataset_type}")
    print(f"  loss_type    : {cfg.LOSS_TYPE}")
    print(f"  is_on_daic   : {is_on_daic}")
    print(f"  base_dir     : {cfg.BASE_DIR}")
    print(f"  scenario     : {cfg.SCENARIO_PATH}")
    print(f"  train samples: {cfg.TRAIN_SAMPLES}")
    print(f"  val samples  : {cfg.VAL_SAMPLES}")
    print(f"  test samples : {cfg.TEST_SAMPLES}")

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check for full_field_200_samples.")
    parser.add_argument("--dataset-type", choices=["real", "synthetic"], default="synthetic")
    parser.add_argument("--daic", action="store_true")
    args = parser.parse_args()

    print(f"\n{'═' * 55}")
    print(f"  full_field_200_samples — environment check")
    print(f"  dataset-type: {args.dataset_type}   daic: {args.daic}")
    print(f"{'═' * 55}")

    check_cuda()
    imports_ok = check_imports()
    paths_ok = check_paths(args.dataset_type, args.daic)

    print(f"\n{'═' * 55}")
    if imports_ok and paths_ok:
        print("  ALL CHECKS PASSED — ready to run.")
    else:
        print("  SOME CHECKS FAILED — fix the issues above before submitting.")
    print(f"{'═' * 55}\n")


if __name__ == "__main__":
    main()
