"""
Sanity check for the full_training_pipeline experiment.

Run before submitting to DAIC:
    python check_env.py --dataset-type synthetic
    python check_env.py --dataset-type real
    python check_env.py --dataset-type synthetic --daic
"""
from __future__ import annotations

import argparse
import pathlib
import sys

_src = pathlib.Path(__file__).resolve().parents[1]
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


def check_imports() -> None:
    _section("Key imports")
    imports = [
        ("torch", "torch"),
        ("artist", "artist"),
        ("paint", "paint"),
        ("h5py", "h5py"),
        ("tqdm", "tqdm"),
        ("full_training_pipeline.config", "full_training_pipeline.config"),
        ("full_training_pipeline.model", "full_training_pipeline.model"),
        ("full_training_pipeline.train", "full_training_pipeline.train"),
        ("artist_extensions.kinematic_reconstructors", "artist_extensions.kinematic_reconstructors"),
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


def check_paths(dataset_type: str, is_on_daic: bool) -> None:
    _section("Paths")
    from full_training_pipeline.config import build_config
    config = build_config(dataset_type=dataset_type, is_on_daic=is_on_daic)

    required_files = [
        ("Scenario file",       config.scenario_path),
        ("Coarse checkpoint",   config.coarse_checkpoint_path),
        ("Benchmark CSV",       config.benchmark_csv),
    ]
    required_dirs = [
        ("Calibration props dir", config.calibration_properties_dir),
        ("Flux image dir",        config.flux_image_dir),
    ]
    if dataset_type == "synthetic":
        required_dirs += [
            ("Synthetic train dir", config.synthetic_data_base_dir / "train"),
            ("Synthetic val dir",   config.synthetic_data_base_dir / "val"),
            ("Synthetic test dir",  config.synthetic_data_base_dir / "test"),
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
    print(f"  dataset_type : {config.dataset_type}")
    print(f"  loss_type    : {config.loss_type}")
    print(f"  is_on_daic   : {is_on_daic}")
    print(f"  output_dir   : {config.output_dir}")
    print(f"  checkpoint   : {config.coarse_checkpoint_path.name}")
    print(f"  max_epochs   : {config.max_epochs}")
    print(f"  lr           : {config.learning_rate}")

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check for full_training_pipeline.")
    parser.add_argument("--dataset-type", choices=["real", "synthetic"], default="synthetic")
    parser.add_argument("--daic", action="store_true")
    args = parser.parse_args()

    print(f"\n{'═' * 55}")
    print(f"  full_training_pipeline — environment check")
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
