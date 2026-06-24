"""Preflight check for the profiling experiment on DAIC.

Run this INSIDE the apptainer container, ideally in a GPU session, before submitting
the real job. It verifies — and reports PASS / WARN / FAIL for — everything the run
needs:

  * torch + CUDA, and whether the GPU is an A40,
  * every ARTIST / PAINT module the drivers import,
  * every ``constants`` / ``config_dictionary`` key the drivers reference (the most
    likely thing to drift between ARTIST versions),
  * the datasets (PAINT heliostats, tower file, training benchmark, availability JSON),
  * that enough heliostats are eligible for the largest requested field size,
  * write permission for the scenario + output directories.

Exit code is 0 only if there are no FAILs, so it can gate the sbatch (``set -e``).

Usage (inside container, GPU session recommended):
    srun --gres=gpu:a40:1 --cpus-per-task=4 --mem=16G --time=00:20:00 --pty bash
    cd .../master-thesis/src
    apptainer exec --nv --bind /tudelft.net:/tudelft.net <sif> \
        python profiling_experiment/check_daic.py --daic --sizes 1 10 20 50
"""

from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import sys

# Repo root resolved from this file — no imports required, so filesystem checks run
# even if the ARTIST import is what's broken.
_HERE = pathlib.Path(__file__).resolve().parent
_SRC = _HERE.parent
_REPO = _SRC.parent

# ANSI markers (fall back to plain text if not a tty).
_TTY = sys.stdout.isatty()
_MARK = {
    "OK": ("\033[32m[ OK ]\033[0m" if _TTY else "[ OK ]"),
    "WARN": ("\033[33m[WARN]\033[0m" if _TTY else "[WARN]"),
    "FAIL": ("\033[31m[FAIL]\033[0m" if _TTY else "[FAIL]"),
}

_results: list[tuple[str, str, str]] = []  # (status, label, detail)


def record(status: str, label: str, detail: str = "") -> None:
    _results.append((status, label, detail))
    line = f"{_MARK[status]} {label}"
    print(f"{line}   {detail}" if detail else line)


def _try(label: str, fn, fatal: bool = True) -> object:
    """Run a check fn; convert exceptions into a FAIL/WARN record. Returns fn() or None."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 — preflight wants every failure, not a crash
        record("FAIL" if fatal else "WARN", label, f"{type(exc).__name__}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_torch_and_gpu() -> None:
    import torch

    record("OK", "torch import", f"version {torch.__version__}, cuda build {torch.version.cuda}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        status = "OK" if "a40" in name.lower() else "WARN"
        record(status, "CUDA GPU", f"{name} "
               f"({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB)")
    else:
        record("WARN", "CUDA GPU",
               "no GPU visible — run inside an `srun --gres=gpu:a40:1 ... --pty bash` "
               "session with `apptainer exec --nv` for an accurate check")


def check_artist_imports() -> None:
    modules = {
        "artist": [],
        "artist.io": ["PaintCalibrationDataParser"],
        "artist.optim": ["KinematicsReconstructor"],
        "artist.optim.loss": ["FocalSpotLoss"],
        "artist.raytracing": ["HeliostatRayTracer"],
        "artist.scenario": ["Scenario"],
        "artist.data_parser": ["paint_scenario_parser"],
        "artist.scenario.h5_scenario_generator": ["H5ScenarioGenerator"],
        "artist.util": ["constants", "config_dictionary", "set_logger_config"],
        "artist.util.env": ["get_device", "setup_distributed_environment"],
        "artist.util.environment_setup": ["get_device"],
        "paint.util.paint_mappings": ["UTIS_KEY"],
    }
    for mod_name, attrs in modules.items():
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001
            record("FAIL", f"import {mod_name}", f"{type(exc).__name__}: {exc}")
            continue
        missing = [a for a in attrs if not hasattr(mod, a)]
        if missing:
            record("FAIL", f"import {mod_name}", f"missing names: {missing}")
        else:
            record("OK", f"import {mod_name}", ", ".join(attrs) if attrs else "")


def check_required_constants() -> None:
    from artist.util import config_dictionary, constants

    required_constants = [
        "initial_learning_rate_rotation_deviation", "initial_learning_rate_initial_angles",
        "initial_learning_rate_initial_stroke_length", "tolerance", "max_epoch", "batch_size",
        "log_step", "early_stopping_delta", "early_stopping_patience", "early_stopping_window",
        "scheduler_type", "reduce_on_plateau", "gamma", "lr_min", "lr_max", "step_size_up",
        "reduce_factor", "patience", "threshold", "cooldown", "optimization", "scheduler",
        "data_parser", "heliostat_data_mapping", "kinematics_reconstruction_raytracing", "device",
    ]
    missing = [c for c in required_constants if not hasattr(constants, c)]
    record("FAIL" if missing else "OK", "artist.util.constants keys",
           f"missing: {missing}" if missing else f"all {len(required_constants)} present")

    required_cfg = ["sun_key", "light_source_distribution_is_normal", "fit_nurbs_from_normals"]
    missing_cfg = [c for c in required_cfg if not hasattr(config_dictionary, c)]
    record("FAIL" if missing_cfg else "OK", "artist.util.config_dictionary keys",
           f"missing: {missing_cfg}" if missing_cfg else f"all {len(required_cfg)} present")


def check_artist_methods() -> None:
    from artist.data_parser import paint_scenario_parser
    from artist.raytracing import HeliostatRayTracer
    from artist.scenario import Scenario

    pairs = [
        (paint_scenario_parser, "extract_paint_tower_measurements"),
        (paint_scenario_parser, "extract_paint_heliostats_fitted_surface"),
        (Scenario, "load_scenario_from_hdf5"),
        (Scenario, "get_number_of_heliostat_groups_from_hdf5"),
        (HeliostatRayTracer, "trace_rays"),  # needed for the runtime monkeypatch
    ]
    missing = [f"{obj.__name__ if hasattr(obj,'__name__') else obj}.{attr}"
               for obj, attr in pairs if not hasattr(obj, attr)]
    record("FAIL" if missing else "OK", "ARTIST methods used by drivers",
           f"missing: {missing}" if missing else "all present")


def check_sibling_modules() -> None:
    for p in (str(_SRC), str(_SRC / "one_heliostat_demo" / "single_heliostat")):
        if p not in sys.path:
            sys.path.insert(0, p)
    for name in ("create_all_scenarios", "config", "selection", "profiling"):
        try:
            importlib.import_module(name)
            record("OK", f"project module '{name}'")
        except Exception as exc:  # noqa: BLE001
            record("FAIL", f"project module '{name}'", f"{type(exc).__name__}: {exc}")
    try:
        importlib.import_module("utils.evaluation")
        record("OK", "project module 'utils.evaluation'")
    except Exception as exc:  # noqa: BLE001
        record("FAIL", "project module 'utils.evaluation'", f"{type(exc).__name__}: {exc}")


def check_datasets(daic: bool) -> None:
    # Resolve dataset paths the same way the drivers do, with a repo-relative fallback.
    paint_dir = _REPO / "datasets" / "paint" / "heliostats"
    try:
        import create_all_scenarios as cas
        paint_dir = cas.DAIC_PAINT_DIR if daic else cas.LOCAL_PAINT_DIR
    except Exception:  # noqa: BLE001 — fall back to repo-relative path
        pass

    targets = [
        (paint_dir, "dir", "PAINT heliostats dir"),
        (paint_dir / "WRI1030197-tower-measurements.json", "file", "tower measurements JSON"),
        (_REPO / "datasets" / "paint"
         / "benchmark_split-balanced_train-100_validation-50_deflectometry"
         / "calibration_properties", "dir", "benchmark calibration_properties"),
        (_REPO / "datasets" / "paint"
         / "benchmark_split-balanced_train-100_validation-50_deflectometry"
         / "flux_image", "dir", "benchmark flux_image"),
        (_REPO / "datasets" / "paint" / "splits"
         / "benchmark_split-balanced_train-100_validation-50_deflectometry.csv", "file",
         "benchmark split CSV"),
        (_SRC / "utils" / "deflectometry_availability.json", "file", "deflectometry availability JSON"),
    ]
    for path, kind, label in targets:
        exists = path.is_dir() if kind == "dir" else path.is_file()
        if not exists:
            record("FAIL", label, f"missing: {path}")
        elif kind == "dir":
            n = sum(1 for _ in path.iterdir())
            record("OK" if n > 0 else "FAIL", label, f"{n} entries  ({path})")
        else:
            record("OK", label, str(path))


def check_container_binds(sif: str) -> None:
    if pathlib.Path("/tudelft.net").exists():
        record("OK", "/tudelft.net bind mount", "accessible")
    else:
        record("WARN", "/tudelft.net bind mount",
               "not visible — add `--bind /tudelft.net:/tudelft.net` to apptainer exec")
    if pathlib.Path(sif).is_file():
        record("OK", "apptainer image (.sif)", sif)
    else:
        record("WARN", "apptainer image (.sif)", f"not found at {sif} (override with --sif)")


def check_eligible_pool(daic: bool, sizes: list[int]) -> None:
    import selection

    pool = selection.eligible_fitting_list(daic=daic)
    need = max(sizes)
    record("OK" if len(pool) >= need else "FAIL", "eligible heliostat pool",
           f"{len(pool)} eligible (need >= {need} for sizes {sizes})")


def check_write_perms(daic: bool) -> None:
    try:
        import create_all_scenarios as cas
        base = cas.DAIC_BASE_DIR if daic else cas.LOCAL_BASE_DIR
    except Exception:  # noqa: BLE001
        base = _REPO
    for sub in ("scenarios/profiling", "outputs/new_mapping_function/profiling_experiment"):
        d = base / sub
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
            record("OK", f"write access {sub}", str(d))
        except Exception as exc:  # noqa: BLE001
            record("FAIL", f"write access {sub}", f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Preflight check for the DAIC profiling experiment.")
    p.add_argument("--daic", action="store_true", help="Use DAIC dataset paths.")
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 10, 20, 50])
    p.add_argument("--sif", default="/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/artist-local.sif")
    args = p.parse_args()

    print(f"Profiling experiment preflight  (repo={_REPO}, daic={args.daic})\n")

    _try("torch + GPU", check_torch_and_gpu)
    _try("ARTIST imports", check_artist_imports)
    _try("ARTIST constants", check_required_constants)
    _try("ARTIST methods", check_artist_methods)
    _try("project modules", check_sibling_modules)
    _try("container binds", lambda: check_container_binds(args.sif), fatal=False)
    _try("datasets", lambda: check_datasets(args.daic))
    _try("eligible pool", lambda: check_eligible_pool(args.daic, args.sizes))
    _try("write permissions", lambda: check_write_perms(args.daic))

    fails = sum(1 for s, _, _ in _results if s == "FAIL")
    warns = sum(1 for s, _, _ in _results if s == "WARN")
    print("\n" + "=" * 60)
    print(f"Summary: {len(_results) - fails - warns} OK, {warns} WARN, {fails} FAIL")
    if fails:
        print("NOT READY — fix the FAIL items above before submitting the job.")
    else:
        print("READY TO RUN — submit with: sbatch src/sbatch_files/run_profiling_experiment.sh")
    print("=" * 60)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
