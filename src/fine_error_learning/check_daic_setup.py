"""
DAIC readiness check for fine_error_learning.

Run inside the apptainer shell on a DAIC GPU node (or via apptainer exec):

    cd /home/nfs/agrigore/projects/githubProjects/master-thesis/src
    python fine_error_learning/check_daic_setup.py --daic

Verifies, in order:
  1. imports          — torch/CUDA, ARTIST, PAINT, the FEL package modules
  2. paths            — repo, scenarios (68 h5), synthetic dataset splits,
                        stage-1 checkpoints (or a note they will be generated),
                        output root writable
  3. functional probe — loads one scenario (AA23) with a NOMINAL warm start
                        (no checkpoint needed), loads 4 train measurements,
                        builds tokens, runs one model forward and one ARTIST
                        ray trace + focal-spot loss on the selected device

Exit code 0 = all good, 1 = something failed (see the ✗ lines).
Without --daic it checks the LOCAL paths instead (same checks, useful before
committing changes that the DAIC run will pull).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

_here = pathlib.Path(__file__).resolve().parent   # fine_error_learning/
_src = _here.parent                               # src/
sys.path.insert(0, str(_src))

OK, FAIL = "✓", "✗"
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{OK if ok else FAIL}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--daic", action="store_true", help="Check DAIC cluster paths")
    args = p.parse_args()

    print("=" * 70)
    print(f"FEL DAIC SETUP CHECK  ({'DAIC' if args.daic else 'LOCAL'} paths)")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Imports                                                           #
    # ------------------------------------------------------------------ #
    print("\n[1] Imports")
    try:
        import torch
        check("torch import", True, f"v{torch.__version__}")
        check(
            "CUDA available",
            torch.cuda.is_available() or not args.daic,
            torch.cuda.get_device_name(0) if torch.cuda.is_available()
            else "no GPU — OK locally / on a login node, but run this via "
                 "apptainer --nv on a GPU node before submitting",
        )
    except Exception as e:
        print(f"  [{FAIL}] torch import — {e}")
        return 1
    try:
        from artist.scenario.scenario import Scenario  # noqa: F401
        check("ARTIST import", True)
    except Exception as e:
        check("ARTIST import", False, str(e))
    try:
        import paint.util.paint_mappings  # noqa: F401
        check("PAINT import", True)
    except Exception as e:
        check("PAINT import", False, str(e))
    try:
        from fine_error_learning import (  # noqa: F401
            config, data, evaluate, model, pipeline, train, warm_start,
        )
        check("fine_error_learning package import", True)
    except Exception as e:
        check("fine_error_learning package import", False, str(e))
        return 1

    from fine_error_learning import config as cfg

    if args.daic:
        cfg.BASE_DIR = pathlib.Path(
            "/home/nfs/agrigore/projects/githubProjects/master-thesis"
        )
        paint_root = pathlib.Path(
            "/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint"
        )
        cfg.SCENARIO_PATH_TEMPLATE = str(
            cfg.BASE_DIR / "scenarios" / "one_heliostat_scenarios"
            / "{heliostat_id}" / "scenario.h5"
        )
        cfg.SYNTHETIC_DATA_DIR = paint_root / "synthetic" / "balanced_dataset" / "dataset"
        cfg.STAGE1_CHECKPOINT_DIR = (
            cfg.BASE_DIR / "outputs" / "fine_error_learning" / "all63_stage1_synth"
        )
        cfg.OUTPUT_ROOT = cfg.BASE_DIR / "outputs" / "fine_error_learning"

    # ------------------------------------------------------------------ #
    # 2. Paths                                                             #
    # ------------------------------------------------------------------ #
    print("\n[2] Paths")
    check("repo BASE_DIR exists", cfg.BASE_DIR.exists(), str(cfg.BASE_DIR))

    scen_dir = pathlib.Path(cfg.SCENARIO_PATH_TEMPLATE.split("{heliostat_id}")[0])
    n_scen = len(list(scen_dir.glob("*/scenario.h5"))) if scen_dir.exists() else 0
    check("scenario files (expect 68)", n_scen == 68, f"found {n_scen} under {scen_dir}")

    data_dir = pathlib.Path(cfg.SYNTHETIC_DATA_DIR)
    check("synthetic dataset dir", data_dir.exists(), str(data_dir))
    if data_dir.exists():
        for split in ("train", "val", "test"):
            sdir = data_dir / split
            n_hel = len([d for d in sdir.iterdir() if d.is_dir()]) if sdir.exists() else 0
            check(f"dataset split '{split}' (62 heliostats)", n_hel == 62,
                  f"found {n_hel}")
        check("perturbations.json", (data_dir / "perturbations.json").exists())

    ckpt_dir = pathlib.Path(cfg.STAGE1_CHECKPOINT_DIR)
    n_ckpt = (
        len(list(ckpt_dir.glob("*/stage1_checkpoint.pt"))) if ckpt_dir.exists() else 0
    )
    check(
        "stage-1 checkpoints (62; missing is OK — the sbatch step-0 generates them)",
        n_ckpt == 62, f"found {n_ckpt} under {ckpt_dir}",
    )
    if n_ckpt != 62:
        print("      → will be generated by run_fel_daic_matrix.sh step 0")

    try:
        cfg.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        probe = cfg.OUTPUT_ROOT / ".write_test"
        probe.touch()
        probe.unlink()
        check("output root writable", True, str(cfg.OUTPUT_ROOT))
    except Exception as e:
        check("output root writable", False, str(e))

    # ------------------------------------------------------------------ #
    # 3. Functional probe (nominal warm start — no checkpoint needed)      #
    # ------------------------------------------------------------------ #
    print("\n[3] Functional probe (AA23, nominal warm start, 4 rays)")
    if failures:
        print("  skipped — fix the ✗ items above first")
    else:
        try:
            from artist.util import get_device
            from fine_error_learning import data as fel_data
            from fine_error_learning import pipeline as fel_pipeline
            from fine_error_learning import warm_start as fel_warm_start
            from fine_error_learning.model import FelTransformerModel

            cfg.WARM_START = "nominal"
            device = get_device()
            print(f"  device: {device}")

            state = fel_warm_start.load_warm_start_state("AA23", cfg, device)
            check("scenario + warm-start state load", True,
                  f"dist to tower {state.hel_dist_m:.1f} m")

            measurements = fel_data.load_measurements(
                data_dir / "train", "AA23",
                state.heliostat_group, state.scenario, device,
            )
            check("measurement load", measurements is not None,
                  f"{measurements.n if measurements else 0} train measurements")

            mean, std = fel_data.compute_scaler([measurements])
            flux, scalars, mask = fel_data.build_model_tokens(
                measurements, k=4, scaler_mean=mean, scaler_std=std
            )
            m = FelTransformerModel(d_model=32, n_heads=2, d_ff=64, d_img=16).to(device)
            delta = m(
                flux.unsqueeze(0), scalars.unsqueeze(0), mask.unsqueeze(0),
                state.theta_kr.to(device).unsqueeze(0),
                state.heliostat_position.to(device).unsqueeze(0),
            )
            check("model forward (Δθ=0 at init)", bool((delta == 0).all()),
                  f"shape {tuple(delta.shape)}")

            state.scenario.set_number_of_rays(4)
            pred_flux, bitmap_res, sampler_idx = fel_pipeline.predict_flux(
                state=state,
                theta_final=state.theta_kr.to(device),
                incident_rays=measurements.incident_rays[:4],
                motor_positions=measurements.motor_positions[:4],
                target_indices=measurements.target_indices[:4],
                device=device,
                random_seed=0,
            )
            lps, _ = fel_pipeline.focal_spot_centroid_loss(
                predicted_flux=pred_flux,
                focal_spots=measurements.focal_spots[:4][sampler_idx],
                target_indices=measurements.target_indices[:4][sampler_idx],
                bitmap_resolution=bitmap_res,
                scenario=state.scenario,
                device=device,
            )
            import math
            err_mrad = math.sqrt(lps.mean().item()) / state.hel_dist_m * 1000.0
            check(
                "ray trace + focal-spot loss", math.isfinite(err_mrad),
                f"nominal-kinematics centroid error {err_mrad:.2f} mrad "
                "(nominal vs perturbed GT: expect roughly the pre-training "
                "level, ~5-25 mrad)",
            )
        except Exception as e:
            import traceback
            check("functional probe", False, repr(e))
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    if failures:
        print(f"RESULT: {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        print("=" * 70)
        return 1
    print("RESULT: all checks passed — ready for sbatch.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
