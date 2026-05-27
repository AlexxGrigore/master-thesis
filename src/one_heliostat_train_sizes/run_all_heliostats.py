"""
Runner: train-size sensitivity sweep for all 5 field heliostats, then compare.

Runs main.py sequentially for each heliostat (they share the GPU so serial is correct),
then calls compare_heliostats.py to produce the combined plot and table.

Usage
-----
    # locally
    python one_heliostat_train_sizes/run_all_heliostats.py

    # on DAIC (called by sbatch)
    python one_heliostat_train_sizes/run_all_heliostats.py --daic

    # custom output parent
    python one_heliostat_train_sizes/run_all_heliostats.py --output-parent outputs/my_run
"""
import argparse
import datetime
import pathlib
import subprocess
import sys

HELIOSTATS = ["AC36", "AG33", "AO34", "AW36", "BE35"]

_HERE = pathlib.Path(__file__).resolve().parent
_SRC  = _HERE.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the one-heliostat train-size sweep for all field heliostats."
    )
    parser.add_argument("--daic", action="store_true", help="Use DAIC cluster paths.")
    parser.add_argument(
        "--output-parent",
        type=pathlib.Path,
        default=None,
        help="Parent directory for per-heliostat output dirs. "
             "Defaults to <base_dir>/outputs/one_hel_train_sizes (local) or "
             "/home/nfs/agrigore/.../outputs/one_hel_train_sizes (DAIC).",
    )
    parser.add_argument(
        "--heliostats",
        nargs="+",
        default=HELIOSTATS,
        help="Heliostat IDs to run (default: all 5).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Pass --smoke-test to each main.py call (quick end-to-end check).",
    )
    parser.add_argument(
        "--no-deflectometry", dest="ideal_scenario", action="store_true",
        help="Train using ideal (flat) scenarios instead of deflectometry-fitted ones.",
    )
    args = parser.parse_args()

    if args.output_parent is None:
        if args.daic:
            base = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
        else:
            base = _SRC.parent
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_parent = base / "outputs" / f"one_hel_train_sizes_{timestamp}"

    args.output_parent.mkdir(parents=True, exist_ok=True)
    comparison_dir = args.output_parent / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    failed = []
    for hid in args.heliostats:
        out_dir = args.output_parent / hid
        cmd = [
            python, str(_HERE / "main.py"),
            "--heliostat-id", hid,
            "--output-dir", str(out_dir),
        ]
        if args.daic:
            cmd.append("--daic")
        if args.smoke_test:
            cmd.append("--smoke-test")
        if args.ideal_scenario:
            cmd.append("--no-deflectometry")

        print(f"\n{'='*60}")
        print(f"  Running heliostat {hid}  →  {out_dir}")
        print(f"{'='*60}\n", flush=True)

        result = subprocess.run(cmd, cwd=str(_SRC))
        if result.returncode != 0:
            print(f"[WARNING] main.py returned non-zero exit code for {hid}: {result.returncode}")
            failed.append(hid)

    # Always attempt comparison even if some runs failed.
    completed = [hid for hid in args.heliostats if hid not in failed]
    completed_dirs = [str(args.output_parent / hid) for hid in completed]

    if len(completed_dirs) >= 2:
        print(f"\n{'='*60}")
        print(f"  Generating comparison plots  →  {comparison_dir}")
        print(f"{'='*60}\n", flush=True)

        compare_cmd = [
            python, str(_HERE / "compare_heliostats.py"),
            "--dirs", *completed_dirs,
            "--out", str(comparison_dir),
        ]
        subprocess.run(compare_cmd, cwd=str(_SRC), check=True)
    else:
        print(f"[INFO] Only {len(completed_dirs)} run(s) completed — skipping comparison.")

    if failed:
        print(f"\n[WARNING] The following heliostats failed: {failed}")
        sys.exit(1)

    print("\nAll done.")


if __name__ == "__main__":
    main()
