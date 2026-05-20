"""
Sequential overnight runner for full_field_200_samples experiments.

Runs three experiments back-to-back, logging each to a shared log file:
  1. Pixel loss — synthetic data, no backlash
  2. Pixel loss — synthetic data, with backlash (±3 mrad)
  3. Pixel loss — real PAINT data

Usage
-----
    python run_experiments.py
    python run_experiments.py --smoke-test   # quick end-to-end sanity check
    python run_experiments.py --daic         # use DAIC cluster paths
"""
import argparse
import datetime
import pathlib
import subprocess
import sys

_here = pathlib.Path(__file__).resolve().parent

EXPERIMENTS = [
    {
        "name":        "pixel_synth_no_backlash",
        "description": "Pixel loss | synthetic data | no backlash",
        "args":        ["--dataset-type", "synthetic", "--loss-type", "pixel"],
    },
    {
        "name":        "pixel_synth_backlash",
        "description": "Pixel loss | synthetic data | backlash ±3 mrad",
        "args":        ["--dataset-type", "synthetic", "--loss-type", "pixel",
                        "--backlash-mrad", "3.0"],
    },
    {
        "name":        "pixel_real",
        "description": "Pixel loss | real PAINT data",
        "args":        ["--dataset-type", "real", "--loss-type", "pixel"],
    },
]


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(log_path: pathlib.Path, msg: str) -> None:
    line = f"[{_timestamp()}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run overnight experiment sequence.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Pass --smoke-test to every experiment (fast end-to-end check).")
    parser.add_argument("--daic", action="store_true",
                        help="Pass --daic to every experiment (use cluster paths).")
    args = parser.parse_args()

    log_path = _here / "overnight.log"
    _log(log_path, f"=== Overnight run started — {len(EXPERIMENTS)} experiments ===")
    if args.smoke_test:
        _log(log_path, "SMOKE TEST mode — short epochs for sanity check.")

    results = []
    for i, exp in enumerate(EXPERIMENTS, 1):
        _log(log_path, f"--- [{i}/{len(EXPERIMENTS)}] Starting: {exp['description']} ---")
        t_start = datetime.datetime.now()

        cmd = [sys.executable, str(_here / "main.py")] + exp["args"]
        if args.smoke_test:
            cmd.append("--smoke-test")
        if args.daic:
            cmd.append("--daic")

        _log(log_path, f"Command: {' '.join(cmd)}")

        # Stream stdout+stderr to terminal and log file simultaneously.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with open(log_path, "a") as lf:
            for line in proc.stdout:
                print(line, end="", flush=True)
                lf.write(line)
        proc.wait()

        elapsed = (datetime.datetime.now() - t_start).total_seconds() / 60
        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        _log(log_path, f"--- [{i}/{len(EXPERIMENTS)}] Finished: {exp['description']} — {status} in {elapsed:.1f} min ---")
        results.append((exp["description"], status, elapsed))

        if proc.returncode != 0:
            _log(log_path, "Experiment failed — stopping sequence.")
            break

    _log(log_path, "=== Summary ===")
    for desc, status, elapsed in results:
        _log(log_path, f"  {status:30s}  {elapsed:6.1f} min  |  {desc}")

    total_min = sum(e for _, _, e in results)
    _log(log_path, f"Total elapsed: {total_min:.1f} min")
    _log(log_path, "=== Done ===")

    if any(s != "OK" for _, s, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
