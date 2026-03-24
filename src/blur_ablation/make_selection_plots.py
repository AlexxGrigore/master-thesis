"""Standalone script to regenerate heliostat selection plots from saved run outputs.

Usage (from project root src/):
    python -m blur_ablation.make_selection_plots \
        --run_dir ../outputs/good_runs/blur_ablation_20260317_000052 \
        --scenario ../scenarios/deflectometry_scenario/deflectometry_scenario.h5 \
        --benchmark_csv ../datasets/paint_benchmarks/splits/benchmark_split-balanced_train-10_validation-30.csv
"""

import argparse
import csv
import json
import pathlib

import h5py

from blur_ablation.plotting import plot_field_heatmap, plot_field_coordinates

HELIOSTATS_PER_CELL = 2


def load_all_positions(scenario_path: pathlib.Path) -> dict[str, tuple[float, float]]:
    """Read (east_m, north_m) for every heliostat directly from the HDF5 file."""
    positions = {}
    with h5py.File(scenario_path, "r") as f:
        for name, group in f["heliostats"].items():
            pos = group["position"][:]  # shape (4,) — (E, N, U, 1) homogeneous
            positions[name] = (float(pos[0]), float(pos[1]))
    return positions


def load_deflectometry_names(benchmark_csv: pathlib.Path) -> set[str]:
    """Return the set of heliostat names that appear in the benchmark CSV."""
    names = set()
    with open(benchmark_csv, newline="") as f:
        for row in csv.DictReader(f):
            names.add(row["HeliostatId"])
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--scenario",
        default=pathlib.Path("../scenarios/deflectometry_scenario/deflectometry_scenario.h5"),
        type=pathlib.Path,
    )
    parser.add_argument(
        "--benchmark_csv",
        default=pathlib.Path("../datasets/paint_benchmarks/splits/benchmark_split-balanced_train-10_validation-30.csv"),
        type=pathlib.Path,
    )
    args = parser.parse_args()

    run_dir: pathlib.Path = args.run_dir
    scenario_path: pathlib.Path = args.scenario
    benchmark_csv: pathlib.Path = args.benchmark_csv

    for p in [run_dir, scenario_path, benchmark_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p}")

    with open(run_dir / "selected_heliostats.json") as f:
        selected_heliostats = json.load(f)
    print(f"Loaded {len(selected_heliostats)} selected heliostats.")

    print(f"Loading heliostat positions from {scenario_path} …")
    all_positions = load_all_positions(scenario_path)
    print(f"  {len(all_positions)} heliostats in scenario.")

    deflectometry_names = load_deflectometry_names(benchmark_csv)
    print(f"  {len(deflectometry_names)} heliostats with deflectometry data.")

    plot_field_heatmap(
        selected_heliostats=selected_heliostats,
        output_path=run_dir / "fig4_field_heatmap.png",
        heliostats_per_cell=HELIOSTATS_PER_CELL,
    )
    print(f"  Fig 4 saved -> {run_dir / 'fig4_field_heatmap.png'}")

    plot_field_coordinates(
        selected_heliostats=selected_heliostats,
        all_heliostat_positions=all_positions,
        output_path=run_dir / "fig5_field_coordinates.png",
        deflectometry_names=deflectometry_names,
    )
    print(f"  Fig 5 saved -> {run_dir / 'fig5_field_coordinates.png'}")


if __name__ == "__main__":
    main()
