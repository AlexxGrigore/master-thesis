"""
Regenerate fig7 and fig8 from existing blur-ablation JSON outputs.

Usage:
    python blur_ablation/regenerate_new_figs.py \
        --results_dir ../../outputs/NewRuns/blur_ablation_20260330_081442
"""

import argparse
import json
import pathlib
import sys

_pkg = pathlib.Path(__file__).parent
_src = _pkg.parent
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_pkg))

from blur_ablation.plotting import plot_rays_convergence_by_surface, plot_surface_pts_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.parent.parent
                / "outputs" / "NewRuns" / "blur_ablation_20260330_081442",
    )
    args = parser.parse_args()
    results_dir = args.results_dir

    with open(results_dir / "sweep_results.json") as f:
        records = json.load(f)

    with open(results_dir / "selected_heliostats.json") as f:
        selected = json.load(f)

    with open(results_dir / "optimal_sigma.json") as f:
        raw = json.load(f)

    heliostat_distances = {h["name"]: h["distance_m"] for h in selected}
    # optimal_sigma.json keys are strings ("10", "25", …) → convert to int.
    optimal_sigmas = {int(k): float(v) for k, v in raw.items()}

    print(f"Records: {len(records)}")
    print(f"Optimal sigmas: {optimal_sigmas}")

    plot_rays_convergence_by_surface(
        records=records,
        heliostat_distances=heliostat_distances,
        optimal_sigmas=optimal_sigmas,
        output_path=results_dir / "fig7_rays_convergence_by_surface.png",
    )
    print(f"Saved: {results_dir / 'fig7_rays_convergence_by_surface.png'}")

    plot_surface_pts_grid(
        records=records,
        heliostat_distances=heliostat_distances,
        optimal_sigmas=optimal_sigmas,
        output_path=results_dir / "fig8_surface_pts_grid.png",
    )
    print(f"Saved: {results_dir / 'fig8_surface_pts_grid.png'}")


if __name__ == "__main__":
    main()
