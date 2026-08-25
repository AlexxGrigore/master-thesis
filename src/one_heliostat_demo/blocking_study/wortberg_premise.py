"""Test Wortberg's stated mechanism for the contour loss, using the blocking raytracer.

`src/artist_extensions/contour_loss.py` states the premise of the contour loss as a
three-link causal chain:

    "Instead of collapsing each flux image to its centre of mass (which occlusion by
     neighbouring heliostats corrupts - flux is eaten from the LOWER part of the spot,
     pulling the COM upward), this loss extracts the upper contour ..."

    L1. occlusion eats flux from the LOWER part of the SPOT
    L2. therefore the flux COM is pulled UPWARD
    L3. therefore the UPPER CONTOUR is the robust feature

Wortberg could not test this: his raytracer had no blocking. ARTIST does. Every link is
now measurable by tracing each calibration sample twice, with identical ray seeds, once
with blocking off and once with the neighbours in place, and comparing:

    L1  where the lost flux sits inside the spot (deficit centroid vs spot centroid)
    L2  the SIGNED vertical component of the flux COM shift
    L3  the contour COM shift against the flux COM shift, i.e. exactly the quantity the
        contour loss claims to protect

This needs no training, so it isolates the physics from every optimizer effect.

Usage
-----
    python wortberg_premise.py BE25 --max-samples 20
    python wortberg_premise.py BE25 AA27 --hypothesis lower --rays 20
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import statistics
import sys

import h5py
import torch

_here = pathlib.Path(__file__).resolve().parent
_src = _here.parents[1]
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_here))

from artist.flux import get_center_of_mass  # noqa: E402
from artist.geometry import bitmap_coordinates_to_target_coordinates  # noqa: E402
from artist.io.paint_calibration_parser import PaintCalibrationDataParser  # noqa: E402
from artist.scenario.scenario import Scenario  # noqa: E402
from artist.util import indices, set_logger_config  # noqa: E402

from artist_extensions.contour_loss import (  # noqa: E402
    ContourExtractor,
    contour_center_of_mass,
)
from utils.evaluation import build_heliostat_data_mapping  # noqa: E402

from gate import BENCHMARK, _paths, blocker_surfaces, trace_one_sample  # noqa: E402

log = logging.getLogger(__name__)

# Contour extractor settings, taken verbatim from single_heliostat/config.py so this
# measures the SAME feature the training actually optimises.
CONTOUR_TAU = 0.58
CONTOUR_ETA = 70.0
CONTOUR_SMOOTHING_ROUNDS = 2
CONTOUR_GAUSS_SIGMA = 3.0
CONTOUR_GAUSS_KSIZE = 13


def _metre_coordinates(
    bitmap_coordinates: torch.Tensor,
    target_area_index: torch.Tensor,
    resolution: torch.Tensor,
    solar_tower,
    device: torch.device,
) -> torch.Tensor:
    """Map bitmap pixel coordinates onto the target plane in ENU metres."""
    return bitmap_coordinates_to_target_coordinates(
        bitmap_coordinates=bitmap_coordinates,
        bitmap_resolution=resolution,
        solar_tower=solar_tower,
        target_area_indices=target_area_index.unsqueeze(0),
        device=device,
    )[0]


def run(
    heliostat_id: str,
    hypothesis: str,
    paint_dir: pathlib.Path,
    scenario_root: pathlib.Path,
    output_root: pathlib.Path,
    max_samples: int,
    surface_points_per_facet: int,
    number_of_rays: int,
    device: torch.device,
) -> dict:
    """Measure all three links of the Wortberg premise for one heliostat."""
    scenario_dir = scenario_root / heliostat_id
    scenario_path = scenario_dir / "scenario.h5"
    if not scenario_path.exists():
        scenario_path = scenario_dir / "scenario_ideal.h5"

    with h5py.File(scenario_path) as file_handle:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=file_handle,
            device=device,
            number_of_surface_points_per_facet=torch.tensor(
                [surface_points_per_facet, surface_points_per_facet]
            ),
        )
    scenario.set_number_of_rays(number_of_rays)
    heliostat_group = scenario.heliostat_field.heliostat_groups[0]
    heliostat_index = heliostat_group.names.index(heliostat_id)

    mapping = [
        entry
        for entry in build_heliostat_data_mapping(
            paint_dir / "splits" / f"{BENCHMARK}.csv",
            paint_dir / BENCHMARK / "calibration_properties",
            paint_dir / BENCHMARK / "flux_image",
            "test",
        )
        if entry[0] == heliostat_id
    ]
    _, centroids, rays, motor_positions, _, target_mask = PaintCalibrationDataParser(
        centroid_extraction_method="UTIS"
    ).parse_data_for_reconstruction(
        heliostat_data_mapping=mapping,
        heliostat_group=heliostat_group,
        scenario=scenario,
        device=device,
    )

    target_centre = scenario.solar_tower.get_centers_of_target_areas(
        target_area_indices=torch.tensor([0], device=device), device=device
    )[0]
    distance_m = float(
        torch.norm(heliostat_group.positions[heliostat_index][:3] - target_centre[:3])
    )
    resolution = torch.tensor([indices.bitmap_resolution] * 2)
    extractor = ContourExtractor(
        tau=CONTOUR_TAU,
        eta=CONTOUR_ETA,
        smoothing_rounds=CONTOUR_SMOOTHING_ROUNDS,
        gaussian_sigma=CONTOUR_GAUSS_SIGMA,
        gaussian_kernel_size=CONTOUR_GAUSS_KSIZE,
    ).to(device)

    number_of_samples = min(max_samples, rays.shape[0])
    log.info(
        f"{heliostat_id}: {number_of_samples} samples, hypothesis '{hypothesis}', "
        f"{distance_m:.0f} m to target, scenario {scenario_path.name}"
    )

    records = []
    for sample in range(number_of_samples):
        # Paired traces: identical ray seed, blocking off then on. Aimed at the measured
        # focal spot so the beam actually reaches the target (see GATE_RESULTS.md).
        flux_open, _, _, _ = trace_one_sample(
            scenario, heliostat_group, heliostat_index,
            motor_positions[sample], rays[sample], target_mask[sample],
            None, device, centroid=centroids[sample],
        )
        surfaces = blocker_surfaces(
            heliostat_group, hypothesis, scenario, rays[sample], device
        )
        flux_blocked, _, _, blocking_factor = trace_one_sample(
            scenario, heliostat_group, heliostat_index,
            motor_positions[sample], rays[sample], target_mask[sample],
            surfaces, device, centroid=centroids[sample],
        )

        open_image, blocked_image = flux_open[0], flux_blocked[0]
        blocked_fraction = 1.0 - float(blocking_factor.item())

        # --- L2: signed vertical shift of the FLUX centre of mass -------------------
        com_open = get_center_of_mass(bitmaps=flux_open, device=device)
        com_blocked = get_center_of_mass(bitmaps=flux_blocked, device=device)
        metres_open = _metre_coordinates(
            com_open, target_mask[sample], resolution, scenario.solar_tower, device
        )
        metres_blocked = _metre_coordinates(
            com_blocked, target_mask[sample], resolution, scenario.solar_tower, device
        )
        flux_shift = metres_blocked[:3] - metres_open[:3]
        # The up component is the one Wortberg's claim is about.
        flux_shift_up_mrad = float(flux_shift[2]) / distance_m * 1000.0
        flux_shift_total_mrad = float(torch.norm(flux_shift)) / distance_m * 1000.0

        # --- L3: same shift, but for the UPPER CONTOUR feature ----------------------
        contour_open = extractor(flux_open)
        contour_blocked = extractor(flux_blocked)
        if float(contour_open.sum()) < 1e-8 or float(contour_blocked.sum()) < 1e-8:
            log.warning(f"  sample {sample}: empty contour, skipped")
            continue
        contour_com_open = _metre_coordinates(
            contour_center_of_mass(contour_open), target_mask[sample],
            resolution, scenario.solar_tower, device,
        )
        contour_com_blocked = _metre_coordinates(
            contour_center_of_mass(contour_blocked), target_mask[sample],
            resolution, scenario.solar_tower, device,
        )
        contour_shift = contour_com_blocked[:3] - contour_com_open[:3]
        contour_shift_up_mrad = float(contour_shift[2]) / distance_m * 1000.0
        contour_shift_total_mrad = float(torch.norm(contour_shift)) / distance_m * 1000.0

        # --- L1: where inside the spot does the lost flux sit? ----------------------
        # Positive part of (open - blocked) is the deficit. Compare its centroid with the
        # spot centroid: a NEGATIVE value means the deficit sits BELOW the spot centre,
        # which is what "eaten from the lower part of the spot" requires.
        deficit = (open_image - blocked_image).clamp(min=0)
        if float(deficit.sum()) > 1e-8:
            deficit_com = get_center_of_mass(bitmaps=deficit.unsqueeze(0), device=device)
            deficit_metres = _metre_coordinates(
                deficit_com, target_mask[sample], resolution, scenario.solar_tower, device
            )
            deficit_below_m = float(deficit_metres[2] - metres_open[2])
        else:
            deficit_below_m = float("nan")

        records.append(
            {
                "sample": sample,
                "blocked_fraction": blocked_fraction,
                "flux_shift_up_mrad": flux_shift_up_mrad,
                "flux_shift_total_mrad": flux_shift_total_mrad,
                "contour_shift_up_mrad": contour_shift_up_mrad,
                "contour_shift_total_mrad": contour_shift_total_mrad,
                "deficit_below_spot_centre_m": deficit_below_m,
            }
        )

    def column(key: str) -> list[float]:
        return [r[key] for r in records if r[key] == r[key]]

    def median_or_none(values: list[float]) -> float | None:
        return statistics.median(values) if values else None

    blocked = column("blocked_fraction")
    flux_up = column("flux_shift_up_mrad")
    flux_total = column("flux_shift_total_mrad")
    contour_up = column("contour_shift_up_mrad")
    contour_total = column("contour_shift_total_mrad")
    deficit = column("deficit_below_spot_centre_m")

    summary = {
        "heliostat_id": heliostat_id,
        "hypothesis": hypothesis,
        "scenario": str(scenario_path),
        "distance_m": distance_m,
        "number_of_samples": len(records),
        "number_of_rays": number_of_rays,
        "blocked_fraction_median": median_or_none(blocked),
        "L1_deficit_below_spot_centre_m_median": median_or_none(deficit),
        "L1_samples_deficit_below": sum(1 for d in deficit if d < 0),
        "L2_flux_shift_up_mrad_median": median_or_none(flux_up),
        "L2_samples_com_moved_up": sum(1 for v in flux_up if v > 0),
        "L2_flux_shift_total_mrad_median": median_or_none(flux_total),
        "L3_contour_shift_total_mrad_median": median_or_none(contour_total),
        "L3_contour_shift_up_mrad_median": median_or_none(contour_up),
        "L3_robustness_ratio": (
            median_or_none(contour_total) / median_or_none(flux_total)
            if flux_total and median_or_none(flux_total) > 0 else None
        ),
        "records": records,
    }

    out_dir = output_root / heliostat_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"wortberg_premise_{hypothesis}.json").write_text(
        json.dumps(summary, indent=2)
    )

    n = len(records)
    print()
    def fmt(value, spec: str) -> str:
        return "n/a" if value is None else format(value, spec)

    blocked_str = fmt(
        summary["blocked_fraction_median"] * 100
        if summary["blocked_fraction_median"] is not None else None, ".1f"
    )
    print(f"  {heliostat_id}  hypothesis={hypothesis}  n={n}  blocked median {blocked_str}%")
    print(f"    L1  deficit sits {fmt(summary['L1_deficit_below_spot_centre_m_median'], '+.3f')} m "
          f"vs spot centre (negative = below, as Wortberg requires)   "
          f"below in {summary['L1_samples_deficit_below']}/{n}")
    print(f"    L2  flux COM vertical shift {fmt(summary['L2_flux_shift_up_mrad_median'], '+.3f')} mrad "
          f"(positive = up, as Wortberg requires)   up in {summary['L2_samples_com_moved_up']}/{n}")
    print(f"    L3  flux COM total shift    {fmt(summary['L2_flux_shift_total_mrad_median'], '.3f')} mrad")
    print(f"        contour COM total shift {fmt(summary['L3_contour_shift_total_mrad_median'], '.3f')} mrad"
          f"   ratio {fmt(summary['L3_robustness_ratio'], '.2f')}  (<1 = contour more robust)")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the Wortberg contour-loss premise.")
    parser.add_argument("heliostat_ids", nargs="+")
    parser.add_argument("--daic", action="store_true")
    parser.add_argument("--hypothesis", default="lower")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--surface-points", type=int, default=25)
    parser.add_argument("--rays", type=int, default=20)
    args = parser.parse_args()

    set_logger_config()
    logging.getLogger().setLevel(logging.WARNING)
    log.setLevel(logging.INFO)
    torch.manual_seed(7)

    paint_dir, scenario_root, output_root = _paths(args.daic)
    for hid in args.heliostat_ids:
        run(
            heliostat_id=hid,
            hypothesis=args.hypothesis,
            paint_dir=paint_dir,
            scenario_root=scenario_root,
            output_root=output_root,
            max_samples=args.max_samples,
            surface_points_per_facet=args.surface_points,
            number_of_rays=args.rays,
            device=torch.device("cpu"),
        )


if __name__ == "__main__":
    main()
