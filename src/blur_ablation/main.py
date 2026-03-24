import datetime
import json
import logging
import math
import pathlib
import random
import sys

_pkg = pathlib.Path(__file__).parent   # .../src/blur_ablation/
_src = _pkg.parent                     # .../src/
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_pkg))

# ===================================================================
# Configuration
# ===================================================================

IS_ON_DAIC = True
SMOKE_TEST = not IS_ON_DAIC  # 2 heliostats, rays={10,20}, sigmas={0,5}

if IS_ON_DAIC:
    import matplotlib
    matplotlib.use("Agg")

import torch
import h5py
import paint.util.paint_mappings as paint_mappings
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import set_logger_config
from artist.util.environment_setup import get_device

from utils.checkpointing import load_kinematic_parameters
from utils.evaluation import build_heliostat_data_mapping
from blur_ablation.sweep import run_blur_sweep
from blur_ablation.plotting import plot_heatmap, plot_line_plots, plot_sigma_sweep, plot_field_heatmap, plot_field_coordinates

set_logger_config()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

random.seed(42)

if IS_ON_DAIC:
    BASE_DIR = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
    BENCHMARK_DIR = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/src/paint_benchmarks")
else:
    BASE_DIR = pathlib.Path(__file__).parent.parent.parent
    BENCHMARK_DIR = BASE_DIR / "datasets" / "paint_benchmarks"

BENCHMARK_NAME = "benchmark_split-balanced_train-10_validation-30"
SCENARIO_PATH = BASE_DIR / "scenarios" / "deflectometry_scenario" / "deflectometry_scenario.h5"

KINEMATIC_CHECKPOINT = (
    BASE_DIR / "outputs" / "basic_kinematic_parameters"
    / "focal_spot_loss_deflectometry_only" / "all_kinematic_parameters.json"
)

_run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = (
    BASE_DIR / "outputs" / "local_runs" / f"blur_ablation_{_run_timestamp}"
    if not IS_ON_DAIC
    else BASE_DIR / "outputs" / f"blur_ablation_{_run_timestamp}"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_log_handler = logging.FileHandler(OUTPUT_DIR / "blur_ablation.log")
_log_handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
logging.getLogger().addHandler(_log_handler)

BENCHMARK_CSV = BENCHMARK_DIR / "splits" / f"{BENCHMARK_NAME}.csv"
CALIBRATION_PROPERTIES_DIR = BENCHMARK_DIR / "datasets" / BENCHMARK_NAME / "calibration_properties"
FLUX_IMAGE_DIR = BENCHMARK_DIR / "datasets" / BENCHMARK_NAME / "flux_image"

# Sweep parameters.
RAYS_CONFIGS = [10, 20, 50]
SIGMA_CONFIGS = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
REF_RAYS = 200        # rays for high-quality reference
SAMPLE_LIMIT = 10     # measurements per heliostat (train split)

# Heliostat selection grid.
DISTANCE_BANDS = [("near", 0, 100), ("mid", 100, 175), ("far", 175, float("inf"))]
QUADRANTS = ["N", "E", "S", "W"]
HELIOSTATS_PER_CELL = 2  # 2 × 12 cells = 24 heliostats (~25 total)

# Smoke test overrides.
if SMOKE_TEST:
    RAYS_CONFIGS = [10, 20]
    SIGMA_CONFIGS = [0.0, 5.0]
    REF_RAYS = 50
    SAMPLE_LIMIT = 3
    HELIOSTATS_PER_CELL = 1
    log.info("[SMOKE TEST] Reduced configs active.")

print(f"Output directory: {OUTPUT_DIR}")
print(f"Scenario: {SCENARIO_PATH}")
print(f"Checkpoint: {KINEMATIC_CHECKPOINT}")


# ===================================================================
# Device
# ===================================================================

device = get_device()
print(f"Device: {device}")


# ===================================================================
# Load train mapping (all deflectometry heliostats)
# ===================================================================

full_train_mapping = build_heliostat_data_mapping(
    benchmark_csv=BENCHMARK_CSV,
    calibration_properties_dir=CALIBRATION_PROPERTIES_DIR,
    flux_image_dir=FLUX_IMAGE_DIR,
    split="train",
)
print(f"Full train mapping: {len(full_train_mapping)} heliostats")

# Index by heliostat name for fast lookup.
mapping_by_name = {name: (cal, flux) for name, cal, flux in full_train_mapping}


# ===================================================================
# Load scenario (25×25) to read heliostat positions
# ===================================================================

print("Loading scenario (25×25) to read heliostat positions …")
with h5py.File(SCENARIO_PATH, "r") as f:
    scenario_pos = Scenario.load_scenario_from_hdf5(
        scenario_file=f,
        device=device,
        number_of_surface_points_per_facet=torch.tensor([25, 25]),
    )

# Collect (name, distance_m, bearing_deg) for each heliostat in the scenario.
heliostat_info: list[dict] = []
for heliostat_group in scenario_pos.heliostat_field.heliostat_groups:
    positions = heliostat_group.positions[:, :3].cpu()  # [N, 3] — (E, N, U) in metres
    for idx, name in enumerate(heliostat_group.names):
        if name not in mapping_by_name:
            continue  # not in benchmark
        e, n, u = positions[idx].tolist()
        dist = math.sqrt(e ** 2 + n ** 2)  # horizontal ground distance
        bearing_deg = math.degrees(math.atan2(e, n)) % 360  # N=0, E=90, S=180, W=270
        if bearing_deg < 45 or bearing_deg >= 315:
            quadrant = "N"
        elif bearing_deg < 135:
            quadrant = "E"
        elif bearing_deg < 225:
            quadrant = "S"
        else:
            quadrant = "W"
        # Assign distance band.
        band = "far"
        for label, lo, hi in DISTANCE_BANDS:
            if lo <= dist < hi:
                band = label
                break
        heliostat_info.append({
            "name": name,
            "distance_m": dist,
            "bearing_deg": bearing_deg,
            "quadrant": quadrant,
            "band": band,
            "east_m": e,
            "north_m": n,
        })

print(f"Heliostat info collected for {len(heliostat_info)} heliostats.")

# ===================================================================
# Heliostat selection: sample HELIOSTATS_PER_CELL per (band × quadrant)
# ===================================================================

from collections import defaultdict
cell_buckets: dict[tuple, list] = defaultdict(list)
for h in heliostat_info:
    cell_buckets[(h["band"], h["quadrant"])].append(h)

selected_heliostats: list[dict] = []
for band, _, _ in DISTANCE_BANDS:
    for q in QUADRANTS:
        cell = cell_buckets.get((band, q), [])
        if cell:
            chosen = random.sample(cell, min(HELIOSTATS_PER_CELL, len(cell)))
            selected_heliostats.extend(chosen)

print(f"\nSelected {len(selected_heliostats)} heliostats:")
for h in selected_heliostats:
    print(f"  {h['name']:8s}  dist={h['distance_m']:6.1f} m  {h['band']:4s}  {h['quadrant']}")

heliostat_distances = {h["name"]: h["distance_m"] for h in selected_heliostats}
selected_names = set(heliostat_distances.keys())

# Build filtered mapping with only selected heliostats.
selected_mapping = [
    (name, mapping_by_name[name][0][:SAMPLE_LIMIT], mapping_by_name[name][1][:SAMPLE_LIMIT])
    for name in (h["name"] for h in selected_heliostats)
    if name in mapping_by_name
]
print(f"Selected mapping: {len(selected_mapping)} heliostats, up to {SAMPLE_LIMIT} measurements each.")

# Save heliostat metadata.
with open(OUTPUT_DIR / "selected_heliostats.json", "w") as fp:
    json.dump(selected_heliostats, fp, indent=2)


# ===================================================================
# Data parser
# ===================================================================

data_parser = PaintCalibrationDataParser(
    sample_limit=SAMPLE_LIMIT,
    centroid_extraction_method=paint_mappings.UTIS_KEY,
)


# ===================================================================
# Load both scenario instances and apply pre-trained kinematics
# ===================================================================

def _load_scenario_with_checkpoint(surface_pts: list[int]) -> Scenario:
    with h5py.File(SCENARIO_PATH, "r") as f:
        sc = Scenario.load_scenario_from_hdf5(
            scenario_file=f,
            device=device,
            number_of_surface_points_per_facet=torch.tensor(surface_pts),
        )
    load_kinematic_parameters(sc, KINEMATIC_CHECKPOINT, device)
    return sc


print(f"\nLoading scenario 50×50 (reference) + applying checkpoint …")
scenario_50 = _load_scenario_with_checkpoint([50, 50])

print(f"Loading scenario 25×25 (test configs) + applying checkpoint …")
scenario_25 = _load_scenario_with_checkpoint([25, 25])


# ===================================================================
# Run sweep
# ===================================================================

print(f"\nRunning sweep: {len(RAYS_CONFIGS)} ray configs × {len(SIGMA_CONFIGS)} sigma configs")
print(f"  rays configs: {RAYS_CONFIGS}")
print(f"  sigma configs: {SIGMA_CONFIGS}")
print(f"  reference rays: {REF_RAYS} (50×50)")

records = run_blur_sweep(
    scenario_50=scenario_50,
    scenario_25=scenario_25,
    selected_mapping=selected_mapping,
    data_parser=data_parser,
    rays_configs=RAYS_CONFIGS,
    sigma_configs=SIGMA_CONFIGS,
    ref_rays=REF_RAYS,
    device=device,
)

# Save raw results.
with open(OUTPUT_DIR / "sweep_results.json", "w") as fp:
    json.dump(records, fp, indent=2)
print(f"\nResults saved to {OUTPUT_DIR / 'sweep_results.json'} ({len(records)} records).")


# ===================================================================
# Determine optimal sigma (minimises mean MSE across all heliostats)
# ===================================================================

import pandas as pd

df = pd.DataFrame(records)
blur_df = df[df["sigma"] > 0]
if not blur_df.empty:
    sigma_mse = blur_df.groupby("sigma")["mse"].mean()
    optimal_sigma = float(sigma_mse.idxmin())
    print(f"\nOptimal sigma: {optimal_sigma} (mean MSE = {sigma_mse[optimal_sigma]:.4f})")
else:
    optimal_sigma = 0.0

with open(OUTPUT_DIR / "optimal_sigma.json", "w") as fp:
    json.dump({"optimal_sigma": optimal_sigma}, fp)


# ===================================================================
# Generate plots
# ===================================================================

print("\nGenerating plots …")

plot_heatmap(
    records=records,
    heliostat_distances=heliostat_distances,
    output_path=OUTPUT_DIR / "fig1_heatmap_rays_vs_distance.png",
    optimal_sigma=optimal_sigma,
)
print("  Fig 1: heatmap saved.")

plot_line_plots(
    records=records,
    heliostat_distances=heliostat_distances,
    output_path=OUTPUT_DIR / "fig2_line_plots.png",
    optimal_sigma=optimal_sigma,
)
print("  Fig 2: line plots saved.")

plot_sigma_sweep(
    records=records,
    heliostat_distances=heliostat_distances,
    output_path=OUTPUT_DIR / "fig3_sigma_sweep.png",
    fixed_n_rays=RAYS_CONFIGS[0],
)
print("  Fig 3: sigma sweep saved.")

all_heliostat_positions = {h["name"]: (h["east_m"], h["north_m"]) for h in heliostat_info}

plot_field_heatmap(
    selected_heliostats=selected_heliostats,
    output_path=OUTPUT_DIR / "fig4_field_heatmap.png",
    heliostats_per_cell=HELIOSTATS_PER_CELL,
)
print("  Fig 4: field heatmap saved.")

plot_field_coordinates(
    selected_heliostats=selected_heliostats,
    all_heliostat_positions=all_heliostat_positions,
    output_path=OUTPUT_DIR / "fig5_field_coordinates.png",
)
print("  Fig 5: field coordinates saved.")

print(f"\nAll outputs written to: {OUTPUT_DIR}")
