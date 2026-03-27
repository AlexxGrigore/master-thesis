import datetime
import json
import logging
import math
import pathlib
import sys

_pkg = pathlib.Path(__file__).parent   # .../src/blur_ablation/
_src = _pkg.parent                     # .../src/
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_pkg))

# ===================================================================
# Configuration
# ===================================================================

IS_ON_DAIC = True
SMOKE_TEST = not IS_ON_DAIC  # reduced rays/sigmas/sample_limit/surface_configs
# SMOKE_TEST = False

if IS_ON_DAIC:
    import matplotlib
    matplotlib.use("Agg")

import time

import numpy as np
import torch
import h5py
import paint.util.paint_mappings as paint_mappings
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.scenario.scenario import Scenario
from artist.util import set_logger_config
from artist.util.environment_setup import get_device

from utils.checkpointing import load_kinematic_parameters
from utils.evaluation import build_heliostat_data_mapping
from blur_ablation.sweep import run_blur_sweep, trace_flux_for_mapping
from blur_ablation.plotting import (
    plot_heatmap, plot_line_plots, plot_sigma_sweep,
    plot_field_heatmap, plot_field_coordinates,
    plot_surface_pts_comparison,
)

set_logger_config()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

if IS_ON_DAIC:
    BASE_DIR = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis")
    BENCHMARK_DIR = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/src/paint_benchmarks")
else:
    BASE_DIR = pathlib.Path(__file__).parent.parent.parent
    BENCHMARK_DIR = BASE_DIR / "datasets" / "paint_benchmarks"

BENCHMARK_NAME = "benchmark_split-balanced_train-10_validation-30"

# 18-heliostat scenario — pre-selected stratified subset (blur_ablation, random.seed(42)).
SCENARIO_PATH = BASE_DIR / "scenarios" / "blur_ablation_scenario" / "blur_ablation_scenario.h5"

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
SURFACE_CONFIGS = [25, 50, 75, 100]   # N in N×N surface points per facet
RAYS_CONFIGS    = [10, 20, 50]
SIGMA_CONFIGS   = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
REF_RAYS        = 200        # rays for high-quality reference
SAMPLE_LIMIT    = 10         # measurements per heliostat (train split)

# Used only for plot labelling (fig4) — reflects how the scenario was originally built.
HELIOSTATS_PER_CELL = 2

# Smoke test overrides.
if SMOKE_TEST:
    SURFACE_CONFIGS = [25, 50]
    RAYS_CONFIGS    = [10, 20]
    SIGMA_CONFIGS   = [0.0, 5.0]
    REF_RAYS        = 50
    SAMPLE_LIMIT    = 3
    log.info("[SMOKE TEST] Reduced configs active.")

print(f"Output directory: {OUTPUT_DIR}")
print(f"Scenario: {SCENARIO_PATH}")
print(f"Checkpoint: {KINEMATIC_CHECKPOINT}")
print(f"Surface configs: {SURFACE_CONFIGS}")


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
# Load one scenario per surface config, each with pre-trained kinematics
# ===================================================================

def _load_scenario_with_checkpoint(surface_pts: int) -> Scenario:
    with h5py.File(SCENARIO_PATH, "r") as f:
        sc = Scenario.load_scenario_from_hdf5(
            scenario_file=f,
            device=device,
            number_of_surface_points_per_facet=torch.tensor([surface_pts, surface_pts]),
        )
    load_kinematic_parameters(sc, KINEMATIC_CHECKPOINT, device)
    return sc


print(f"\nLoading reference scenario ({SURFACE_CONFIGS[0]}×{SURFACE_CONFIGS[0]}) for metadata …")
_ref_scenario = _load_scenario_with_checkpoint(SURFACE_CONFIGS[0])
print("Done.")


# ===================================================================
# Build heliostat metadata from scenario (for mapping and plots)
# ===================================================================

heliostat_info: list[dict] = []
for heliostat_group in _ref_scenario.heliostat_field.heliostat_groups:
    positions = heliostat_group.positions[:, :3].cpu()  # [N, 3] — (E, N, U) in metres
    for idx, name in enumerate(heliostat_group.names):
        e, n, u = positions[idx].tolist()
        dist = math.sqrt(e ** 2 + n ** 2)
        heliostat_info.append({
            "name": name,
            "distance_m": dist,
            "east_m": e,
            "north_m": n,
        })

# Assign distance band and lateral column (needed for fig4).
DISTANCE_BANDS = [("near", 0, 100), ("mid", 100, 175), ("far", 175, float("inf"))]
for h in heliostat_info:
    h["band"] = next(
        label for label, lo, hi in DISTANCE_BANDS if lo <= h["distance_m"] < hi
    )

for band_label, _, _ in DISTANCE_BANDS:
    band_h = [h for h in heliostat_info if h["band"] == band_label]
    if not band_h:
        continue
    east_vals = np.array([h["east_m"] for h in band_h])
    p33 = float(np.percentile(east_vals, 33.33))
    p66 = float(np.percentile(east_vals, 66.67))
    for h in band_h:
        h["column"] = "left" if h["east_m"] < p33 else ("mid" if h["east_m"] <= p66 else "right")

print(f"Heliostats in scenario: {len(heliostat_info)}")

# Free reference scenario — no longer needed after metadata extraction.
del _ref_scenario
if device.type == "cuda":
    torch.cuda.empty_cache()

heliostat_distances = {h["name"]: h["distance_m"] for h in heliostat_info}

# Build mapping limited to heliostats present in both scenario and benchmark.
selected_mapping = [
    (name, mapping_by_name[name][0][:SAMPLE_LIMIT], mapping_by_name[name][1][:SAMPLE_LIMIT])
    for name in (h["name"] for h in heliostat_info)
    if name in mapping_by_name
]
print(f"Selected mapping: {len(selected_mapping)} heliostats, up to {SAMPLE_LIMIT} measurements each.")

# Save heliostat metadata.
with open(OUTPUT_DIR / "selected_heliostats.json", "w") as fp:
    json.dump(heliostat_info, fp, indent=2)


# ===================================================================
# Data parser
# ===================================================================

data_parser = PaintCalibrationDataParser(
    sample_limit=SAMPLE_LIMIT,
    centroid_extraction_method=paint_mappings.UTIS_KEY,
)


# ===================================================================
# GPU memory & speed benchmark (single heliostat, all surface configs)
# ===================================================================

def _bench_config(scenario: Scenario, n_rays: int, surface_pts: int, heliostat_mapping: list) -> dict:
    scenario.set_number_of_rays(n_rays)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    trace_flux_for_mapping(
        scenario=scenario,
        heliostat_data_mapping=heliostat_mapping,
        data_parser=data_parser,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else float("nan")
    return {
        "surface_pts": surface_pts,
        "n_rays": n_rays,
        "wall_time_s": round(elapsed, 3),
        "peak_gpu_mem_gb": round(peak_mem_gb, 4),
    }

bench_heliostat_mapping = selected_mapping[:1]
print(f"\nGPU benchmark on heliostat '{bench_heliostat_mapping[0][0]}' …")
bench_results = []
for sp in SURFACE_CONFIGS:
    _bench_scenario = _load_scenario_with_checkpoint(sp)
    for n_rays in [10, REF_RAYS]:
        r = _bench_config(_bench_scenario, n_rays, sp, bench_heliostat_mapping)
        bench_results.append(r)
        print(f"  {sp:3d}×{sp:<3d}  {n_rays:4d} rays: {r['wall_time_s']:.3f} s,  peak GPU: {r['peak_gpu_mem_gb']:.4f} GB")
    del _bench_scenario
    if device.type == "cuda":
        torch.cuda.empty_cache()

with open(OUTPUT_DIR / "benchmark_gpu.json", "w") as fp:
    json.dump(bench_results, fp, indent=2)
print(f"  Saved → {OUTPUT_DIR / 'benchmark_gpu.json'}")


# ===================================================================
# Run sweep (surface_pts × n_rays × sigma)
# ===================================================================

print(f"\nRunning sweep: {len(SURFACE_CONFIGS)} surface configs × {len(RAYS_CONFIGS)} ray configs × {len(SIGMA_CONFIGS)} sigma configs")
print(f"  surface configs: {SURFACE_CONFIGS}")
print(f"  rays configs: {RAYS_CONFIGS}")
print(f"  sigma configs: {SIGMA_CONFIGS}")
print(f"  reference rays: {REF_RAYS}")

records = run_blur_sweep(
    scenario_loader=_load_scenario_with_checkpoint,
    surface_configs=SURFACE_CONFIGS,
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
# Determine optimal sigma per surface config
# ===================================================================

import pandas as pd

df = pd.DataFrame(records)
optimal_sigmas: dict[int, float] = {}
for sp in SURFACE_CONFIGS:
    blur_df = df[(df["surface_pts"] == sp) & (df["sigma"] > 0)]
    if not blur_df.empty:
        sigma_mse = blur_df.groupby("sigma")["mse"].mean()
        optimal_sigmas[sp] = float(sigma_mse.idxmin())
        print(f"  Optimal sigma for {sp}×{sp}: {optimal_sigmas[sp]} (mean MSE = {sigma_mse[optimal_sigmas[sp]]:.4f})")
    else:
        optimal_sigmas[sp] = 0.0

with open(OUTPUT_DIR / "optimal_sigma.json", "w") as fp:
    json.dump(optimal_sigmas, fp, indent=2)


# ===================================================================
# Generate plots
# ===================================================================

print("\nGenerating plots …")

# Figs 1–3: per-surface-config (one set per N×N in a subdirectory).
for sp in SURFACE_CONFIGS:
    sp_records = [r for r in records if r["surface_pts"] == sp]
    sp_dir = OUTPUT_DIR / f"surface_{sp}x{sp}"
    sp_dir.mkdir(parents=True, exist_ok=True)
    opt_sigma = optimal_sigmas[sp]

    plot_heatmap(
        records=sp_records,
        heliostat_distances=heliostat_distances,
        output_path=sp_dir / "fig1_heatmap_rays_vs_distance.png",
        optimal_sigma=opt_sigma,
    )
    plot_line_plots(
        records=sp_records,
        heliostat_distances=heliostat_distances,
        output_path=sp_dir / "fig2_line_plots.png",
        optimal_sigma=opt_sigma,
    )
    plot_sigma_sweep(
        records=sp_records,
        heliostat_distances=heliostat_distances,
        output_path=sp_dir / "fig3_sigma_sweep.png",
        fixed_n_rays=RAYS_CONFIGS[0],
    )
    print(f"  Figs 1–3 for {sp}×{sp} saved to {sp_dir.name}/")

# Fig 4 & 5: field layout (surface-config-independent).
all_heliostat_positions = {h["name"]: (h["east_m"], h["north_m"]) for h in heliostat_info}

plot_field_heatmap(
    selected_heliostats=heliostat_info,
    output_path=OUTPUT_DIR / "fig4_field_heatmap.png",
    heliostats_per_cell=HELIOSTATS_PER_CELL,
)
print("  Fig 4: field heatmap saved.")

plot_field_coordinates(
    selected_heliostats=heliostat_info,
    all_heliostat_positions=all_heliostat_positions,
    output_path=OUTPUT_DIR / "fig5_field_coordinates.png",
)
print("  Fig 5: field coordinates saved.")

# Fig 6: cross-surface-config comparison — sigma vs MSE, one line per surface config.
plot_surface_pts_comparison(
    records=records,
    rays_configs=RAYS_CONFIGS,
    output_path=OUTPUT_DIR / "fig6_surface_pts_comparison.png",
)
print("  Fig 6: surface pts comparison saved.")

print(f"\nAll outputs written to: {OUTPUT_DIR}")
