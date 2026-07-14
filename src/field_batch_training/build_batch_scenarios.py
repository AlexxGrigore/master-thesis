"""Build the field-batch scenario files: 13 HDF5 scenarios of <=100 heliostats each.

Slices the 1277 heliostats of the 50-20-20 benchmark alphabetically into batches
of 100 (last batch 77) and creates one multi-heliostat ARTIST scenario per batch,
with ideal surfaces (canting from the PAINT properties) — same construction as
scenarios/full_benchmark_ideal/ideal_1277.h5, just batched.

Inputs (must exist — produced by download_paint_benchmark.py):
  PAINT_DIR/splits/benchmark_split-balanced_train-50_validation-20.csv
  PAINT_DIR/heliostats/WRI1030197-tower-measurements.json
  PAINT_DIR/heliostats/{HID}/Properties/{HID}-heliostat-properties.json

Outputs:
  SCENARIO_DIR/field_batch_{00..12}.h5
  SCENARIO_DIR/field_batches_manifest.json   (batch -> heliostat list)

Usage:
  python build_batch_scenarios.py                 # local paths
  python build_batch_scenarios.py --daic          # DAIC paths
  python build_batch_scenarios.py --batch-id 3    # only one batch (resumable)
  python build_batch_scenarios.py --batch-size 100
"""
import argparse
import csv
import json
import logging
import pathlib
import sys
import time

import torch

_here = pathlib.Path(__file__).resolve().parent   # field_batch_training/
_src = _here.parent                                # src/
sys.path.insert(0, str(_src))

from artist.io import paint_scenario_parser  # noqa: E402
from artist.scenario.h5_scenario_generator import H5ScenarioGenerator  # noqa: E402
from artist.util import constants as config_dictionary, set_logger_config  # noqa: E402
from artist.util import get_device  # noqa: E402
from artist.util.config import LightSourceConfig, LightSourceListConfig  # noqa: E402

log = logging.getLogger(__name__)

BENCHMARK = "benchmark_split-balanced_train-50_validation-20"
TOWER_FILE = "WRI1030197-tower-measurements.json"
_N_CONTROL_POINTS = torch.tensor([20, 20])   # same as the other scenario creators


def _paths(daic: bool):
    if daic:
        paint = pathlib.Path("/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint")
        scen = pathlib.Path("/home/nfs/agrigore/projects/githubProjects/master-thesis/scenarios/field_batches")
    else:
        root = _src.parent
        paint = root / "datasets" / "paint"
        scen = root / "scenarios" / "field_batches"
    return paint, scen


def _make_light_source_config() -> LightSourceListConfig:
    return LightSourceListConfig(
        light_source_list=[
            LightSourceConfig(
                light_source_key="sun_1",
                light_source_type=config_dictionary.sun_key,
                number_of_rays=10,
                distribution_type=config_dictionary.light_source_distribution_is_normal,
                mean=0.0,
                covariance=4.3681e-06,
            )
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Create batched multi-heliostat field scenarios.")
    p.add_argument("--daic", action="store_true", help="Use DAIC paths.")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--batch-id", type=int, default=None,
                   help="Build only this batch (default: all). Resumable per batch.")
    p.add_argument("--force", action="store_true", help="Overwrite existing batch files.")
    args = p.parse_args()

    set_logger_config()
    device = get_device()
    paint_dir, scen_dir = _paths(args.daic)
    scen_dir.mkdir(parents=True, exist_ok=True)

    # 1. heliostat list — alphabetical, from the benchmark split CSV
    csv_path = paint_dir / "splits" / f"{BENCHMARK}.csv"
    with open(csv_path) as fh:
        heliostats = sorted({r["HeliostatId"] for r in csv.DictReader(fh)})
    log.info(f"{len(heliostats)} heliostats in {BENCHMARK}")

    batches = [heliostats[i:i + args.batch_size]
               for i in range(0, len(heliostats), args.batch_size)]
    log.info(f"{len(batches)} batches of sizes {[len(b) for b in batches]}")

    # 2. verify inputs; collect (hid, properties_path) tuples per batch
    heliostats_dir = paint_dir / "heliostats"
    tower_file = heliostats_dir / TOWER_FILE
    if not tower_file.exists():
        sys.exit(f"Missing tower measurements: {tower_file}")

    def props_path(hid: str) -> pathlib.Path:
        return heliostats_dir / hid / "Properties" / f"{hid}-heliostat-properties.json"

    missing = [h for b in batches for h in b if not props_path(h).exists()]
    if missing:
        sys.exit(f"Missing Properties for {len(missing)} heliostats "
                 f"(run download_paint_benchmark.py first): {missing[:10]} ...")

    # 3. tower / power-plant configs — shared by every batch
    (power_plant_config,
     target_area_list_planar_config,
     target_area_list_cylindrical_config) = (
        paint_scenario_parser.extract_paint_tower_measurements(
            tower_measurements_path=tower_file, device=device,
        )
    )

    manifest = {}
    todo = range(len(batches)) if args.batch_id is None else [args.batch_id]
    t0 = time.time()
    for bi in todo:
        batch = batches[bi]
        out = scen_dir / f"field_batch_{bi:02d}.h5"
        manifest[f"{bi:02d}"] = batch
        if out.exists() and not args.force:
            log.info(f"[{bi:02d}] exists, skipped: {out}")
            continue
        log.info(f"[{bi:02d}] building {out.name} with {len(batch)} heliostats "
                 f"({batch[0]}..{batch[-1]})")
        heliostat_list_config, prototype_config = (
            paint_scenario_parser.extract_paint_heliostats_ideal_surface(
                paths=[(hid, props_path(hid)) for hid in batch],
                power_plant_position=power_plant_config.power_plant_position,
                number_of_nurbs_control_points=_N_CONTROL_POINTS,
                device=device,
            )
        )
        H5ScenarioGenerator(
            file_path=out,
            power_plant_config=power_plant_config,
            target_area_list_planar_config=target_area_list_planar_config,
            target_area_list_cylindrical_config=target_area_list_cylindrical_config,
            light_source_list_config=_make_light_source_config(),
            prototype_config=prototype_config,
            heliostat_list_config=heliostat_list_config,
        ).generate_scenario()
        log.info(f"[{bi:02d}] done ({time.time() - t0:.0f}s elapsed)")

    if args.batch_id is None:
        with open(scen_dir / "field_batches_manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2)
        log.info(f"manifest written: {scen_dir / 'field_batches_manifest.json'}")


if __name__ == "__main__":
    main()
