"""
Build per-heliostat IDEAL-surface scenario HDF5 files for every heliostat in a
benchmark CSV — e.g. the field-wide 50/20/20 split (1277 heliostats).

Why a separate script rather than `--only-ideal` on create_full_field_scenarios.py:
that flag DROPS heliostats that have deflectometry data (`defl_list = []`) instead of
routing them to the ideal builder, so it does not give uniform coverage of an
arbitrary heliostat list. This script always builds ideal surfaces, for every
heliostat requested, regardless of deflectometry availability — deliberately, so a
field-wide run uses one consistent surface model throughout (comparable pointing
accuracy field-wide, no confound between "kinematic error" and "which surface model
this heliostat happened to get"). It reuses create_one_scenario() from
create_full_field_scenarios.py rather than duplicating the tower/scenario-generation
code, and writes into the SAME default location that script and _find_scenario()
already use, so nothing downstream needs to know this script exists.

Resumable: skips a heliostat whose scenario file already exists, unless --force.

Usage
-----
    python build_ideal_scenarios_for_benchmark.py \
        --benchmark-csv ../../../datasets/paint/splits/benchmark_split-balanced_train-50_validation-20.csv
    python build_ideal_scenarios_for_benchmark.py --smoke-test
"""

import argparse
import logging
import pathlib
import sys
import time

import pandas as pd
import torch
from tqdm import tqdm

_here = pathlib.Path(__file__).resolve().parent          # daic_full_field/
_src  = _here.parent.parent                               # src/
sys.path.insert(0, str(_src))
sys.path.insert(0, str(_here.parent))                     # one_heliostat_demo/
sys.path.insert(0, str(_here))

from artist.io import paint_scenario_parser               # noqa: E402
from artist.util import set_logger_config                 # noqa: E402
from artist.util import get_device                        # noqa: E402

from create_full_field_scenarios import create_one_scenario  # noqa: E402

log = logging.getLogger(__name__)

_LOCAL_BASE       = _src.parent
_LOCAL_PAINT_HELS = _LOCAL_BASE / "datasets" / "paint" / "heliostats"
_DEFAULT_OUT      = _LOCAL_BASE / "scenarios" / "full_field_one_heliostat_scenarios"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-csv", type=pathlib.Path, required=True,
                   help="Benchmark split CSV; every unique HeliostatId gets a scenario.")
    p.add_argument("--heliostats-dir", type=pathlib.Path, default=_LOCAL_PAINT_HELS)
    p.add_argument("--output-dir", type=pathlib.Path, default=_DEFAULT_OUT)
    p.add_argument("--force", action="store_true", help="Overwrite existing scenario files.")
    p.add_argument("--smoke-test", action="store_true", help="First 3 heliostats only.")
    return p.parse_args()


def main() -> None:
    set_logger_config()
    logging.getLogger().setLevel(logging.INFO)
    torch.manual_seed(7)
    args = _parse_args()

    hids = sorted(pd.read_csv(args.benchmark_csv)["HeliostatId"].unique())
    if args.smoke_test:
        hids = hids[:3]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(args.output_dir / "build_ideal_scenarios.log")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info(f"Benchmark CSV   : {args.benchmark_csv}")
    log.info(f"Heliostats      : {len(hids)}")
    log.info(f"Output dir      : {args.output_dir}")

    tower_file = args.heliostats_dir / "WRI1030197-tower-measurements.json"
    if not tower_file.exists():
        log.error(f"Tower measurements file not found: {tower_file}")
        sys.exit(1)

    device = get_device()
    log.info(f"Device: {device}")
    power_plant_config, target_area_list_planar_config, target_area_list_cylindrical_config = (
        paint_scenario_parser.extract_paint_tower_measurements(
            tower_measurements_path=tower_file, device=device
        )
    )

    missing_props, ok, skipped_existing, failed = [], [], [], []
    t0 = time.time()
    for hid in tqdm(hids, desc="Building ideal scenarios", unit="hel"):
        props_files = sorted(
            (args.heliostats_dir / hid / "Properties").glob(f"{hid}-heliostat-properties.json")
        )
        if not props_files:
            log.warning(f"  {hid}: properties JSON not found — skipped")
            missing_props.append(hid)
            continue

        out_path = args.output_dir / "ideal" / hid / "scenario_ideal.h5"
        if out_path.exists() and not args.force:
            skipped_existing.append(hid)
            continue

        try:
            create_one_scenario(
                hid=hid,
                props_path=props_files[0],
                defl_path=None,          # ideal surface, uniformly, regardless of availability
                out_path=out_path,
                power_plant_config=power_plant_config,
                target_area_list_planar_config=target_area_list_planar_config,
                target_area_list_cylindrical_config=target_area_list_cylindrical_config,
                device=device,
            )
            ok.append(hid)
        except Exception as exc:  # noqa: BLE001 — one bad heliostat must not abort the batch
            log.error(f"  {hid}: FAILED — {exc}", exc_info=True)
            failed.append(hid)

    elapsed_min = (time.time() - t0) / 60.0
    log.info(
        f"Done in {elapsed_min:.1f} min — built {len(ok)}, already existed {len(skipped_existing)}, "
        f"missing properties {len(missing_props)}, failed {len(failed)}"
    )
    if missing_props:
        log.warning(f"Missing properties: {missing_props}")
    if failed:
        log.warning(f"Failed: {failed}")
    print(f"\nBuilt {len(ok)} + {len(skipped_existing)} already present "
          f"= {len(ok) + len(skipped_existing)}/{len(hids)} scenarios ready under {args.output_dir}")


if __name__ == "__main__":
    main()
