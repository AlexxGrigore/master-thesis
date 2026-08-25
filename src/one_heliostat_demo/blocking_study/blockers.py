"""Select the heliostats that can possibly block a given heliostat's reflected beam.

The exact blocking is computed by ARTIST during ray tracing. This module only has to
produce a conservative SUPERSET of candidates, so that the neighbourhood scenario stays
small (a couple of dozen heliostats) without ever dropping a real blocker.

The test is a BOX in front of the trained heliostat, measured along the beam to the target:
a candidate is kept when it lies between 0 and `forward_m` along that beam AND within
`lateral_m` of the beam line sideways. The test is orientation free, so it holds for any
neighbour state (stowed, aimed at any target), which is the whole point: the neighbourhood
scenario must be built before the hypothesis is chosen.

Why a box rather than a semicircle. Measured across all 1277 heliostats and 3 targets:

    rule                  median   p90   max  neighbours
    semicircle 50 m           93   175   189
    semicircle 30 m           38    66    70
    box 30 m x +/-6 m         11    17    20

A semicircle grows as area, so most of what it catches sits far off to the side where the
beam, being only one mirror wide, can never reach. The lateral cut removes that waste for
free. Both are safe; the box is roughly 3.5x cheaper to ray trace.

The defaults (30 m forward, 6 m lateral) are set against the measured worst case in the
real field: the analytic reach `x_max` never exceeds 23.4 m anywhere (see
`blocking_reach.analytic_reach` and the field-wide sweep), and no real blocker was ever
found further than 3.6 m sideways, about one mirror width. Both defaults therefore carry a
comfortable margin. For a tighter, per-heliostat radius use `blocking_reach.analytic_reach`
instead of the fixed 30 m.

Positions come from the PAINT properties JSONs through ARTIST's own WGS84 to local ENU
conversion, so they agree with the scenario file exactly.

Usage
-----
    from blockers import load_field, select_blockers
    field = load_field(paint_heliostats_dir)
    blockers = select_blockers("BE25", field)
"""

from __future__ import annotations

import glob
import json
import pathlib
from dataclasses import dataclass

import torch

from artist.geometry import coordinates

# PAINT calibration targets. The receiver is a cylinder; its centre is enough for a
# conservative screen.
TARGET_NAMES = (
    "solar_tower_juelich_upper",
    "solar_tower_juelich_lower",
    "multi_focus_tower",
    "receiver",
)

TOWER_FILE = "WRI1030197-tower-measurements.json"


@dataclass
class Field:
    """Heliostat positions and target centres in local ENU metres."""

    positions: dict[str, torch.Tensor]  # heliostat id -> [3]
    mirror_radius: dict[str, float]  # heliostat id -> half mirror diagonal [m]
    targets: dict[str, torch.Tensor]  # target name -> [3]


def _to_enu(
    wgs84: list[float], power_plant_position: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Convert a single [lat, lon, alt] triple to local ENU metres."""
    point = torch.tensor([wgs84], dtype=torch.float64, device=device)
    return coordinates.convert_wgs84_coordinates_to_local_enu(
        point, power_plant_position, device=device
    )[0][:3].to(torch.float32)


def load_field(
    heliostats_dir: pathlib.Path, device: torch.device | None = None
) -> Field:
    """
    Read every PAINT heliostat properties JSON plus the tower measurements.

    Parameters
    ----------
    heliostats_dir : pathlib.Path
        The PAINT `heliostats/` directory (contains one folder per heliostat and the tower
        measurements JSON).
    device : torch.device | None
        The device on which to place the tensors (default is None -> CPU).

    Returns
    -------
    Field
        Heliostat positions, mirror radii and target centres in local ENU metres.
    """
    device = device or torch.device("cpu")
    tower = json.loads((heliostats_dir / TOWER_FILE).read_text())
    plant = torch.tensor(
        tower["power_plant_properties"]["coordinates"],
        dtype=torch.float64,
        device=device,
    )

    targets = {
        name: _to_enu(tower[name]["coordinates"]["center"], plant, device)
        for name in TARGET_NAMES
        if name in tower
    }

    positions: dict[str, torch.Tensor] = {}
    radii: dict[str, float] = {}
    pattern = str(heliostats_dir / "*" / "Properties" / "*-heliostat-properties.json")
    for path in sorted(glob.glob(pattern)):
        hid = pathlib.Path(path).parts[-3]
        props = json.loads(pathlib.Path(path).read_text())
        positions[hid] = _to_enu(props["heliostat_position"], plant, device)
        radii[hid] = 0.5 * float(
            (props["width"] ** 2 + props["height"] ** 2) ** 0.5
        )

    return Field(positions=positions, mirror_radius=radii, targets=targets)


def _along_and_lateral(
    point: torch.Tensor, origin: torch.Tensor, target: torch.Tensor
) -> tuple[float, float]:
    """Decompose `point - origin` into (distance along the beam, sideways offset).

    Both measured in the horizontal plane, since the box is defined on the ground: the
    vertical part of the geometry is already accounted for by the reach itself.
    """
    beam = (target - origin)[:2]
    length = torch.norm(beam)
    if length < 1e-9:
        return 0.0, float("inf")
    direction = beam / length
    offset = (point - origin)[:2]
    along = float(torch.dot(offset, direction))
    lateral = float(torch.norm(offset - along * direction))
    return along, lateral


def select_blockers(
    heliostat_id: str,
    field: Field,
    target_names: tuple[str, ...] = TARGET_NAMES,
    forward_m: float = 30.0,
    lateral_m: float = 6.0,
) -> dict[str, list[str]]:
    """
    Select the candidate blockers of one heliostat, per target, using the box rule.

    A candidate is kept when it lies between 0 and `forward_m` along the beam to the
    target, and within `lateral_m` of that beam line sideways.

    Parameters
    ----------
    heliostat_id : str
        The heliostat whose reflected beam may be blocked.
    field : Field
        The field geometry from `load_field`.
    target_names : tuple[str, ...]
        The targets to screen against (default is all four PAINT targets).
    forward_m : float
        Depth of the box along the beam (default is 30.0 m, comfortably beyond the 23.4 m
        worst-case analytic reach anywhere in the real field).
    lateral_m : float
        Half-width of the box (default is 6.0 m; no real blocker was ever measured further
        than 3.6 m sideways, about one mirror width).

    Returns
    -------
    dict[str, list[str]]
        Target name -> sorted list of candidate blocker ids. The key "union" holds the
        union over all screened targets, which is what the scenario needs.
    """
    origin = field.positions[heliostat_id]

    per_target: dict[str, list[str]] = {}
    union: set[str] = set()
    for name in target_names:
        if name not in field.targets:
            continue
        target = field.targets[name]
        hits = []
        for candidate, position in field.positions.items():
            if candidate == heliostat_id:
                continue
            along, lateral = _along_and_lateral(position, origin, target)
            if 0.5 < along <= forward_m and lateral < lateral_m:
                hits.append(candidate)
        per_target[name] = sorted(hits)
        union |= set(hits)

    per_target["union"] = sorted(union)
    return per_target


def beam_elevation_deg(heliostat_id: str, target_name: str, field: Field) -> float:
    """Elevation angle of the line from the heliostat to the target centre, in degrees."""
    delta = field.targets[target_name] - field.positions[heliostat_id]
    horizontal = float(torch.norm(delta[:2]))
    return float(torch.rad2deg(torch.atan2(delta[2], torch.tensor(horizontal))))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Screen a heliostat for blockers.")
    parser.add_argument("heliostat_ids", nargs="+")
    parser.add_argument("--daic", action="store_true")
    parser.add_argument("--forward", type=float, default=30.0,
                        help="Box depth along the beam [m].")
    parser.add_argument("--lateral", type=float, default=6.0,
                        help="Box half-width sideways [m].")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[3]
    heliostats_dir = (
        pathlib.Path(
            "/tudelft.net/staff-umbrella/StudentsCVlab/agrigore/datasets/paint/heliostats"
        )
        if args.daic
        else root / "datasets" / "paint" / "heliostats"
    )
    field = load_field(heliostats_dir)
    print(f"{len(field.positions)} heliostats, {len(field.targets)} targets")
    for hid in args.heliostat_ids:
        result = select_blockers(hid, field, forward_m=args.forward,
                                 lateral_m=args.lateral)
        elevations = " ".join(
            f"{n.split('_')[-1]}={beam_elevation_deg(hid, n, field):.1f}deg"
            for n in TARGET_NAMES
            if n in field.targets
        )
        print(f"\n{hid}  {elevations}")
        for name, hits in result.items():
            print(f"  {name:28s} {hits}")
