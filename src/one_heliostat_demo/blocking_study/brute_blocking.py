"""Workaround for a correctness bug in ARTIST's LBVH blocking filter.

The bug
-------
`artist.raytracing.blocking.build_linear_bounding_volume_hierarchies` produces a
DISCONNECTED FOREST rather than a single tree for most primitive counts. The traversal in
`lbvh_filter_blocking_planes` starts at node 0 and can therefore only ever reach the
component containing node 0; every primitive in another component is silently dropped, so
blocking is UNDER-reported with no warning.

Measured on synthetic field-like primitives (regular grid, aimed planes):

    n_primitives:   2   3   4   5   6   7   8   9  10  11 ... 19  20  21 ... 36  37
    disjoint roots: 1   1   1   1   2   1   1   2   2   2      1   1   1     10   1
    reachable:      2   3   4   5   3   7   8   6   7   8     19  20  21      8   37

n = 6 is broken (3 of 6 primitives reachable), and 6 is exactly the size of a
one-heliostat-plus-five-blockers scenario, so this is not an exotic corner case. Confirmed
against the real field too: for BE25 with 37 primitives the LBVH filter returned ZERO hits
while a direct evaluation found 4.66 % of rays blocked.

The workaround
--------------
Replace the filter with an exact one: consider every primitive except the ray's own
heliostat. The LBVH exists purely as an acceleration structure, so bypassing it changes
performance, not physics. `soft_ray_blocking_mask` is O(rays x primitives); with a few tens
of primitives (which is all a blocking neighbourhood ever needs, see the census) the cost
is negligible.

Self-exclusion must be kept. It normally happens inside the filter
(`ray_owner_hit != leaf_primitives`). It cannot be dropped: a heliostat's own blocking
plane is a single flat rectangle fitted to its corners, while its surface points sit on
tilted facets, so points lie off that plane by centimetres and rays re-intersect it beyond
the 5 cm `ray_origin_offset`. Leaving self in place produced 29 % spurious self-blocking on
BE25.

Usage
-----
    from brute_blocking import exact_blocking

    with exact_blocking():
        flux, _, _, blocking_factor = ray_tracer.trace_rays(...)

Report `exact_blocking` in any result that depends on blocked fractions, and re-check
whether upstream ARTIST has fixed the builder before relying on it.
"""

from __future__ import annotations

import contextlib
import logging

import torch

from artist.raytracing import blocking as _blocking

log = logging.getLogger(__name__)

_original_filter = _blocking.lbvh_filter_blocking_planes


def _exact_filter(
    points_at_ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    blocking_primitives_corners: torch.Tensor,
    ray_to_heliostat_mapping: torch.Tensor,
    intersection_distances_target: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return every primitive index except the ones owning the traced rays.

    Drop-in replacement for `lbvh_filter_blocking_planes` with the same signature. It
    performs no spatial culling, so it can never miss a blocker; `soft_ray_blocking_mask`
    downstream still decides what is actually occluded.
    """
    number_of_primitives = blocking_primitives_corners.shape[0]
    owners = torch.unique(ray_to_heliostat_mapping)
    keep = torch.ones(number_of_primitives, dtype=torch.bool, device=owners.device)
    valid = owners[(owners >= 0) & (owners < number_of_primitives)]
    if valid.numel() > 0:
        keep[valid.long()] = False
    return torch.nonzero(keep, as_tuple=True)[0]


@contextlib.contextmanager
def exact_blocking():
    """Context manager that swaps ARTIST's buggy LBVH filter for the exact one."""
    _blocking.lbvh_filter_blocking_planes = _exact_filter
    try:
        yield
    finally:
        _blocking.lbvh_filter_blocking_planes = _original_filter


@contextlib.contextmanager
def capture_blocking_mask():
    """Capture the per-ray blocked mask ARTIST computes internally.

    `HeliostatRayTracer.trace_rays` only returns the scalar `blocking_factor`
    (fraction of rays not blocked); the per-ray, per-surface-point soft mask
    it computes along the way (`blocking.soft_ray_blocking_mask`, shape
    ``[number_of_active_heliostats, number_of_rays, number_of_points]``,
    ~0 = unblocked / ~1 = blocked) is discarded. This wraps that function to
    stash its output, the same monkeypatch pattern as `exact_blocking`.

    Usage
    -----
        with capture_blocking_mask() as captured:
            ray_tracer.trace_rays(...)
        blocked = captured["blocked"]  # absent if there were no blockers to test

    Averaging over the ray dimension gives a per-surface-point blocked
    fraction, directly usable to shade the mirror's own surface (as opposed
    to the flux it forms on the target).
    """
    captured: dict[str, torch.Tensor] = {}
    original = _blocking.soft_ray_blocking_mask

    def _wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        captured["blocked"] = out.detach().clone()
        return out

    _blocking.soft_ray_blocking_mask = _wrapped
    try:
        yield captured
    finally:
        _blocking.soft_ray_blocking_mask = original


@contextlib.contextmanager
def capture_target_intersections():
    """Capture the per-ray target-bitmap intersection coordinates.

    Wraps `artist.raytracing.geometry.line_plane_intersections` (the same
    monkeypatch pattern as `exact_blocking` / `capture_blocking_mask`) to
    stash its return value: `(bitmap_e, bitmap_u, distances,
    angle_reduced_intensities)`, each shaped
    ``[number_of_active_heliostats, number_of_rays, number_of_points]``.
    `trace_rays` uses these internally to splat the flux bitmap but never
    returns them, so there is otherwise no way to ask "where on the target
    does THIS specific mirror surface point's ray land" -- the question that
    matters for checking whether the mirror-to-target mapping is a direct
    copy or a flipped/rotated image (as for any converging mirror).

    Usage
    -----
        from artist.raytracing import geometry as _geometry
        with capture_target_intersections() as captured:
            _geometry.line_plane_intersections(...)  # or via trace_rays
        e, u = captured["bitmap_e"], captured["bitmap_u"]
    """
    from artist.raytracing import geometry as _geometry

    captured: dict[str, torch.Tensor] = {}
    original = _geometry.line_plane_intersections

    def _wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        bitmap_e, bitmap_u, distances, intensities = out
        captured["bitmap_e"] = bitmap_e.detach().clone()
        captured["bitmap_u"] = bitmap_u.detach().clone()
        captured["distances"] = distances.detach().clone()
        captured["angle_reduced_intensities"] = intensities.detach().clone()
        return out

    _geometry.line_plane_intersections = _wrapped
    try:
        yield captured
    finally:
        _geometry.line_plane_intersections = original


def lbvh_is_connected(blocking_primitives_corners: torch.Tensor, device=None) -> bool:
    """True if ARTIST's LBVH for this primitive set is a single connected tree.

    Diagnostic helper: lets a run assert up front whether the stock filter would have been
    trustworthy for a given scenario size.
    """
    from artist.util import constants

    tree = _blocking.build_linear_bounding_volume_hierarchies(
        blocking_primitives_corners=blocking_primitives_corners, device=device
    )
    left = tree[constants.left_node]
    right = tree[constants.right_node]
    is_leaf = tree[constants.is_leaf]
    primitive_index = tree[constants.primitive_index]

    reachable: set[int] = set()
    seen: set[int] = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node < 0 or node in seen:
            continue
        seen.add(node)
        if bool(is_leaf[node]):
            reachable.add(int(primitive_index[node]))
        else:
            stack.append(int(left[node]))
            stack.append(int(right[node]))
    return len(reachable) == blocking_primitives_corners.shape[0]
