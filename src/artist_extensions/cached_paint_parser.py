"""
CachedPaintCalibrationDataParser — drop-in replacement for PaintCalibrationDataParser
that caches parse_data_for_reconstruction results in CPU RAM after the first call.

Eliminates repeated NFS reads on every training epoch when data lives on a network
filesystem (e.g. /tudelft.net on DAIC). Moving cached tensors CPU→GPU each epoch
is orders of magnitude faster than re-reading thousands of files over the network.

Usage
-----
    parser = CachedPaintCalibrationDataParser(
        sample_limit=100,
        centroid_extraction_method="utis",
    )
    # First call reads from disk and populates the cache.
    # All subsequent calls with the same group return cached tensors.
"""
from __future__ import annotations

import logging

import torch
from artist.data_parser.paint_calibration_parser import PaintCalibrationDataParser
from artist.util.environment_setup import get_device

log = logging.getLogger(__name__)


class CachedPaintCalibrationDataParser(PaintCalibrationDataParser):
    """
    Wraps PaintCalibrationDataParser with an in-memory cache.

    The cache key is the tuple of heliostat names in the group plus the bitmap
    resolution so that the stored tensors are always valid for subsequent calls.

    Each parser instance maintains its own independent cache, so train / val / test
    parsers must be separate objects (not shared) to avoid cross-split contamination.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # key: (group_names_tuple, bitmap_res_tuple) -> tuple of CPU tensors
        self._cache: dict[tuple, tuple[torch.Tensor, ...]] = {}

    @property
    def cache_size_mb(self) -> float:
        total = sum(
            t.element_size() * t.numel()
            for tensors in self._cache.values()
            for t in tensors
        )
        return total / 1024 ** 2

    def parse_data_for_reconstruction(
        self,
        heliostat_data_mapping,
        heliostat_group,
        scenario,
        bitmap_resolution=torch.tensor([256, 256]),
        device=None,
    ):
        res_tuple = tuple(int(x) for x in bitmap_resolution.tolist())
        key = (tuple(heliostat_group.names), res_tuple)

        if key not in self._cache:
            log.info(
                "CachedPaintParser: cache miss — loading %d heliostats from disk.",
                len(heliostat_group.names),
            )
            result = super().parse_data_for_reconstruction(
                heliostat_data_mapping=heliostat_data_mapping,
                heliostat_group=heliostat_group,
                scenario=scenario,
                bitmap_resolution=bitmap_resolution,
                device=torch.device("cpu"),
            )
            self._cache[key] = tuple(t.cpu() for t in result)
            log.info(
                "CachedPaintParser: cache populated — %.1f MB in RAM.",
                self.cache_size_mb,
            )

        target_device = get_device(device)
        return tuple(t.to(target_device) for t in self._cache[key])
