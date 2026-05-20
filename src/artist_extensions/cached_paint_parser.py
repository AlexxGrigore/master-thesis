"""PaintCalibrationDataParser with an in-memory parse cache.

Parses each (heliostat group, bitmap resolution) combination at most once.
Subsequent calls return the cached tensors moved to the requested device.
This avoids repeated PNG decoding and coordinate transforms when the same
validation data is queried every epoch.
"""
from __future__ import annotations

import torch

from artist.io.paint_calibration_parser import PaintCalibrationDataParser
from artist.util import get_device


class CachedPaintCalibrationDataParser(PaintCalibrationDataParser):
    """PaintCalibrationDataParser with an in-memory result cache."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cache: dict[tuple, tuple[torch.Tensor, ...]] = {}

    @property
    def cache_size_mb(self) -> float:
        """Approximate memory used by cached tensors, in megabytes."""
        total_bytes = sum(
            t.nbytes for tensors in self._cache.values() for t in tensors
        )
        return total_bytes / 1e6

    def parse_data_for_reconstruction(
        self,
        heliostat_data_mapping,
        heliostat_group,
        scenario,
        bitmap_resolution=torch.tensor([256, 256]),
        device=None,
    ):
        key = (
            tuple(heliostat_group.names),
            tuple(int(x) for x in bitmap_resolution.tolist()),
        )
        if key not in self._cache:
            result = super().parse_data_for_reconstruction(
                heliostat_data_mapping=heliostat_data_mapping,
                heliostat_group=heliostat_group,
                scenario=scenario,
                bitmap_resolution=bitmap_resolution,
                device=torch.device("cpu"),
            )
            self._cache[key] = tuple(t.cpu() for t in result)

        target = get_device(device)
        return tuple(t.to(target) for t in self._cache[key])
