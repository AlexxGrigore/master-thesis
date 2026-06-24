"""Lightweight, CUDA-synced profiling harness for the upscaling experiment.

We only care about two resources: wall-clock time and GPU VRAM. Both are easy to
measure wrong, so this module centralises the correct way:

  * Time — GPU kernels launch asynchronously, so a bare ``time.perf_counter()`` stops
    before the GPU is actually done. We ``torch.cuda.synchronize()`` on both ends.
  * VRAM — ``reset_peak_memory_stats()`` before each phase, then read
    ``max_memory_allocated`` (real tensors) and ``max_memory_reserved`` (what
    ``nvidia-smi`` shows) after. These are reset per phase so each N is independent.

CPU host RSS is reported once at the end (process high-water mark; not resettable),
since the binding constraint we care about is GPU VRAM.

Nothing here touches ARTIST source. ``accumulate_raytrace_time`` monkeypatches
``HeliostatRayTracer.trace_rays`` at runtime *in this driver only* to isolate how much
of training time is spent inside the ray tracer (forward pass).
"""

from __future__ import annotations

import contextlib
import functools
import gc
import json
import pathlib
import resource
import sys
import time
from typing import Any

import torch


def _sync() -> None:
    """Block until all queued GPU work is finished (no-op on CPU)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def free_cuda() -> None:
    """Release GPU memory back to the allocator between sweep iterations.

    Call this AFTER a phase has been measured/recorded and AFTER the large objects
    (scenario, reconstructor, …) have been ``del``-eted in the *caller's* scope —
    deleting them here would not work, since this function only holds borrowed
    references. ``gc.collect()`` breaks any reference cycles (autograd graphs,
    optimizer state) so their CUDA tensors are actually freed, then
    ``empty_cache()`` returns the blocks to the driver. Without this, each N's peak
    VRAM would include the previous N's leftovers, and large N could OOM on garbage.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _cpu_peak_rss_gb() -> float:
    """Process peak resident set size in GB.

    ``ru_maxrss`` is kilobytes on Linux (DAIC) and bytes on macOS.
    """
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**2 if sys.platform.startswith("linux") else 1024**3
    return round(maxrss / divisor, 4)


class Profiler:
    """Collects per-phase time + peak VRAM records and dumps them to JSON."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    @contextlib.contextmanager
    def measure(self, name: str):
        """Context manager timing a phase and capturing its peak VRAM.

        Resets the CUDA peak-memory counter on entry so each phase's VRAM figure is
        independent of every other phase.
        """
        gpu = torch.cuda.is_available()
        if gpu:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        _sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            record: dict[str, Any] = {"seconds": round(time.perf_counter() - start, 4)}
            if gpu:
                record["peak_vram_alloc_gb"] = round(
                    torch.cuda.max_memory_allocated() / 2**30, 4
                )
                record["peak_vram_reserved_gb"] = round(
                    torch.cuda.max_memory_reserved() / 2**30, 4
                )
            self.records[name] = record

    def annotate(self, name: str, **fields: Any) -> None:
        """Attach extra metadata (n_heliostats, paths, …) to a recorded phase."""
        self.records.setdefault(name, {}).update(fields)

    def save(self, path: pathlib.Path, meta: dict[str, Any] | None = None) -> None:
        """Write all records (+ optional run metadata + CPU peak RSS) to JSON."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": meta or {},
            "cpu_peak_rss_gb": _cpu_peak_rss_gb(),
            "gpu_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "records": self.records,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)


@contextlib.contextmanager
def accumulate_raytrace_time(ray_tracer_cls: type):
    """Temporarily wrap ``ray_tracer_cls.trace_rays`` to sum its (synced) runtime.

    Yields a dict ``{"seconds": float, "calls": int}`` updated live. Captures only the
    forward ray-tracing time, not the autograd backward pass. Restores the original
    method on exit, so ARTIST is left untouched.
    """
    original = ray_tracer_cls.trace_rays
    state = {"seconds": 0.0, "calls": 0}

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        _sync()
        start = time.perf_counter()
        result = original(self, *args, **kwargs)
        _sync()
        state["seconds"] += time.perf_counter() - start
        state["calls"] += 1
        return result

    ray_tracer_cls.trace_rays = wrapped
    try:
        yield state
    finally:
        ray_tracer_cls.trace_rays = original
