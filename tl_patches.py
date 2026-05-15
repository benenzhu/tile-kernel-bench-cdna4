"""Local monkey-patches over tilelang -- imported at the top of bench.py.

We don't touch upstream tilelang from this repo; instead we re-bind the
hot functions at import time. Each patch is small, additive, and
should disappear once the upstream change lands.

Patches:
  - tilelang.profiler.bench._bench_with_cupti: wrap each fn() call in
    a torch.profiler.record_function marker so we can recover per-iter
    device time and apply mean/median/min/max. Upstream uses
    profiler.key_averages() which collapses everything into a mean.
    Read aggregation from the env var TL_BENCH_RETURN_MODE
    (default "median").
"""
from __future__ import annotations

import os

import torch
import tilelang.profiler.bench as _tlbench


_ITER_MARKER = "tilelang_bench_iter"


def _patched_bench_with_cupti(fn, cache, n_repeat: int) -> float:
    """Drop-in replacement for tilelang's _bench_with_cupti that records a
    per-iter marker so we can aggregate via mean/median/min/max instead of
    being limited to the average key_averages() reports.
    """
    return_mode = os.environ.get("TL_BENCH_RETURN_MODE", "median")
    if return_mode not in ("min", "max", "mean", "median"):
        raise ValueError(f"TL_BENCH_RETURN_MODE={return_mode!r} invalid")

    with _tlbench.suppress_stdout_stderr():
        schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=schedule,
        )

        with profiler:
            for _ in range(2):
                for _ in range(n_repeat):
                    cache.zero_()
                    with torch.profiler.record_function(_ITER_MARKER):
                        fn()
                profiler.step()

    # Per-iter device time = the marker's device_time_total (sum of every
    # CUDA kernel launched within the with-block). cache.zero_() lives
    # outside the marker, so it's naturally excluded.
    iter_us = [
        e.device_time_total
        for e in profiler.events()
        if e.name == _ITER_MARKER
    ][-n_repeat:]

    if iter_us:
        times = torch.tensor(iter_us, dtype=torch.float)
        kernel_time_us = getattr(times, return_mode)().item()
    else:
        # Fallback to legacy aggregation if markers unavailable.
        excluded = "at::native::vectorized_elementwise"
        total = excluded_t = 0.0
        for event in profiler.key_averages():
            total += event.self_device_time_total
            if excluded in event.key:
                excluded_t += event.self_device_time_total
        kernel_time_us = (total - excluded_t) / n_repeat

    return kernel_time_us * 1e-3


_tlbench._bench_with_cupti = _patched_bench_with_cupti
