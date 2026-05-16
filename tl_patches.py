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
_MIN_REPEAT_DEFAULT = 200


def _patched_bench_with_cupti(fn, cache, n_repeat: int) -> float:
    """Drop-in replacement for tilelang's _bench_with_cupti that records a
    per-iter marker so we can aggregate via mean/median/min/max instead of
    being limited to the average key_averages() reports.

    Also enforces a minimum sample count (TL_BENCH_MIN_REPEAT, default 100)
    so the median has enough data even for slow kernels where do_bench's
    auto-tuned n_repeat would only give a handful of samples.
    """
    return_mode = os.environ.get("TL_BENCH_RETURN_MODE", "median")
    if return_mode not in ("min", "max", "mean", "median"):
        raise ValueError(f"TL_BENCH_RETURN_MODE={return_mode!r} invalid")

    min_repeat = int(os.environ.get("TL_BENCH_MIN_REPEAT", _MIN_REPEAT_DEFAULT))
    n_repeat = max(n_repeat, min_repeat)

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


# # ---------------------------------------------------------------------------
# # Widen the HIP resource-usage recorder window so the cython execution
# # backend works too. Upstream's JITKernel._compile_and_create_adapter only
# # wraps tilelang.lower() with reset_recorder/pop_recorded. lower() compiles
# # hipcc only when enable_device_compile=True (tvm_ffi backend). With the
# # cython backend hipcc runs LATER inside the adapter constructor, after the
# # recorder has already been popped -> kernel.resource_usage is empty.
# #
# # Re-pop after the original method returns; if anything landed there (the
# # late hipcc call), promote it onto the kernel.
# # ---------------------------------------------------------------------------
# from tilelang import JITKernel as _JITKernel  # noqa: E402
# from tilelang.jit.adapter.utils import is_hip_target as _is_hip  # noqa: E402

# _orig_compile = _JITKernel._compile_and_create_adapter


# def _patched_compile_and_create_adapter(self, *args, **kwargs):
#     adapter = _orig_compile(self, *args, **kwargs)
#     if _is_hip(self.target):
#         from tilelang.jit.adapter.hip_resource_info import pop_recorded
#         late = pop_recorded()
#         if late and not getattr(self, "_resource_usage", None):
#             self._resource_usage = late
#     return adapter


# _JITKernel._compile_and_create_adapter = _patched_compile_and_create_adapter


# def get_kernel_resources(kernel):
#     """Return ``(n_regs, n_spills_total)`` for a HIP kernel, or
#     ``(None, None)`` on other targets / cache-loaded kernels with no
#     captured remarks.

#     `n_spills_total` is `VGPRs Spill + ScratchSize_bytes // 4` -- treating
#     one scratch dword the same as one spilled VGPR for accounting, since
#     both end up in main memory and cost roughly the same to access.
#     """
#     info = getattr(kernel, "_primary_resource_usage", lambda: None)()
#     if info is None:
#         return None, None
#     try:
#         scratch_bytes = int(info.extra.get("ScratchSize [bytes/lane]", "0"))
#     except (ValueError, AttributeError):
#         scratch_bytes = 0
#     return info.n_regs, info.n_spills + scratch_bytes // 4
