"""3D elementwise C = A + B on a large (>4 GB) tensor.

Existence check for the BufferLoad/BufferStore path on tensors that cross
the 32-bit byte-offset boundary -- regressions there are silent in
unit tests that use small shapes. fp16 (512, 512, 16384) = 8 GB per
tensor, 24 GB total; well within MI355 HBM but >4 GB.
"""
import torch
import tilelang
import tilelang.language as T


OP_NAME = "add_3d_large"
DTYPE = "fp16"


@tilelang.jit(out_idx=[-1])
def add3d(D0, D1, D2, block_D2, threads=128):
    dtype = T.float16

    @T.prim_func
    def main(
        A: T.Tensor((D0, D1, D2), dtype),  # type: ignore
        B: T.Tensor((D0, D1, D2), dtype),  # type: ignore
        C: T.Tensor((D0, D1, D2), dtype),  # type: ignore
    ):
        # One block per (D0, D1) row pair; threads + block_D2 cover the inner dim.
        with T.Kernel(D0, D1, T.ceildiv(D2, block_D2), threads=threads) as (bx, by, bz):
            for k in T.Parallel(block_D2):
                C[bx, by, bz * block_D2 + k] = (
                    A[bx, by, bz * block_D2 + k] + B[bx, by, bz * block_D2 + k]
                )

    return main


_SHAPES = [
    # (D0, D1, D2, block_D2). 512*512*16384 fp16 = 8 GB per tensor.
    (512, 512, 16384, 1024),
]


CASES = [dict(D0=D0, D1=D1, D2=D2, block_D2=bk) for D0, D1, D2, bk in _SHAPES]


def bench_one(case, check):
    D0, D1, D2 = case["D0"], case["D1"], case["D2"]
    block_D2 = case["block_D2"]
    kernel = add3d(D0, D1, D2, block_D2)

    a = torch.randn(D0, D1, D2, device="cuda", dtype=torch.float16)
    b = torch.randn(D0, D1, D2, device="cuda", dtype=torch.float16)

    if check:
        c = kernel(a, b)
        ref = a + b
        torch.testing.assert_close(c, ref, rtol=1e-3, atol=1e-3)

    profiler = kernel.get_profiler()
    latency_ms = profiler.do_bench(backend="cupti", input_tensors=[a, b])

    # Memory-bound: read A + read B + write C, all fp16.
    bytes_moved = 3 * D0 * D1 * D2 * 2
    tbps = bytes_moved / (latency_ms * 1e-3) / 1e12
    return {
        "shape_str": f"{D0}x{D1}x{D2}",
        "tile_str": f"D2blk{block_D2}",
        "dtype": DTYPE,
        "latency_ms": latency_ms,
        "tbps": tbps,
        "tflops": 0.0,
    }
