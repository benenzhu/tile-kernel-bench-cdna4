"""RMSNorm kernel, lifted from tilelang/examples/norm/rms_norm.py.

Memory-bound op: TFLOPS column reports 0; compare on TB/s.
"""
import torch
import tilelang
import tilelang.language as T


OP_NAME = "rmsnorm"
DTYPE = "bf16"


@tilelang.jit(out_idx=[-1])
def rms_norm(M, N, blk_m):
    dtype = T.bfloat16

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore
        B: T.Tensor((M, N), dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            A_shared = T.alloc_shared((blk_m, N), dtype)
            A_pow_local = T.alloc_fragment((blk_m, N), dtype)
            A_local = T.alloc_fragment((blk_m, N), dtype)
            A_powsum = T.alloc_fragment((blk_m,), dtype)

            T.copy(A[bx * blk_m:(bx + 1) * blk_m, :], A_shared)
            T.copy(A_shared, A_local)
            for i, j in T.Parallel(blk_m, N):
                A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
            T.reduce_sum(A_pow_local, A_powsum, dim=1)
            for i in T.Parallel(blk_m):
                A_powsum[i] = T.rsqrt(A_powsum[i] / N + 1e-12)
            for i, j in T.Parallel(blk_m, N):
                A_local[i, j] *= A_powsum[i]
            T.copy(A_local, B[bx * blk_m:(bx + 1) * blk_m, :])

    return main


_SHAPES = [
    # (M, N, blk_m)
    (4096, 8192, 1),
    (16384, 8192, 1),
    (32768, 8192, 1),
]


CASES = [dict(M=M, N=N, blk_m=blk_m) for M, N, blk_m in _SHAPES]


def bench_one(case, check):
    M, N, blk_m = case["M"], case["N"], case["blk_m"]
    kernel = rms_norm(M, N, blk_m)

    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    if check:
        y = kernel(x)
        ref = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-12).to(x.dtype)
        torch.testing.assert_close(y, ref, rtol=1e-2, atol=1e-1)

    profiler = kernel.get_profiler()
    latency_ms = profiler.do_bench(backend="cupti", input_tensors=[x])

    # Memory-bound: read A + write B, both bf16.
    bytes_moved = 2 * M * N * 2
    tbps = bytes_moved / (latency_ms * 1e-3) / 1e12
    return {
        "shape_str": f"{M}x{N}",
        "tile_str": f"blk_m{blk_m}",
        "dtype": DTYPE,
        "latency_ms": latency_ms,
        "tbps": tbps,
        "tflops": 0.0,
    }
