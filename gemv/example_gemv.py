"""Vectorized split-K GEMV (out = A @ B.T), lifted from
tilelang/examples/gemv/example_gemv.py (`splitk_gemv_vectorized`).

A is a length-K vector; B is (N, K); C is length N. Memory-bound op
common in LLM decode (qkv/proj/lm_head). Tile params pinned, no autotune.
"""
import torch
import tilelang as tl
import tilelang.language as T

from tvm import DataType  # bundled with tvm-ffi via tilelang


OP_NAME = "gemv"
DTYPE = "fp16"


@tl.jit(out_idx=[-1])
def gemv_kernel(
    N: int,
    K: int,
    BLOCK_N: int,
    reduce_threads: int,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float,
):
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def main(
        A: T.Tensor((K,), dtype),  # type: ignore
        B: T.Tensor((N, K), dtype),  # type: ignore
        C: T.Tensor((N,), dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
            C_accum = T.alloc_local((1,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            T.atomic_add(C_shared[tn], C_accum[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]

    return main


_SHAPES = [
    # (N, K, BLOCK_N, reduce_threads)
    (8192, 8192, 64, 16),
    (32768, 8192, 64, 16),   # lm_head-ish
    (8192, 32768, 64, 16),   # up_proj-ish
]


CASES = [dict(N=N, K=K, BLOCK_N=bN, reduce_threads=rt) for N, K, bN, rt in _SHAPES]


def bench_one(case, check):
    N, K = case["N"], case["K"]
    BLOCK_N, rt = case["BLOCK_N"], case["reduce_threads"]
    kernel = gemv_kernel(N, K, BLOCK_N, rt)

    a = torch.randn(K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)

    if check:
        c = kernel(a, b)
        ref = (b.float() @ a.float()).to(torch.float16)
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)

    profiler = kernel.get_profiler()
    latency_ms = profiler.do_bench(backend="cupti", input_tensors=[a, b])

    flops = 2.0 * N * K
    # Bytes: A (K fp16) + B (N*K fp16) + C (N fp16). B dominates.
    bytes_moved = (K + N * K + N) * 2
    tflops = flops / (latency_ms * 1e-3) / 1e12
    tbps = bytes_moved / (latency_ms * 1e-3) / 1e12
    return {
        "shape_str": f"{N}x{K}",
        "tile_str": f"N{BLOCK_N}_rt{rt}",
        "dtype": DTYPE,
        "latency_ms": latency_ms,
        "tbps": tbps,
        "tflops": tflops,
    }
