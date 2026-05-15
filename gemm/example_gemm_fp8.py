"""FP8 GEMM kernel for AMD/CDNA, lifted from
tilelang/examples/gemm_fp8/example_tilelang_gemm_amd.py.

The upstream example uses tilelang.autotune over a 72-config grid; we
pin one sane config so it slots into the bench harness without burning
hours on tuning. Tile params are still surfaced via CASES so they show
up in the comparison table.
"""
import torch
import tilelang
import tilelang.language as T

from tilelang.utils import determine_fp8_type, determine_torch_fp8_type


OP_NAME = "gemm_fp8"
DTYPE = "fp8"


@tilelang.jit(out_idx=[-1])
def fp8_matmul(M, N, K, block_M, block_N, block_K, num_stages, num_threads, k_pack):
    dtype = determine_fp8_type()
    accum_dtype = T.float32

    @T.prim_func
    def gemm_fp8_ss(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True,
                       k_pack=k_pack, policy=T.GemmWarpPolicy.FullRow)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_fp8_ss


CASES = [
    dict(M=1024, N=1024, K=1024,
         block_M=128, block_N=128, block_K=128,
         num_stages=0, num_threads=256, k_pack=2),
    dict(M=2048, N=2048, K=2048,
         block_M=128, block_N=128, block_K=128,
         num_stages=0, num_threads=256, k_pack=2),
    dict(M=4096, N=4096, K=4096,
         block_M=128, block_N=128, block_K=128,
         num_stages=0, num_threads=256, k_pack=2),
    dict(M=8192, N=8192, K=8192,
         block_M=128, block_N=128, block_K=128,
         num_stages=0, num_threads=256, k_pack=2),
]


def bench_one(case, check):
    M, N, K = case["M"], case["N"], case["K"]
    kernel = fp8_matmul(
        M, N, K,
        case["block_M"], case["block_N"], case["block_K"],
        case["num_stages"], case["num_threads"], case["k_pack"],
    )

    # Tilelang's auto-supplied tensor path goes through dlpack, which doesn't
    # yet handle torch.float8_*. Build the fp8 inputs ourselves and pass them
    # to the profiler explicitly.
    torch_fp8 = determine_torch_fp8_type()
    a = (torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.01).to(torch_fp8)
    b = (torch.randn(N, K, dtype=torch.float16, device="cuda") * 0.01).to(torch_fp8)

    if check:
        c = kernel(a, b)
        ref = (a.half() @ b.half().T).to(torch.float32)
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-1)

    profiler = kernel.get_profiler()
    latency_ms = profiler.do_bench(backend="cupti", input_tensors=[a, b])

    flops = 2.0 * M * N * K
    # Bytes: A (fp8=1B), B (fp8=1B), C (fp32=4B).
    bytes_moved = (M * K) * 1 + (N * K) * 1 + (M * N) * 4
    tflops = flops / (latency_ms * 1e-3) / 1e12
    tbps = bytes_moved / (latency_ms * 1e-3) / 1e12

    return {
        "shape_str": f"{M}x{N}x{K}",
        "tile_str": f"{case['block_M']}x{case['block_N']}x{case['block_K']}",
        "dtype": DTYPE,
        "latency_ms": latency_ms,
        "tbps": tbps,
        "tflops": tflops,
    }
