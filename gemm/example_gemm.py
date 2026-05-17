import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K,
           num_stages=3, num_threads=128,
           dtype=T.bfloat16, accum_dtype=T.float32):
    """NN layout: B is stored as (K, N), no transpose."""

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=num_threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, k_pack=2)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


@tilelang.jit(out_idx=[-1])
def matmul_nt(M, N, K, block_M, block_N, block_K,
              num_stages=3, num_threads=128,
              dtype=T.bfloat16, accum_dtype=T.float32):
    """NT layout: B is stored as (N, K) and passed with transpose_B=True. This
    is the convention most LLM stacks use for weights so K stays contiguous."""

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=num_threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, k_pack=2, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main():
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)

    import torch

    a = torch.randn(1024, 1024).cuda().bfloat16()
    b = torch.randn(1024, 1024).cuda().bfloat16()

    c = kernel(a, b)

    ref_c = a @ b

    print("c:")
    print(c)
    print("ref_c:")
    print(ref_c)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")

    # Get CUDA Source
    print("CUDA Source:")
    print(kernel.get_kernel_source())

    # benchmark
    profiler = kernel.get_profiler()
    latency = profiler.do_bench(backend="cupti")
    # latency = profiler.do_bench()
    print(f"tilelang Latency: {latency}ms")


def run_regression_perf():
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)
    profiler = kernel.get_profiler()
    return profiler.do_bench(backend="cupti")


# ---------------------------------------------------------------------------
# bench-module API (consumed by ../bench.py)
# ---------------------------------------------------------------------------

OP_NAME = "gemm"
DTYPE = "bf16"

_SHAPES = [
    # (M, N, K, block_M, block_N, block_K, num_stages, num_threads)
    (1024, 1024, 16384, 128, 128, 32, 3, 128),
    (2048, 2048, 2048, 128, 128, 32, 3, 128),
    (4096, 4096, 4096, 128, 128, 32, 3, 128),
    (8192, 8192, 8192, 128, 128, 32, 3, 128),
    (1024, 8192, 8192, 128, 128, 32, 3, 128),
    (8192, 8192, 1024, 128, 128, 32, 3, 128),
    (4096, 4096, 8192, 128, 128, 32, 3, 128),
    # Big-K compute-heavy case; needs num_stages=2 to fit 64 KB CDNA4 LDS
    # (per-stage = 2 * 256 * 64 * 2 B = 64 KB; 2 stages = 128 KB).
    # 512 threads/block gives more lanes to cover the 256x256 tile.
    (8192, 8192, 16384, 256, 256, 64, 2, 512),
]


def _make_cases():
    out = []
    for M, N, K, bM, bN, bK, ns, nt in _SHAPES:
        # Order matters: NN then NT for the same shape so they print
        # adjacent without needing to sort the report afterwards.
        for transpose_b in (False, True):
            out.append(dict(
                M=M, N=N, K=K,
                block_M=bM, block_N=bN, block_K=bK,
                num_stages=ns,
                num_threads=nt,
                transpose_b=transpose_b,
            ))
    return out


CASES = _make_cases()


def bench_one(case, check):
    import torch
    M, N, K = case["M"], case["N"], case["K"]
    transpose_b = case["transpose_b"]
    layout = "NT" if transpose_b else "NN"

    factory = matmul_nt if transpose_b else matmul
    kernel = factory(
        M, N, K,
        case["block_M"], case["block_N"], case["block_K"],
        num_stages=case.get("num_stages", 3),
        num_threads=case.get("num_threads", 128),
    )

    if check:
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        if transpose_b:
            b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
            ref = a @ b.T
        else:
            b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
            ref = a @ b
        c = kernel(a, b)
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)

    profiler = kernel.get_profiler()
    latency_ms = profiler.do_bench(backend="cupti")

    flops = 2.0 * M * N * K
    bytes_moved = (M * K + N * K + M * N) * 2  # fp16, same total either way
    tflops = flops / (latency_ms * 1e-3) / 1e12
    tbps = bytes_moved / (latency_ms * 1e-3) / 1e12
    return {
        "shape_str": f"{M}x{N}x{K}_{layout}",
        "tile_str": f"{case['block_M']}x{case['block_N']}x{case['block_K']}",
        "dtype": DTYPE,
        "latency_ms": latency_ms,
        "tbps": tbps,
        "tflops": tflops,
        "_kernel": kernel,
    }


if __name__ == "__main__":
    main()
