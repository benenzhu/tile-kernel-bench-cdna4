import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main():
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)

    import torch

    a = torch.randn(1024, 1024).cuda().half()
    b = torch.randn(1024, 1024).cuda().half()

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
DTYPE = "fp16"

CASES = [
    dict(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32),
    dict(M=2048, N=2048, K=2048, block_M=128, block_N=128, block_K=32),
    dict(M=4096, N=4096, K=4096, block_M=128, block_N=128, block_K=32),
    dict(M=8192, N=8192, K=8192, block_M=128, block_N=128, block_K=32),
    dict(M=1024, N=8192, K=8192, block_M=128, block_N=128, block_K=32),
    dict(M=8192, N=8192, K=1024, block_M=128, block_N=128, block_K=32),
    dict(M=4096, N=4096, K=8192, block_M=128, block_N=128, block_K=32),
]


def bench_one(case, check):
    import torch
    M, N, K = case["M"], case["N"], case["K"]
    kernel = matmul(M, N, K, case["block_M"], case["block_N"], case["block_K"])

    if check:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        c = kernel(a, b)
        ref = a @ b
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)

    profiler = kernel.get_profiler()
    latency_ms = profiler.do_bench(backend="cupti")

    flops = 2.0 * M * N * K
    bytes_moved = (M * K + K * N + M * N) * 2  # fp16 inputs and output
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


if __name__ == "__main__":
    main()
