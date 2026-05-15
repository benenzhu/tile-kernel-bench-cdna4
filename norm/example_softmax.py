"""Online softmax kernel, lifted from
tilelang/examples/online_softmax/online_softmax.py.

Memory-bound op: TFLOPS column reports 0 (no useful matmul); compare on TB/s.
"""
import torch
import tilelang
import tilelang.language as T


OP_NAME = "softmax"
DTYPE = "bf16"


@tilelang.jit(out_idx=[1])
def softmax_kernel(M, N, dtype: T.dtype = T.bfloat16):
    BN = min(tilelang.next_power_of_2(N), 8192)
    NN = tilelang.cdiv(N, BN)
    accum_dtype = T.float32
    scale = 1.44269504  # log2(e)

    @T.prim_func
    def main(
        X: T.Tensor([M, N], dtype),  # type: ignore
        Y: T.Tensor([M, N], dtype),  # type: ignore
    ):
        with T.Kernel(M, threads=128) as (i_m,):
            x = T.alloc_fragment([BN], dtype)
            y = T.alloc_fragment([BN], dtype)
            lse = T.alloc_fragment([1], accum_dtype)
            max_x = T.alloc_fragment([1], dtype)
            exp_x = T.alloc_fragment([BN], accum_dtype)
            sum_exp_x = T.alloc_fragment([1], accum_dtype)
            T.fill(lse, -T.infinity(accum_dtype))

            for i_n in T.Pipelined(0, NN):
                T.copy(X[i_m, i_n * BN:(i_n + 1) * BN], x)
                T.reduce_max(x, max_x, dim=0, clear=True)
                for j in T.Parallel(BN):
                    exp_x[j] = T.exp2(x[j] * scale - max_x[0] * scale)
                T.reduce_sum(exp_x, sum_exp_x, dim=0, clear=True)
                lse[0] = max_x[0] * scale + T.log2(
                    T.exp2(lse[0] - max_x[0] * scale) + sum_exp_x[0])

            for i_n in T.Pipelined(0, NN):
                T.copy(X[i_m, i_n * BN:(i_n + 1) * BN], x)
                for j in T.Parallel(BN):
                    y[j] = T.exp2(x[j] * scale - lse[0])
                T.copy(y, Y[i_m, i_n * BN:(i_n + 1) * BN])

    return main


_SHAPES = [
    # (M, N)
    (4096, 8192),
    (16384, 8192),
    (32768, 8192),
]


CASES = [dict(M=M, N=N) for M, N in _SHAPES]


def bench_one(case, check):
    M, N = case["M"], case["N"]
    kernel = softmax_kernel(M, N)

    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    if check:
        y = kernel(x)
        ref = x.softmax(dim=1)
        torch.testing.assert_close(y, ref, rtol=1e-2, atol=1e-2)

    profiler = kernel.get_profiler()
    latency_ms = profiler.do_bench(backend="cupti", input_tensors=[x])

    # Memory-bound: read X + write Y, both bf16.
    bytes_moved = 2 * M * N * 2
    tbps = bytes_moved / (latency_ms * 1e-3) / 1e12
    return {
        "shape_str": f"{M}x{N}",
        "tile_str": "-",
        "dtype": DTYPE,
        "latency_ms": latency_ms,
        "tbps": tbps,
        "tflops": 0.0,
        "_kernel": kernel,
    }
