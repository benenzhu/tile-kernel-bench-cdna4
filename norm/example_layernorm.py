"""LayerNorm forward kernel, lifted from tilelang/examples/norm/layernorm.py.

Memory-bound op: TFLOPS column reports 0; compare on TB/s. Backward path
in the upstream example is left out (this bench targets fwd only).
"""
import torch
import tilelang
import tilelang.language as T


OP_NAME = "layernorm"
DTYPE = "bf16"


@tilelang.jit(out_idx=[-3, -2, -1])
def _layernorm_fwd(N, D, eps=1e-5, blk_m=1, threads=256,
                   in_dtype="bfloat16", out_dtype="bfloat16"):
    accum_dtype = "float"

    @T.prim_func
    def main(
        X: T.Tensor((N, D), in_dtype),  # type: ignore
        gamma: T.Tensor((D,), in_dtype),  # type: ignore
        beta: T.Tensor((D,), in_dtype),  # type: ignore
        Y: T.Tensor((N, D), out_dtype),  # type: ignore
        Mean: T.Tensor((N,), accum_dtype),  # type: ignore
        Rstd: T.Tensor((N,), accum_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(N, blk_m), threads=threads) as bx:
            X_smem = T.alloc_shared((blk_m, D), in_dtype)
            G_smem = T.alloc_shared((D,), in_dtype)
            B_smem = T.alloc_shared((D,), in_dtype)
            X_local = T.alloc_fragment((blk_m, D), accum_dtype)
            X_sq_local = T.alloc_fragment((blk_m, D), accum_dtype)
            sum_row = T.alloc_fragment((blk_m,), accum_dtype)
            sumsq_row = T.alloc_fragment((blk_m,), accum_dtype)
            mean_row = T.alloc_fragment((blk_m,), accum_dtype)
            rstd_row = T.alloc_fragment((blk_m,), accum_dtype)

            T.copy(X[bx * blk_m, 0], X_smem)
            T.copy(gamma, G_smem)
            T.copy(beta, B_smem)

            for i, j in T.Parallel(blk_m, D):
                X_local[i, j] = T.Cast(accum_dtype, X_smem[i, j])
            for i, j in T.Parallel(blk_m, D):
                X_sq_local[i, j] = X_local[i, j] * X_local[i, j]

            T.reduce_sum(X_local, sum_row, dim=1)
            T.reduce_sum(X_sq_local, sumsq_row, dim=1)

            inv_D = T.float32(1.0) / T.Cast(accum_dtype, D)
            for i in T.Parallel(blk_m):
                mean_row[i] = sum_row[i] * inv_D
                rstd_row[i] = T.rsqrt(sumsq_row[i] * inv_D - mean_row[i] * mean_row[i]
                                      + T.Cast(accum_dtype, eps))
                Mean[bx * blk_m + i] = mean_row[i]
                Rstd[bx * blk_m + i] = rstd_row[i]

            for i, j in T.Parallel(blk_m, D):
                norm = (X_local[i, j] - mean_row[i]) * rstd_row[i]
                X_smem[i, j] = T.Cast(
                    out_dtype,
                    norm * T.Cast(accum_dtype, G_smem[j]) + T.Cast(accum_dtype, B_smem[j]),
                )

            T.copy(X_smem, Y[bx * blk_m, 0])

    return main


_SHAPES = [
    # (N, D)
    (4096, 8192),
    (16384, 8192),
    (32768, 8192),
]


CASES = [dict(N=N, D=D) for N, D in _SHAPES]


def bench_one(case, check):
    N, D = case["N"], case["D"]
    kernel = _layernorm_fwd(N, D)

    x = torch.randn(N, D, device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn(D, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(D, device="cuda", dtype=torch.bfloat16)

    if check:
        y, mean, rstd = kernel(x, gamma, beta)
        ref = torch.nn.functional.layer_norm(x.float(), (D,), gamma.float(), beta.float()).to(x.dtype)
        torch.testing.assert_close(y, ref, rtol=1e-2, atol=1e-2)

    profiler = kernel.get_profiler()
    latency_ms = profiler.do_bench(backend="cupti", input_tensors=[x, gamma, beta])

    # Bandwidth model: read X (N*D bf16) + read gamma+beta (2*D bf16, tiny)
    # + write Y (N*D bf16) + write Mean,Rstd (N fp32, tiny). Dominated by X+Y.
    bytes_moved = (2 * N * D + 2 * D) * 2 + 2 * N * 4
    tbps = bytes_moved / (latency_ms * 1e-3) / 1e12
    return {
        "shape_str": f"{N}x{D}",
        "tile_str": "blk_m1",
        "dtype": DTYPE,
        "latency_ms": latency_ms,
        "tbps": tbps,
        "tflops": 0.0,
    }
