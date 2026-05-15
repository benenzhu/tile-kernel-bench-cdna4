"""DeepSeek MLA decode kernel for AMD/CDNA, lifted from
tilelang/examples/deepseek_mla/amd/benchmark_mla_decode_amd_tilelang.py.

Upstream uses tilelang.autotune over a 192-config grid; we pin one config
that mirrors the example defaults so it slots into the bench harness.

Verification (`check=True`) materialises a reference attention via
einsum and is RAM-heavy for large kv_ctx -- disable it for the bigger
shapes if memory is tight.
"""
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T

from einops import rearrange, einsum


OP_NAME = "mla_decode"
DTYPE = "fp16"


@tilelang.jit(
    out_idx=[6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashmla_decode(batch, heads, kv_head_num, seqlen_kv, dim, pe_dim,
                    block_N, block_H, num_split, threads=128):
    scale = (1.0 / (dim + pe_dim)) ** 0.5 * 1.44269504  # log2(e)
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // kv_head_num
    VALID_BLOCK_H = min(block_H, kv_group_num)
    assert kv_head_num == 1, "kv_head_num must be 1"

    @T.prim_func
    def main_split(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        glse: T.Tensor([batch, heads, num_split], dtype),
        Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        # flash_attn_split
        with T.Kernel(batch, heads // min(block_H, kv_group_num), num_split, threads=threads) as (bx, by, bz):
            Q_local = T.alloc_fragment([block_H, dim], dtype)
            Q_pe_local = T.alloc_fragment([block_H, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            cur_kv_head = by // (kv_group_num // block_H)
            T.use_swizzle(10)
            T.copy(Q[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :], Q_local)
            T.copy(Q_pe[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, :], Q_pe_local)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv((seqlen_kv // num_split), block_N)
            for k in T.Pipelined(loop_range, num_stages=0):
                kv_start = (seqlen_kv // num_split) * bz + k * block_N
                kv_end = (seqlen_kv // num_split) * bz + (k + 1) * block_N
                T.copy(KV[bx, kv_start:kv_end, cur_kv_head, :], KV_shared)
                T.copy(K_pe[bx, kv_start:kv_end, cur_kv_head, :], K_pe_shared)
                T.clear(acc_s)
                T.gemm(Q_local, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(Q_pe_local, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, acc_s_cast)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(acc_s_cast, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] /= logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, glse[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, bz])
            T.copy(acc_o, Output_partial[bx, by * VALID_BLOCK_H:(by + 1) * VALID_BLOCK_H, bz, :])

        # combine
        with T.Kernel(heads, batch, threads=128) as (by, bz):
            po_local = T.alloc_fragment([dim], dtype)
            o_accum_local = T.alloc_fragment([dim], accum_dtype)
            lse_local_split = T.alloc_var(accum_dtype)
            lse_logsum_local = T.alloc_var(accum_dtype)
            lse_max_local = T.alloc_var(accum_dtype)
            scale_local = T.alloc_var(accum_dtype)

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            lse_max_local = -T.infinity(accum_dtype)
            for k in T.serial(num_split):
                lse_max_local = T.max(lse_max_local, glse[bz, by, k])
            for k in T.Pipelined(num_split, num_stages=1):
                lse_local_split = glse[bz, by, k]
                lse_logsum_local += T.exp2(lse_local_split - lse_max_local)
            lse_logsum_local = T.log2(lse_logsum_local) + lse_max_local
            for k in T.serial(num_split):
                for i in T.Parallel(dim):
                    po_local[i] = Output_partial[bz, by, k, i]
                lse_local_split = glse[bz, by, k]
                scale_local = T.exp2(lse_local_split - lse_logsum_local)
                for i in T.Parallel(dim):
                    o_accum_local[i] += po_local[i] * scale_local
            for i in T.Parallel(dim):
                Output[bz, by, i] = o_accum_local[i]

    return main_split


def _ref_program(q, q_pe, kv, k_pe):
    dim = q.shape[-1]
    pe_dim = q_pe.shape[-1]
    num_head_groups = q.shape[1] // kv.shape[2]
    scale = (dim + pe_dim) ** 0.5
    q = rearrange(q, "b (h g) d -> b g h d", g=num_head_groups)
    q_pe = rearrange(q_pe, "b (h g) d -> b g h d", g=num_head_groups)
    kv = rearrange(kv, "b n h d -> b h n d")
    k_pe = rearrange(k_pe, "b n h d -> b h n d")
    query = torch.concat([q, q_pe], dim=-1)
    key = torch.concat([kv, k_pe], dim=-1)
    scores = einsum(query, key, "b g h d, b h s d -> b g h s")
    attention = F.softmax(scores / scale, dim=-1)
    out = einsum(attention, kv, "b g h s, b h s d -> b g h d")
    out = rearrange(out, "b g h d -> b (h g) d")
    return out


CASES = [
    # (batch, heads, kv_heads, kv_ctx, dim, pe_dim, BLOCK_N, BLOCK_H, num_split, threads)
    dict(batch=64, heads=128, kv_heads=1, kv_ctx=4096, dim=512, pe_dim=64,
         BLOCK_N=32, BLOCK_H=64, num_split=4, threads=128),
    dict(batch=128, heads=128, kv_heads=1, kv_ctx=8192, dim=512, pe_dim=64,
         BLOCK_N=32, BLOCK_H=64, num_split=4, threads=128),
]


def bench_one(case, check):
    batch = case["batch"]
    heads = case["heads"]
    kv_heads = case["kv_heads"]
    kv_ctx = case["kv_ctx"]
    dim = case["dim"]
    pe_dim = case["pe_dim"]
    BLOCK_N = case["BLOCK_N"]
    BLOCK_H = case["BLOCK_H"]
    num_split = case["num_split"]
    threads = case["threads"]

    kernel = flashmla_decode(batch, heads, kv_heads, kv_ctx, dim, pe_dim,
                             BLOCK_N, BLOCK_H, num_split, threads=threads)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)

    if check:
        inputs = profiler._get_inputs()
        out = kernel(*inputs)
        # First 4 inputs are Q, Q_pe, KV, K_pe; remaining are workspace tensors.
        ref = _ref_program(*inputs[:4])
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    latency_ms = profiler.do_bench(warmup=500)

    qk_flops = 2.0 * batch * heads * kv_ctx * (dim + pe_dim)
    pv_flops = 2.0 * batch * heads * kv_ctx * dim
    flops = qk_flops + pv_flops
    # Bytes: Q + Q_pe + KV + K_pe (read) + Output (write).
    fp16_b = 2
    bytes_moved = (
        batch * heads * dim * fp16_b
        + batch * heads * pe_dim * fp16_b
        + batch * kv_ctx * kv_heads * dim * fp16_b
        + batch * kv_ctx * kv_heads * pe_dim * fp16_b
        + batch * heads * dim * fp16_b
    )
    tflops = flops / (latency_ms * 1e-3) / 1e12
    tbps = bytes_moved / (latency_ms * 1e-3) / 1e12

    shape_str = (f"b{batch}_h{heads}_kv{kv_ctx}_d{dim}_pe{pe_dim}")
    tile_str = f"N{BLOCK_N}_H{BLOCK_H}_split{num_split}_t{threads}"
    return {
        "shape_str": shape_str,
        "tile_str": tile_str,
        "dtype": DTYPE,
        "latency_ms": latency_ms,
        "tbps": tbps,
        "tflops": tflops,
        "_kernel": kernel,
    }
