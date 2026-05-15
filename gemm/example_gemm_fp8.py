"""FP8 GEMM kernels for AMD/CDNA, lifted from
tilelang/examples/gemm_fp8/example_tilelang_gemm_amd*.py.

Two flavours, both NT layout:
  * fp8_matmul              -- plain shared-memory tilemma path.
  * fp8_matmul_preshuffle   -- B is pre-shuffled into the layout the AMD
                                matrix-core (mfma) emitter expects.

Upstream uses tilelang.autotune; we pin one config so it slots into the
bench harness without burning hours on tuning. CASES sweeps each shape
with `preshuffle in (False, True)` so the two variants print adjacent.
"""
import torch
import tilelang
import tilelang.language as T

from tilelang.layout import make_swizzled_layout
from tilelang.rocm.intrinsics.mfma_macro_generator import MatrixCorePreshuffleIntrinEmitter
from tilelang.tileop.base import GemmWarpPolicy
from tilelang.utils import determine_fp8_type, determine_torch_fp8_type


OP_NAME = "gemm_fp8"
DTYPE = "fp8"


@tilelang.jit(out_idx=[-1])
def fp8_matmul(M, N, K, block_M, block_N, block_K, num_stages, num_threads, k_pack):
    """Plain fp8 NT GEMM (no preshuffle)."""
    dtype = determine_fp8_type()
    accum_dtype = T.float32

    @T.prim_func
    def gemm_fp8_ss(
        A: T.Tensor((M, K), dtype),  # type: ignore
        B: T.Tensor((N, K), dtype),  # type: ignore
        C: T.Tensor((M, N), accum_dtype),  # type: ignore
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


@tilelang.jit(out_idx=[-1])
def fp8_matmul_preshuffle(M, N, K, block_M, block_N, block_K, num_stages, num_threads, k_pack):
    """fp8 NT GEMM with B pre-shuffled into the mfma intrinsic layout.

    Lifted from tilelang/examples/gemm_fp8/example_tilelang_gemm_amd_fp8_preshuffle.py.
    Caller must run `_shuffle_b()` on B before passing it to the kernel.
    """
    in_dtype = determine_fp8_type()
    accum_dtype = T.float32
    out_dtype = T.float32

    warp_size = 64
    num_warps = num_threads // warp_size
    policy = GemmWarpPolicy.Square
    m_warp, n_warp = policy.compute_warp_partition(block_M, block_N, num_warps)

    warp_row_tiles = block_M // m_warp
    warp_col_tiles = block_N // n_warp

    mfma = MatrixCorePreshuffleIntrinEmitter(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=m_warp,
        block_col_warps=n_warp,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=block_K,
        k_pack=k_pack,
        b_preshuffle=True,
    )
    micro_size_y = mfma.micro_size_y
    micro_size_k = mfma.micro_size_k
    pack_size_k = micro_size_k * k_pack

    A_shape = (M, K)
    A_shared_shape = (block_M, block_K)
    B_shape = (N // micro_size_y, K // pack_size_k, micro_size_y, pack_size_k)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),  # type: ignore
        B: T.Tensor(B_shape, in_dtype),  # type: ignore
        C: T.Tensor((M, N), out_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope="shared")
            A_local = T.alloc_local((mfma.warp_rows * mfma.local_size_a * k_pack), in_dtype)
            B_local = T.alloc_local((mfma.warp_cols * mfma.local_size_b * k_pack), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.annotate_layout({
                A_shared: make_swizzled_layout(A_shared),
                C_local: mfma.make_mfma_store_layout(C_local),
            })

            num_ko = K // block_K
            num_ki = block_K // (k_pack * micro_size_k)

            T.clear(C_local)
            for ko in T.Pipelined(num_ko, num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                for ki in T.serial(0, num_ki):
                    mfma.ldmatrix_a(A_local, A_shared, ki)
                    mfma.ldmatrix_b(B_local, B, ki + ko * num_ki, pid_m=by, pid_n=bx)
                    mfma.mfma(A_local, B_local, C_local, ki)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _shuffle_b(b: torch.Tensor, k_pack: int) -> torch.Tensor:
    """Reshape B (N, K) into the (N/16, K/(32*k_pack), 16, 32*k_pack) layout
    the preshuffle kernel expects. Mirrors `shuffle_weight(..., is_transpose=True)`
    from the upstream example.
    """
    BN, BK = 16, 32 * k_pack
    N, K = b.shape
    assert N % BN == 0 and K % BK == 0, (N, K, BN, BK)
    return b.view(N // BN, BN, K // BK, BK).permute(0, 2, 1, 3).contiguous()


def _make_cases():
    shapes = [
        # (M, N, K, block_M, block_N, block_K, num_stages, num_threads, k_pack)
        (1024, 1024, 1024, 128, 128, 128, 0, 256, 2),
        (2048, 2048, 2048, 128, 128, 128, 0, 256, 2),
        (4096, 4096, 4096, 128, 128, 128, 0, 256, 2),
        (8192, 8192, 8192, 128, 128, 128, 0, 256, 2),
    ]
    out = []
    for M, N, K, bM, bN, bK, ns, nt, kp in shapes:
        # Order matters: plain then preshuffle, so they print adjacent.
        for preshuffle in (False, True):
            out.append(dict(
                M=M, N=N, K=K,
                block_M=bM, block_N=bN, block_K=bK,
                num_stages=ns, num_threads=nt, k_pack=kp,
                preshuffle=preshuffle,
            ))
    return out


CASES = _make_cases()


def bench_one(case, check):
    M, N, K = case["M"], case["N"], case["K"]
    preshuffle = case["preshuffle"]
    suffix = "_pre" if preshuffle else ""

    factory = fp8_matmul_preshuffle if preshuffle else fp8_matmul
    kernel = factory(
        M, N, K,
        case["block_M"], case["block_N"], case["block_K"],
        case["num_stages"], case["num_threads"], case["k_pack"],
    )

    # Tilelang's auto-supplied tensor path goes through dlpack which doesn't
    # yet handle torch.float8_*; build the fp8 inputs ourselves and pass them
    # explicitly. (The cython execution backend is what makes the kernel call
    # itself work -- see TILELANG_EXECUTION_BACKEND in CI.)
    torch_fp8 = determine_torch_fp8_type()
    a = (torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.01).to(torch_fp8)
    b = (torch.randn(N, K, dtype=torch.float16, device="cuda") * 0.01).to(torch_fp8)
    b_kernel = _shuffle_b(b, k_pack=case["k_pack"]) if preshuffle else b

    if check:
        c = kernel(a, b_kernel)
        ref = (a.half() @ b.half().T).to(torch.float32)
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-1)

    profiler = kernel.get_profiler()
    latency_ms = profiler.do_bench(backend="cupti", input_tensors=[a, b_kernel])

    flops = 2.0 * M * N * K
    # Bytes: A (fp8=1B), B (fp8=1B), C (fp32=4B). Preshuffle changes B's
    # layout but not its byte count.
    bytes_moved = (M * K) * 1 + (N * K) * 1 + (M * N) * 4
    tflops = flops / (latency_ms * 1e-3) / 1e12
    tbps = bytes_moved / (latency_ms * 1e-3) / 1e12

    return {
        "shape_str": f"{M}x{N}x{K}{suffix}",
        "tile_str": f"{case['block_M']}x{case['block_N']}x{case['block_K']}",
        "dtype": DTYPE,
        "latency_ms": latency_ms,
        "tbps": tbps,
        "tflops": tflops,
    }
