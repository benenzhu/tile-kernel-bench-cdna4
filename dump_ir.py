"""Dump per-pass IR for the example_gemm matmul kernel.

Re-uses the matmul function body from gemm/example_gemm.py but re-applies
the @tilelang.jit decorator with `tl.enable_dump_ir = True` so every pass
in the lowering pipeline writes its IR snapshot to ./dump_ir/.
"""
import os
import shutil
import sys

import tilelang
import tilelang.language as T

DUMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "dump_ir"))

# Clean before each run so the dumps reflect only this invocation.
if os.path.isdir(DUMP_DIR):
    shutil.rmtree(DUMP_DIR)
os.makedirs(DUMP_DIR, exist_ok=True)

PASS_CONFIGS = {
    "tl.enable_dump_ir": True,
    "tl.dump_ir_path": DUMP_DIR,
}


@tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)
def matmul(M, N, K, block_M, block_N, block_K,
           num_stages=3, num_threads=128,
           dtype=T.bfloat16, accum_dtype=T.float32):
    """NN layout: B is stored as (K, N), no transpose. Verbatim copy of
    /root/tile-kernel-bench-cdna4/gemm/example_gemm.py:matmul."""

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, k_pack=2)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def main():
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)
    print(f"\nDump directory: {DUMP_DIR}")
    print("Per-pass IRs written. Sorted listing:\n")
    for fname in sorted(os.listdir(DUMP_DIR)):
        full = os.path.join(DUMP_DIR, fname)
        size = os.path.getsize(full)
        print(f"  {fname:80s}  {size:>8d} bytes")


if __name__ == "__main__":
    main()
