"""Sweep GEMM shapes through the tilelang kernel and print a flydsl-style table.

Output format (matches FlyDSL CI):

    op                     shape                              dtype            TB/s     TFLOPS
    ---------------------- ---------------------------------- ---------- ---------- ----------
    gemm                   1024x1024x1024_tile128x128x32      fp16            X.XXX    XXX.XXX
"""
import argparse
from dataclasses import dataclass

import torch

from example_gemm import matmul

DTYPE_BYTES = {torch.float16: 2, torch.bfloat16: 2}
DTYPE_NAME = {torch.float16: "fp16", torch.bfloat16: "bf16"}


@dataclass
class Case:
    M: int
    N: int
    K: int
    block_M: int = 128
    block_N: int = 128
    block_K: int = 64


DEFAULT_CASES = [
    Case(1024, 1024, 1024),
    Case(1024, 8192, 8192),
    Case(2048, 2048, 2048),
    Case(4096, 4096, 4096),
    Case(4096, 4096, 8192),
    Case(8192, 8192, 1024),
    Case(8192, 8192, 8192),
]


def bench_one(case: Case, dtype, check: bool):
    M, N, K = case.M, case.N, case.K
    kernel = matmul(M, N, K, case.block_M, case.block_N, case.block_K)

    if check:
        a = torch.randn(M, K, device="cuda", dtype=dtype)
        b = torch.randn(K, N, device="cuda", dtype=dtype)
        c = kernel(a, b)
        ref = a @ b
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)

    profiler = kernel.get_profiler()
    latency_ms = profiler.do_bench(backend="cupti")

    flops = 2.0 * M * N * K
    bytes_moved = (M * K + K * N + M * N) * DTYPE_BYTES[dtype]
    tflops = flops / (latency_ms * 1e-3) / 1e12
    tbps = bytes_moved / (latency_ms * 1e-3) / 1e12  # TB/s
    return tbps, tflops


def parse_case(s: str) -> Case:
    # Format: M,N,K[,bM,bN,bK]
    parts = [int(x) for x in s.split(",")]
    if len(parts) == 3:
        return Case(*parts)
    if len(parts) == 6:
        return Case(*parts)
    raise argparse.ArgumentTypeError(
        f"shape must be M,N,K or M,N,K,bM,bN,bK (got {s!r})"
    )


def print_header():
    # widths chosen to match flydsl: op(22) shape(34) dtype(10) tbs(10) tflops(10)
    print(f"{'op':<22} {'shape':<34} {'dtype':<10} {'TB/s':>10} {'TFLOPS':>10}")
    print(f"{'-'*22} {'-'*34} {'-'*10} {'-'*10} {'-'*10}")


def print_row(op, shape, dtype, tbs, tflops):
    tflops_str = "-" if tflops is None else f"{tflops:>10.3f}"
    print(f"{op:<22} {shape:<34} {dtype:<10} {tbs:>10.3f} {tflops_str}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shapes",
        type=parse_case,
        nargs="*",
        default=None,
        help="Override cases, e.g. --shapes 1024,1024,1024 4096,4096,4096,128,128,64",
    )
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    cases = args.shapes if args.shapes else DEFAULT_CASES
    dtype = torch.float16

    # Run all cases first; tilelang prints INFO logs during compile/bench, so
    # printing the table inline gets interleaved. Collect results and dump
    # the table at the very end.
    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] benchmarking "
              f"{case.M}x{case.N}x{case.K} ...", flush=True)
        tbs, tflops = bench_one(case, dtype=dtype, check=not args.no_check)
        shape = (
            f"{case.M}x{case.N}x{case.K}"
            f"_tile{case.block_M}x{case.block_N}x{case.block_K}"
        )
        results.append((shape, DTYPE_NAME[dtype], tbs, tflops))

    print()
    print("=" * 80)
    print("Benchmarks")
    print("=" * 80)
    print()
    print_header()
    for shape, dtype_name, tbs, tflops in results:
        print_row("gemm", shape, dtype_name, tbs, tflops)


if __name__ == "__main__":
    main()
