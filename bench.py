"""Single bench entrypoint that sweeps every registered op over its CASES.

Each op module exports:
    OP_NAME: str
    CASES: list[dict]
    bench_one(case, check) -> dict with keys
        shape_str, tile_str, dtype, latency_ms, tbps, tflops

Run:
    python bench.py                       # all ops, all cases
    python bench.py --ops gemm gemm_fp8   # just these
    python bench.py --output-csv out.csv  # also dump CSV
"""
import argparse
import csv
import importlib
import sys
import traceback

# Apply local tilelang monkey-patches BEFORE any op module imports tilelang.
import tl_patches  # noqa: F401


# Registered op modules. Order = bench order = table order.
OP_MODULES = [
    "gemm.example_gemm",                      # fp16, sweeps NN + NT layouts
    "gemm.example_gemm_fp8",                  # fp8 NT, plain + preshuffled B
    "gemv.example_gemv",                      # fp16 vectorized split-K gemv
    "norm.example_softmax",                   # bf16, memory-bound
    "norm.example_rmsnorm",                   # bf16, memory-bound
    "norm.example_layernorm",                 # bf16, memory-bound
    "attention.example_flash_attn_fwd",       # fp16 MHA fwd, ±causal
    "mla.example_mla_decode",
    "elementwise.example_add_3d_large",       # fp16 3D add, >4 GB BufferLoad guard
]


def print_header():
    print(f"{'op':<14} {'shape':<28} {'dtype':<6} {'tile':<22} "
          f"{'TB/s':>10} {'TFLOPS':>10}")
    print(f"{'-'*14} {'-'*28} {'-'*6} {'-'*22} {'-'*10} {'-'*10}")


def print_row(r):
    print(f"{r['op']:<14} {r['shape_str']:<28} {r['dtype']:<6} {r['tile_str']:<22} "
          f"{r['tbps']:>10.3f} {r['tflops']:>10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", nargs="*", default=None,
                    help="Subset by OP_NAME, e.g. --ops gemm mla_decode")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--output-csv", type=str, default=None)
    args = ap.parse_args()

    # Discover ops first so we fail fast on import errors.
    ops = []
    for path in OP_MODULES:
        mod = importlib.import_module(path)
        ops.append(mod)
    if args.ops:
        ops = [m for m in ops if m.OP_NAME in set(args.ops)]
        if not ops:
            print(f"No ops match {args.ops}", file=sys.stderr)
            sys.exit(2)

    results = []
    failures = []
    total = sum(len(m.CASES) for m in ops)
    idx = 0
    for mod in ops:
        for case in mod.CASES:
            idx += 1
            print(f"[{idx}/{total}] {mod.OP_NAME}: {case}", flush=True)
            try:
                r = mod.bench_one(case, check=not args.no_check)
                r["op"] = mod.OP_NAME
                results.append(r)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                traceback.print_exc()
                failures.append((mod.OP_NAME, case, repr(e)))

    print()
    print("=" * 90)
    print("Benchmarks")
    print("=" * 90)
    print()
    print_header()
    for r in results:
        print_row(r)

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["op", "shape_str", "tile_str", "dtype",
                        "latency_ms", "tbps", "tflops"])
            for r in results:
                w.writerow([r["op"], r["shape_str"], r["tile_str"], r["dtype"],
                            f"{r['latency_ms']:.6f}", f"{r['tbps']:.6f}",
                            f"{r['tflops']:.6f}"])
        print(f"\nWrote CSV: {args.output_csv}")

    if failures:
        print(f"\n{len(failures)} case(s) failed:", file=sys.stderr)
        for op, case, err in failures:
            print(f"  - {op} {case}: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
