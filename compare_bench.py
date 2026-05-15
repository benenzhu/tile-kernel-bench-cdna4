"""Compare two bench CSVs (baseline vs current) and report TFLOPS / TB/s deltas.

CSV columns expected:
    op, shape_str, tile_str, dtype, latency_ms, tbps, tflops

Match key is (op, shape_str, dtype) -- tile_str is *not* part of the key,
so retuning the tile (or any other hyper-param) and getting more TFLOPS
shows up as an improvement, not a "shape missing" mismatch.

Exits 1 if any matched shape regresses TFLOPS by more than --threshold (%).
"""
import argparse
import csv
import sys


KEY_COLS = ("op", "shape_str", "dtype")


def load_csv(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = tuple(r[c] for c in KEY_COLS)
            entry = {
                "tile": r["tile_str"],
                "tbps": float(r["tbps"]),
                "tflops": float(r["tflops"]),
                "latency_ms": float(r["latency_ms"]),
            }
            # If duplicates exist for same shape (different tiles), keep the
            # best TFLOPS so the baseline is the most generous comparison.
            if key in rows and rows[key]["tflops"] >= entry["tflops"]:
                continue
            rows[key] = entry
    return rows


def fmt_delta(base, cur):
    if base == 0:
        return "n/a"
    pct = (cur - base) / base * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("current")
    ap.add_argument("--threshold", type=float, default=5.0,
                    help="Fail if TFLOPS regresses by more than this %% (default 5)")
    args = ap.parse_args()

    base = load_csv(args.baseline)
    cur = load_csv(args.current)

    keys = sorted(set(base) | set(cur))
    regressions = []

    cols = (
        f"{'op':<12} {'shape':<28} {'dtype':<6} "
        f"{'tile base':<22} {'tile cur':<22} "
        f"{'TFLOPS base':>12} {'TFLOPS cur':>12} {'ΔTFLOPS':>10} "
        f"{'TB/s base':>10} {'TB/s cur':>10} {'ΔTB/s':>10}"
    )
    print("=" * 90)
    print(f"Compare baseline={args.baseline}  current={args.current}")
    print(f"Match key: {KEY_COLS} (tile is NOT a match key)")
    print(f"Regression threshold: {args.threshold:.2f}%")
    print("=" * 90)
    print(cols)
    print("-" * len(cols))

    for k in keys:
        op, shape, dt = k
        b = base.get(k)
        c = cur.get(k)
        if b is None:
            print(f"{op:<12} {shape:<28} {dt:<6} "
                  f"{'-':<22} {c['tile']:<22} "
                  f"{'n/a':>12} {c['tflops']:>12.3f} {'NEW':>10} "
                  f"{'n/a':>10} {c['tbps']:>10.3f} {'NEW':>10}")
            continue
        if c is None:
            print(f"{op:<12} {shape:<28} {dt:<6} "
                  f"{b['tile']:<22} {'-':<22} "
                  f"{b['tflops']:>12.3f} {'n/a':>12} {'MISSING':>10} "
                  f"{b['tbps']:>10.3f} {'n/a':>10} {'MISSING':>10}")
            continue
        tflops_delta = fmt_delta(b["tflops"], c["tflops"])
        tbps_delta = fmt_delta(b["tbps"], c["tbps"])
        print(f"{op:<12} {shape:<28} {dt:<6} "
              f"{b['tile']:<22} {c['tile']:<22} "
              f"{b['tflops']:>12.3f} {c['tflops']:>12.3f} {tflops_delta:>10} "
              f"{b['tbps']:>10.3f} {c['tbps']:>10.3f} {tbps_delta:>10}")

        pct = (c["tflops"] - b["tflops"]) / b["tflops"] * 100.0 if b["tflops"] else 0
        if pct < -args.threshold:
            regressions.append((op, shape, dt, b, c, pct))

    print()
    if regressions:
        print(f"FAIL: {len(regressions)} shape(s) regressed > {args.threshold:.2f}%:")
        for op, shape, dt, b, c, pct in regressions:
            print(f"  - {op} {shape} {dt}: {b['tflops']:.3f} ({b['tile']}) -> "
                  f"{c['tflops']:.3f} ({c['tile']}) TFLOPS ({pct:+.2f}%)")
        sys.exit(1)
    else:
        print(f"OK: no TFLOPS regression beyond {args.threshold:.2f}%")


if __name__ == "__main__":
    main()
