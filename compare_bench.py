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


def _maybe_int(v):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except ValueError:
        return None


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
                # Resource counts are HIP-only; absent from older CSVs.
                "n_regs": _maybe_int(r.get("n_regs")),
                "n_spills": _maybe_int(r.get("n_spills")),
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
    ap.add_argument("--markdown", type=str, default=None,
                    help="Also write a markdown summary to this path (PR comment body)")
    args = ap.parse_args()

    base = load_csv(args.baseline)
    cur = load_csv(args.current)

    keys = sorted(set(base) | set(cur))
    regressions = []
    improvements = []
    new_shapes = []
    missing_shapes = []
    # Compiler resource changes (informational; do NOT fail the run).
    regs_changed = []   # (op, shape, dt, b_regs, c_regs)
    spills_changed = []  # (op, shape, dt, b_spills, c_spills)

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

    # Capture table rows so we can also emit a markdown copy later.
    table_rows = []

    def emit(line):
        print(line)
        table_rows.append(line)

    for k in keys:
        op, shape, dt = k
        b = base.get(k)
        c = cur.get(k)
        if b is None:
            emit(f"{op:<12} {shape:<28} {dt:<6} "
                 f"{'-':<22} {c['tile']:<22} "
                 f"{'n/a':>12} {c['tflops']:>12.3f} {'NEW':>10} "
                 f"{'n/a':>10} {c['tbps']:>10.3f} {'NEW':>10}")
            new_shapes.append((op, shape, dt, c))
            continue
        if c is None:
            emit(f"{op:<12} {shape:<28} {dt:<6} "
                 f"{b['tile']:<22} {'-':<22} "
                 f"{b['tflops']:>12.3f} {'n/a':>12} {'MISSING':>10} "
                 f"{b['tbps']:>10.3f} {'n/a':>10} {'MISSING':>10}")
            missing_shapes.append((op, shape, dt, b))
            continue
        tflops_delta = fmt_delta(b["tflops"], c["tflops"])
        tbps_delta = fmt_delta(b["tbps"], c["tbps"])
        emit(f"{op:<12} {shape:<28} {dt:<6} "
             f"{b['tile']:<22} {c['tile']:<22} "
             f"{b['tflops']:>12.3f} {c['tflops']:>12.3f} {tflops_delta:>10} "
             f"{b['tbps']:>10.3f} {c['tbps']:>10.3f} {tbps_delta:>10}")

        # Pick the primary perf metric per row: TFLOPS for compute-bound ops,
        # TB/s for memory-bound (softmax/norm/etc emit tflops=0 in CSV).
        if b["tflops"] > 0 and c["tflops"] > 0:
            metric, b_v, c_v = "TFLOPS", b["tflops"], c["tflops"]
        else:
            metric, b_v, c_v = "TB/s", b["tbps"], c["tbps"]
        pct = (c_v - b_v) / b_v * 100.0 if b_v else 0
        if pct < -args.threshold:
            regressions.append((op, shape, dt, b, c, pct, metric, b_v, c_v))
        elif pct > args.threshold:
            improvements.append((op, shape, dt, b, c, pct, metric, b_v, c_v))

        # Resource deltas (any non-zero change is reported -- compiler-side
        # wins/regressions matter even when perf is unchanged).
        if b["n_regs"] is not None and c["n_regs"] is not None and b["n_regs"] != c["n_regs"]:
            regs_changed.append((op, shape, dt, b["n_regs"], c["n_regs"]))
        if b["n_spills"] is not None and c["n_spills"] is not None and b["n_spills"] != c["n_spills"]:
            spills_changed.append((op, shape, dt, b["n_spills"], c["n_spills"]))

    print()
    if new_shapes:
        print(f"NEW: {len(new_shapes)} shape(s) added vs baseline:")
        for op, shape, dt, c in new_shapes:
            metric, value = ("TFLOPS", c["tflops"]) if c["tflops"] > 0 else ("TB/s", c["tbps"])
            print(f"  * {op} {shape} {dt}: {value:.3f} ({c['tile']}) {metric}")
        print()
    if missing_shapes:
        print(f"REMOVED: {len(missing_shapes)} shape(s) gone vs baseline:")
        for op, shape, dt, b in missing_shapes:
            metric, value = ("TFLOPS", b["tflops"]) if b["tflops"] > 0 else ("TB/s", b["tbps"])
            print(f"  * {op} {shape} {dt}: was {value:.3f} ({b['tile']}) {metric}")
        print()
    if improvements:
        print(f"WINS: {len(improvements)} shape(s) improved > {args.threshold:.2f}%:")
        for op, shape, dt, b, c, pct, metric, b_v, c_v in improvements:
            print(f"  + {op} {shape} {dt}: {b_v:.3f} ({b['tile']}) -> "
                  f"{c_v:.3f} ({c['tile']}) {metric} ({pct:+.2f}%)")
        print()

    # Resource changes -- separate "down" (compiler win) and "up" (compiler regression).
    regs_down = [t for t in regs_changed if t[4] < t[3]]
    regs_up   = [t for t in regs_changed if t[4] > t[3]]
    spills_down = [t for t in spills_changed if t[4] < t[3]]
    spills_up   = [t for t in spills_changed if t[4] > t[3]]
    if regs_down:
        print(f"VGPR DOWN: {len(regs_down)} shape(s) (compiler win):")
        for op, shape, dt, b_n, c_n in regs_down:
            print(f"  - {op} {shape} {dt}: {b_n} -> {c_n} VGPR ({c_n - b_n:+d})")
        print()
    if regs_up:
        print(f"VGPR UP: {len(regs_up)} shape(s) (compiler regression):")
        for op, shape, dt, b_n, c_n in regs_up:
            print(f"  + {op} {shape} {dt}: {b_n} -> {c_n} VGPR ({c_n - b_n:+d})")
        print()
    if spills_down:
        print(f"SPILLS DOWN: {len(spills_down)} shape(s) (compiler win):")
        for op, shape, dt, b_n, c_n in spills_down:
            print(f"  - {op} {shape} {dt}: {b_n} -> {c_n} spill+sc/4 ({c_n - b_n:+d})")
        print()
    if spills_up:
        print(f"SPILLS UP: {len(spills_up)} shape(s) (compiler regression):")
        for op, shape, dt, b_n, c_n in spills_up:
            print(f"  + {op} {shape} {dt}: {b_n} -> {c_n} spill+sc/4 ({c_n - b_n:+d})")
        print()

    if regressions:
        print(f"FAIL: {len(regressions)} shape(s) regressed > {args.threshold:.2f}%:")
        for op, shape, dt, b, c, pct, metric, b_v, c_v in regressions:
            print(f"  - {op} {shape} {dt}: {b_v:.3f} ({b['tile']}) -> "
                  f"{c_v:.3f} ({c['tile']}) {metric} ({pct:+.2f}%)")
    else:
        print(f"OK: no perf regression beyond {args.threshold:.2f}%")

    if args.markdown:
        _write_markdown(
            args.markdown, args.threshold, cols, table_rows,
            regressions, improvements, new_shapes, missing_shapes,
            regs_down, regs_up, spills_down, spills_up,
        )

    if regressions:
        sys.exit(1)


def _md_table(headers, rows):
    """Render a markdown table from a list of header strings and rows
    (each row is a list of cell strings, same length as headers)."""
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return out


def _perf_table(section_rows):
    """Build a perf table (Regressions / Wins).

    section_rows: iterable of (op, shape, dtype, b, c, pct, metric, b_v, c_v).
    The 'Tile' column is included only if at least one row has a tile change.
    For rows where tile is unchanged, that cell shows just the tile.
    """
    rows = list(section_rows)
    any_tile_changed = any(r[3]["tile"] != r[4]["tile"] for r in rows)
    headers = ["op", "shape", "dtype", "base", "current", "Δ", "metric"]
    if any_tile_changed:
        headers.append("tile")
    out_rows = []
    for op, shape, dt, b, c, pct, metric, b_v, c_v in rows:
        row = [
            f"`{op}`",
            f"`{shape}`",
            dt,
            f"{b_v:.3f}",
            f"**{c_v:.3f}**",
            f"**{pct:+.2f}%**",
            metric,
        ]
        if any_tile_changed:
            row.append(
                f"`{b['tile']}` → `{c['tile']}`"
                if b["tile"] != c["tile"]
                else f"`{b['tile']}`"
            )
        out_rows.append(row)
    return _md_table(headers, out_rows)


def _resource_table(section_rows, unit_label):
    """Build a resource table (VGPR / Spills). No tile column."""
    headers = ["op", "shape", "dtype", "base", "current", "Δ"]
    out_rows = []
    for op, shape, dt, b_n, c_n in section_rows:
        out_rows.append([
            f"`{op}`",
            f"`{shape}`",
            dt,
            str(b_n),
            f"**{c_n}**",
            f"**{c_n - b_n:+d} {unit_label}**",
        ])
    return _md_table(headers, out_rows)


def _write_markdown(path, threshold, header_line, table_rows,
                    regressions, improvements, new_shapes, missing_shapes,
                    regs_down=None, regs_up=None,
                    spills_down=None, spills_up=None):
    """Render a PR-comment-friendly markdown summary.

    Important headlines (regressions, wins, new shapes, compiler-resource
    deltas) live OUTSIDE the collapsible block so the reader doesn't have
    to expand to see them. The full per-row table is inside <details>.
    """
    regs_down = regs_down or []
    regs_up = regs_up or []
    spills_down = spills_down or []
    spills_up = spills_up or []
    if regressions:
        status = f"**Status: FAIL** ({len(regressions)} regression(s) > {threshold:.2f}%)"
    else:
        status = f"**Status: OK** (no regression beyond {threshold:.2f}%)"

    lines = []
    lines.append(status)
    lines.append("")

    if regressions:
        lines.append(f"### Regressions ({len(regressions)})")
        lines.extend(_perf_table(regressions))
        lines.append("")

    if improvements:
        lines.append(f"### Wins ({len(improvements)})")
        lines.extend(_perf_table(improvements))
        lines.append("")

    if new_shapes:
        lines.append(f"### New shapes ({len(new_shapes)})")
        rows = []
        for op, shape, dt, c in new_shapes:
            metric, value = ("TFLOPS", c["tflops"]) if c["tflops"] > 0 else ("TB/s", c["tbps"])
            rows.append([f"`{op}`", f"`{shape}`", dt, f"{value:.3f}", metric, f"`{c['tile']}`"])
        lines.extend(_md_table(["op", "shape", "dtype", "value", "metric", "tile"], rows))
        lines.append("")

    if missing_shapes:
        lines.append(f"### Removed shapes ({len(missing_shapes)})")
        rows = []
        for op, shape, dt, b in missing_shapes:
            metric, value = ("TFLOPS", b["tflops"]) if b["tflops"] > 0 else ("TB/s", b["tbps"])
            rows.append([f"`{op}`", f"`{shape}`", dt, f"{value:.3f}", metric])
        lines.extend(_md_table(["op", "shape", "dtype", "was", "metric"], rows))
        lines.append("")

    if regs_down:
        lines.append(f"### VGPR down ({len(regs_down)}) — compiler win")
        lines.extend(_resource_table(regs_down, "VGPR"))
        lines.append("")
    if regs_up:
        lines.append(f"### VGPR up ({len(regs_up)}) — compiler regression")
        lines.extend(_resource_table(regs_up, "VGPR"))
        lines.append("")
    if spills_down:
        lines.append(f"### Spills down ({len(spills_down)}) — compiler win")
        lines.extend(_resource_table(spills_down, "spill+sc/4"))
        lines.append("")
    if spills_up:
        lines.append(f"### Spills up ({len(spills_up)}) — compiler regression")
        lines.extend(_resource_table(spills_up, "spill+sc/4"))
        lines.append("")

    lines.append(f"<details><summary>Full compare table ({len(table_rows)} rows)</summary>")
    lines.append("")
    lines.append("```")
    lines.append(header_line)
    lines.append("-" * len(header_line))
    lines.extend(table_rows)
    lines.append("```")
    lines.append("")
    lines.append("</details>")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
