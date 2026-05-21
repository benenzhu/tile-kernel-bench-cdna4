# Chunk-Block-Aware Thread Binding for B's T.copy — Work Log

Goal: make B's g2s into LDS land 64 lanes in one 1024B contiguous segment so
buffer_load_dwordx4 ... lds becomes legal. Constraint: keep XOR swizzle for
bank-conflict-free reads.

Repo: /root/tilelang2 (branch feat-c-remove_vmcnt0).
Bench: /root/tile-kernel-bench-cdna4 (scripts/iter_buffer_load.sh).

## Status snapshot

- baseline now.cu emitted for B (`example_gemm.py` 1024 NN): 16 lanes × 8 K-rows,
  crosses tc boundary at lane 8 → +1920B jump.
- `ComputeChunkBlockAwarePlanCandidate` exists in src/op/parallel.cc:720 but is
  *not* taking effect for B. User says "previous implementation, not quite right".

## Investigation plan

1. Verify whether existing CBA actually runs (it's only reached from
   `ComputePlanCandidate`, which is only entered from the `kFree` branch when
   `source_buffer` is not defined).
2. For T.copy(B_global, B_shared), source_buffer is B_shared (write buffer
   with a layout). So the code goes through `ComputeLoopLayoutFromBuffer`
   (line 608), NOT `ComputePlanCandidate`. CBA never runs.
3. New approach: intercept the binding *inside or before* the buffer-path,
   reshape it to be chunk-block aware.

## Iteration log

### Round 1 result (committed)

Diagnosis: existing CBA in `ComputePlanCandidate` is reached only when
`source_buffer` is undefined. For B's T.copy at kCommon/kStrict, the
source_buffer chain takes precedence (when B_shared has a layout in
T.layout_map), so CBA never runs; the default PlanLoopPartition flatten
produces the 16-lane × 8-K-row binding that crosses tc.

Fix: add an early hook in `ParallelOp::InferLayout` that calls CBA
*before* source_buffer dispatch (and at all 3 levels so the first level
where the layout map is populated wins). The hook re-derives `vec_size`
inline using the same logic as `ComputePlanCandidate`.

Validation:
- example_gemm.py NN 1024^3 + K=16384: correctness PASS
- iter_buffer_load.sh NT 8192² K=8192 256x256x64 stages=2 threads=512:
  correctness PASS, **1111.38 TFLOPS** (above 1000 floor; NT B last-dim
  K=64 = 1 bank cycle so CBA gate naturally skips, perf unchanged)
- NN 8192³ 128x128x32 stages=3 threads=128: correctness PASS, 370 TFLOPS
  (was wrong values before; ptx_cp_async path, no buffer_load_lds yet)

Forward fragment dump:
  Fragment([32, 128] -> [32], thread: 128,
    forward_thread = _i % 16 * 8 + _j % 64 // 8,
    forward_index = [_j // 64 * 16 + _i // 16 * 8 + _j % 8])
  → 8 lanes/row × 8 K-rows per warp ✓

### Round 2 result (committed)

Diagnosis path: lower_ptx_async_copy.cc's `IsLdsLaneContiguous` check
rejects the post-Forward LDS index (XOR still in last dim). It then
emits `ptx_cp_async` (not LDS), so M9 in LowerTileOp never gets a
chance to apply the SwizzleDelta swap.

Fix: relax the IsLdsLaneContiguous gate. Always emit `ptx_cp_async_lds`
when shared + 16B + non-predicated. M9 then sees the LDS-variant call
and either (a) rewrites it via swap (LDS path retained), or (b)
downgrades to `ptx_cp_async` (same args, just different opcode) when
the post-swap index is still non-affine.

For B (FullBank, tc=2 in NN K_inner=128 case): swap math works because
- last-dim post-swap = `(K%16)*64 + (N%64)/8*16` simplifies to `t*8` ✓
- outer dims (stage, tc, ts) are constants per-warp under CBA binding
- result: LDS dst = `i*1024 + tx*8` (lane-contiguous), global src
  carries the XOR per-lane.

For A (HalfBank, ts boundary in M dim): M9 downgrades because the
post-swap dim 2 (`(M_loc + tx/4)/8`) is a step-function in tx — A's
binding spans 16 M-rows per warp but `ts = M/8` switches at M=8 so
warp lanes straddle two `ts` indices. Fixing A needs CBA-style binding
on the stride dim (open).

Validation:
- NT 8192² K=8192 256x256x64 stages=2 threads=512: **1116 TFLOPS,
  correctness PASS** (above 1000 floor)
- NN 8192³ 128x128x32 stages=3 threads=128: **443 TFLOPS** (was 370 with
  only Round-1 CBA, was wrong values pre-M10), PASS
- NN 8192³ 256x256x32 stages=3 threads=256: 602 TFLOPS, PASS
- NN 8192³ 256x256x64 stages=2 threads=256: 760 TFLOPS, PASS
- NN 4096³ 128x128x32 stages=3 threads=128: 405 TFLOPS, PASS

Files touched:
- `src/op/parallel.cc` + `parallel.h`: early CBA hook (Round 1)
- `src/transform/lower_ptx_async_copy.cc`: always-emit-LDS, let M9 decide

### Final NT verification

```
[iter] running NT 8192x8192x8192 tile 256x256x64 stages=2 threads=512
[iter] correctness: PASS
[iter] latency: 0.9826 ms
[iter] TFLOPS:   1119.02
```

Hard floor (≥1000 TFLOPS) maintained.

### Round 3 ideas (not yet implemented)

1. Fix A's binding so warp stays within one ts plane (CBA on stride
   dim). Hard: HalfBank `index` extent is 128 elements but A's
   continuous extent is 32, so the "ts gap" is structural.
2. Reduce HalfBank's index padding (32 vs 128 in A's case).
3. Try different vec sizes for A so the warp footprint matches a single
   `ts*index_extent`.

The biggest remaining gap (NN ~760 vs NT ~1116) is most likely the
combination of A still going through software cp_async + the smaller
amount of MFMA per cp_async at this NN tile.


With CBA binding, each warp's 64 lanes land in 8 separate 128B LDS rows.
Lane t lands at `(t>>3)*128B + ((t&7) XOR (t>>3))*16B`. AMD
`buffer_load_dwordx4 ... lds` requires `lane k @ base + k*16B` linear, so
the XOR shuffle must be pushed to the global-source side via SwizzleDelta.

M9 (lower_tile_op.cc:996-1006) currently only handles single-dim swap
(`swizzle_delta = const * thread_var`). The new binding's delta is
`((t&7) XOR (t>>3) - (t&7)) * 16B`, a 2-bit XOR shuffle across thread
bit segments. Need to extend M9 to recognize this pattern and emit the
right per-lane source offset.


Hypothesis: CBA helper at parallel.cc:720 is in the source but doesn't fire
for B because either (a) `source_buffer` (B_shared) wins the dispatch and
calls `ComputeLoopLayoutFromBuffer` instead of `ComputePlanCandidate`, or (b)
CBA fires but its gate returns empty.

Action: add a one-shot fprintf in:
  - ParallelOp::InferLayout when source_buffer.defined() branch is taken
  - ParallelOp::ComputeChunkBlockAwarePlanCandidate entry
  - CBA gate failure points

Then rebuild + run dump_ir.py once. Inspect stderr.

