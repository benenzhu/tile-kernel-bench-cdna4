# Thread Binding Walkthrough — From `T.copy` / `T.gemm` to MFMA

A study guide. Pair this doc with `dump_ir.py` so you can open each pass
snapshot in `dump_ir/` and verify the claims yourself.

Target program: `gemm/example_gemm.py::matmul(1024, 1024, 1024, 128, 128, 32)`.
Each thread block has 128 threads, processes a `128 × 128` output tile,
copies `(128, 32)` A and `(32, 128)` B sub-tiles per K iteration. bf16 in,
fp32 accum.

Top-level question the doc answers: **how does `threadIdx.x` end up at a
specific address in `B_shared` (LDS write) and at a specific MFMA register
slot for `B_local` (MFMA read)?**

---

## Setup: dump every pass

```bash
python dump_ir.py
```

Produces `dump_ir/NNN_pass_name.py`. We will reference these by ID.
Important ones for thread binding:

| ID  | Pass                          | What you learn                                            |
| --- | ----------------------------- | --------------------------------------------------------- |
| 010 | PipelinePlanning              | Pipeline stages annotated, buffer shapes get a stage dim. |
| 011 | InjectSoftwarePipeline        | Prologue / steady / epilogue split for pipelining.        |
| 013 | LayoutInference               | **Layout decisions are sealed here.** Shared-mem layouts (swizzle) + Fragment layouts (warp tiles). |
| 020 | LowerTileOp                   | **Thread binding becomes concrete here.** `T.copy` → cp_async, `T.gemm` → per-warp MFMA loads + `tvm_mfma`. |
| 036 | FlattenBuffer                 | 4D `(stage, ?, ?, ?)` shared addressing collapses to 1D. |
| 039 | VectorizeLoop                 | Inner loops fold into `T.vectorized(8)` chunks.           |
| 080 | LowerTVMBuiltin               | What you'll see in the emitted HIP (cp_async / MFMA intrinsics). |

For the actual emitted HIP source: `kernel.get_kernel_source()`.

---

## Part 1 — Shared-memory copy: `T.copy(B[..., ...], B_shared)`

The chain to follow:

```
T.copy(Call)                                                            (010, 011)
    ↓ rocm::Copy::Lower in src/backend/rocm/op/copy.cc
T.gemm op leaves T.copy untouched; T.copy stays a Call through LayoutInference
    ↓ LowerTileOp (020) dispatches CopyOp.Lower
        - MakeSIMTLoop          → `for k(32), for n(128): B_shared[k,n] = B[...,...]`
        - ParallelLoopFuser     → `for kn(4096): ...`           (1D fused parallel For)
        - ParallelOp::InferLayout → loop_layout (Fragment)        ← thread binding decided here
        - LowerParallelLoop     → serial nest with threadIdx substituted
        - InjectPTXAsyncCopy    → ptx_cp_async OR ptx_cp_async_lds (DMA)
    ↓
T.ptx_cp_async(...)                                                     (020 output)
    ↓ codegen_hip.cc
tl::cp_async_gs<16>(dst_addr, src_addr)              (slow path)
   or tl::cp_async_gs_lds<16>(dst_addr, src_addr)    (DMA fast path)
```

### Step 1a — Pre-layout shape (open `dump_ir/013_tl.LayoutInference.py`)

Find `B_shared`. You'll see it allocated as `(3, 32, 128)` — the leading 3 is
the pipeline stage dim added by pass 010. The `(32, 128)` is `(block_K, block_N)`.

Verify: the gemm op's Python `InferLayout` returns
`{self.B: make_swizzled_layout(self.B), ...}`. That layout's `InputShape` is
`(3, 32, 128)`; its `OutputShape` is `(3, 2, 4, 512)` — `(stage, tc, ts, idx)`.

- `tc = 2`: continuous N (128 bf16 = 256 bytes) ÷ one LDS bank cycle (128 B) = 2 chunk-block planes.
- `ts = 4`: 32 rows ÷ 8 rows per FullBank ts-group = 4.
- `idx = 512`: per-(tc, ts) plane has 8 rows × 8 chunks × 8 vec = 512 elements.

### Step 1b — Loop fusion + binding (`src/op/copy.cc:333` `MakeSIMTLoop` + `src/op/parallel.cc` flow)

LowerTileOp (`020`) dispatches the T.copy node. Inside
`rocm::Copy::LowerCPAsync` (`src/backend/rocm/op/copy.cc:95`):

```cpp
auto simt_loop = op.MakeSIMTLoop(analyzer);              // 2D parallel For
auto fused_loop = ParallelLoopFuser::Fuse(simt_loop);    // 1D parallel For (extent=4096)
auto par_op = ParallelOp(fused_loop);
par_op->InferLayout(...);                                // fills loop_layout_
LowerParallelLoop(par_op->GetRoot(), par_op->GetLoopLayout(), thread_var, ...);
InjectPTXAsyncCopy(lowered, ..., enable_buffer_load_lds=true_on_gfx950);
```

The fused 1D loop has `loop_var ∈ [0, 4096) = block_K * block_N`.

Inside `ParallelOpNode::InferLayout` (`src/op/parallel.cc:255`), the path
taken for a plain global→shared copy:
- no source fragment (B_global is global, B_shared is non-fragment) →
  `source_buffer.defined() == false`
- falls into the `level == kFree` branch (line 407) →
  `candidate_from_plan = ComputePlanCandidate(T)`

`ComputePlanCandidate` (`src/op/parallel.cc:670`) computes:
```
vector_size = 8                       (bf16, 16-byte chunk)
num_thread  = 128
flat        = loop_var                (already 1D fused)
access_idx  = flat / 8                (8 elements per chunk)
thd         = access_idx % 128        ← THIS IS THE THREAD BINDING
idx         = (access_idx / 128) * 8 + flat % 8
```

So `thread t` owns the elements where `(flat / 8) % 128 == t`. Decompose
`flat = k*128 + n` (the original 2D):
- `flat / 8 = k*16 + n/8`
- `thd = (k*16 + n/8) % 128`
- For k=0: `thd = n/8 ∈ [0, 16)` ← **16 lanes span the N dim**. lane 0..7 in
  tc=0, lane 8..15 in tc=1.

This is the "default plan" binding. It's bank-conflict-free on the LDS READ
side (the swizzle handles that) but **straddles the tc boundary**, so the
LDS WRITE side is not lane-contiguous — see Step 1d.

### Step 1c — `LowerParallelLoop` substitutes threadIdx (`src/transform/loop_partition.cc:64`)

`PartitionLoop` rewrites the parallel For into a *serial* For whose body uses
`thread_var` directly. For our case:
- New serial loop: `for outer in 0..(4096 / (128 * 8)) = 0..4` (i.e., 4 chunks per thread)
- Inside body: `flat = outer * (128 * 8) + thread_var * 8 + vec` (vec becomes the vectorized inner)
- Reverse-engineer k, n from flat: `k = flat / 128`, `n = flat % 128`
- `B_shared[stage, k, n] = B[...]` becomes
  `B_shared[stage, (outer*1024 + thread_var*8 + vec)/128, ...]`

Open `dump_ir/020_tl.LowerTileOp.py`. Find the line storing into `B_shared`
inside an `unroll(4) × vectorized(8)` nest. You'll see:

```
B_shared[0,
         (thread_binding % 16 * 8 + (i * 8 + vec) % 8) // 64,   # tc dim
         ((i * 8 + vec) // 8 * 8 + thread_binding // 16) // 8,   # ts dim
         (8-chunk swizzle inner indexing)]
```

Map the variables: `i` is the unrolled outer count (0..3), `vec` is the
vectorized lane (0..7), `thread_binding` is `threadIdx.x`. Substitute
`thread_binding=t`, `vec=0`, `i=0`:
- `tc = ((t%16)*8 + 0) // 64`. For `t<8`: tc=0. For `8≤t<16`: tc=1. ← **straddles!**
- `ts = (0 + t//16) // 8` = 0 for t<128. (`ts` will grow via i.)
- The big inner expression is the FullBank swizzle: `((k%8) * 8 + (j/8)⊕(k%8)) * 8 + vec`.

### Step 1d — Why M9 / `buffer_load_dwordx4 ... lds` doesn't fire

`InjectPTXAsyncCopy` (`src/transform/lower_ptx_async_copy.cc:552`) checks
`IsLdsLaneContiguous(dst_index)` (line 442). The check substitutes the
thread var with 0, 1, 2, …, 1023 and confirms the dst byte address is
exactly `lane * stride + base` for some constant stride. **For the default
binding above:**

| lane (t) | tc | ts | s (in ts plane = t/8 mod 8) | byte_offset (elems × 2) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 16 |
| ... |  |  |  | (stride 16 ✓) |
| 7 | 0 | 0 | 0 | 112 |
| **8** | **1** | 0 | 1 | **144** ← gap = 32, not 16 |

`IsLdsLaneContiguous` returns false → emits `ptx_cp_async` (regular global
load + shared store), NOT `ptx_cp_async_lds` (DMA). In the HIP source you
see `tl::cp_async_gs<16>` instead of `tl::cp_async_gs_lds<16>`.

Cause: FullBank's `c_swizzle = c⊕s` permutes columns differently per row,
so any wave whose lanes span >1 row sees a non-affine stride.

### Step 1e — Codegen

Open the HIP source via `kernel.get_kernel_source()`. Search for
`cp_async_gs`. For every copy you'll see `tl::cp_async_gs<16>(...)` — the
**slow path**. None of `cp_async_gs_lds<16>` (the DMA variant). That's the
behavioral evidence for Step 1d.

If you flip the binding so a wave stays inside one row, the address would
be `lane * 16 + base` and you'd see `cp_async_gs_lds<16>` instead. That's
what M9 swizzle-swap was trying to engineer (see `lower_tile_op.cc:936`
for the multi-dim swizzle handling and the M9-safe downgrade comment).

### Recap: the binding pipeline for `T.copy`

```
T.copy Call
   ↓ (Step 1a) make_swizzled_layout decides FullBank with tc=2, ts=4, idx=512
   ↓ (Step 1b) ParallelOpNode::InferLayout via ComputePlanCandidate
                  picks thd = (flat / vec) % num_thread  ← default flatten policy
   ↓ (Step 1c) PartitionLoop substitutes threadIdx, emits serial+vectorized For
   ↓ (Step 1d) InjectPTXAsyncCopy: IsLdsLaneContiguous decides cp_async vs cp_async_lds
   ↓ (Step 1e) codegen_hip.cc: ptx_cp_async → tl::cp_async_gs<16>
                                ptx_cp_async_lds → tl::cp_async_gs_lds<16>  (DMA)
```

Open this in order: `dump_ir/013_*.py` → see swizzle layout; `dump_ir/020_*.py`
→ see thread substitution; then the HIP source for the final form.

---

## Part 2 — MFMA gemm: `T.gemm(A_shared, B_shared, C_local)`

The chain is fundamentally different — there is no `parallel For` to bind.
The MFMA emitter directly hand-writes the per-warp loop structure.

### Step 2a — Layout decisions (`tilelang/rocm/op/gemm/gemm_mfma.py:40`)

`GemmMFMA.infer_layout()` returns:
```python
self.A: make_swizzled_layout(self.A),                          # 2D HalfBank or QuarterBank
self.B: make_swizzled_layout(self.B),                          # 2D FullBank (tc=2)
self.C: mfma_emitter.make_mfma_store_layout(self.C)            # Fragment, per-lane register map
```

A_shared and B_shared get plain shared-mem swizzle layouts (no thread info).
C_local gets a **Fragment** which IS the thread-to-register binding. Open
`dump_ir/013_tl.LayoutInference.py` and find `C_local` in the
`layout_map = {...}` annotation — it's a Fragment, not a Layout.

`make_mfma_store_layout` for f32_16×16×32_bf16:
- Each MFMA produces a `16×16` fp32 tile owned by 64 lanes (one wavefront).
- Each lane holds `16*16/64 = 4` fp32 elements.
- A `128×128` block tile = `8 × 8 = 64` MFMA tiles, distributed over warps.
  With 128 threads / 64 lanes per wave = 2 wavefronts.

So C_local's Fragment encodes: `(C row, C col) → (warp_id, lane_id, reg_idx)`.

### Step 2b — Per-warp partition

The MFMA emitter (`tilelang/intrinsics/mfma_macro_generator.py` and friends)
takes the block tile `(128, 128)` and distributes MFMA tiles across warps
using `block_row_warps × block_col_warps` decided by `policy.compute_warp_partition`.

For 128 threads, the policy typically picks `m_warp=1, n_warp=2` or
`m_warp=2, n_warp=1`. The result determines `warp_row_tiles` and
`warp_col_tiles`. From the gen IR for our case (open `dump_ir/020_*.py` at
the `_gemm_ssr` block):

```
for i in range(4):                         ← warp_row_tiles / mfma_m = 4
    for local_id in T.vectorized(8):       ← per lane: load 8 bf16 = 16 B
        A_local[i*8 + local_id] = A_shared_3[ko%3, 0, addr_expr_for_A]
for j in range(8):                         ← warp_col_tiles / mfma_n = 8
    for local_id in T.vectorized(8):
        B_local[j*8 + local_id] = B_shared_3[ko%3, addr_expr_for_B]
for kp, i, j in T.grid(1, 4, 8):
    T.tvm_mfma("f32_16x16x32_bf16", "row", "row",
               "bfloat16x8", "bfloat16x8", "float32x4",
               B_local.data, j, A_local.data, i, C_local.data, i*8 + j)
```

This tells you:
- **Warp tile**: 4 MFMA tiles down (m direction), 8 across (n direction).
  Total per warp: 64 MFMA × `16×16` output → `64×128` per warp. With 2
  warps stacked along m: `128×128` ✓.
- **Per lane**: loads `4 * 8 = 32` bf16 for A, `8 * 8 = 64` bf16 for B,
  accumulates `4 * 8 * 4 = 128` fp32 in C_local. (Reads: `C_local` is
  128 elements per lane in this kernel.)

### Step 2c — Decoding the lane → LDS address for A (open `020_*.py` ~line 155)

The A load:
```
A_local[i*8 + local_id] = A_shared[
    ko%3,                                          # pipeline stage
    0,                                             # outer (A's tc=1, so degenerate)
    (warp_off + i*16 + thread_binding % 16) // 8,  # ts of HalfBank
    ((warp_off + i*16 + thread_binding % 16) % 8 * 32
     + ((thread_binding % 64 // 16 * 8 + local_id) // 16
        + (...) % 8 // 4) % 2 * 16
     + ...
     + (thread_binding % 64 // 16 * 8 + local_id) % 8)
]
where warp_off = thread_binding % 128 // 64 * 64   # m_warp lookup: top half vs bottom half
```

Decompose by lane index (`t = threadIdx.x`):
- `t // 64` = warp_id along M (0 or 1)
- `t % 64` = lane within wave (0..63)
- `t % 16` = the MFMA "row in tile" coordinate
- `t % 64 // 16` = the MFMA "k slot" coordinate (0..3, 4 k slots per MFMA)

Compared to the copy path, this is **directly written by the MFMA emitter**
in Python — no `ParallelOp::InferLayout`, no flatten/partition policy. The
emitter knows the `mfma_load_a_intrin` lane pattern (`v[k_slot, row_in_tile]`
→ register) and computes the LDS address that yields the correct mapping.

For B, the address has both `j*16` (the column tile) AND a row direction
shaped by `t % 16`, plus the FullBank `tc` selector. Same machinery, just
different intrinsic.

### Step 2d — `tvm_mfma` intrinsic

```
T.tvm_mfma("f32_16x16x32_bf16", "row", "row",
           "bfloat16x8", "bfloat16x8", "float32x4",
           B_local.data, j,
           A_local.data, i,
           C_local.data, i*8 + j)
```

This is the device intrinsic that becomes
`__builtin_amdgcn_mfma_f32_16x16x32_bf16` in HIP codegen. It tells the
hardware: "execute MFMA, A from `A_local[i*8 .. i*8+7]` (8 bf16 per lane),
B from `B_local[j*8 .. j*8+7]`, accumulate into `C_local[i*8+j*4 .. ]`."

Open `dump_ir/080_tir.LowerTVMBuiltin.py` to see how `T.tvm_mfma` is
already passed-through as a builtin Call.

### Recap: MFMA binding

```
T.gemm Call
   ↓ (Step 2a) GemmMFMA.infer_layout sets {A: swizzle, B: swizzle, C: Fragment(MFMA store)}
   ↓ (Step 2b) GemmMFMA.lower (Python) emits per-warp loops with hardcoded
                 LDS addresses for A_local, B_local + tvm_mfma calls
   ↓ (Step 2c) thread var pieces (t/64, t%16, t%64//16) directly index into
                 swizzled LDS — bypasses ParallelOp::InferLayout
   ↓ (Step 2d) tvm_mfma → builtin → MFMA instruction in HIP
```

The key contrast with Part 1: **`T.gemm` does its own thread binding in
Python**; it doesn't generate a `T.Parallel` loop that needs flattening.
The thread-binding "policy" you'd want to change for MFMA lives in
`tilelang/intrinsics/mfma_macro_generator.py`, not in `parallel.cc`.

---

## Suggested study path

1. **Run** `python dump_ir.py` once and keep `dump_ir/` open in your editor.
2. **Compare** `013_tl.LayoutInference.py` vs `020_tl.LowerTileOp.py`. The
   diff between these two is "policy decisions become substituted
   expressions". Look at how `T.copy(...)` (a Call) in 013 becomes the
   unrolled+vectorized loop nest in 020.
3. **Pick one address** in `020_*.py` (e.g., the B_shared store on line 82
   of `020_tl.LowerTileOp.py`). Substitute concrete `thread_binding = 0,
   1, 2, ...` by hand. Verify the lane → byte stride. This is exactly what
   `IsLdsLaneContiguous` does (Step 1d).
4. **Look at the FullBank math** in `/root/tilelang2/src/layout/gemm_layouts.cc:498`
   `MakeFullBankSwizzleLayout2D`. Now you understand where the
   `(c⊕s) * 16` terms in the IR come from.
5. **Read `src/op/parallel.cc:670` `ComputePlanCandidate`** and trace by hand
   for the fused 1D copy. Confirm `thd = (flat/8) % 128` matches what you
   see in the IR.
6. **For MFMA**, open `tilelang/rocm/op/gemm/gemm_mfma.py` and
   `tilelang/intrinsics/mfma_macro_generator.py`. The Python code
   constructs the per-warp loops directly — no IR-level partition pass to
   trace. Cross-reference with `_gemm_ssr` blocks in `020_*.py`.

---

## Quick reference: lane decomposition cheat-sheet

For `num_threads = 128`, gfx950 wavefront = 64 lanes:

| Expression in IR              | Meaning                                  |
| ----------------------------- | ---------------------------------------- |
| `thread_binding`              | `threadIdx.x` (0..127)                    |
| `thread_binding // 64`        | Warp index along M (0 or 1)              |
| `thread_binding % 64`         | Lane within wave (0..63)                 |
| `thread_binding % 16`         | "row in MFMA 16×16 tile" coordinate       |
| `thread_binding % 64 // 16`   | "k slot in MFMA tile" coordinate (0..3)   |
| `thread_binding % 8`          | "chunk-in-row" coordinate in CBA binding |
| `thread_binding // 8`         | "row" coordinate in CBA binding          |

For the default copy binding (Step 1b):
- `thread_binding // 16` = row in block_N=128, `thread_binding % 16` = chunk-in-row

For CBA-aware binding (the change we made in `parallel.cc:ComputeChunkBlockAwarePlanCandidate`):
- `thread_binding // 8` = row, `thread_binding % 8` = chunk-in-row (per tc plane)
