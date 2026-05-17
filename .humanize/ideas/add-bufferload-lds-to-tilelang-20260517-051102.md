# Land Buffer-Load-LDS Behind The Existing Swizzle-Swap Flag

## Original Idea

add bufferload_lds to tilelang
  在/root/tilelang2里面，在zty_opt_can_run_1120flops分支里面有完整的实现，然后我也拷贝了一份到/root/backuptilelang你可以参考.
  目前代码变化了很多，所以需要你重新看下怎么做,你的验证命令就只用跑bash scripts/iter_buffer_load.sh ，然后之前gen出来的实现在_fast.cpp里面

## Primary Direction: End-to-End Vertical Slice Gated by Env Flag

### Rationale

Build a single tracer-bullet path that takes ONE matmul shape (the bench script's 8192×8192×8192 NT, tile 256×256×64) all the way from `T.copy` to `buffer_load_dwordx4 ... lds`, gated by `TL_ENABLE_ROCM_SWIZZLE_SWAP=1` (the flag the bench script already exports). Other shapes/tile sizes intentionally fall back to the old emission path until the slice is green. This is distinct from porting strategies because it scopes the surface area to one byte-for-byte verifiable kernel (`_fast.cpp`) before generalizing, and is robust to disagreement about which layers of the stack are already in place on the current branch.

### Approach Summary

Treat `_fast.cpp` as a pinned acceptance template and `bash scripts/iter_buffer_load.sh` as the only validation harness. Drive a vertical slice through the stack for exactly the one shape the bench exercises:

1. **Audit, do not assume** — for each of the six known files (`src/tl_templates/hip/copy.h`, `src/op/builtin.{h,cc}`, `src/transform/inject_ptx_async_copy.cc`, `src/target/codegen_hip.cc`, `src/transform/lower_tile_op.cc`, `tilelang/transform/hoist_buffer_resource.py`), `git show feat-c-remove_vmcnt0:<path>` and diff against the reference branch. Resolve the conflicting reports from exploration: some agents found the device-side template + ops + injection logic already present on the current branch (uncommitted in the working tree), others found them missing from HEAD. Establish ground truth before touching anything.

2. **Wire only the gap** — once the audit lists the missing rungs, fill exactly those, in this priority order so each step is independently testable: (a) device template `cp_async_gs_lds_with_rsrc<N>` in `hip/copy.h`; (b) builtin ops `ptx_cp_async_lds`, `ptx_make_buffer_resource`, `ptx_cp_async_lds_rsrc` in `op/builtin.{h,cc}`; (c) HIP codegen handlers in `target/codegen_hip.cc` emitting the template calls; (d) `IsLdsContiguous` decision in `inject_ptx_async_copy.cc`; (e) Python `HoistBufferResource` pass + registration in `tilelang/transform/__init__.py` + insertion point in `tilelang/engine/phase.py`; (f) swizzle-swap in `lower_tile_op.cc`. Each step is gated behind an env-var check (`std::getenv("TL_ENABLE_ROCM_SWIZZLE_SWAP")`) at the highest layer that lets the bench toggle the entire feature on/off.

3. **Diff against `_fast.cpp` after every step** — recompile via `bash scripts/iter_buffer_load.sh` (which already wipes the JIT cache + rebuilds tilelang editable + emits the matmul `.cpp` via `TILELANG_HIP_SAVE_TEMP_FILES=1`). Compare the emitted `.hipi`/`.cpp` against `_fast.cpp` to confirm progress: the hoisted `make_wave_buffer_resource` + `readfirstlane` pair must appear in the prologue, and the inner-loop `tl::cp_async_gs_lds_with_rsrc<16>(...)` calls must replace the baseline `cp.async` emission.

4. **Acceptance** = the script reports `correctness: PASS` AND non-trivial TFLOPS (target ≈1120 per the reference branch commit message). VGPR count and spill count from the script's printout are secondary signals that the LDS path is engaged (buffer_load_lds bypasses VGPRs, so register pressure should drop visibly).

### Objective Evidence

- `/root/tile-kernel-bench-cdna4/scripts/iter_buffer_load.sh` lines 119-125: exports `TL_ENABLE_ROCM_SWIZZLE_SWAP=1` (enabled by default; `--no-swap` to disable). Target case at lines 26-34: 8192×8192×8192 NT, tile 256×256×64, 2 stages, 512 threads. Calls `gemm/example_gemm.py::matmul_nt` and validates via `torch.testing.assert_close(rtol=1e-2, atol=1e-2)`.
- `/root/tile-kernel-bench-cdna4/_fast.cpp` lines 10-13 establish the prologue acceptance template:
  ```
  auto __rsrc_A = make_wave_buffer_resource((const void*)(A));
  uint32_t __base_A = __builtin_amdgcn_readfirstlane((uint32_t)(uintptr_t)(A));
  ```
  Lines 27, 32, 38, 44 show the loop-body call shape `tl::cp_async_gs_lds_with_rsrc<16>(lds_addr, global_addr, __rsrc_A, __base_A)` with the swizzled global index baked into the second argument.
- Reference implementation on `/root/tilelang2` branch `zty_opt_can_run_1120flops`:
  - `src/tl_templates/hip/copy.h` lines 16-187: `as3_uint32_ptr`, `llvm_amdgcn_raw_buffer_load_lds`, `cp_async_gs_lds<N>`, `cp_async_gs_lds_with_rsrc<N>` device templates using `buffer_load_dwordx4 %1, %2, 0 offen lds` inline asm.
  - `src/op/builtin.h` lines 388-409: declarations for `ptx_cp_async_lds`, `ptx_make_buffer_resource`, `ptx_cp_async_lds_rsrc`.
  - `src/transform/inject_ptx_async_copy.cc` lines 52-96 (`IsLdsContiguous` sample-point linearity check) and lines 178-184 (decision: `is_rocm_ && !predicated && bytes == 16 && IsLdsContiguous(dst_offset)` → emit `ptx_cp_async_lds`).
  - `src/target/codegen_hip.cc` lines 770-809: handlers emitting `tl::cp_async_gs_lds_with_rsrc<N>` and `make_wave_buffer_resource`. Lines 1148-1169: `AttrStmt` handlers for `buffer_resource_var` / `buffer_base_var`.
  - `tilelang/transform/hoist_buffer_resource.py` (239 LOC): `_collect_buffer_vars`, `_rewrite_calls` (`ptx_cp_async_lds` → `ptx_cp_async_lds_rsrc`), `_fix_amd_wait_counts` (vmcnt scaling by `loads_per_group`).
  - `tilelang/transform/__init__.py` adds `from .hoist_buffer_resource import HoistBufferResource`; `tilelang/engine/phase.py` calls it after `InjectPTXAsyncCopy()` and before codegen.
- Conflicting evidence about current-branch state (must be resolved during audit step):
  - One explorer confirmed `git show feat-c-remove_vmcnt0:src/transform/inject_ptx_async_copy.cc` reports the file is on disk but **not in HEAD** of the branch — meaning the current working tree contains uncommitted infrastructure. `git status` shows extensive staged/modified files matching this pattern.
  - Another explorer reported the entire stack (templates, ops, injection, codegen, swizzle-swap) is **present** on disk in the current branch, with only the Python `HoistBufferResource` pass plumbing missing.
  - These reports are mutually consistent if the working tree has staged-but-uncommitted changes that introduce most of the stack; the audit step must distinguish committed-HEAD state from working-tree state before deciding which steps remain.
- Build/iterate infrastructure already in place: `iter_buffer_load.sh` wipes `/root/.tilelang/cache/0.1.9_*` and reinstalls tilelang editable (`USE_ROCM=ON pip install -e . --no-deps --no-build-isolation`); `TILELANG_HIP_SAVE_TEMP_FILES=1` produces inspectable `.s` / `.hipi` / `.cpp` artifacts (matches the `tmpuj5rx1ou-*` files already in the bench repo).

### Known Risks

- **Audit before edit, or repeat past mistakes.** The conflicting explorer reports mean blind code edits risk re-implementing existing logic or breaking partially-landed work in the dirty working tree.
- **Single-shape regression risk.** Gating the LDS path on the env flag protects against breaking other kernels, but only if the gate is at a high enough layer (injection decision, not codegen). Gating at codegen alone leaks `ptx_cp_async_lds_rsrc` calls into other targets.
- **Layout assumption.** Swizzle-swap in `lower_tile_op.cc` requires `layout_map_[buffer]->HasSwizzle()` and `layout_map_[buffer]->SwizzleDelta()`. If layout inference fails for the target shape, the optimization silently degrades to the VGPR path with no diagnostic.
- **AMD vmcnt semantics.** The hoisting pass scales `async_wait_inflight_count(N)` by `loads_per_group` because AMD vmcnt is per-load (not per-commit-group). If a parallel CUDA-side change to vmcnt handling has landed on the current branch, the scaling may double-fire and stall waves longer than intended.
- **Path refactoring landmines from cross-branch porting.** Some explorers reported `src/target/codegen_hip.cc` and `src/transform/inject_ptx_async_copy.cc` moved to `src/backend/rocm/codegen/...` or `3rdparty/tvm/...` on related branches. Verify the current branch's actual layout before quoting line numbers from the reference.
- **Bench-only validation.** `iter_buffer_load.sh` only tests one shape. The vertical slice does not exercise predicated copies, non-16-byte sizes, or non-contiguous LDS patterns; regressions outside the slice will not show up in iteration.

## Alternative Directions Considered

### Alt-1: Cherry-Pick + Conflict-Resolution Port
- Gist: Identify the minimal commit set on `zty_opt_can_run_1120flops` that introduces buffer_load_lds (5 core commits: `6feaabd5`, `3aa3f2d9`, `077bf1e9`, `0af0260c`, `af39a726`), cherry-pick them onto `feat-c-remove_vmcnt0`, and resolve conflicts hunk-by-hunk. Inherits author attribution and lets git do textual heavy lifting.
- Objective Evidence:
  - Merge-base `808b0cefcc3de6ffb114327be741fadb4534cbf7` (Feb 2025); current branch is 196 commits ahead, reference is 64 commits ahead.
  - `git log zty_opt_can_run_1120flops --oneline` shows snapshot-style "add"/"zz" commits — high textual entanglement.
  - 5 commits between `6feaabd5` and `af39a726` carry the feature; commits like `1545d49e..84baaf45` mix in unrelated `lower_tile_op.cc` changes.
- Why not primary: Snapshot-style commits make cherry-picking entangle the buffer_load_lds feature with the rest of the "1120 TFLOPS stack" (vmcnt0 removal, swizzle-swap refinements, register tuning). Conflict surface is large and not bounded by the iteration loop the user actually runs.

### Alt-2: Template-First Bottom-Up Port
- Gist: Walk the stack from the device-runtime layer (`src/tl_templates/hip/copy.h`) upward through codegen and IR transforms in dependency order. Smallest physical surface starts the chain.
- Objective Evidence:
  - One explorer reported all six layers — device template, ops, injection decision, codegen, swizzle-swap, hoisting pass — already present on the current branch, with line citations into `src/tl_templates/hip/copy.h:16-187`, `src/op/builtin.h:388-409`, `src/transform/inject_ptx_async_copy.cc:52-96, 178-184`, `src/target/codegen_hip.cc:770-809`, `src/transform/lower_tile_op.cc:905-960`, `tilelang/transform/hoist_buffer_resource.py`.
  - `_fast.cpp` in the bench repo demonstrates the device template `cp_async_gs_lds_with_rsrc<16>` was emitted at least once, supporting the "already substantially complete" claim.
- Why not primary: If this explorer is correct, the "port" is effectively done and the remaining work is verification only — the primary direction subsumes this by starting with an audit. If this explorer is wrong (other explorers report layers missing from HEAD), bottom-up still walks a path that's longer than the vertical slice needs.

### Alt-3: Python-Pass-First Top-Down Port
- Gist: Start with `tilelang/transform/hoist_buffer_resource.py` (the highest-level new artifact), wire its pipeline registration, and let downstream missing pieces surface as failures. Smallest delta if the rest of the stack is already present.
- Objective Evidence:
  - Reference branch's `tilelang/transform/__init__.py` adds exactly one line: `from .hoist_buffer_resource import HoistBufferResource`. `tilelang/engine/phase.py` adds two lines: `mod = tilelang.transform.HoistBufferResource()(mod)` + `print_pass(...)` in `OptimizeForTarget()`.
  - Pass is 239 LOC, self-contained, uses standard `prim_func_pass` decorator + `ir_transform`.
  - Sibling Python passes (`HoistBroadcastValues`, `DecoupleTypeCast`, `LegalizeNegativeIndex`) follow identical registration pattern.
- Why not primary: Only optimal if the entire C++ side (templates, ops, codegen, injection decision, swizzle-swap) is already committed on the current branch — which the conflicting explorer reports prevent us from asserting up-front.

### Alt-4: Filtered Patch Extraction From Reference Branch
- Gist: Generate `git diff 808b0cef zty_opt_can_run_1120flops -- <6 files>` (~816 LOC), split mechanically into "buffer_load_lds-relevant hunks" vs "1120 TFLOPS side-effects" (vmcnt0 removal, register tuning), apply only the former onto the current branch.
- Objective Evidence:
  - Composite 6-file patch size measured at 816 lines (124L copy.h + 67L builtin + ~116L codegen + 263L inject+lower + 239L hoist pass + miscellany).
  - Related branches (`feat-buffer-load-cdna4` at 438 LOC, `feat-b-buffer-optimize`) exist and represent prior partial extraction attempts.
  - `git apply --check` against the current branch is expected to fail because file paths may have moved (`src/target/codegen_hip.cc` → `src/backend/rocm/codegen/codegen_hip.cc` on some related branches).
- Why not primary: Hunk-level filtering is mechanical only if hunks are physically disjoint; in practice the vmcnt scaling logic and the hoisting pass share the same file. Path refactoring also breaks `git apply` cleanly.

### Alt-5: Re-Implement Against Current-Branch Abstractions
- Gist: Treat the reference implementation as a behavioral spec and re-design on top of the current branch's newer abstractions (`tileop/` directory, `kEnableAsyncCopy` / `kAsyncCopyNoImplicitCommitWait` annotation keys, refactored `inject_ptx_async_copy.cc` / `lower_ptx_async_copy.cc` / `ptx_async_copy_injector.h` split).
- Objective Evidence:
  - `src/op/builtin.h` (per `git diff` Phase 2) on the current branch already adds annotation keys absent from the reference: `kLoopPreferAsync`, `kParallelAsyncWithoutAsyncCommitWait`, `kAsyncCopyNoImplicitCommitWait`, `kPipelineMbarPhaseExpr`, `kEnableAsyncCopy`.
  - Current branch's `src/transform/` directory has `inject_ptx_async_copy.cc`, `lower_ptx_async_copy.cc`, `ptx_async_copy_injector.h` — a header/impl split absent on the reference, suggesting designed async-copy infrastructure.
  - `tileop/` directory hierarchy on current branch consolidates `gemm_mfma.py`, `gemm_mma.py`, etc. — a cleaner integration point than the reference's `rocm/op/`.
- Why not primary: Higher cognitive load and design risk than the vertical slice. Worth doing only if the audit reveals that the reference impl's hoisting model is fundamentally incompatible with the new async-copy infra.

## Synthesis Notes

The primary direction is intentionally conservative because the explorers disagree on the most load-bearing question — whether the current branch's HEAD already contains the buffer_load_lds stack or only the working tree does. The primary's first step (audit `git show feat-c-remove_vmcnt0:<path>` per file) is borrowed from Alt-4's diff-driven thinking and is the cheapest way to resolve that disagreement before any code is written. If the audit confirms Alt-2's claim that the C++ side is committed, the primary collapses to Alt-3 (Python plumbing only) — fold in Alt-3's specific edits to `tilelang/transform/__init__.py` and `tilelang/engine/phase.py`. If the audit confirms instead that HEAD is missing significant C++ infrastructure, fold in Alt-1's commit list as the canonical source of truth for what to port. Alt-5's re-implementation angle becomes load-bearing only if the audit also surfaces that the new async-copy infra (`lower_ptx_async_copy.cc`, the `kEnableAsyncCopy` annotation keys) conflicts with the reference's hoisting model — in which case the primary's "wire only the gap" step needs to be expanded to "redesign the gap on new abstractions". The env-flag gating (`TL_ENABLE_ROCM_SWIZZLE_SWAP`) is non-negotiable regardless of which alt folds in, because the bench script already exports it and it is the only available knob to keep other kernels on the baseline path during iteration.
