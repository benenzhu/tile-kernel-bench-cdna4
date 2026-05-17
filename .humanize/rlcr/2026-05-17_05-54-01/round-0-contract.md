# Round 0 Contract

## Objective
Initialize the RLCR loop infrastructure and complete M1 (function-to-site mapping audit, no code commits per plan). This round does NOT advance any of M2-M8; those land in subsequent rounds.

## Dual-Repo Caveat (READ THIS FIRST)
The RLCR loop is anchored in the bench repo `/root/tile-kernel-bench-cdna4` on branch `debug_vm_cnt0`. But per the plan's AC-3, the actual implementation commits land in `/root/tilelang2` on branch `feat-c-remove_vmcnt0`. This is an unavoidable mismatch — the bench repo only contains the test harness; the compiler being modified is a sibling checkout.

Consequences for the loop:
- Per-round commits in the bench repo will only touch RLCR metadata (`goal-tracker.md`, `round-N-contract.md`, `round-N-summary.md`, `bitlesson.md`, the plan file). No production code changes here.
- Per-round commits in `/root/tilelang2` on `feat-c-remove_vmcnt0` carry the actual milestone work; those are out of scope for Codex's `codex review --base <branch>` on this repo.
- Each round summary in the bench repo will name the corresponding `/root/tilelang2` commit hash so the work is traceable.
- Codex review during the loop can verify summaries and metadata coherence, but cannot directly diff the tilelang2 commits. Verification of code correctness comes from `bash scripts/iter_buffer_load.sh` runs reported in the summaries.

## Deliverables for Round 0
1. Populated `goal-tracker.md` Mutable section: Active Tasks rows for task-M1..task-M8 with status (M1=in_progress→completed in this round; M2-M8=pending).
2. M1 function-to-site mapping note (this contract carries it; copied into the summary).
3. `bitlesson.md` initialized (script created the template; no entries yet — Action: none for Round 0).
4. `round-0-summary.md` filled in with the M1 mapping and the "M2 starts in Round 1" handoff.
5. One commit in `/root/tile-kernel-bench-cdna4` capturing all the above metadata. No `/root/tilelang2` commits this round.

## M1 Function→Site Mapping (feat-c-remove_vmcnt0)

| Functional Piece | Reference Path (zty_opt) | Target Insertion Site (feat-c-remove_vmcnt0) | Notes |
|---|---|---|---|
| Device template `cp_async_gs_lds_with_rsrc<N>` | `src/tl_templates/hip/copy.h:165-187` | `src/tl_templates/hip/copy.h` (extend existing file) | `make_wave_buffer_resource` already defined at line 22 — reuse, do not duplicate |
| TIR builtin ops `ptx_cp_async_lds`, `ptx_make_buffer_resource`, `ptx_cp_async_lds_rsrc` | `src/op/builtin.{h,cc}` | `src/op/builtin.{h,cc}` (same paths) | Add declarations + registrations; mirror the style of neighboring ops on this branch |
| HIP codegen handlers (`VisitExpr_`/`EmitCall_`) | `src/target/codegen_hip.cc` | `src/backend/rocm/codegen/codegen_hip.cc` | **PATH CHANGED** — refactored to backend-specific directory |
| Async-copy injection decision | `src/transform/inject_ptx_async_copy.cc` | Primary: `src/backend/rocm/op/copy.cc` (already invokes `InjectPTXAsyncCopy` at ~line 131; ROCm-scoped). Fallback: `src/transform/lower_ptx_async_copy.cc` (the renamed 745-LOC pass) | **RENAMED + SPLIT**. The ROCm-side `op/copy.cc` is the cleanest insertion point — it is the ROCm-specific entry to the injector and already carries the right scope |
| HoistBufferResource Python pass | `tilelang/transform/hoist_buffer_resource.py` | `tilelang/transform/hoist_buffer_resource.py` (new file) | Register in `tilelang/transform/__init__.py` (sibling to existing `HoistBroadcastValues`, `DecoupleTypeCast`); insertion point in pipeline will be located when M6 starts (likely `tilelang/engine/phase.py` or wherever `__init__.py` chains passes) |
| Swizzle-swap optimization | `src/transform/lower_tile_op.cc` lines 905-960 | `src/transform/lower_tile_op.cc` (same path) | Existing file; add `HasSwizzle()`-conditional swap that moves XOR from LDS store indices to global load indices |

Additional verified facts:
- /root/tilelang2 current HEAD is `zty_opt_can_run_1120flops` (the reference). Branch switch to `feat-c-remove_vmcnt0` happens at the start of Round 1 (before any code commit). Round 0 stays read-only across both branches.
- `tilelang/transform/__init__.py` on `feat-c-remove_vmcnt0` already imports `HoistBroadcastValues` and `DecoupleTypeCast` (lines ~9-10) — registration pattern for `HoistBufferResource` is identical.
- `src/backend/rocm/op/copy.cc:131` shows `InjectPTXAsyncCopy(lowered_loop, /*enable_auto_async_copy=*/true, ...)` as the existing ROCm-side entry — the buffer_load_lds routing decision can be made before/after this call.

## Out of Scope for Round 0
- Any code changes in `/root/tilelang2` (M2-M8 work).
- Switching `/root/tilelang2` to `feat-c-remove_vmcnt0` (deferred to Round 1).
- Running the bench (no code changes yet — would only re-measure the reference).
- BitLesson additions (nothing surprising learned yet).

## Exit Criteria for Round 0
- M1 mapping table captured in this contract AND in `round-0-summary.md`.
- `goal-tracker.md` Mutable section reflects task-M1 = completed, task-M2..M8 = pending.
- One commit in the bench repo with all Round 0 metadata.
- Summary names this commit hash.
