# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal

Land a working AMD gfx950 `buffer_load_dwordx4 ... lds` (direct global-to-LDS DMA, bypassing VGPRs) emission path in tilelang on the `feat-c-remove_vmcnt0` branch of `/root/tilelang2`, so that the benchmark in `/root/tile-kernel-bench-cdna4` reaches a hard floor of ≥1000 TFLOPS on the target shape, with each well-defined sub-step landed as its own commit directly on `feat-c-remove_vmcnt0` to keep revertibility cheap.

Reference implementation lives on `/root/tilelang2` branch `zty_opt_can_run_1120flops` (and at `/root/backuptilelang`). It must be studied, not copied verbatim. Design choices are flexible — the only invariants are: (a) the bench validates correctness AND ≥1000 TFLOPS, (b) the emitted kernel uses the `buffer_load ... lds` instruction family for the inner G→S copies on the target shape, and (c) each milestone is a single self-contained commit.

Operating principle (per user directive 2026-05-17): **treat the reference branch `zty_opt_can_run_1120flops` as a separate repository.** The two branches have diverged so heavily that one-to-one file mapping is not the point. The unit of port is the *function* (device template, builtin ops, codegen handler, injection decision, hoisting pass, swizzle-swap), not the file path. For each function, identify where on `feat-c-remove_vmcnt0`'s structure the equivalent logically belongs in the current architecture — that may or may not be the same file path as the reference uses.

Ground-truth observations (verified during plan generation against `feat-c-remove_vmcnt0`'s committed HEAD, useful as starting points):
- Present (vanilla, no buffer_load_lds tokens yet): `src/tl_templates/hip/copy.h`, `src/op/builtin.h`, `src/op/builtin.cc`, `src/transform/lower_tile_op.cc`. `make_wave_buffer_resource` already appears in one file (likely an earlier buffer-resource helper landed on this branch — reuse it, do not duplicate).
- Missing from HEAD at the reference's paths: `src/target/codegen_hip.cc`, `src/transform/inject_ptx_async_copy.cc`, `tilelang/transform/hoist_buffer_resource.py`. The first two were refactored elsewhere on this branch. Per the user directive above, do not waste effort trying to find an exact replacement file — instead, identify the HIP codegen entry point and the async-copy injection pass on this branch and graft the new behavior there in whatever shape the current structure prefers. The Python hoisting pass is a new file regardless.

### Acceptance Criteria
<!-- Each criterion must be independently verifiable -->
<!-- Claude must extract or define these in Round 0 -->


- AC-1: `bash scripts/iter_buffer_load.sh` from `/root/tile-kernel-bench-cdna4`, with `/root/tilelang2` checked out at `feat-c-remove_vmcnt0` HEAD after the port, produces correctness PASS AND reported TFLOPS ≥ 1000 on the default shape (8192×8192×8192 NT, tile 256×256×64, stages=2, threads=512).
  - Positive Tests (expected to PASS):
    - Running the bench prints `[iter] correctness: PASS` (per the `torch.testing.assert_close(rtol=1e-2, atol=1e-2)` check inside the script).
    - The printed `[iter] TFLOPS:` line shows a numeric value ≥ 1000.0.
  - Negative Tests (expected to FAIL):
    - Reverting any of the milestone commits and rerunning the bench produces either correctness FAIL or TFLOPS strictly below the pre-port baseline measured at the start of the port.
    - Running the bench against the pre-port HEAD of `feat-c-remove_vmcnt0` (no port commits applied) reports TFLOPS strictly below 1000.
  - AC-1.1: The TFLOPS gate is a HARD floor; values below 1000.0 fail the criterion regardless of how close they get.
    - Positive: 1000.0, 1080.5, 1124.7 → PASS.
    - Negative: 999.9, 850.0 → FAIL.

- AC-2: The emitted HIP source for the bench kernel uses the `buffer_load ... lds` instruction family for the inner G→S copies, with the buffer resource descriptor hoisted out of the inner loop (one `make_wave_buffer_resource` per global tensor at kernel entry; one `readfirstlane`-based base address per tensor at kernel entry).
  - Positive Tests (expected to PASS):
    - Re-running the bench with `TILELANG_HIP_SAVE_TEMP_FILES=1` (already set by the script) produces a `tmp*-gfx950.s` and/or generated `.cpp` whose inner loop calls `tl::cp_async_gs_lds_with_rsrc<16>(...)` (or an equivalent symbol that lowers to `buffer_load_dwordx4 ... offen lds`).
    - The generated `.cpp` shows the `make_wave_buffer_resource((const void*)(A))` and `__builtin_amdgcn_readfirstlane(...)` initializations OUTSIDE the K loop (kernel prologue), matching the shape demonstrated by `/root/tile-kernel-bench-cdna4/_fast.cpp` lines 10-13.
    - Disassembly of the resulting `.s` contains at least one occurrence of `buffer_load_dwordx4` with the `lds` modifier in the inner loop body.
  - Negative Tests (expected to FAIL):
    - A generated `.cpp` where the inner-loop copies still go through register-staging intrinsics (e.g. `tl::cp_async_*` variants that do not name `gs_lds`) for the target shape.
    - A generated `.cpp` where `make_wave_buffer_resource` is called INSIDE the K loop on every iteration (resource not hoisted).

- AC-3: Every milestone in `## Dependencies and Sequence` lands as exactly one git commit on the `feat-c-remove_vmcnt0` branch of `/root/tilelang2`, with a commit message that names the milestone, and the bench is runnable (at least: tilelang rebuilds without error) after each commit.
  - Positive Tests (expected to PASS):
    - `git log feat-c-remove_vmcnt0 --oneline` after completion shows one commit per milestone, in milestone order.
    - For each milestone commit `C`, checking out `C` and running `pip install -e . --no-deps --no-build-isolation` in `/root/tilelang2` exits 0.

---

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan + dual-repo caveat noted in Round 0 contract | RLCR loop runs in bench repo `debug_vm_cnt0`, but implementation commits land in `/root/tilelang2` `feat-c-remove_vmcnt0`. Codex review can verify summaries but not directly diff tilelang2 commits. | None — AC-3 still requires per-milestone commits in tilelang2, just acknowledges they are outside the bench repo. |
| 0 | M1 reframed and swizzle-swap promoted (already reflected in plan file) | User directive during pre-loop quiz: treat reference as separate repo; land swizzle-swap unconditionally. | None — plan was edited before Round 0 init. |

#### Active Tasks
<!-- Mainline tasks only: each task must directly advance the current round objective and carry routing metadata -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| task-M1 | AC-3 | completed (pending verification) | analyze | claude | Function→site mapping captured in round-0-contract.md. No code commits per plan. |
| task-M2 | AC-2, AC-3, AC-4 | pending | coding | claude | Round 1 deliverable. Extend `src/tl_templates/hip/copy.h` with `cp_async_gs_lds_with_rsrc<N>`; reuse existing `make_wave_buffer_resource`. Commit in `/root/tilelang2` on `feat-c-remove_vmcnt0`. |
| task-M3 | AC-2, AC-3, AC-4 | pending | coding | claude | Round 2 deliverable. Add `ptx_cp_async_lds_rsrc` (+helpers) to `src/op/builtin.{h,cc}`. |
| task-M4 | AC-2, AC-3, AC-4 | pending | coding | claude | Round 3 deliverable. Add codegen handler in `src/backend/rocm/codegen/codegen_hip.cc`. |
| task-M5 | AC-2, AC-3, AC-4 | pending | coding | claude | Round 4 deliverable. Insertion site: `src/backend/rocm/op/copy.cc` (primary) or `src/transform/lower_ptx_async_copy.cc` (fallback). |
| task-M6 | AC-2, AC-3, AC-4 | pending | coding | claude | Round 5 deliverable. New file `tilelang/transform/hoist_buffer_resource.py`; descriptor-hoist only. |
| task-M6.5 | AC-1, AC-3 | pending (conditional) | coding | claude | Triggered only if M6 bench shows wait-count correctness failure. |
| task-M7 | AC-1, AC-2, AC-3, AC-4 | pending | coding | claude | Round 6 deliverable. Swizzle-swap in `src/transform/lower_tile_op.cc`. In-scope by default. |
| task-M8 | AC-1, AC-2, AC-3, AC-4 | pending | coding | claude | Round 7+ deliverable. Perf gate run + iteration. |

### Blocking Side Issues
<!-- Only issues that directly block current mainline progress belong here -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|
| (none) | - | - | - |

### Queued Side Issues
<!-- Non-blocking issues stay queued and must NOT replace the round objective -->
| Issue | Discovered Round | Why Not Blocking | Revisit Trigger |
|-------|-----------------|------------------|-----------------|
| /root/tilelang2 currently checked out at `zty_opt_can_run_1120flops` (reference). Branch switch to `feat-c-remove_vmcnt0` needed before any M2+ commit. | 0 | Round 0 is read-only across both branches; switch happens at start of Round 1. | Start of Round 1. |
| The pipeline insertion point for `HoistBufferResource` on this branch not yet located (likely `tilelang/engine/phase.py` or chained in `__init__.py`). | 0 | Not needed until M6 (Round 5). | Start of Round 5 (M6). |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|
| AC-3 setup | task-M1 (audit) | 0 | (pending Codex review) | round-0-contract.md mapping table; verified via `git ls-tree` + `git grep` against `feat-c-remove_vmcnt0` |

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|
| (none) | - | - | - | - |

