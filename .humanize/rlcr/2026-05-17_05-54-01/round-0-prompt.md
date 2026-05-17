Read and execute below with ultrathink

## Goal Tracker Setup (REQUIRED FIRST STEP)

Before starting implementation, you MUST initialize the Goal Tracker:

1. Read @/root/tile-kernel-bench-cdna4/.humanize/rlcr/2026-05-17_05-54-01/goal-tracker.md
2. If the "Ultimate Goal" section says "[To be extracted...]", extract a clear goal statement from the plan
3. If the "Acceptance Criteria" section says "[To be defined...]", define 3-7 specific, testable criteria
4. Populate the "Active Tasks" table with MAINLINE tasks from the plan, mapping each to an AC and filling Tag/Owner
5. Record any already-known side issues in either "Blocking Side Issues" or "Queued Side Issues"
6. Write the updated goal-tracker.md

## Round Contract Setup (REQUIRED BEFORE CODING)

Before starting implementation, create @/root/tile-kernel-bench-cdna4/.humanize/rlcr/2026-05-17_05-54-01/round-0-contract.md with:

1. **One mainline objective** for this round
2. **Target ACs** (1-2 ACs only)
3. **Blocking side issues in scope** for this round
4. **Queued side issues out of scope** for this round
5. **Round success criteria**

Use this contract to keep the round focused. Do NOT let non-blocking bugs or cleanup work replace the mainline objective.

**IMPORTANT**: The IMMUTABLE SECTION can only be modified in Round 0. After this round, it becomes read-only.

---

## Implementation Plan

For all tasks that need to be completed, please use the Task system (TaskCreate, TaskUpdate, TaskList).

Every task MUST start with exactly one lane tag:
- `[mainline]` for plan-derived work that directly advances the round objective
- `[blocking]` for issues that prevent the mainline objective from succeeding safely
- `[queued]` for non-blocking bugs, cleanup, or follow-up work

Rules:
- `[mainline]` tasks are the primary success condition for the round
- `[blocking]` tasks may be resolved in the round only if they truly block mainline progress
- `[queued]` tasks must NOT become the round objective and do NOT need to be cleared before moving on
- If a new issue is not blocking the current objective, tag it `[queued]` and keep moving on the mainline

## Task Tag Routing (MUST FOLLOW)

Each task must have one routing tag from the plan: `coding` or `analyze`.

- Tag `coding`: Claude executes the task directly.
- Tag `analyze`: Claude must execute via `/humanize:ask-codex`, then integrate Codex output.
- Keep Goal Tracker "Active Tasks" columns **Tag** and **Owner** aligned with execution (`coding -> claude`, `analyze -> codex`).
- If a task has no explicit tag, default to `coding` (Claude executes directly).

# Port Buffer-Load-LDS Onto feat-c-remove_vmcnt0, Slimmed And Milestone-Committed

## Goal Description

Land a working AMD gfx950 `buffer_load_dwordx4 ... lds` (direct global-to-LDS DMA, bypassing VGPRs) emission path in tilelang on the `feat-c-remove_vmcnt0` branch of `/root/tilelang2`, so that the benchmark in `/root/tile-kernel-bench-cdna4` reaches a hard floor of ≥1000 TFLOPS on the target shape, with each well-defined sub-step landed as its own commit directly on `feat-c-remove_vmcnt0` to keep revertibility cheap.

Reference implementation lives on `/root/tilelang2` branch `zty_opt_can_run_1120flops` (and at `/root/backuptilelang`). It must be studied, not copied verbatim. Design choices are flexible — the only invariants are: (a) the bench validates correctness AND ≥1000 TFLOPS, (b) the emitted kernel uses the `buffer_load ... lds` instruction family for the inner G→S copies on the target shape, and (c) each milestone is a single self-contained commit.

Operating principle (per user directive 2026-05-17): **treat the reference branch `zty_opt_can_run_1120flops` as a separate repository.** The two branches have diverged so heavily that one-to-one file mapping is not the point. The unit of port is the *function* (device template, builtin ops, codegen handler, injection decision, hoisting pass, swizzle-swap), not the file path. For each function, identify where on `feat-c-remove_vmcnt0`'s structure the equivalent logically belongs in the current architecture — that may or may not be the same file path as the reference uses.

Ground-truth observations (verified during plan generation against `feat-c-remove_vmcnt0`'s committed HEAD, useful as starting points):
- Present (vanilla, no buffer_load_lds tokens yet): `src/tl_templates/hip/copy.h`, `src/op/builtin.h`, `src/op/builtin.cc`, `src/transform/lower_tile_op.cc`. `make_wave_buffer_resource` already appears in one file (likely an earlier buffer-resource helper landed on this branch — reuse it, do not duplicate).
- Missing from HEAD at the reference's paths: `src/target/codegen_hip.cc`, `src/transform/inject_ptx_async_copy.cc`, `tilelang/transform/hoist_buffer_resource.py`. The first two were refactored elsewhere on this branch. Per the user directive above, do not waste effort trying to find an exact replacement file — instead, identify the HIP codegen entry point and the async-copy injection pass on this branch and graft the new behavior there in whatever shape the current structure prefers. The Python hoisting pass is a new file regardless.

## Acceptance Criteria

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
    - Reverting the last commit (`git revert HEAD`) without aborting leaves the tree in a state where the bench's pre-revert outputs still build (the revert chain is clean).
  - Negative Tests (expected to FAIL):
    - A single commit bundling two or more milestones together.
    - A commit message that does not name its milestone.
    - A milestone whose commit fails `pip install -e .` (broken intermediate build).

- AC-4: No non-bench regression on the default smoke checks the user already runs in this branch — at minimum, the bench's correctness gate (rtol=1e-2, atol=1e-2) must continue to pass on the target shape AND `import tilelang` must still succeed in a fresh Python invocation after each milestone.
  - Positive Tests (expected to PASS):
    - `python -c "import tilelang"` exits 0 after each milestone commit.
    - The bench's correctness assertion does not regress between any consecutive milestone commits that touch emission (correctness PASS held the prior commit ⇒ still PASS the next).
  - Negative Tests (expected to FAIL):
    - `python -c "import tilelang"` raises ImportError after any milestone commit.
    - The bench reports correctness FAIL at the final milestone even if TFLOPS is high (correctness is a strict precondition for AC-1).

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
A complete, slimmed port of the buffer_load_lds emission stack onto `feat-c-remove_vmcnt0`: device-side template (`cp_async_gs_lds_with_rsrc<N>`), TIR builtin ops (`ptx_cp_async_lds`, `ptx_make_buffer_resource`, `ptx_cp_async_lds_rsrc` or equivalent names of the implementer's choosing), HIP codegen handlers, async-copy injection decision (with a lane-contiguity check), Python hoisting pass that lifts resource descriptors to the kernel prologue, and the swizzle-swap optimization in `lower_tile_op.cc` (which moves XOR swizzle from LDS store to global load for layouts with `HasSwizzle()`). Swizzle-swap is part of the planned scope — it lands unconditionally as its own milestone, not as a fallback gated on bench TFLOPS. The AMD vmcnt wait-count fixup in the hoisting pass remains conditional on whether bench correctness requires it. Each layer lands as its own milestone commit on `feat-c-remove_vmcnt0`. Other kernels (non-NT shapes, predicated copies, non-16-byte tile sizes) may continue to emit through the existing baseline path.

### Lower Bound (Minimum Acceptable Scope)
The minimum machinery needed for the bench's target shape to emit at least one `buffer_load_dwordx4 ... offen lds` call per inner-loop tile fetch with a hoisted resource descriptor, AND with swizzle-swap engaged on the global-load side (per Upper Bound; swizzle-swap is in scope by default), hit correctness PASS, and report TFLOPS ≥ 1000. If the implementer can hit the gate by, for example, bypassing the IR-level injection pass and emitting the call directly from a tighter codegen pattern matcher, that is acceptable. The AMD vmcnt fixup remains the only conditional layer — omit it if `feat-c-remove_vmcnt0`'s own async-wait infra already produces correct waits.

### Allowed Choices
- Can use: any tilelang-internal abstraction available on `feat-c-remove_vmcnt0` (`tileop/` directory, new annotation keys like `kEnableAsyncCopy`, the split `inject_ptx_async_copy.cc` / `lower_ptx_async_copy.cc` / `ptx_async_copy_injector.h` infra); inline asm in `tl_templates/hip/copy.h`; new TIR builtins under `tl::` namespace; new Python passes registered in `tilelang/transform/__init__.py`; reusing or extending the existing `make_wave_buffer_resource` helper that is already present on this branch.
- Can use: study the reference branch's diff, read its files, lift design ideas, copy small self-contained snippets (e.g. the inline-asm body for `buffer_load_dwordx4 ... lds`, the `IsLdsContiguous` analysis function) where the same logic is reusable.
- Cannot use: `git cherry-pick` of any commit from `zty_opt_can_run_1120flops` (or any related branch). The user's directive is explicit: study the reference, then write a slimmed port. Cherry-pick is not permitted because it both inherits unwanted commits and obscures what is and is not needed.
- Cannot use: env-var gating like `TL_ENABLE_ROCM_SWIZZLE_SWAP` (the user explicitly said to ignore this; the reference branch does not consume it either, and the bench script exports it as no-op).
- Cannot use: feature branches off `feat-c-remove_vmcnt0`. Per the user, commits go DIRECTLY on `feat-c-remove_vmcnt0`.
- Cannot use: time estimates, line ranges, or git-history-rewriting operations (rebase squash, force push, amend of already-shared commits) without an explicit user request.

> **Note on Deterministic Designs**: The user's flexibility directive (`实现方案都是可变的`) plus the hard perf gate make this a perf-deterministic plan but design-open. The Allowed Choices section is correspondingly broad on HOW and tight on the two forbidden tools (cherry-pick, feature branches).

## Feasibility Hints and Suggestions

### Conceptual Approach

The reference branch tells us the *shape* of a working solution; treat it as a separate repo whose paths are incidental. The target branch tells us where things logically belong now. The slimmed port walks the layers bottom-up, committing per layer, validating per layer where possible:

1. **Map functions to insertion sites, not paths to paths.** Without writing any code, decide for each functional piece (device template, builtin ops, codegen handler, injection decision, hoisting pass, swizzle-swap) where on `feat-c-remove_vmcnt0` the equivalent functionality belongs. Identify the HIP codegen entry point on this branch (whatever it is called) and the async-copy injection pass on this branch. Confirm what `make_wave_buffer_resource` is already doing on this branch so we can reuse it.

2. **Add the device-side primitive.** Extend `src/tl_templates/hip/copy.h` with `cp_async_gs_lds_with_rsrc<N>` (and the no-rsrc variant only if codegen needs it). The inline-asm body can be lifted verbatim from the reference because it is hardware-pinned to gfx950 and has no branch-specific dependencies. Reuse the existing `make_wave_buffer_resource` on this branch instead of redeclaring it. Commit.

3. **Declare the TIR builtin ops.** Add `ptx_cp_async_lds_rsrc` (and `ptx_make_buffer_resource` if not already implied by the existing helper) in `src/op/builtin.h` + `src/op/builtin.cc`, mirroring the registration style this branch uses for nearby ops. Pick names that do not collide with what the branch already has. Commit.

4. **Wire HIP codegen.** In the HIP codegen entry point (identified in step 1), add `VisitExpr_` / `EmitCall_` handlers for the new ops that emit `tl::cp_async_gs_lds_with_rsrc<N>(...)`. Verify with a unit-style hand-built TIR Call that the right C++ text comes out. Commit.

5. **Wire the injection decision.** In the async-copy injection pass (identified in step 1), add a contiguity check — port the reference's `IsLdsContiguous` analysis as the simplest correct option — and route to the new op when the target is gfx950 AND the copy is 16-byte AND the LDS offset is lane-contiguous. Other cases keep emitting the existing baseline. Commit.

6. **Add the hoisting pass.** Create `tilelang/transform/hoist_buffer_resource.py`, register it in `tilelang/transform/__init__.py`, slot it into the pipeline in `tilelang/engine/phase.py` after the injection pass. Start with the descriptor-hoist half only; defer the AMD vmcnt-fixup half unless bench correctness shows wait counts are wrong on this branch. Commit.

7. **Land swizzle-swap.** Port the swizzle-swap optimization into `src/transform/lower_tile_op.cc` (or wherever the equivalent layout-lowering happens on this branch): for layouts with `HasSwizzle()`, move XOR swizzle from LDS store indices to global load indices so LDS writes remain lane-contiguous. This is in-scope by default — not a fallback. Commit.

8. **Final gate.** `bash scripts/iter_buffer_load.sh` from the bench repo. Confirm correctness PASS + TFLOPS ≥ 1000. Inspect a saved `.cpp` to confirm the kernel matches the AC-2 shape. Commit any final cleanup or audit notes.

### Relevant References

- `/root/tile-kernel-bench-cdna4/_fast.cpp` — the pinned acceptance template; the emitted kernel after the port must structurally match its prologue (`make_wave_buffer_resource` + `readfirstlane` per tensor at kernel entry) and inner-loop shape (`tl::cp_async_gs_lds_with_rsrc<16>` calls).
- `/root/tile-kernel-bench-cdna4/scripts/iter_buffer_load.sh` — the only validation harness; wipes `/root/.tilelang/cache/0.1.9_*`, rebuilds tilelang editable, runs the bench, prints correctness + TFLOPS + VGPR + spills.
- `/root/tile-kernel-bench-cdna4/gemm/example_gemm.py` — the matmul_nt entry the bench calls.
- `/root/tilelang2` branch `zty_opt_can_run_1120flops` — the working reference. Specifically: `src/tl_templates/hip/copy.h` (device template + inline asm), `src/op/builtin.{h,cc}` (op declarations), `src/target/codegen_hip.cc` (codegen handlers — note this file is NOT at this path on `feat-c-remove_vmcnt0`), `src/transform/inject_ptx_async_copy.cc` (injection decision + `IsLdsContiguous`), `src/transform/lower_tile_op.cc` (swizzle-swap), `tilelang/transform/hoist_buffer_resource.py` (resource hoisting + AMD wait fixup).
- `/root/backuptilelang` — a frozen snapshot of the working reference; useful for `diff` without worrying about `/root/tilelang2` HEAD moving.
- `/root/tilelang2/tilelang/transform/__init__.py` and `/root/tilelang2/tilelang/engine/phase.py` on `feat-c-remove_vmcnt0` — the pipeline registration sites for the new Python pass.

## Dependencies and Sequence

### Milestones
1. M1 — Function-to-site mapping (no production code edits). Treat the reference branch as a separate repo whose paths are incidental. For each functional piece, decide where on `feat-c-remove_vmcnt0` it logically belongs in the current architecture: identify the HIP codegen entry point on this branch (whatever its file is called); identify the async-copy injection pass on this branch; read the existing `make_wave_buffer_resource` and decide whether to extend it or add a sibling. M1 itself does not commit code — its output is the function→site mapping carried into M2's commit message.
   - Phase A: identify the HIP codegen entry point on this branch.
   - Phase B: identify the async-copy injection pass on this branch.
   - Phase C: read the existing `make_wave_buffer_resource` and decide reuse vs sibling.
2. M2 — Device-side template lands in `src/tl_templates/hip/copy.h`: add `cp_async_gs_lds_with_rsrc<N>` (and the no-rsrc variant only if codegen will call it) reusing the existing resource maker; commit on `feat-c-remove_vmcnt0`. Build must succeed (`pip install -e .`).
   - Step 1: extend `copy.h` with the inline-asm-bearing template.
   - Step 2: smoke-build tilelang.
   - Step 3: commit with message naming M2 and the function→site mapping from M1.
3. M3 — TIR builtin op declarations land in `src/op/builtin.h` + `src/op/builtin.cc`: register `ptx_cp_async_lds_rsrc` (and any helpers needed); commit. Build must succeed.
   - Step 1: declare and register ops.
   - Step 2: build; commit.
4. M4 — HIP codegen handlers land in the entry point identified in M1: emit `tl::cp_async_gs_lds_with_rsrc<N>(...)` for the new op; commit. Build must succeed.
   - Step 1: add the codegen handler.
   - Step 2: write a tiny hand-built TIR Call test (or use an existing fixture) to confirm the right text comes out.
   - Step 3: commit.
5. M5 — Async-copy injection decision lands in the pass identified in M1: add an `IsLdsContiguous`-style check, route 16-byte gfx950 G→S copies to the new op; commit. Build must succeed AND `import tilelang` must still succeed.
   - Step 1: add the contiguity analysis.
   - Step 2: add the routing.
   - Step 3: commit.
6. M6 — `tilelang/transform/hoist_buffer_resource.py` created and wired: pass file added, registered in `tilelang/transform/__init__.py`, inserted into `tilelang/engine/phase.py` after the injection pass. Start with descriptor-hoist only; do NOT port the AMD wait-count fixup yet. Commit.
   - Step 1: write the pass.
   - Step 2: register + slot into pipeline.
   - Step 3: run the bench. If correctness FAIL with wait-count error, M6.5 below; else commit and continue.
   - Step 3a (conditional M6.5): if bench correctness fails due to incorrect async waits on AMD, port the reference's AMD vmcnt scaling into the hoisting pass; commit as its own milestone.
7. M7 — Swizzle-swap lands (in-scope by default, not a fallback). Port the swizzle-swap optimization into `src/transform/lower_tile_op.cc` (or the equivalent layout-lowering site on this branch): for layouts with `HasSwizzle()`, move XOR swizzle from LDS store indices to global load indices so LDS writes remain lane-contiguous. Commit. Build must succeed.
   - Step 1: add swizzle-swap logic for `HasSwizzle()` layouts.
   - Step 2: smoke-build; commit.
8. M8 — Final perf gate. Run `bash scripts/iter_buffer_load.sh`. Confirm correctness PASS and TFLOPS ≥ 1000. Inspect a saved `.cpp` to confirm the kernel structurally matches the AC-2 shape (`make_wave_buffer_resource` + `readfirstlane` in prologue; `cp_async_gs_lds_with_rsrc<16>` in inner loop). Commit any final cleanup / audit notes. If TFLOPS still falls short, iterate (rebench → small tweak → rebench), committing each tweak.
   - Step 1: bench.
   - Step 2: inspect saved `.cpp`.
   - Step 3: commit cleanup / additional tweaks as separate commits.

Dependencies:
- M1 blocks M2-M7 (function→site mapping is the prerequisite for every coding milestone).
- M2 (device template) must land before M4 (codegen) — codegen calls the template by name.
- M3 (op declarations) must land before M4 (codegen) and M5 (injection) — both reference the new op.
- M4 (codegen) must land before M5 (injection) — otherwise the IR refers to an op without a printer.
- M5 (injection) must land before M6 (hoisting) — hoisting transforms calls produced by the injection pass.
- M6 must land before M7 — swizzle-swap interacts with the hoisted-descriptor emission path.
- M6.5 (wait-count fixup) is conditional on M6 bench correctness behavior; lands between M6 and M7 if needed.
- M7 must land before M8 — perf depends on swizzle-swap engaging on lane-contiguous LDS writes.
- M8 may iterate (rebench → tweak → rebench), but every iteration commits.

## Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task-M1 | Map each functional piece (device template, builtin ops, codegen handler, injection decision, hoisting pass, swizzle-swap) to its insertion site on `feat-c-remove_vmcnt0`. Identify the HIP codegen entry point and the async-copy injection pass on this branch. Inspect existing `make_wave_buffer_resource` usage. Produce a one-paragraph mapping note. No code commits. | AC-3 (sets up commit hygiene) | analyze | - |
| task-M2 | Extend `src/tl_templates/hip/copy.h` with `cp_async_gs_lds_with_rsrc<N>` template (inline asm lifted from reference); build tilelang; commit on `feat-c-remove_vmcnt0`. | AC-2, AC-3, AC-4 | coding | task-M1 |
| task-M3 | Declare and register `ptx_cp_async_lds_rsrc` (and any needed siblings) in `src/op/builtin.{h,cc}`; build; commit. | AC-2, AC-3, AC-4 | coding | task-M2 |
| task-M4 | Add HIP codegen handler emitting `tl::cp_async_gs_lds_with_rsrc<N>(...)` at the entry point identified by M1; verify with a hand-built TIR Call; commit. | AC-2, AC-3, AC-4 | coding | task-M3 |
| task-M5 | Add lane-contiguity check (port `IsLdsContiguous`) and route 16-byte gfx950 G→S copies to the new op in the async-copy injection pass identified by M1; commit. | AC-2, AC-3, AC-4 | coding | task-M4 |
| task-M6 | Create `tilelang/transform/hoist_buffer_resource.py` (descriptor-hoist only), register in `tilelang/transform/__init__.py`, insert into `tilelang/engine/phase.py`; run the bench to determine if M6.5 is needed; commit. | AC-2, AC-3, AC-4 | coding | task-M5 |
| task-M6.5 | (Conditional) If M6 bench shows AMD wait-count correctness issue, port the reference's `_fix_amd_wait_counts` into the hoisting pass; commit. | AC-1 (correctness), AC-3 | coding | task-M6 |
| task-M7 | Port swizzle-swap optimization into `src/transform/lower_tile_op.cc` (or equivalent layout-lowering site on this branch) for layouts with `HasSwizzle()`: move XOR swizzle from LDS store indices to global load indices. In-scope by default. Build; commit. | AC-1, AC-2, AC-3, AC-4 | coding | task-M6 |
| task-M8 | Final bench-and-verify run: `bash scripts/iter_buffer_load.sh`; confirm correctness PASS + TFLOPS ≥ 1000; inspect saved `.cpp` for AC-2 shape match; confirm one commit per milestone via `git log feat-c-remove_vmcnt0 --oneline`. If TFLOPS short, iterate with small tweaks (each its own commit). | AC-1, AC-2, AC-3, AC-4 | coding | task-M7 |

## Claude-Codex Deliberation

### Agreements
- (Codex unavailable — see Convergence Status below. The points here reflect Claude's self-review against the draft and the user's directives.)
- The plan keeps the draft's primary direction (vertical slice gated by perf-on-the-bench), while honoring the user's three explicit overrides: no cherry-pick (draft Alt-1 is rejected outright), slim down (do not port the reference's full machinery; treat its layers as an upper bound), and commit per milestone (every milestone in the sequence corresponds to exactly one commit, with import/build smoke after each).
- The env-flag gating that the draft made central (`TL_ENABLE_ROCM_SWIZZLE_SWAP`) is dropped per the user's "ignore — not useful" answer. The bench script still exports the flag harmlessly.
- The audit-first ordering matches the draft's primary direction Step 1 and is reinforced by ground-truth evidence collected during plan generation: `feat-c-remove_vmcnt0` HEAD genuinely lacks the buffer_load_lds machinery (zero tokens for `cp_async_gs_lds`, `ptx_cp_async_lds`, `HoistBufferResource`, `IsLdsContiguous`), so the work is real and not a no-op as one gen-idea explorer suggested.

### Resolved Disagreements
- Draft suggested optionally porting onto a feature branch off the target (`/root/tilelang2`'s `feat-c-remove_vmcnt0`) — user explicitly chose direct commits on the target branch; plan reflects that.
- Draft listed cherry-pick as Alt-1 — user explicitly forbade it; plan moves cherry-pick into "Cannot use".
- Draft was ambivalent about the perf gate ("≈1120 TFLOPS per the reference branch commit message" framed as aspirational) — user picked a hard floor of 1000 TFLOPS; AC-1.1 makes this binding.
- Draft framed env-flag gating as "non-negotiable" in its Synthesis Notes — user said to ignore it; plan drops it as an explicit "Cannot use".
- Original Phase-7 plan framed M1 as "path resolution" (find moved file paths) — user said during the start-rlcr-loop quiz to treat the reference branch as a separate repo and port functionality, not paths. Plan reframes M1 as function-to-site mapping.
- Original Phase-7 plan made swizzle-swap a conditional M7 that only landed if TFLOPS < 1000 — user said during the start-rlcr-loop quiz to land it unconditionally. Plan promotes swizzle-swap to a standard M7 milestone and splits the perf-gate run into its own M8.

### Convergence Status
- Final Status: `partially_converged`. The user chose to continue Claude-only after Codex CLI was reported missing in Phase 3. There was no Phase 5 convergence loop. The plan was instead refined by direct user dialogue in Phase 6, which resolved every open question the draft surfaced (target branch, perf gate, env gate, commit strategy). No items remain `needs_user_decision`.

## Pending User Decisions

- (None — all four decisions surfaced during plan generation were resolved by the user during Phase 6 dialogue. The resolved choices are encoded in Path Boundaries and Acceptance Criteria above.)

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "M1", "M2", "Milestone", "Phase", "Step", "task-M", or similar workflow markers — these belong in this plan document only.
- Commit messages MAY name milestones (per AC-3) for traceability, since commits ARE the milestone delivery mechanism. Code itself uses domain-appropriate naming (e.g. `BufferResourceHoist`, `cp_async_gs_lds_with_rsrc`, `IsLdsContiguous`) drawn from the reference branch's vocabulary where the names are already domain-correct.
- Implementation must not introduce env-var gating like `TL_ENABLE_ROCM_SWIZZLE_SWAP` — feature is always-on for gfx950 16-byte lane-contiguous copies (lower layers in the injection pass already act as the natural gate).
- No `git cherry-pick` (per user). Reading the reference's diffs/files and rewriting is the intended workflow.
- Each milestone is one commit on `feat-c-remove_vmcnt0`. Do not bundle, do not split.

--- Original Design Draft Start ---

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

--- Original Design Draft End ---

---

## BitLesson Selection (REQUIRED FOR EACH TASK)

Before executing each task or sub-task, you MUST:

1. Read @/root/tile-kernel-bench-cdna4/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task to select relevant lesson IDs
3. Follow the selected lesson IDs (or `NONE`) during implementation

Include a `## BitLesson Delta` section in your summary with:
- Action: none|add|update
- Lesson ID(s): NONE or comma-separated IDs
- Notes: what changed and why (required if action is add or update)

Reference: @/root/tile-kernel-bench-cdna4/.humanize/bitlesson.md

---

## Goal Tracker Rules

Throughout your work, you MUST maintain the Goal Tracker:

1. **Before starting a round**: Re-anchor on the original plan and current round contract
2. **Before starting a task**: Mark the relevant mainline task as "in_progress" in Active Tasks
   - Confirm Tag/Owner routing is correct before execution
3. **Active Tasks** are MAINLINE tasks only - side issues do not belong there
4. **Blocking Side Issues** are reserved for issues that truly stop mainline progress
5. **Queued Side Issues** are non-blocking and must not take over the round
6. **After completing a mainline task**: Move it to "Completed and Verified" with evidence (but mark as "pending verification")
7. **If you discover the plan has errors**:
   - Do NOT silently change direction
   - Add entry to "Plan Evolution Log" with justification
   - Explain how the change still serves the Ultimate Goal
8. **If you need to defer a task**:
   - Move it to "Explicitly Deferred" section
   - Provide strong justification
   - Explain impact on Acceptance Criteria
9. **If you discover new issues**:
   - Add to "Blocking Side Issues" only if mainline progress is blocked
   - Otherwise add to "Queued Side Issues" or keep them as `[queued]` tasks/backlog

---

Note: You MUST NOT try to exit `start-rlcr-loop` loop by lying or edit loop state file or try to execute `cancel-rlcr-loop`

After completing the work, please:
0. If you have access to the `code-simplifier` agent, use it to review and optimize the code you just wrote
1. Finalize @/root/tile-kernel-bench-cdna4/.humanize/rlcr/2026-05-17_05-54-01/goal-tracker.md (this is Round 0, so you are initializing it - see "Goal Tracker Setup" above)
2. Write your round contract into @/root/tile-kernel-bench-cdna4/.humanize/rlcr/2026-05-17_05-54-01/round-0-contract.md
3. Commit your changes with a descriptive commit message
4. Write your work summary into @/root/tile-kernel-bench-cdna4/.humanize/rlcr/2026-05-17_05-54-01/round-0-summary.md
