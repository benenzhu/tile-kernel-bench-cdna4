# Round 0 Summary

## What Was Implemented

Round 0 = infrastructure initialization + M1 audit (no code commits, per plan).

- RLCR loop infrastructure set up by the setup script (Codex CLI now installed and working).
- M1 (function-to-site mapping) completed read-only against `/root/tilelang2`. Mapping table captured in `round-0-contract.md` (key insertion sites identified for every M2..M7 deliverable).
- Goal Tracker populated: Immutable section was filled by the setup script from the plan's goal + ACs. I filled the Mutable section: Active Tasks (M1..M8), Plan Evolution Log (dual-repo caveat + plan-already-edited note), Queued Side Issues (branch-switch reminder, hoist-pass insertion-point TBD).
- BitLesson knowledge base remains empty (template created by setup script; no lessons yet, none earned by Round 0).
- Dual-repo caveat explicitly recorded: RLCR runs in bench repo `/root/tile-kernel-bench-cdna4` on `debug_vm_cnt0`; implementation commits will land in `/root/tilelang2` on `feat-c-remove_vmcnt0`. Each future round summary will name the corresponding tilelang2 commit hash.

## M1 Mapping Result (the substantive deliverable for Round 0)

| Functional Piece | Insertion Site on `feat-c-remove_vmcnt0` | Notes |
|---|---|---|
| Device template `cp_async_gs_lds_with_rsrc<N>` | `src/tl_templates/hip/copy.h` (extend) | `make_wave_buffer_resource` already at line 22 — reuse |
| Builtin ops (`ptx_cp_async_lds`, `ptx_make_buffer_resource`, `ptx_cp_async_lds_rsrc`) | `src/op/builtin.{h,cc}` (extend) | |
| HIP codegen handlers | `src/backend/rocm/codegen/codegen_hip.cc` | **Path changed** vs reference (was `src/target/codegen_hip.cc`) |
| Async-copy injection decision | Primary: `src/backend/rocm/op/copy.cc` (already calls `InjectPTXAsyncCopy` at ~line 131); Fallback: `src/transform/lower_ptx_async_copy.cc` (745 LOC) | **Renamed + split** vs reference. ROCm-side `op/copy.cc` is cleanest insertion point |
| `HoistBufferResource` Python pass | New file `tilelang/transform/hoist_buffer_resource.py`; register in `tilelang/transform/__init__.py` (sibling to `HoistBroadcastValues`, `DecoupleTypeCast`) | Pipeline insertion point TBD when M6 starts |
| Swizzle-swap | `src/transform/lower_tile_op.cc` (extend) | Same path as reference |

## Files Created/Modified

In the bench repo (`/root/tile-kernel-bench-cdna4`):
- `.humanize/rlcr/2026-05-17_05-54-01/` (RLCR session dir created by setup script; gitignored)
  - `goal-tracker.md` — Mutable section populated (Plan Evolution Log + Active Tasks + Queued Side Issues + Completed)
  - `round-0-contract.md` — written (contract + M1 mapping table)
  - `round-0-summary.md` — this file
- `.humanize/bitlesson.md` — created by setup script (template only; gitignored)
- `.humanize/plans/add-bufferload-lds-to-tilelang-20260517-052752.md` — committed by the setup script (`--track-plan-file`)

In `/root/tilelang2`:
- No changes. /root/tilelang2 is still on `zty_opt_can_run_1120flops`; branch switch to `feat-c-remove_vmcnt0` happens at start of Round 1.

## Commits This Round

**One commit in the bench repo: `a665666` ("Add humanize local folder into gitignore").** This was a corrective housekeeping commit, not a code change:
- The setup script committed the plan file and the idea file under `.humanize/` via `--track-plan-file`. The session's Stop hook then flagged these as tracked-but-should-be-local Humanize state.
- Fix per hook guidance: appended `.humanize*` to `.gitignore`, ran `git rm --cached -r .humanize`, committed the gitignore + the index removals. Working-tree files remain on disk; they are simply no longer tracked.
- All RLCR state files (`goal-tracker.md`, `round-0-*.md`, `bitlesson.md`, plan, idea) now live entirely off-git. Codex reads them from disk.
- No `/root/tilelang2` commits this round (M1 is no-code-commits per the plan).

This is the expected behavior for Round 0 (init + audit only). M2 in Round 1 will be the first round with an actual `/root/tilelang2` commit on `feat-c-remove_vmcnt0`; the bench-repo side will continue producing only on-disk metadata that Codex reads directly. Future rounds should not produce bench-repo commits unless the bench script itself is being edited.

## Validation

- `git ls-tree -r feat-c-remove_vmcnt0` confirmed the relocated HIP codegen path (`src/backend/rocm/codegen/codegen_hip.cc`).
- `git ls-tree feat-c-remove_vmcnt0 src/transform/` confirmed `lower_ptx_async_copy.cc` is the new async-copy pass (no `inject_ptx_async_copy.cc`); `ptx_async_copy_injector.h` is the small statement-level entry header.
- `git grep "InjectPTXAsyncCopy\|PTXAsyncCopyInjector" feat-c-remove_vmcnt0` surfaced `src/backend/rocm/op/copy.cc` as the ROCm-side caller — confirmed cleanest insertion point.
- `git grep "make_wave_buffer_resource" feat-c-remove_vmcnt0` confirmed the helper is defined at `src/tl_templates/hip/copy.h:22` and reused at lines 107, 132.
- `git show feat-c-remove_vmcnt0:tilelang/transform/__init__.py` confirmed import pattern for sibling Python passes — `HoistBufferResource` will follow the same form.

No bench run this round — would only re-measure the reference branch (currently checked out in /root/tilelang2) and not test anything new.

## Remaining Items

All of M2..M8 (deferred to Rounds 1..7+ per the per-milestone-commit plan):
- Round 1: M2 device template
- Round 2: M3 builtin ops
- Round 3: M4 HIP codegen
- Round 4: M5 injection decision (in `rocm/op/copy.cc`)
- Round 5: M6 hoisting pass (also locate pipeline insertion point)
- Round 6: M7 swizzle-swap
- Round 7+: M8 perf gate + iteration

Branch switch reminder: start of Round 1 must `cd /root/tilelang2 && git checkout feat-c-remove_vmcnt0` before any commit.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 0 is pure initialization + read-only audit; no failure mode encountered, no novel solution found, nothing reusable to record yet. BitLesson template was created by the setup script (`.humanize/bitlesson.md`).
