# Code Review - Round 0

## Original Implementation Plan

**IMPORTANT**: The original plan that Claude is implementing is located at:
@.humanize/plans/add-bufferload-lds-to-tilelang-20260517-052752.md

You MUST read this plan file first to understand the full scope of work before conducting your review.
This plan contains the complete requirements and implementation details that Claude should be following.

Based on the original plan and @/root/tile-kernel-bench-cdna4/.humanize/rlcr/2026-05-17_05-54-01/round-0-prompt.md, Claude claims to have completed the work. Please conduct a thorough critical review to verify this.

---
Below is Claude's summary of the work completed:
<!-- CLAUDE's WORK SUMMARY START -->
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
<!-- CLAUDE's WORK SUMMARY  END  -->
---

## Development History (Integral Context)

Accumulated commits since loop start (oldest first):
```
3935c22 remove caches
ff52b5f remove caches
aec54f2 add a flag to save temp files
dc8e068 add a flag to save temp files
80da99b add a flag to save temp files
320cdb6 add a flag to save temp files
08969ef add
a665666 Add humanize local folder into gitignore
```

### Recent Round Files
Read these files before conducting your review to understand the trajectory of work:
(first round, no prior history)

Use this history to identify patterns across rounds: recurring issues, stalled progress, or drift from the mainline objective. Weight recent rounds more heavily but watch for systemic trends in the full commit log.

## Part 1: Implementation Review

- Your task is to conduct a deep critical review, focusing on finding implementation issues and identifying gaps between "plan-design" and actual implementation.
- Relevant top-level guidance documents, phased implementation plans, and other important documentation and implementation references are located under @docs.
- If Claude planned to defer any tasks to future phases in its summary, DO NOT follow its lead. Instead, you should force Claude to complete ALL tasks as planned.
  - Such deferred tasks are considered incomplete work and should be flagged in your review comments, requiring Claude to address them.
  - If Claude planned to defer any tasks, please explore the codebase in-depth and draft a detailed implementation plan. This plan should be included in your review comments for Claude to follow.
  - Your review should be meticulous and skeptical. Look for any discrepancies, missing features, incomplete implementations.
- If Claude does not plan to defer any tasks, but honestly admits that some tasks are still pending (not yet completed), you should also include those pending tasks in your review.
  - Your review should elaborate on those unfinished tasks, explore the codebase, and draft an implementation plan.
  - A good engineering implementation plan should be **singular, directive, and definitive**, rather than discussing multiple possible implementation options.
  - The implementation plan should be **unambiguous**, internally consistent, and coherent from beginning to end, so that **Claude can execute the work accurately and without error**.

## Part 2: Goal Alignment Check (MANDATORY)

Read @/root/tile-kernel-bench-cdna4/.humanize/rlcr/2026-05-17_05-54-01/goal-tracker.md and verify:

1. **Acceptance Criteria Progress**: For each AC, is progress being made? Are any ACs being ignored?
2. **Forgotten Items**: Are there tasks from the original plan that are not tracked in Active/Completed/Deferred?
3. **Deferred Items**: Are deferrals justified? Do they block any ACs?
4. **Plan Evolution**: If Claude modified the plan, is the justification valid?

Include a brief Goal Alignment Summary in your review:
```
ACs: X/Y addressed | Forgotten items: N | Unjustified deferrals: N
```

## Part 3: Required Finding Classification

You MUST classify your findings into these lanes:
- **Mainline Gaps**: plan-derived work or AC progress that is missing, incomplete, or regressing
- **Blocking Side Issues**: bugs or implementation issues that block the current mainline objective from succeeding safely
- **Queued Side Issues**: valid non-blocking follow-up issues that should be documented but must NOT take over the next round

Also include a one-line verdict:
```
Mainline Progress Verdict: ADVANCED / STALLED / REGRESSED
```

This verdict line is mandatory. If you omit it, the Humanize stop hook will block the round and require the review to be rerun.

If Claude mostly worked on queued side issues and failed to advance the mainline, say so explicitly.

## Part 4: ## Goal Tracker Update Requests (YOUR RESPONSIBILITY)

Claude should normally keep the **mutable section** of `goal-tracker.md` up to date directly. If Claude's summary contains a "Goal Tracker Update Request" section, or if you detect tracker drift during review, YOU must:

1. **Evaluate the tracker state**: Is the mutable section still aligned with the Ultimate Goal and current AC progress?
2. **If correction is needed**: Update @/root/tile-kernel-bench-cdna4/.humanize/rlcr/2026-05-17_05-54-01/goal-tracker.md yourself with the requested changes:
   - Move tasks between Active/Completed/Deferred sections as appropriate
   - Add entries to "Plan Evolution Log" with round number and justification
   - Add new issues to "Blocking Side Issues" or "Queued Side Issues" as appropriate
   - **NEVER modify the IMMUTABLE SECTION** (Ultimate Goal and Acceptance Criteria)
3. **If you reject a requested tracker change**: Include in your review why it was rejected

Common update requests you should handle:
- Task completion: Move from "Active Tasks" to "Completed and Verified"
- New blocking issues: Add to "Blocking Side Issues"
- New queued issues: Add to "Queued Side Issues"
- Plan changes: Add to "Plan Evolution Log" with your assessment
- Deferrals: Only allow with strong justification; add to "Explicitly Deferred"

## Part 5: Output Requirements

- In short, your review comments can include: problems/findings/blockers; claims that don't match reality; implementation plans for deferred work (to be implemented now); implementation plans for unfinished work; goal alignment issues.
- Your output should be structured so Claude can tell which items are mainline gaps, blocking side issues, and queued side issues.
- If after your investigation the actual situation does not match what Claude claims to have completed, or there is pending work to be done, output your review comments to @/root/tile-kernel-bench-cdna4/.humanize/rlcr/2026-05-17_05-54-01/round-0-review-result.md.
- **CRITICAL**: Only output "COMPLETE" as the last line if ALL tasks from the original plan are FULLY completed with no deferrals
  - DEFERRED items are considered INCOMPLETE - do NOT output COMPLETE if any task is deferred
  - UNFINISHED items are considered INCOMPLETE - do NOT output COMPLETE if any task is pending
  - The ONLY condition for COMPLETE is: all original plan tasks are done, all ACs are met, no deferrals or pending work allowed
- The word COMPLETE on the last line will stop Claude.
