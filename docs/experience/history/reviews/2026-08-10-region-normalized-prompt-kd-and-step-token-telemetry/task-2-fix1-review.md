> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-2-fix1-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 2 Fix Round 1 Review

**Scope:** `task-2-fix1-review-package.diff` only (rewritten `test_forward_kl_prompt_region_mean_invariant_to_prompt_repetition`; production code unchanged).

**Fix report tests:** targeted `-k prompt_region_mean_invariant` → 1 passed; full suite → 60 passed (confirmed in report; not re-run here).

## Findings

| ID | Verdict | Notes |
|---|---|---|
| F1 | **ADDRESSED** | Test now uses 2-token non-empty response region with logits distinct from prompt (`*2.0` vs `*0.5`), so per-token KL differs across regions. Prompt repeated 3 vs 30 with identical per-token KL. Asserts new loss invariant (`short == long`) and matches manual `response_mean + w * prompt_mean`. Adds `_old_shared_denominator` sanity check proving old formula drifts (`old_short != old_long`). Satisfies brief requirement to distinguish formulas. |

## New Breakage

None observed in the fix diff. Change is test-only; no API or dispatch edits.

## Overall Verdict

**PASS** — F1 fully fixed; anti-regression test now meaningfully guards region-normalized combination vs shared-denominator regression.
