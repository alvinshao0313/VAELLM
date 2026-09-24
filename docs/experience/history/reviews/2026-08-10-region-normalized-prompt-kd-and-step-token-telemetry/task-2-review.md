> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-2-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 2 Review: Region-Level Combination in Dense Loss Dispatch

## Verdict

- **Spec:** ❌ (one required test property unmet)
- **Quality:** Changes Requested (one blocking test deficiency + minor notes)

## Spec Compliance Matrix

| Requirement | Status | Notes |
|---|---|---|
| Keep `mask` as response mask; add `prompt_mask` + `prompt_kd_weight` to both dispatchers | ✅ | Both `compute_dense_loss_from_logits` and `compute_dense_loss_from_offloaded_teacher` updated; backward compatible |
| Formula `L = L_response + w * L_prompt` with independent per-region means | ✅ | `_combine_region_loss` calls each region's loss fn separately; each divides by its own `mask.sum().clamp_min(1.0)` (verified in `_masked_token_kl_mean`) |
| MUST NOT use shared weighted-mask denominator | ✅ | Regions computed independently; no fractional mask passed to underlying fns |
| Weight zero skips prompt, preserves response-only scalar + student grads | ✅ | `_combine_region_loss` short-circuits before calling `prompt_loss_fn()`; `test_zero_prompt_weight_matches_response_only_value_and_gradient` verifies scalar + grad |
| CE evaluated exactly once for CE+KD families | ✅ | `kd`/`kd_top*`/`dual_kd`/`dual_kd_top*`/`eakld_kd` build `kd_region` first, then `ce*(1-a) + kd_region*a`; `test_kd_ce_not_double_counted_across_regions` confirms `d(loss)/d(ce) = 1-a` |
| EAKLD separate gamma per region; existing `eakld/*` telemetry stays response-region | ✅ | Response call passes `telemetry_out`; prompt call passes `telemetry_out=None`; offloaded prompt computes its own gamma via `compute_teacher_entropy_mean_and_gamma(teacher_logits_cpu, prompt_mask, ...)` |
| All listed pure criteria covered | ✅ | rkl, dual_rkl, kl, dual_kl, r_kl_top*, dual_r_kl_top*, kl_top*, dual_kl_top*, mse, eakld, eakld_top* all branch through `_combine_region_loss` |
| All listed CE+KD families covered | ✅ | kd, kd_top*, dual_kd, dual_kd_top*, eakld_kd |
| Positive weight with missing prompt_mask must raise | ✅ | `_validate_prompt_weight` raises `ValueError`; `test_positive_weight_without_prompt_mask_is_error` |
| Negative weight must raise | ✅ | `_validate_prompt_weight` raises; `test_negative_prompt_weight_is_error` |
| Empty regions must be finite (zero contribution) | ✅ | All-zero mask → `denom = clamp_min(1.0)`, numerator 0 → result 0, finite; 4 empty-region tests cover both sides for KL + EAKLD |
| Manual forward-KL test (response + 0.03*prompt) | ✅ | `test_forward_kl_region_combination_matches_manual_means` |
| Anti-regression test: repetition must not change prompt mean/coefficient, AND must distinguish new from old shared-denominator formula | ❌ | See Finding F1 |
| p=0 compatibility (scalar + student grads) | ✅ | `test_zero_prompt_weight_matches_response_only_value_and_gradient` |
| CE anti-double-count test with large CE scalar | ✅ | `ce_tensor=1000.0`, grad = `1-alpha` |
| EAKLD prompt call does not overwrite response telemetry | ✅ | `test_eakld_prompt_call_does_not_overwrite_response_telemetry` compares against response-only telemetry |
| Only Task2 production files changed | ✅ | `e2e_common/dense_loss.py` + 2 test files; no fractional-mask dependency (uses `build_distill_token_regions` with binary masks) |
| No auto-commit | ✅ | Report confirms no commits |

## Findings

### F1 (Blocking) — Anti-regression test does NOT distinguish new formula from old shared-denominator formula

**Location:** `tests/test_distill_losses.py`, `test_forward_kl_prompt_region_mean_invariant_to_prompt_repetition`

**Issue:** The brief explicitly requires: *"This test must distinguish the new formula from the old shared-denominator formula."* The test uses an **all-zero response mask** (`short_response = torch.zeros(1, 3)`, `long_response = torch.zeros(1, 30)`) and asserts only `short_loss == long_loss` (relative equality, not absolute value).

Under the **old** shared-denominator formula with fractional mask `m = resp + w*prompt`:
- `L_old = sum(kl * m) / sum(m) = (w * P * klp) / (w * P) = klp` — **independent of P**.

Under the **new** region-normalized formula:
- `L_new = 0 + w * mean(klp) = w * klp` — also **independent of P**.

Both formulas produce P-invariant results when response is empty, so `short_loss == long_loss` passes for **both** formulas. The test does not distinguish them. The distinguishing power only emerges when **response tokens are present**: the old formula's denominator `sum(resp + w*prompt)` changes with P, shifting the weighted average, while the new formula's `L_response + w*L_prompt` stays constant.

**Fix:** Add non-empty response tokens (with per-token KL distinct from prompt per-token KL) to both the short and long cases, keeping the response region identical across the two cases. Then:
- New formula: `loss_short == loss_long` (response mean + `w * prompt_mean` unchanged).
- Old formula: `loss_short != loss_long` (shared denominator shifts with prompt count).

Optionally also assert the absolute value equals `w * klp` (which would fail for the old formula's `klp`), but the response-token approach is the robust distinguishing test.

### F2 (Minor) — Validation runs before `sft`/`origin` early return

**Location:** `e2e_common/dense_loss.py`, `compute_dense_loss_from_logits`

`_validate_prompt_weight` is called at the top of the dispatcher, before the `sft`/`origin` branch returns `ce_loss`. This means `compute_dense_loss_from_logits(loss_type="sft", ce_loss=..., prompt_kd_weight=0.03)` raises even though `sft` never uses `prompt_mask`. This is defensible (consistent global validation) and not a spec violation, but callers passing a leftover positive weight to `sft` will get a surprising error. Acceptable as-is; flagged for awareness.

### F3 (Minor) — Offloaded prompt EAKLD recomputes gamma from `teacher_logits_cpu`

**Location:** `e2e_common/dense_loss.py`, `_prompt_eakld` in `compute_dense_loss_from_offloaded_teacher`

The prompt gamma is recomputed from `teacher_logits_cpu` + `prompt_mask` on every call. The report acknowledges this and defers precomputed prompt gamma to Task 5. Correct for now; the offloaded prompt path is also not exercised with positive weight in Task 2 tests (deferred to Task 5 per report). Not a spec violation.

### F4 (Minor) — `_combine_region_loss` redundant `prompt_mask is None` check

When `weight > 0`, `_validate_prompt_weight` already guaranteed `prompt_mask is not None`, so the `prompt_mask is None` check in `_combine_region_loss` only fires for the `weight == 0` default path. Harmless defensive guard; no action needed.

## Quality Notes

- `_combine_region_loss` + `_validate_prompt_weight` are clean, well-named helpers that substantially reduce per-branch duplication.
- Lazy `prompt_loss_fn` lambda correctly avoids computing prompt loss when `weight == 0`.
- Mask device/dtype handling delegated to `_default_token_mask` in underlying fns — consistent with existing patterns.
- Smoke test correctly updated: binary regions, `neither_region` zero-gradient check, `prompt_kd_weight=0.03` exercised across all `DENSE_LOSS_TYPES`.
- Test coverage is otherwise thorough: 10 new region-dispatch tests + updated smoke tests, 60 passed per report.

## Summary

The production implementation in `e2e_common/dense_loss.py` is correct and well-structured — the formula, CE-once invariant, EAKLD telemetry isolation, validation, and empty-region finiteness all hold. The single blocking issue is **F1**: the anti-regression test does not fulfill its explicit requirement to distinguish the new region-normalized formula from the old shared-denominator formula, because it uses an empty response mask under which both formulas are P-invariant. Adding non-empty response tokens with distinct per-token KL will make the test genuinely discriminating. Once F1 is fixed, this task should pass gate.
