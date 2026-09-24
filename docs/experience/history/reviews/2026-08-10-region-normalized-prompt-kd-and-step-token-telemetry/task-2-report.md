> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-2-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 2 Report: Implement Region-Level Combination in Dense Loss Dispatch

## Status

**DONE**

## Summary

Added `prompt_mask` and `prompt_kd_weight` parameters to both
`compute_dense_loss_from_logits()` and `compute_dense_loss_from_offloaded_teacher()`
in `e2e_common/dense_loss.py`. Each pure logit criterion now computes the response
region first, then (only when weight > 0) computes the prompt region separately and
combines as `L_response + w * L_prompt`. CE+KD families form the regional KD scalar
first, then mix CE exactly once. EAKLD response calls keep existing telemetry; prompt
EAKLD calls pass `telemetry_out=None` so `eakld/*` fields stay response-region only.

## Files Changed

### `e2e_common/dense_loss.py`

- Imported `compute_teacher_entropy_mean_and_gamma` for offloaded prompt gamma.
- Added `_validate_prompt_weight()`: rejects negative weight; rejects positive
  weight without `prompt_mask`.
- Added `_combine_region_loss()`: returns response loss immediately when weight is
  zero or prompt mask is absent; otherwise adds `weight * prompt_loss_fn()`.
- `compute_dense_loss_from_logits()`: added `prompt_mask` / `prompt_kd_weight`
  kwargs. Every pure logit branch (`rkl`, `dual_rkl`, `kl`, `dual_kl`,
  `r_kl_top*`, `dual_r_kl_top*`, `kl_top*`, `dual_kl_top*`, `mse`, `eakld`,
  `eakld_top*`) computes response first, then combines via `_combine_region_loss`.
  CE+KD branches (`kd`, `kd_top*`, `dual_kd`, `dual_kd_top*`, `eakld_kd`) build
  regional KD first, then apply `ce*(1-alpha) + kd_region*alpha` once. EAKLD prompt
  calls pass `telemetry_out=None`.
- `compute_dense_loss_from_offloaded_teacher()`: added `prompt_mask` /
  `prompt_kd_weight` kwargs. Response EAKLD uses existing gamma/entropy/count and
  writes telemetry. Prompt EAKLD computes prompt gamma on-the-fly from
  `teacher_logits_cpu` + `prompt_mask` and calls the CPU-teacher EAKLD reducer with
  `telemetry_out=None`. `eakld_kd` mixes CE once after regional combination.

### `tests/test_distill_losses.py`

Added 10 region-dispatch tests:

- `test_forward_kl_region_combination_matches_manual_means`: manual
  `response_mean + 0.03 * prompt_mean` equality.
- `test_forward_kl_prompt_region_mean_invariant_to_prompt_repetition`: 3 vs 30
  repeated prompt tokens with identical per-token KL → same loss (distinguishes
  region-normalized from shared-denominator formula).
- `test_zero_prompt_weight_matches_response_only_value_and_gradient`: scalar and
  student gradient equality at `prompt_kd_weight=0`.
- `test_empty_prompt_mask_with_positive_weight_contributes_zero`: all-zero prompt
  mask + positive weight equals response-only, finite.
- `test_empty_response_mask_leaves_only_weighted_prompt_loss_kl`: all-zero
  response mask leaves only `w * prompt_mean` (forward KL).
- `test_empty_response_mask_leaves_only_weighted_prompt_loss_eakld`: same for
  EAKLD.
- `test_empty_prompt_mask_eakld_positive_weight_remains_finite`: all-zero prompt
  mask + positive weight, EAKLD stays finite with finite gradient.
- `test_negative_prompt_weight_is_error`.
- `test_positive_weight_without_prompt_mask_is_error`.
- `test_kd_ce_not_double_counted_across_regions`: large CE scalar (1000), verifies
  `d(loss)/d(ce) = 1 - alpha`, proving CE counted once.
- `test_eakld_prompt_call_does_not_overwrite_response_telemetry`: telemetry matches
  response-only EAKLD telemetry.

### `tests/smoke/test_loss_pipeline_smoke.py`

- Replaced `build_distill_token_mask(..., prompt_kd_weight=0.1)` (removed in Task 1)
  with `build_distill_token_regions(...)`, passing `response_mask` as `mask` and
  `prompt_mask` with `prompt_kd_weight=0.03` to the dispatcher.
- Updated zero-gradient assertion to check positions outside both regions
  (`neither_region`), since prompt-region positions now legitimately receive
  gradient from the prompt loss.

## Self-Review

| Requirement | Met? | Notes |
|---|---|---|
| Keep `mask` as response mask | Yes | Backward compatible |
| Add `prompt_mask` + `prompt_kd_weight` to both dispatchers | Yes | |
| Manual forward-KL test (response + 0.03*prompt) | Yes | |
| Anti-regression repetition test | Yes | 3 vs 30 tokens, identical per-token KL |
| p=0 compatibility (value + gradient) | Yes | |
| Empty-region tests (KL + EAKLD) | Yes | 4 tests covering both empty sides |
| Negative weight error | Yes | |
| Positive weight without prompt_mask error | Yes | |
| Pure logit: response first, skip prompt at w=0 | Yes | `_combine_region_loss` short-circuits |
| Applied to all listed pure criteria | Yes | rkl, dual_rkl, kl, dual_kl, r_kl_top*, dual_r_kl_top*, kl_top*, dual_kl_top*, mse, eakld, eakld_top* |
| CE+KD: regional KD first, CE once | Yes | kd, kd_top*, dual_kd, dual_kd_top*, eakld_kd |
| CE anti-double-count test | Yes | grad w.r.t. CE = 1 - alpha |
| EAKLD response keeps telemetry, prompt does not | Yes | Tested |
| Only allowed files modified | Yes | 3 files |
| No git commit | Yes | Worktree only |

## Concerns

1. The offloaded dispatcher computes prompt gamma on-the-fly from
   `teacher_logits_cpu` + `prompt_mask` inside `_prompt_eakld()`. Task 5 will
   replace this with precomputed prompt scalar arguments corresponding to the new
   `TeacherTargetBatch` fields. This keeps Task 2 self-contained while preserving
   the region-normalized contract.

2. The offloaded prompt path is not yet exercised with positive prompt weight in
   Task 2 tests; Task 5 adds dense-versus-offload equality tests with positive
   prompt weight.

## Test Summary

`PYTHONPATH=. pytest tests/test_distill_losses.py tests/smoke/test_loss_pipeline_smoke.py -q`
→ **60 passed in 5.15s**

## Commits

None (per instructions).

---

## Fix Round 1/5

### Finding F1 (Important/Spec)

`test_forward_kl_prompt_region_mean_invariant_to_prompt_repetition` used an all-zero
response mask, so both the new region-normalized formula and the old
shared-denominator formula were independent of prompt length P. The test failed to
distinguish the two formulas as the brief requires.

### Fix

Rewrote the test to include a non-empty response region (2 tokens) whose per-token
KL differs from the prompt region's per-token KL. The prompt region is repeated
(3 vs 30 tokens) with identical per-token KL across prompt positions. Now:

- New region-normalized loss: `L_response + w * L_prompt` is invariant to P
  (each region divides by its own token count, and all prompt per-token KL are
  equal). Asserted `short_loss == long_loss` and equality with the independently
  computed `response_mean + w * prompt_mean`.
- Old shared-denominator formula `(sum((r + w*p) * kl) / sum(r + w*p))` now drifts
  with P because the response per-token KL differs from the prompt per-token KL.
  Added a sanity assertion that the old formula's short vs long values are NOT
  close, proving the test distinguishes the two formulas.

### Covering tests

```
PYTHONPATH=. pytest tests/test_distill_losses.py -q -k prompt_region_mean_invariant
```
Output: `1 passed, 56 deselected in 4.84s`

```
PYTHONPATH=. pytest tests/test_distill_losses.py tests/smoke/test_loss_pipeline_smoke.py -q
```
Output: `60 passed in 5.31s`

### Commits

None.
