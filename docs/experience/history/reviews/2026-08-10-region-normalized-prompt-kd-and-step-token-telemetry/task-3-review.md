> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-3-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 3 Review: Convert Category Distillation to Region-Normalized Loss

## Verdict

**Spec: ✅ PASS** — all brief checklist items satisfied.
**Quality: High** — focused, meaningful tests; clean refactor; no scope creep.

## Spec Compliance

| Brief item | Status | Evidence |
|---|---|---|
| One shared `build_token_regions(reference_logits)` calling `build_distill_token_regions()` on original labels + attention | ✅ | `train_utils/lora_training.py:750-755` |
| Common combiner with verbatim control flow (response first; short-circuit on `w==0`; else `response + w*prompt`) | ✅ | `train_utils/lora_training.py:757-762` |
| All pure tokenwise branches route through combiner (`rkl`, `dual_rkl`, `kl`, `r_kl_top*`, `dual_r_kl_top*`, `kl_top*`, `mse`, `dual_kl`, `dual_kl_top*`, `eakld_top*`, `eakld`) | ✅ | 11 branches in diff; each wraps criterion in `lambda mask: ...` and calls `combine_region_loss` |
| All CE+KD branches build regional KD first, then apply `ori_loss*(1-alpha)+distill_loss*alpha` once (`kd_top*`, `kd`, `dual_kd_top*`, `dual_kd`, `eakld_kd`) | ✅ | 5 branches; alpha mix appears exactly once each |
| EAKLD called twice with different masks when `w>0`, once when `w==0` | ✅ | Combiner short-circuit + test `test_eakld_positive_prompt_weight_calls_criterion_twice_with_different_masks` / `..._zero_..._once_on_response` |
| SFT/origin, hidden, pre-MLP, teacher staging, LoRA merge/restore unchanged | ✅ | Diff only touches import line, `build_token_regions`, `combine_region_loss`, and branch bodies; staging/merge helpers untouched |
| Static audit: no `build_distill_token_mask` / `build_token_mask` in `lora_training.py`; no call passes `prompt_kd_weight` into `build_distill_token_mask()` | ✅ | `rg` returns no matches in `lora_training.py`; only `prompt_kd_weight` usages are ctor validation + combiner |

## Quality

- Tests are substantive, not smoke-only:
  - `test_kl_region_combination_matches_manual_means` independently recomputes teacher/student logits and asserts `allclose(loss, response_mean + w*prompt_mean)` — verifies the combiner math, not just finiteness.
  - `test_zero_prompt_weight_matches_response_only_value` confirms short-circuit equivalence.
  - EAKLD tests mock `compute_eakld`, record masks, assert call count, non-emptiness, and disjointness (`(response+prompt).max() <= 1.0`).
  - `test_kd_ce_counted_once_across_regions` asserts gradient flow to student scale (CE mixed once).
- `build_distill_token_regions` produces disjoint regions by construction (`labels.ne(-100)` vs `labels.eq(-100) & attention`), both causal-shifted — sound.
- Combiner correctly differentiable when prompt mask is all-zero (masked losses clamp denom to 1.0 → differentiable zero), matching Task 2's `dense_loss.py` contract.

## Test Reproduction (bitvae env)

```
PYTHONPATH=. pytest tests/test_cat_eval_adapter_match.py -q -k RegionNormalized
→ 5 passed, 23 deselected in 5.25s

PYTHONPATH=. pytest tests/test_cat_eval_adapter_match.py tests/smoke/test_one_step_train_smoke.py -q -k "not dense_eakld and not cpu_offload_eakld"
→ 31 passed, 2 deselected in 6.20s
```

Matches report exactly.

## Findings

1. **Pre-existing eakld smoke failures (out of scope, correctly flagged).**
   `tests/smoke/test_one_step_train_smoke.py::test_dense_eakld_one_step_trainer_smoke` and `..._cpu_offload_eakld...` fail with `TypeError: build_distill_token_mask() got an unexpected keyword argument 'prompt_kd_weight'` at `compressed_e2e_fintuning/trainer.py:407`. This is Task 1 breakage in the e2e decoder trainer path, not touched by Task 3. Report's concern #1 is accurate. Reproduced: 2 failed.

2. **No Task 3 defects found.** Branch coverage complete, combiner matches brief verbatim, EAKLD double-call behavior proven by focused test, static audit clean.

## ⚠️ Warnings

- None for Task 3. The e2e trainer breakage (finding 1) is explicitly out of scope per global constraints ("E2E trainer still broken until later tasks"); do not fix here.

## Commits

None (per instructions). ✅
