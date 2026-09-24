> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-3-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 3 Report: Convert Category Distillation to Region-Normalized Loss

## Status

**DONE**

## Summary

Converted `CustomSFTTrainer.compute_loss()` in `train_utils/lora_training.py`
from the old single-mask `build_distill_token_mask(..., prompt_kd_weight=...)`
path to a shared region-normalized combination. One shared
`build_token_regions(reference_logits)` helper now calls
`build_distill_token_regions()` on the original `labels` + `attention_mask`.
One shared `combine_region_loss(loss_for_mask, regions)` implements the
verbatim control flow from the brief:

```
response_loss = loss_for_mask(response_mask)
if prompt weight is zero: return response_loss
prompt_loss = loss_for_mask(prompt_mask)
return response_loss + prompt weight * prompt_loss
```

All pure tokenwise branches (`rkl`, `dual_rkl`, `kl`, `r_kl_top*`,
`dual_r_kl_top*`, `kl_top*`, `mse`, `dual_kl`, `dual_kl_top*`, `eakld_top*`,
`eakld`) route through the combiner. CE+KD branches (`kd_top*`, `kd`,
`dual_kd_top*`, `dual_kd`, `eakld_kd`) build the regional KD scalar first,
then apply `ori_loss * (1 - alpha) + distill_loss * alpha` exactly once.
SFT/origin, hidden loss, pre-MLP hidden loss, teacher-logit staging, and
LoRA merge/restore logic are unchanged.

## Files Changed

### `train_utils/lora_training.py`

- Imports: replaced `build_distill_token_mask` with
  `build_distill_token_regions` (the now-unused `build_distill_token_mask`
  import was removed).
- Replaced the local `build_token_mask(reference_logits)` (which passed
  `prompt_kd_weight` into `build_distill_token_mask`) with
  `build_token_regions(reference_logits)` calling
  `build_distill_token_regions()` on `full_inputs` labels/attention.
- Added `combine_region_loss(loss_for_mask, regions)` implementing the
  brief's verbatim control flow (response first; short-circuit when
  `prompt_kd_weight == 0.0`; otherwise add `weight * prompt_loss`).
- Refactored every pure tokenwise branch to call
  `combine_region_loss(lambda mask: compute_xxx(..., mask=mask, ...), regions)`.
- Refactored every CE+KD branch to build `distill_loss` via
  `combine_region_loss(...)`, then mix CE once with `alpha`.
- EAKLD branches: with positive prompt weight the EAKLD criterion is now
  called twice (response mask, then prompt mask); with zero weight it is
  called once on the response mask only. No telemetry is collected on the
  prompt call (consistent with the pre-existing lora_training behavior of
  not passing `telemetry_out`).

### `tests/test_cat_eval_adapter_match.py`

Added `CatDistillRegionNormalizedLossTest` with 5 tests:

- `test_eakld_positive_prompt_weight_calls_criterion_twice_with_different_masks`:
  mocks `compute_eakld` and asserts it is called exactly twice, with
  disjoint response/prompt masks, both non-empty.
- `test_eakld_zero_prompt_weight_calls_criterion_once_on_response`:
  mocks `compute_eakld` and asserts it is called exactly once on a
  non-empty (response) mask.
- `test_kl_region_combination_matches_manual_means`: recomputes
  teacher/student logits independently and asserts the trainer loss equals
  `response_mean + weight * prompt_mean`.
- `test_zero_prompt_weight_matches_response_only_value`: asserts the
  zero-weight loss equals the response-only forward-KL.
- `test_kd_ce_counted_once_across_regions`: runs the `kd` branch with
  positive prompt weight and asserts finiteness plus gradient flow to the
  student scale (CE is mixed once via `alpha`).

## Static Audit

`train_utils/lora_training.py`:

- No occurrence of `build_distill_token_mask` or `build_token_mask` remains.
- The only region builder is the shared `build_token_regions` calling
  `build_distill_token_regions()` (which internally calls
  `build_distill_token_mask()` without `prompt_kd_weight`).
- No call passes `prompt_kd_weight` into `build_distill_token_mask()`.

## Test Summary

```
PYTHONPATH=. pytest tests/test_cat_eval_adapter_match.py -q -k RegionNormalized
→ 5 passed, 23 deselected in 5.14s
```

```
PYTHONPATH=. pytest tests/test_cat_eval_adapter_match.py tests/smoke/test_one_step_train_smoke.py -q -k "not dense_eakld and not cpu_offload_eakld"
→ 31 passed, 2 deselected in 6.18s
```

## Concerns

1. `tests/smoke/test_one_step_train_smoke.py::test_dense_eakld_one_step_trainer_smoke`
   and `test_cpu_offload_eakld_one_step_trainer_smoke` fail, but the
   failures originate in `compressed_e2e_fintuning/trainer.py`
   (`VAEDecoderE2ETrainer._build_distill_token_mask` still passes
   `prompt_kd_weight=self.prompt_kd_weight` to `build_distill_token_mask`).
   This is pre-existing breakage introduced by Task 1's removal of the
   `prompt_kd_weight` parameter from `build_distill_token_mask`, in a file
   outside Task 3's scope. Task 3 only touches the category LoRA path in
   `train_utils/lora_training.py` (`CustomSFTTrainer`); the e2e decoder
   trainer path is a separate dispatch and is not modified here.

2. The new `combine_region_loss` follows the brief's verbatim flow strictly:
   it short-circuits on `prompt_kd_weight == 0.0` and otherwise always
   computes the prompt-region loss (even if the prompt mask is all zeros,
   in which case the masked loss functions return a differentiable zero
   via `denom.clamp_min(1.0)`). This matches `e2e_common/dense_loss.py`'s
  region-normalized contract from Task 2.

## Commits

None (per instructions).
