> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-1-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 1 Report: Replace Fractional Prompt Mask with Two Binary Region Masks

## Status

**DONE**

## Summary

Restored `build_distill_token_mask()` to response-only behavior (3 keyword args, no `prompt_kd_weight`). Added frozen `DistillTokenRegions` dataclass and `build_distill_token_regions()` returning separate binary `response_mask` and `prompt_mask`. Removed fractional prompt-weight tests from `test_distill_losses.py`; kept independent fractional-mask reducer tests (EAKLD telemetry, top-k, CPU teacher paths).

## TDD Evidence

### RED

Added 5 new region tests, then ran one representative test before implementation:

```bash
conda activate bitvae
cd /home/shaoyuantian/program/VAELLM
PYTHONPATH=. pytest tests/test_distill_losses.py::test_distill_regions_single_turn_splits_prompt_and_response -q
```

Result:

```
AttributeError: module 'train_utils.distill_losses' has no attribute 'build_distill_token_regions'
```

### GREEN

Implemented `DistillTokenRegions`, restored `build_distill_token_mask()`, added `build_distill_token_regions()`, then ran full suite:

```bash
PYTHONPATH=. pytest tests/test_distill_losses.py -q
```

Result:

```
46 passed in 5.48s
```

## Files Changed

### `train_utils/distill_losses.py`

- Added `@dataclass(frozen=True) class DistillTokenRegions` with `response_mask` and `prompt_mask`.
- Added helpers `_validate_distill_mask_shape()` and `_apply_causal_shift()`.
- Restored `build_distill_token_mask(*, labels, attention_mask, reference_logits)`:
  - Removed `prompt_kd_weight` parameter and all fractional weighting logic.
  - Labels present: `labels != -100`.
  - Labels absent + attention present: attention validity.
  - Neither: all-ones source weights.
  - Causal left shift; final position zero.
- Added `build_distill_token_regions(*, labels, attention_mask, reference_logits)`:
  - `response_mask` reuses `build_distill_token_mask()`.
  - `prompt_mask`: `labels == -100`, AND attention validity when attention exists, causal shift, final zero; all-zero when labels absent.

### `tests/test_distill_losses.py`

**Added:**

- `test_distill_regions_single_turn_splits_prompt_and_response`
- `test_distill_regions_padding_excludes_prompt_and_response`
- `test_distill_regions_interleaved_prompt_and_response`
- `test_distill_regions_labels_none_keeps_response_and_zero_prompt`
- `test_distill_regions_masks_are_binary_disjoint_with_zero_final`

**Removed** (fractional prompt-weight contract on `build_distill_token_mask`):

- `test_distill_mask_prompt_weight_zero_is_exact_current_behavior`
- `test_distill_mask_assigns_fractional_prompt_weights_after_causal_shift`
- `test_distill_mask_padding_excludes_prompt_weight`
- `test_distill_mask_prompt_weight_one_equals_shifted_attention_validity`
- `test_distill_mask_interleaved_prompt_tokens_use_prompt_weight`
- `test_distill_mask_labels_none_ignores_prompt_kd_weight`
- `test_distill_mask_rejects_negative_prompt_kd_weight`
- `test_distill_mask_accepts_prompt_kd_weight_above_one`
- `test_forward_kl_gradient_respects_fractional_prompt_weights`
- `test_forward_kl_loss_matches_manual_fractional_weighted_mean`

**Kept** independent fractional-mask reducer tests (e.g. `test_eakld_teacher_entropy_gamma_uses_fractional_mask`, `test_eakld_topk_fractional_mask_matches_dense_output_and_gradient`, CPU teacher fractional-mask tests).

## Self-Review

| Requirement | Met? | Notes |
|---|---|---|
| `build_distill_token_mask` 3 args only | Yes | No `prompt_kd_weight`; no fractional output |
| `DistillTokenRegions` frozen dataclass | Yes | Two tensor fields |
| `build_distill_token_regions` keyword-only | Yes | Returns `DistillTokenRegions` |
| Single-turn mask values | Yes | response `[0,0,1,1,1,0]`, prompt `[1,1,0,0,0,0]` |
| Padding mask values | Yes | response `[0,1,1,0,0,0]`, prompt `[1,0,0,0,0,0]` |
| Interleaved mask values | Yes | response `[1,0,1,1,0]`, prompt `[0,1,0,0,0]` |
| Labels-none fallback | Yes | response = existing mask helper; prompt all zero |
| Invariants (binary, disjoint, final zero) | Yes | Tested |
| No fractional prompt mask in production path | Yes | Removed from `build_distill_token_mask` |
| Only allowed files modified | Yes | Two files only |
| No git commit | Yes | Worktree only |

## Concerns

1. **Downstream callers still pass `prompt_kd_weight`** to `build_distill_token_mask()` in `train_utils/lora_training.py`, `compressed_e2e_fintuning/trainer.py`, and smoke tests (`tests/smoke/test_loss_pipeline_smoke.py`). These are out of scope for Task 1 but will raise `TypeError` until later tasks wire `build_distill_token_regions()` and region-normalized weighting. Intentional per plan sequencing.

2. **`_validate_distill_mask_shape` has unused `tensor_name` parameter** — kept for readability; no functional impact.

## Test Summary

`PYTHONPATH=. pytest tests/test_distill_losses.py -q` → **46 passed**

## Commits

None (per instructions).
