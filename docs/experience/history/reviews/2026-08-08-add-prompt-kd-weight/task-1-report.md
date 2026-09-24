> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-1-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 1 Report: Add Weighted-Mask Regression Tests First

**Status:** DONE  
**Date:** 2026-08-10  
**Scope:** Tests only — no production code modified.

---

## Summary

Added 10 regression tests to `tests/test_distill_losses.py` covering prompt-weighted causal mask semantics for Task 2 implementation. All new tests fail against current `build_distill_token_mask()` (no `prompt_kd_weight` parameter). All 32 pre-existing tests in the file still pass.

---

## Files Changed

| File | Change |
|------|--------|
| `tests/test_distill_losses.py` | Added 10 new test functions (lines ~559–843) |

**Not modified:** `train_utils/distill_losses.py` or any other production code.

---

## Tests Added

| # | Test name | Covers |
|---|-----------|--------|
| 1 | `test_distill_mask_prompt_weight_zero_is_exact_current_behavior` | Omitting `prompt_kd_weight` vs explicit `0.0` must `torch.equal`; both match legacy `[0,0,1,1,1,0]` |
| 2 | `test_distill_mask_assigns_fractional_prompt_weights_after_causal_shift` | `[-100,-100,-100,A,B,EOS]` + `0.1` → `[0.1,0.1,1,1,1,0]` |
| 3 | `test_distill_mask_padding_excludes_prompt_weight` | Padding positions (`attention_mask==0`) get weight 0 even when `labels==-100` |
| 4 | `test_distill_mask_prompt_weight_one_equals_shifted_attention_validity` | `prompt_kd_weight=1.0` equals shifted attention validity; last logit 0 |
| 5 | `test_distill_mask_interleaved_prompt_tokens_use_prompt_weight` | Mid-sequence `-100` context tokens receive prompt weight |
| 6 | `test_distill_mask_labels_none_ignores_prompt_kd_weight` | Attention-only and no-metadata fallbacks ignore `prompt_kd_weight` |
| 7 | `test_distill_mask_rejects_negative_prompt_kd_weight` | `< 0` raises `ValueError` matching `prompt_kd_weight` |
| 8 | `test_distill_mask_accepts_prompt_kd_weight_above_one` | `2.0` accepted; prompt positions show `2.0` in mask |
| 9 | `test_forward_kl_gradient_respects_fractional_prompt_weights` | p=0 prompt grad 0; p>0 prompt grad non-zero; response non-zero; padding/last 0 |
| 10 | `test_forward_kl_loss_matches_manual_fractional_weighted_mean` | Manual weighted forward KL equals `compute_forward_kl_loss` with fractional mask |

---

## TDD Evidence (RED)

### Environment

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate bitvae
which python   # /home/shaoyuantian/anaconda3/envs/bitvae/bin/python
python -V      # Python 3.11.13
cd /home/shaoyuantian/program/VAELLM
PYTHONPATH=. pytest tests/test_distill_losses.py -q
```

### Result

```
10 failed, 32 passed in 11.53s
```

### Why failure is expected

Current `build_distill_token_mask()` signature:

```python
def build_distill_token_mask(
    *,
    labels: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    reference_logits: torch.Tensor,
) -> torch.Tensor:
```

All 10 new tests pass `prompt_kd_weight=...`, which raises:

```
TypeError: build_distill_token_mask() got an unexpected keyword argument 'prompt_kd_weight'
```

This is the correct RED state: tests specify the API Task 2 must implement. Once `prompt_kd_weight: float = 0.0` is added with weighted-mask logic, these tests should turn green without changing test code (except possibly the negative-weight test if error message differs slightly).

### Existing tests

All 32 pre-existing tests pass unchanged, confirming no regression in current binary-mask behavior.

---

## Self-Review Checklist

- [x] All 10 test categories from task brief implemented
- [x] Test names follow existing `test_distill_mask_*` / `test_forward_kl_*` conventions
- [x] Exact numeric expectations from plan (`[0.1,0.1,1,1,1,0]`, padding example, etc.)
- [x] No production code modified
- [x] No git commit
- [x] pytest RED confirmed

---

## Concerns

None. Negative-weight test currently fails with `TypeError` instead of `ValueError`; this will resolve when Task 2 adds parameter validation.

---

## Next Step (Task 2)

Implement `prompt_kd_weight: float = 0.0` in `train_utils/distill_losses.py` per plan semantics; re-run `PYTHONPATH=. pytest tests/test_distill_losses.py -q` until all 42 tests pass.
