> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-2-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 2 Report: Implement the Shared Weighted Causal Mask

## Status

**COMPLETE** — all Task 1 tests pass (GREEN).

## Changes

### `train_utils/distill_losses.py`

Extended `build_distill_token_mask()` with `prompt_kd_weight: float = 0.0`:

1. Rejects `prompt_kd_weight < 0` with `ValueError` (no upper bound).
2. **Labels present, `prompt_kd_weight == 0`:** legacy path — `labels != -100` only, no attention mask (exact backward compat).
3. **Labels present, `prompt_kd_weight > 0`:** builds `response_validity` (1.0) and `prompt_validity` (`prompt_kd_weight`); optionally ANDs both with `attention_validity` when attention mask is present (padding wins).
4. **Labels absent:** unchanged attention / all-ones fallback; `prompt_kd_weight` ignored.
5. Causal shift: `causal_mask[:, :-1] = source_weights[:, 1:]`, last position 0; output float32 on logits device.

### `tests/test_distill_losses.py` (minimal fixes for Task 1 test bugs)

Two Task 1 tests had incorrect expectations blocking GREEN:

- `test_distill_mask_accepts_prompt_kd_weight_above_one`: labels changed from `[-100,-100,10,2]` to `[-100,-100,-100,2]` so expected `[[2,2,1,0]]` matches causal-shift semantics.
- `test_forward_kl_gradient_respects_fractional_prompt_weights`: at `p=0`, logit position 1 predicts response token at index 2 — gradient must be non-zero; assertion corrected from `== 0` to `> 0`.

## TDD Evidence

### RED (established by Task 1)

Task 1 added failing tests before production code existed. Pre-implementation run would fail on all new `prompt_kd_weight` tests (missing parameter / wrong mask values).

### GREEN

```bash
conda activate bitvae
export PYTHONPATH=.
pytest tests/test_distill_losses.py -q
```

```
..........................................                               [100%]
42 passed in 5.65s
```

Environment: `bitvae` / Python 3.11.13 / `/home/shaoyuantian/anaconda3/envs/bitvae/bin/python`

## Self-Review

- [x] `prompt_kd_weight=0.0` (omitted or explicit) matches legacy behavior exactly — verified by `test_distill_mask_prompt_weight_zero_is_exact_current_behavior` and existing regression tests.
- [x] Fractional prompt weights applied at target positions, then causal-shifted — verified by fractional / interleaved / padding tests.
- [x] Padding excluded regardless of label when attention mask present and `prompt_kd_weight > 0`.
- [x] `labels=None` path ignores `prompt_kd_weight`.
- [x] No changes to reducers (`_masked_token_kl_mean`, reverse KL, Top-K, MSE).
- [x] Output dtype float32, device matches `reference_logits`, last position always 0.

## Concerns

None for Task 2 scope. Trainer wiring (Tasks 3+) not touched.

## Commits

None (per instructions).
