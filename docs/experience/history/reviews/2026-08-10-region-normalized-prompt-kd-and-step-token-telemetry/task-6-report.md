> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-6-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 6 Report: Shared Logging-Window Token Statistics Accumulator

## Status

**Complete.** New module and tests implemented per brief; all required tests pass.

## Files Created

| File | Purpose |
|------|---------|
| `train_utils/distill_token_stats.py` | `DistillWindowTokenStats` dataclass and `DistillTokenStatsAccumulator` |
| `tests/test_distill_token_stats.py` | TDD test suite covering all brief requirements |

## Public API

- **`DistillWindowTokenStats`** (frozen dataclass): `avg_prompt_tokens_per_sample`, `avg_response_tokens_per_sample`, `global_samples`
- **`DistillTokenStatsAccumulator`**: `update(labels, attention_mask)`, `consume_global(accelerator)`

## Implementation Notes

- Token regions: `valid = attention != 0` (or all-true if `attention_mask is None`); prompt = valid ∧ labels == -100; response = valid ∧ labels != -100. No causal shift.
- `update()` accumulates a detached float32 3-vector `[prompt_total, response_total, sample_count]` on the labels device; no per-micro-batch Python sync.
- `consume_global()` always participates in `accelerator.reduce(..., reduction="sum")`; ranks with no local updates pass a zero tensor on `accelerator.device`. Returns `None` only when reduced global sample count is zero. State resets after every consume (successful or empty window).

## Tests

```
PYTHONPATH=. pytest tests/test_distill_token_stats.py -q
10 passed in 2.02s
```

Coverage per brief checklist:

- [x] Single-sample `[-100,-100,-100,A,B,EOS]` → prompt=3, response=3 (no causal shift)
- [x] Padding: label -100 with attention 0 excluded from prompt
- [x] Asymmetric micro-batches use global weighted average (not average-of-averages)
- [x] Ten optimizer-step updates accumulate before single consume
- [x] Fake accelerator proves distributed global totals used
- [x] No-local-update rank still reduces and returns global averages
- [x] Second consume without updates returns `None`
- [x] Invalid labels rank / shape / device mismatch rejected

## Out of Scope (Tasks 7–8)

Trainer wiring not performed in this task.

## Commits

None (per task instructions).
