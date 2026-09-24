> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-6-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 6: Create Shared Logging-Window Token Statistics Accumulator

**Files:** create `train_utils/distill_token_stats.py`, create `tests/test_distill_token_stats.py`

Create two exact public types in the new module:
- frozen dataclass `DistillWindowTokenStats` with fields `avg_prompt_tokens_per_sample: float`, `avg_response_tokens_per_sample: float`, and `global_samples: int`;
- class `DistillTokenStatsAccumulator` with methods `update(labels, attention_mask)` and `consume_global(accelerator)`.

Implementation contract:
- `update()` requires rank-2 `labels`; `attention_mask` may be `None` or a rank-2 tensor with identical shape/device. If attention is absent, every label position is considered non-padding.
- Define `valid = attention_mask != 0` when attention exists, otherwise all-true. Define response tokens as `valid AND labels != -100`; define prompt/context tokens as `valid AND labels == -100`.
- EOS is counted as response when its label is supervised. Padding with label `-100` is excluded by attention. Do not causal-shift these counts.
- `update()` accumulates one detached three-value device tensor containing prompt-token count, response-token count, and sample count.
- `update()` must not convert device counts to Python numbers; avoid per-micro-batch synchronization.
- All updates remain accumulated until the regular logging boundary consumes them. This naturally includes all gradient-accumulation micro-batches and all optimizer steps in the logging window.
- `consume_global()` uses the Trainer accelerator to perform distributed sum reduction on every rank.
- If a rank has no local update in the current logging window, it must still create a zero three-value tensor on the accelerator device and participate in the collective; it must not return before reduction. Only when the reduced global sample count is zero may the method return no statistics. This prevents DDP deadlock.
- Use float32 for the local three-value tensor; window token/sample counts remain within exact float32 integer range for these training settings.
- Only after reduction may totals be converted to Python values and divided by global sample count.
- State resets only after a successful window consume; it must not reset on ordinary optimizer steps or the special step-1 loss log.

Required tests:
- [ ] For `labels=[-100,-100,-100,A,B,EOS]` and all-one attention, one sample reports prompt=3 and response=3. This explicitly proves telemetry is not using causal KD-mask counts.
- [ ] Padding test proves `labels=-100` with attention zero is not counted as prompt.
- [ ] Two asymmetric micro-batches are summed before division; choose values for which average-of-microbatch-averages would be wrong.
- [ ] Ten synthetic optimizer-step worth of updates can accumulate without consume and produce the correct total-window per-sample means when consumed once.
- [ ] A fake accelerator proves distributed global totals, not rank-local totals, are used.
- [ ] Add a no-local-update distributed test: local accumulator state is empty, but fake/global reduction returns nonzero totals from other ranks; `consume_global()` must still perform reduction and return the global averages rather than returning early.
- [ ] A second consume without new updates in a globally empty window returns no statistics and cannot repeat previous values.
- [ ] Invalid labels rank or labels/attention shape/device mismatch is rejected.
- [ ] Run `pytest tests/test_distill_token_stats.py -q`.

---

