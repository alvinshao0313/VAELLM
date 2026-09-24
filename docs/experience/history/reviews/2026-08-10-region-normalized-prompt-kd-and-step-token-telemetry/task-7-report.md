> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-7-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 7 Report: Category Distillation Emits One Token Average Per Regular Logging Window

## Status

**Complete.** Wired `DistillTokenStatsAccumulator` into `CustomSFTTrainer` and added `_LoraDistillTokenStatsCallback` in `lora_utils`. All required tests pass.

## Commits

None (per task instructions).

## Files Changed

| File | Change |
|------|--------|
| `train_utils/lora_training.py` | Import `DistillTokenStatsAccumulator`; initialize `self.distill_token_stats` in `CustomSFTTrainer.__init__`; call `distill_token_stats.update(original_labels, attention_mask)` once at the top of `compute_loss` when `model.training` is true, using the original `inputs` labels/attention (never teacher/hidden/CPU staging). |
| `train_utils/lora_utils.py` | Added `_LoraDistillTokenStatsCallback` implementing the mandatory `on_step_end` control flow (cadence from `state.logging_steps`, first-observed `window_start_step` init, boundary check, all-rank `consume_global`, window length, empty-stats return, rank-zero-only line write). Registered via `trainer.add_callback(...)` only on the `CustomSFTTrainer` branch, after `_replace_progress_log_callback` returns. |
| `tests/test_cat_eval_adapter_match.py` | Set `trainer.distill_token_stats` on both `__new__`-based test helpers; added `add_callback` to `FakeCustomTrainer` so the selection test still exercises the registration path. |
| `tests/smoke/test_one_step_train_smoke.py` | Set `trainer.distill_token_stats` on the `__new__`-based smoke helper. |
| `tests/test_lora_distill_token_stats_callback.py` | New 7-test suite covering window 1-10, cadence resolution, second window, resume, grad-accum factor 2, nonzero rank, and `model.training` guard. |

## Implementation Notes

- `compute_loss` update reads from the original `inputs` dict (before `teacher_inputs`/`student_inputs`/`full_inputs` mutations), so telemetry never derives from response/prompt causal region masks or teacher forwards.
- Callback reads the Trainer-resolved absolute cadence from `state.logging_steps` and requires it to be a positive integer; raw `args.logging_steps` (which may be a ratio) is never used for boundary decisions.
- `window_start_step` is initialized lazily on the first observed `on_step_end` (`if is None`), so a run resumed from step 7 starts its observed window at step 8 without depending on callback init order.
- The `logging_first_step=True` step-1 event is an `on_log`, not `on_step_end`; with cadence > 1, `1 % cadence != 0` returns before consume, so the accumulator is not split/reset.
- `consume_global` is invoked on every rank at a boundary (collective), then rank-zero-only writes the line via `_log_lora_trainer_message_to_file_handlers`.
- Output line format: `LoRA token stats: step=<s> window_optimizer_steps=<d> avg_prompt_tokens=<.4f> avg_response_tokens=<.4f> global_samples=<d>`.
- `logging_steps=max(1, int(cfg.log_every))` and `logging_first_step=True` left unchanged.

## Tests

```
PYTHONPATH=. pytest tests/test_lora_distill_token_stats_callback.py \
  tests/test_cat_eval_adapter_match.py tests/test_distill_token_stats.py \
  tests/smoke/test_one_step_train_smoke.py -q
50 passed in 5.45s
```

New test coverage per brief checklist:

- [x] Window 1-10: steps 1-9 no consume/line; step 10 one line covering steps 1-10; step-1 special log does not consume/reset
- [x] Cadence resolution: raw `args.logging_steps=0.1` ratio but `state.logging_steps=10` → boundary follows resolved state
- [x] Second window: step 20 reports `window_optimizer_steps=10` covering only steps 11-20
- [x] Gradient-accumulation factor 2: two `compute_loss` calls per optimizer step both accumulated; 10 steps → 20 samples
- [x] Resume from step 7: first observed step 8, boundary at step 10 reports `window_optimizer_steps=3`
- [x] Nonzero rank: `consume_global` invoked (reduce called) but no line written
- [x] `model.training=False` guard: no update when model is in eval mode

## Concerns

None. The callback registration relies on `trainer.add_callback` (standard HF Trainer API); the existing selection test's fake trainer was updated to mirror that interface.
