> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-7-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 7: Category Distillation Emits One Token Average Per Regular Logging Window

**Files:** `train_utils/lora_training.py`, `train_utils/lora_utils.py`, `tests/test_cat_eval_adapter_match.py`, `tests/smoke/test_one_step_train_smoke.py`

Trainer-side requirements:
- [ ] Initialize one `DistillTokenStatsAccumulator` on `CustomSFTTrainer` as attribute `distill_token_stats`.
- [ ] At the top of `CustomSFTTrainer.compute_loss()`, after reading the original input `labels` and only when `model.training` is true, call `distill_token_stats.update(labels, attention_mask)` exactly once for that student micro-batch. Do not derive telemetry from response/prompt causal region masks.
- [ ] Never update from teacher forward, hidden-loss computation, CPU teacher-logit staging, checkpoint recomputation, or callbacks.

Add class `_LoraDistillTokenStatsCallback` in `train_utils/lora_utils.py`. Its constructor receives the concrete trainer object and the existing logger and initializes `window_start_step=None`. Do not rely on `on_train_begin` ordering for resume state.

The callback `on_step_end` control flow is mandatory:
1. Read the Trainer-resolved absolute cadence from `state.logging_steps`; require it to be a positive integer. Do not derive the boundary from raw `args.logging_steps`, because Transformers may represent that argument as a ratio before TrainerState resolves it.
2. On the first observed `on_step_end` only, if `window_start_step is None`, set `window_start_step = state.global_step`. Thus a fresh run starts at step 1, while a run resumed from step 7 starts its observed telemetry window at step 8 without depending on callback initialization order.
3. If `state.global_step <= 0` or `state.global_step % state.logging_steps != 0`, return immediately without collective, consume, reset, or output. Therefore, when the resolved cadence is greater than 1, the existing `logging_first_step=True` event at step 1 does not split the regular window. If the resolved cadence itself is 1, step 1 is a real regular boundary and telemetry is emitted normally.
4. At a regular boundary, every rank calls `trainer.distill_token_stats.consume_global(trainer.accelerator)`.
5. Compute `window_optimizer_steps = state.global_step - window_start_step + 1`, then set the next `window_start_step = state.global_step + 1` after the consume.
6. If global statistics are empty, return after all ranks completed the collective.
7. Only then inspect `state.is_world_process_zero`; rank zero writes one line using `_log_lora_trainer_message_to_file_handlers()`.

The output line must contain fixed prefix `LoRA token stats`, current global step, integer `window_optimizer_steps`, prompt-token average with four decimals, response-token average with four decimals, and integer `global_samples`.

- [ ] Register this callback only for `CustomSFTTrainer`, after `_replace_progress_log_callback()` returns the final trainer instance.
- [ ] Keep `logging_steps=max(1, int(cfg.log_every))` and `logging_first_step=True` unchanged; existing loss/grad-norm cadence must not change.
- [ ] With resolved `state.logging_steps=10`, test steps 1-9 produce no token-statistics line/consume; step 10 produces exactly one line covering steps 1-10. The special step-1 loss log must not consume/reset the accumulator.
- [ ] Add a cadence-resolution test where raw `args.logging_steps` is a non-integer ratio but `state.logging_steps=10`; boundary decisions must follow the resolved state value, not the raw argument.
- [ ] A second window test proves step 20 reports only steps 11-20, not cumulative steps 1-20.
- [ ] Gradient-accumulation factor 2 test proves both micro-batches from every optimizer step are included in the ten-step window sample/token totals.
- [ ] Resume test initializes at a non-boundary global step and reports the correct observed partial first window length rather than claiming 10 steps.
- [ ] Add a callback-ordering test where a nonzero rank still invokes consume/reduce at the boundary but does not write the line.

---

