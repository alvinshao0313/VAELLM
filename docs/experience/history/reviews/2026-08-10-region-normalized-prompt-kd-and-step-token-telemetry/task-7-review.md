> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-7-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 7 Review: Category Logging-Window Token Telemetry

## Spec Compliance: ✅

All brief checklist items satisfied:

- `DistillTokenStatsAccumulator` initialized on `CustomSFTTrainer` as `distill_token_stats` (`lora_training.py:578`).
- `compute_loss` top-of-function update reads original `inputs` labels/attention before any `teacher_inputs`/`student_inputs`/`full_inputs` mutation, guarded by `model.training` (`lora_training.py:607-612`). Exactly one update path; teacher/hidden/CPU-staging/callback paths never call `update`.
- `_LoraDistillTokenStatsCallback` constructor takes `trainer` + `logger`, `window_start_step=None` (`lora_utils.py:130-133`).
- `on_step_end` reads resolved `state.logging_steps`, requires positive int (raises otherwise); lazy `window_start_step` init on first observed step; `global_step<=0 or %cadence!=0` early return; all-rank `consume_global`; `window_optimizer_steps = global_step - window_start_step + 1`; advance `window_start_step`; empty-stats return after collective; rank0-only write via `_log_lora_trainer_message_to_file_handlers` (`lora_utils.py:135-167`).
- Output line format matches: `LoRA token stats: step=%s window_optimizer_steps=%d avg_prompt_tokens=%.4f avg_response_tokens=%.4f global_samples=%d`.
- Registration only on `CustomSFTTrainer` branch, after `_replace_progress_log_callback` (`lora_utils.py:670-673`).
- `logging_steps=max(1, int(cfg.log_every))` and `logging_first_step=True` untouched.
- Tests: window 1-10, cadence resolution (raw ratio vs resolved state), second window 11-20, grad-accum factor 2, resume from step 7→8, nonzero rank consume/no-write, `model.training=False` guard — all present in `tests/test_lora_distill_token_stats_callback.py`.

## Quality

High. Update site correctly uses pre-mutation `inputs`; collective ordering (consume before rank check) is right; `window_start_step` advance happens even on empty stats; test fakes mirror real HF Trainer callback contracts (`state.logging_steps` vs `args.logging_steps`); smoke + adapter tests updated to set `distill_token_stats` and `add_callback`.

## Findings

1. **Minor ordering edge case (non-blocking):** `window_start_step` is initialized before the `global_step <= 0` early-return. If `on_step_end` ever fired with `global_step=0`, `window_start_step` would lock to 0 and the next boundary would report `window_optimizer_steps = global_step + 1` (off by one). Unreachable in real HF Trainer (`on_step_end` fires post-increment at step ≥1) and the brief's prescribed order is (1) cadence, (2) init, (3) boundary check — implementation follows that order, so this is spec-conformant, just worth noting.
2. **Test white-box access:** `test_window_one_to_ten...` asserts `trainer.distill_token_stats._accumulator is not None` (private attr). Acceptable for white-box but fragile if accumulator internals change.
3. **DDP reduce arithmetic untested:** `_FakeAccelerator.reduce` returns `tensor.clone()` without true cross-rank summation; tests verify collective *participation* (call count) but not summed arithmetic across ranks. Matches brief's "all ranks consume / only rank0 writes" requirement; a true multi-rank reduce-sum test would be stronger but isn't required.
4. **No explicit "update exactly once per compute_loss" assertion:** grad-accum test indirectly proves it (20 samples for 20 micro-batches), but no direct call-count assertion on `distill_token_stats.update`.

## ⚠️

No blocking issues. Spec fully met; findings are minor hardening opportunities, not defects. Approve.
