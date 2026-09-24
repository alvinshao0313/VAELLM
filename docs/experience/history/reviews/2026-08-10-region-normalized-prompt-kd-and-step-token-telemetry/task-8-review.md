> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-8-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 8 Review — E2E Logging-Window Token Telemetry

## Spec ✅

All brief items satisfied:
- `DistillTokenStatsAccumulator` initialized on `VAEDecoderE2ETrainer.__init__` (`trainer.py:357`).
- Top-level `compute_loss()` records exactly once from original `labels`/`attention_mask` when `model.training` and `isinstance(labels, torch.Tensor)`, before choice/dense/CPU dispatch (`trainer.py:872-882`). No telemetry in `_compute_legacy_dense_loss` / `_compute_teacher_first_cpu_loss` / `_build_cpu_teacher_targets` — no double-count path.
- MCQA `choice_input_ids` inputs (no ordinary `labels`) skipped via the `isinstance` guard; test confirms `consume_global` returns `None`.
- `E2EDistillTokenStatsCallback` mirrors `_LoraDistillTokenStatsCallback` exactly: `state.logging_steps` validation, `window_start_step` init-on-first-call, `global_step % logging_steps == 0` boundary, reduce-before-rank0 (`consume_global` runs before the `is_world_process_zero` write check), resume semantics. Prefix `E2E token stats`; `%.4f`/`%d` precision matches category output.
- Registered in `runtime.py:1043` immediately after `replace_progress_log_callback(trainer)`, before `eval_after_save_callback.bind_trainer`, passing existing E2E `log`.
- E2E loss logging cadence / `logging_first_step` untouched.
- Dense smoke (cadence 10 + `logging_first_step=True`) yields exactly one token line at step 10 covering steps 1-10, no token line at step 1.
- CPU-offload smoke proves same totals as dense; teacher-target construction adds zero extra counts (monkeypatched `update` counter asserts exactly one call).
- 11 focused tests in `tests/test_e2e_distill_token_stats.py`; report claims 11 pass + 36 related-suite pass.

## Quality

- Faithful mirror of the LoRA callback; helper `_log_e2e_trainer_message_to_file_handlers` reuses the same FileHandler-only emission pattern as `E2ETrainerLogCallback`.
- Test coverage is thorough: boundary, cadence resolution via `state.logging_steps` (not raw `args`), second-window isolation, resume-from-non-boundary partial window, nonzero-rank reduce-but-no-write, dense once, CPU==dense, teacher-target zero-extra, MCQA skip, eval skip, cadence-10 smoke.
- No scope creep; no loss-cadence changes; no auto-commit.

## Findings

- None blocking. The MCQA skip is implemented via the `isinstance(original_labels, torch.Tensor)` guard rather than an explicit `choice_input_ids` branch; the comment at `trainer.py:874-876` documents the intent and the test verifies behavior — acceptable.

## ⚠️

- `tests/test_e2e_dataset_mix.py` has 10 pre-existing failures. The report splits them into (a) `dummy.txt` `FileNotFoundError` — truly unrelated fixture, ⚠️ not blocking; (b) `VAEE2ETrainerPromptKdMaskHelperTest::test_private_helper_forwards_prompt_kd_weight` patches `build_distill_token_mask` / calls `trainer._build_distill_token_mask`, which no longer exists (renamed to `build_distill_token_regions` / `_build_distill_token_regions`). That rename appears in this task's review-package diff, so the test breakage may be attributable to changes bundled with Task 8 rather than purely pre-existing. Per the brief, stale-test fixes in `test_e2e_dataset_mix.py` are in scope when caused by API changes in the file list. Recommend confirming whether the rename landed in an earlier task of the same SDD plan (pre-existing) or in Task 8 itself; if the latter, the stale helper test should be updated here. Not blocking token-telemetry deliverables, but should be triaged.
