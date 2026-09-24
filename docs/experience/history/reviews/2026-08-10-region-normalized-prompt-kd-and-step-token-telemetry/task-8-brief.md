> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-8-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 8: E2E Uses the Same Logging-Window Token Telemetry

**Files:** `compressed_e2e_fintuning/trainer.py`, `compressed_e2e_fintuning/runtime.py`, `tests/test_e2e_dataset_mix.py`, `tests/smoke/test_one_step_train_smoke.py`

- [ ] Initialize the same `DistillTokenStatsAccumulator` on `VAEDecoderE2ETrainer`.
- [ ] In top-level `VAEDecoderE2ETrainer.compute_loss()`, before dispatching to choice/dense/CPU paths, record telemetry exactly once from original `labels` and `attention_mask` when `model.training` is true and labels are present. Do not record in `_compute_legacy_dense_loss()`, `_compute_teacher_first_cpu_loss()`, or `_build_cpu_teacher_targets()`; this prevents dense/offload/teacher double counting.
- [ ] For `choice_input_ids` / MCQA requests where ordinary token `labels` are unavailable, skip token-window telemetry explicitly; do not fabricate prompt/response counts.
- [ ] Add `E2EDistillTokenStatsCallback` with exactly the same regular-boundary condition, `window_start_step` handling, reduce-before-rank0 ordering, and resume semantics as category distillation.
- [ ] Register the E2E callback in `runtime.py` immediately after trainer construction, passing the existing E2E logger.
- [ ] Fixed prefix is `E2E token stats`; fields and numeric precision match category output.
- [ ] Do not modify E2E loss logging cadence or `logging_first_step` behavior.
- [ ] With logging cadence 10, dense smoke proves one token line at step 10 covering steps 1-10 and no token line at the special step-1 loss log.
- [ ] CPU-offload smoke proves the same token totals as dense for identical batches and proves teacher-target construction adds zero extra counts.
- [ ] Resume and DDP tests reuse the same accumulator/callback contract as category distillation.

---

