> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-8-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 8 Report: E2E Uses the Same Logging-Window Token Telemetry

## Status

Implemented and verified. All focused tests pass; no regressions in related suites.

## Commits

None (per instructions).

## Changes

### `compressed_e2e_fintuning/trainer.py`
- Imported `DistillTokenStatsAccumulator` from `train_utils.distill_token_stats`.
- Initialized `self.distill_token_stats = DistillTokenStatsAccumulator()` in `VAEDecoderE2ETrainer.__init__` (alongside the existing EAKLD telemetry state, before `super().__init__`).
- In top-level `VAEDecoderE2ETrainer.compute_loss()`, before the `choice_input_ids` / dense / CPU dispatch, record telemetry exactly once when `model.training` is true and `inputs["labels"]` is a tensor. MCQA requests expose no ordinary rank-2 `labels`, so they are skipped explicitly (no fabricated counts). `_compute_legacy_dense_loss`, `_compute_teacher_first_cpu_loss`, and `_build_cpu_teacher_targets` were left untouched, so dense/offload/teacher-target paths cannot double-count.
- Added `E2EDistillTokenStatsCallback` mirroring `_LoraDistillTokenStatsCallback`: same `state.logging_steps` validation, `window_start_step` handling, `global_step % logging_steps == 0` boundary, **reduce-before-rank0** ordering (`consume_global` runs on every rank before the `is_world_process_zero` write check), and resume semantics. Fixed prefix is `E2E token stats`; fields and `%.4f` / `%d` precision match the category output.
- Added `_log_e2e_trainer_message_to_file_handlers` helper (same FileHandler-only emission pattern as the LoRA helper) so the callback writes only to the run log file, matching `E2ETrainerLogCallback`.

### `compressed_e2e_fintuning/runtime.py`
- Imported `E2EDistillTokenStatsCallback`.
- Registered the callback via `trainer.add_callback(E2EDistillTokenStatsCallback(trainer=trainer, logger=log))` immediately after `replace_progress_log_callback(trainer)` and before `eval_after_save_callback.bind_trainer(trainer)`, passing the existing E2E `log` logger.

### `tests/test_e2e_distill_token_stats.py` (new focused file)
Covers the full contract:
- Callback boundary / window / cadence / resume / nonzero-rank semantics (5 tests mirroring the LoRA callback tests, with prefix `E2E token stats`).
- `compute_loss` dense path updates telemetry exactly once (prompt=2, response=6, samples=1).
- CPU-offload path produces the same totals as dense for identical batches.
- `_build_cpu_teacher_targets` adds zero extra counts (monkeypatched `update` counter asserts exactly one call).
- MCQA `choice_input_ids` inputs skip telemetry (`consume_global` returns `None`).
- `model.eval()` skips telemetry.
- Dense smoke with cadence 10 + `logging_first_step=True`: exactly one token line at step 10 covering steps 1-10, no token line at the special step-1 loss log.

## Test Summary

```
tests/test_e2e_distill_token_stats.py: 11 passed
tests/test_e2e_teacher_first.py + tests/smoke/test_one_step_train_smoke.py +
tests/test_lora_distill_token_stats_callback.py + tests/test_distill_token_stats.py: 36 passed
```

Pre-existing, unrelated failures in `tests/test_e2e_dataset_mix.py` (10 failed) were confirmed not caused by this change:
- `DatasetMixBuilderTest` / `DatasetMixArgsTest::test_parse_args_eval_before_save_requires_tasks_and_save_steps` fail with `FileNotFoundError: dummy.txt` (missing fixture file in the working tree).
- `VAEE2ETrainerPromptKdMaskHelperTest::test_private_helper_forwards_prompt_kd_weight` patches `compressed_e2e_fintuning.trainer.build_distill_token_mask`, a symbol that does not exist in the current trainer (it uses `_build_distill_token_regions`); this test predates the current trainer API.
- The unaffected subsets of `test_e2e_dataset_mix.py` (args, hidden-loss, distill-data: 27 tests) pass.

## Constraints Honored
- conda `bitvae` activated for all commands.
- No git commit performed.
- E2E loss logging cadence and `logging_first_step` behavior left unchanged.
- No telemetry recorded inside `_compute_legacy_dense_loss`, `_compute_teacher_first_cpu_loss`, or `_build_cpu_teacher_targets`.

## Concerns
- The pre-existing `test_e2e_dataset_mix.py` failures should be triaged separately (stale `dummy.txt` fixture and a `build_distill_token_mask` test that no longer matches the trainer API). They are out of scope for Task 8.

---

## Round 1/5 Fix: Prompt-KD Mask Helper Test Contract

### Finding
`tests/test_e2e_dataset_mix.py::VAEE2ETrainerPromptKdMaskHelperTest::test_private_helper_forwards_prompt_kd_weight` patched the obsolete `build_distill_token_mask` / `_build_distill_token_mask` API and expected a `prompt_kd_weight` argument. Production now uses `_build_distill_token_regions` returning `DistillTokenRegions` and does NOT pass `prompt_kd_weight` into mask/region construction.

### Fix
Renamed to `test_private_helper_forwards_regions_without_prompt_kd_weight` and updated to the new contract:
- Calls `trainer._build_distill_token_regions(inputs, reference_logits)`.
- Patches `compressed_e2e_fintuning.trainer.build_distill_token_regions` and returns a sentinel `DistillTokenRegions`.
- Asserts the helper forwards exactly `labels`, `attention_mask`, `reference_logits` — and asserts `prompt_kd_weight` is NOT present in the forwarded kwargs.
- Verifies the returned `response_mask` / `prompt_mask` are the sentinel tensors.

Unrelated pre-existing failures in the same file (`dummy.txt`, Weighted lazy mix `text_format`, `eval_before_save` HfArgumentParser) were left untouched per instructions.

### Commands & Output

```
$ PYTHONPATH=. pytest tests/test_e2e_dataset_mix.py::VAEE2ETrainerPromptKdMaskHelperTest -q
.                                                                        [100%]
1 passed in 5.16s

$ PYTHONPATH=. pytest tests/test_e2e_distill_token_stats.py -q
...........                                                              [100%]
11 passed in 6.43s
```

### Commits
None.
