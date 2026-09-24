> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.ai-bridge/CAT_DISTILL_FINAL_AUDIT.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../README.md) 查阅。

# CAT Distill Final Audit

## Modified files

- docs/cat_train_args.md
- docs/catlora_distill_from_checkpoint.md
- scripts/catlora_distill_4gpu_res0.sh
- scripts/catlora_distill_from_checkpoint.sh
- scripts/catlora_simple.sh
- scripts/catlora_simple2.sh
- tests/smoke/test_one_step_train_smoke.py
- tests/test_cat_checkpoint_residency.py
- tests/test_cat_compressed_lora_scope.py
- tests/test_cat_distill_args_contract.py
- tests/test_cat_eval_adapter_match.py
- tests/test_cat_independent_teacher.py
- tests/test_cat_inline_distributed.py
- tests/test_cat_inline_remaining_lora.py
- tests/test_cat_inline_resume_progress.py
- tests/test_cat_reference_paths.py
- tests/test_cat_remaining_decoder.py
- tests/test_cat_skip_grouping.py
- tests/test_e2e_checkpoint_io_legacy.py
- tests/test_lora_distill_token_stats_callback.py
- tests/test_remaining_lora_dataset_seed_cache.py
- tests/test_temporary_switch_residency.py
- tools/cat_eval.py
- train_utils/activation_utils.py
- train_utils/base_reference.py
- train_utils/cat_after_category_distill.py
- train_utils/cat_checkpoint_distill.py
- train_utils/cat_residual_from_base.py
- train_utils/cat_train_args.py
- train_utils/cat_train_pipeline.py
- train_utils/cat_train_residual_protection.py
- train_utils/cat_train_runtime.py
- train_utils/distill_decoder.py
- train_utils/distill_teacher.py
- train_utils/eval_utils.py
- train_utils/lora_training.py
- train_utils/lora_utils.py
- train_utils/mlp_channel_selection.py

## Fixed contracts

- resume progress
- completed-category explicit skip
- teacher_required single source of truth
- compressed newly_compressed_target_count
- resolved decoder LR metadata
- remaining joint decoder frozen VAE prewarm
- teacher_target_offload equivalence

## Focused tests

Task 0 baseline:

```bash
pytest -q \
  tests/test_cat_distill_args_contract.py \
  tests/test_cat_independent_teacher.py \
  tests/test_cat_remaining_decoder.py \
  tests/test_cat_checkpoint_residency.py \
  tests/test_cat_reference_paths.py \
  tests/test_cat_skip_grouping.py \
  tests/test_cat_inline_remaining_lora.py \
  tests/test_cat_inline_distributed.py \
  tests/test_cat_compressed_lora_scope.py \
  tests/test_cat_eval_adapter_match.py \
  tests/test_remaining_lora_dataset_seed_cache.py \
  tests/test_lora_distill_token_stats_callback.py \
  tests/test_teacher_target_offload.py
```

Result: PASS, 145 passed.

Task 1:

```bash
pytest -q \
  tests/test_cat_inline_resume_progress.py \
  tests/test_cat_inline_remaining_lora.py \
  tests/test_cat_inline_distributed.py
```

Result: PASS, 21 passed.

Task 2:

```bash
pytest -q \
  tests/test_cat_independent_teacher.py \
  tests/test_cat_distill_args_contract.py \
  tests/test_cat_eval_adapter_match.py
```

Result: PASS, 65 passed.

Task 3:

```bash
pytest -q \
  tests/test_cat_compressed_lora_scope.py \
  tests/test_cat_distill_args_contract.py \
  tests/test_cat_remaining_decoder.py
```

Result: PASS, 45 passed.

Task 4:

```bash
pytest -q \
  tests/test_cat_remaining_decoder.py \
  tests/test_cat_inline_remaining_lora.py \
  tests/test_remaining_lora_dataset_seed_cache.py
```

Result: PASS, 20 passed.

Task 5:

```bash
pytest -q tests/test_teacher_target_offload.py
```

Result: PASS, 18 passed.

Task 6:

```bash
pytest -q \
  tests/test_cat_distill_args_contract.py \
  tests/test_cat_independent_teacher.py \
  tests/test_cat_remaining_decoder.py \
  tests/test_cat_checkpoint_residency.py \
  tests/test_cat_reference_paths.py \
  tests/test_cat_skip_grouping.py \
  tests/test_cat_inline_resume_progress.py \
  tests/test_cat_inline_remaining_lora.py \
  tests/test_cat_inline_distributed.py \
  tests/test_cat_compressed_lora_scope.py \
  tests/test_cat_eval_adapter_match.py \
  tests/test_remaining_lora_dataset_seed_cache.py \
  tests/test_lora_distill_token_stats_callback.py \
  tests/test_teacher_target_offload.py \
  tests/test_distill_losses.py \
  tests/test_distill_dynamic_padding.py \
  tests/test_e2e_checkpoint_io_legacy.py
```

Result: PASS, 298 passed.

## Compile

```bash
python -m compileall -q train_utils tools tests
```

Result: PASS.

## Shell syntax

```bash
bash -n scripts/catlora_simple.sh
bash -n scripts/catlora_simple2.sh
bash -n scripts/catlora_distill_from_checkpoint.sh
bash -n scripts/catlora_distill_4gpu_res0.sh
```

Result: PASS.

## Static invariants

- PASS: `train_utils/cat_checkpoint_distill.py` has no `TemporarySwitchLinear` / `original_weight_bank`.
- PASS: `train_utils/lora_training.py` has no `set_temporary(False)` / `disable_adapter()` / `teacher_param_snapshots`.
- PASS: CAT CLI/script checked set has no `--unload_vae_original_weights_on_final_save`.
- PASS: inline VAE construction still uses `original_weight=None`, `always_use_original=False`, `protect_original_weight=False`.
- PASS: `CatResumeDistillProgress` exists.
- PASS: `load_cat_resume_distill_progress` exists and is called by inline `run_cat_train`.
- PASS: `resolve_distill_teacher_required` exists and is used by Trainer and metadata.
- PASS: `newly_compressed_target_count` is passed into `_run_compressed_category_distill`.
- PASS: compressed `decoder` / `both` metadata uses `resolved_decoder_lr`.
- PASS: remaining decoder frozen VAE prewarm helper exists and is called only when decoder targets exist.

## Full pytest

```bash
pytest -q
```

Result: FAIL with unrelated remaining failures: 906 passed, 10 failed, 2 warnings.

## Remaining unrelated failures

- test name: `mix_bit/tests/test_validation.py::test_downstream_metrics_are_not_written_into_allocation_or_objective`
- exact assertion/traceback summary: full-suite traceback enters `mix_bit/validation.py`; `_score_percent_from_row({"metric": {"acc,none": 0.5}})` raises `TypeError: float() argument must be a string or a real number, not 'dict'`.
- why unrelated: failing test is not a CAT focused test; traceback does not enter this round's modified `train_utils` CAT/teacher/decoder/resume files; fix would require changing `mix_bit/validation.py`, which the plan forbids by default.
- production files involved: `mix_bit/validation.py`.

- test name: `tests/test_e2e_dataset_mix.py::DatasetMixBuilderTest::test_build_datasets_mix_changes_with_different_seed`
- exact assertion/traceback summary: `e2e_common/lazy_datasets.py:728` raises `ValueError: Weighted lazy mix with multiple text_format values is not supported in one iterable dataset. Use a single-format mix or one source.`
- why unrelated: traceback only enters `e2e_common/data.py` and `e2e_common/lazy_datasets.py`; no modified CAT production file is involved; single reproduction is stable.
- production files involved: `e2e_common/data.py`, `e2e_common/lazy_datasets.py`.

- test name: `tests/test_e2e_dataset_mix.py::DatasetMixBuilderTest::test_build_datasets_mix_interleaves_and_resizes_sources`
- exact assertion/traceback summary: same `ValueError` from `e2e_common/lazy_datasets.py:728`.
- why unrelated: same as above; no traceback into this round's modified production files; single reproduction is stable.
- production files involved: `e2e_common/data.py`, `e2e_common/lazy_datasets.py`.

- test name: `tests/test_e2e_dataset_mix.py::DatasetMixBuilderTest::test_build_datasets_mix_is_deterministic_for_same_seed`
- exact assertion/traceback summary: same `ValueError` from `e2e_common/lazy_datasets.py:728`.
- why unrelated: same as above; no traceback into this round's modified production files; single reproduction is stable.
- production files involved: `e2e_common/data.py`, `e2e_common/lazy_datasets.py`.

- test name: `tests/test_e2e_dataset_mix.py::DatasetMixBuilderTest::test_build_datasets_mix_limits_train_preprocessing`
- exact assertion/traceback summary: same `ValueError` from `e2e_common/lazy_datasets.py:728`.
- why unrelated: same as above; no traceback into this round's modified production files; single reproduction is stable.
- production files involved: `e2e_common/data.py`, `e2e_common/lazy_datasets.py`.

- test name: `tests/test_e2e_dataset_mix.py::DatasetMixBuilderTest::test_build_datasets_mix_rejects_empty_packed_source`
- exact assertion/traceback summary: same `ValueError` from `e2e_common/lazy_datasets.py:728`.
- why unrelated: same as above; no traceback into this round's modified production files; single reproduction is stable.
- production files involved: `e2e_common/data.py`, `e2e_common/lazy_datasets.py`.

- test name: `tests/test_e2e_dataset_mix.py::DatasetMixBuilderTest::test_build_datasets_mix_repeats_short_source_to_target`
- exact assertion/traceback summary: same `ValueError` from `e2e_common/lazy_datasets.py:728`.
- why unrelated: same as above; no traceback into this round's modified production files; single reproduction is stable.
- production files involved: `e2e_common/data.py`, `e2e_common/lazy_datasets.py`.

- test name: `tests/test_e2e_dataset_mix.py::DatasetMixBuilderTest::test_build_datasets_mix_skips_eval_when_eval_strategy_is_no`
- exact assertion/traceback summary: same `ValueError` from `e2e_common/lazy_datasets.py:728`.
- why unrelated: same as above; no traceback into this round's modified production files; single reproduction is stable.
- production files involved: `e2e_common/data.py`, `e2e_common/lazy_datasets.py`.

- test name: `tests/test_e2e_dataset_mix.py::DatasetMixBuilderTest::test_build_datasets_mix_supports_long_sources_without_eval`
- exact assertion/traceback summary: same `ValueError` from `e2e_common/lazy_datasets.py:728`.
- why unrelated: same as above; no traceback into this round's modified production files; single reproduction is stable.
- production files involved: `e2e_common/data.py`, `e2e_common/lazy_datasets.py`.

- test name: `tests/test_e2e_dataset_mix.py::DistillDataTest::test_prepare_distill_datasets_lazy_mix_returns_iterable`
- exact assertion/traceback summary: assertion failure `source_info["actual_rows"] is not None`; actual value is `None`.
- why unrelated: traceback/assertion stays in `tests/test_e2e_dataset_mix.py` against `e2e_common` / distill data behavior; no CAT train/teacher/decoder/resume/checkpoint path is involved; single reproduction is stable.
- production files involved: `e2e_common/data.py`, `e2e_common/lazy_datasets.py`, `train_utils/lora_data.py` import surface only.

## Final verdict

READY
