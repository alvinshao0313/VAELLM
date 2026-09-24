> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-4-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 4 Report: Category-Distillation Parameter Chain

**Status:** DONE  
**Commits:** none

## What changed

- `tests/test_cat_eval_adapter_match.py`: added `CatDistillPromptKdWeightArgsTest` (default / override / negative / `2.0`); updated trainer-selection fixture to pass `prompt_kd_weight`.
- `train_utils/cat_train_args.py`: `NormalizedCatArgs.distill_prompt_kd_weight`, `_DISTILL_PROMPT_KD_WEIGHT_SPEC` (via `_parse_nonnegative_float_text`), CLI `--distill_prompt_kd_weight` default `default=0.0`, parse + `ResolvedDistillRuntimeConfig.prompt_kd_weight` resolve.
- `train_utils/cat_train_pipeline.py`: `distill_tables` includes the new OverrideTable.
- `train_utils/lora_utils.py`: stage config + resolve + log resolved `prompt_kd_weight`; `_build_lora_trainer` passes it to `CustomSFTTrainer`.
- `train_utils/lora_training.py`: minimal `CustomSFTTrainer.__init__` kwarg stub (store + reject `<0`); **no** `compute_loss` changes (Task 5).

## RED evidence

```text
$ export PYTHONPATH=. && pytest tests/test_cat_eval_adapter_match.py::CatDistillPromptKdWeightArgsTest -q
FFFF
4 failed in 8.94s
```

Failures: unknown CLI arg / missing `prompt_kd_weight` on `ResolvedDistillRuntimeConfig`.

## GREEN evidence

```text
$ export PYTHONPATH=. && pytest tests/test_cat_eval_adapter_match.py::CatDistillPromptKdWeightArgsTest -q
....
4 passed in 5.00s

$ export PYTHONPATH=. && pytest tests/test_cat_eval_adapter_match.py -q
.......................
23 passed in 5.05s
```

## Concerns

- Task 5 still owns wiring `prompt_kd_weight` into KD `compute_loss` branches; current trainer only accepts/validates/stores the value.
