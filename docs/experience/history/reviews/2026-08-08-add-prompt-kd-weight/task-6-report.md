> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-6-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 6 Report: Add E2E CLI and Runtime Chain

**Status:** DONE  
**Commits:** none

## What changed

- `tests/test_e2e_dataset_mix.py`: added `VAEE2EPromptKdWeightArgsTest` — default `0.0`, accept `0.05`/`2.0`, reject negative (`SystemExit`); MCQA allows `0.0`, rejects nonzero.
- `compressed_e2e_fintuning/args.py`: `VAEDecoderE2EArguments.prompt_kd_weight` + CLI `--prompt_kd_weight` (float, default `0.0`); validate `>= 0`; MCQA rejects nonzero (choice KD has no token mask).
- `compressed_e2e_fintuning/runtime.py`: log resolved `prompt_kd_weight` and fixed `response_kd_weight=1.0`; pass `prompt_kd_weight=float(args.prompt_kd_weight)` into `VAEDecoderE2ETrainer`.
- `compressed_e2e_fintuning/trainer.py`: minimal `__init__` stub — store `prompt_kd_weight`, reject `<0`. No dense/CPU/gamma wiring (Task 7).

## RED evidence

```text
$ export PYTHONPATH=. && pytest tests/test_e2e_dataset_mix.py::VAEE2EPromptKdWeightArgsTest -q
FFFFFF
6 failed in 6.77s
```

Failures: unknown CLI / missing `prompt_kd_weight` on `VAEDecoderE2EArguments`.

## GREEN evidence

```text
$ export PYTHONPATH=. && pytest tests/test_e2e_dataset_mix.py::VAEE2EPromptKdWeightArgsTest -q
......
6 passed in 6.61s

$ export PYTHONPATH=. && pytest tests/test_e2e_dataset_mix.py -q
9 failed, 38 passed in 11.13s
```

Task-6 class: all 6 green. Full-file 9 failures are pre-existing and unrelated:
- `test_parse_args_eval_before_save_requires_tasks_and_save_steps` still uses removed `--eval_before_save` (CLI is `--eval_after_save`).
- Several `DatasetMixBuilderTest` cases fail on dataset load/`dummy.txt` mocking, not on prompt KD args.

## Concerns

- Task 7 still owns wiring `prompt_kd_weight` into E2E dense / CPU-offload / teacher gamma mask paths; trainer currently only accepts/validates/stores the value.
- Full-file pytest is not clean due to pre-existing failures above; do not treat them as Task 6 regressions.
