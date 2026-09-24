> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-7-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 7 Report: Unify E2E Dense, CPU-Offload and Teacher Gamma Mask

**Status:** DONE  
**Commits:** none

## What changed

- `compressed_e2e_fintuning/trainer.py`:
  - Added private `_build_distill_token_mask(inputs, reference_logits)` — sole production call site of shared `build_distill_token_mask`, always passes `self.prompt_kd_weight`.
  - Wired into `_compute_legacy_dense_loss`, `_compute_teacher_first_cpu_loss`, and `_build_cpu_teacher_targets` gamma_mask (hard requirement: gamma and KL share weights).
  - `__init__` already accepted/validated `prompt_kd_weight` from Task 6; unchanged semantics.
- `tests/test_e2e_dataset_mix.py`: `VAEE2ETrainerPromptKdMaskHelperTest` — light `__new__` fixture asserts private helper forwards `prompt_kd_weight=0.1` to shared mask.
- `tests/smoke/test_loss_pipeline_smoke.py`: dense + CPU-offload EAKLD use `prompt_kd_weight=0.1` fractional mask; assert weights in `(0,1)`; all `DENSE_LOSS_TYPES` forward/backward; dense vs offload loss/telemetry/grad match.
- `tests/smoke/test_one_step_train_smoke.py`: `_build_e2e_trainer` accepts `prompt_kd_weight`; dense + CPU-offload EAKLD one-step each use `0.1`; pre-MLP `__new__` fixture sets `prompt_kd_weight=0.0`.

## RED evidence

```text
$ export PYTHONPATH=. && pytest tests/test_e2e_dataset_mix.py::VAEE2ETrainerPromptKdMaskHelperTest -q
F
AttributeError: 'VAEDecoderE2ETrainer' object has no attribute '_build_distill_token_mask'
1 failed in 5.34s
```

## GREEN evidence

```text
$ export PYTHONPATH=. && rg -n "build_distill_token_mask" compressed_e2e_fintuning/trainer.py
28:    build_distill_token_mask,
402:    def _build_distill_token_mask(
407:        return build_distill_token_mask(
460:                gamma_mask = self._build_distill_token_mask(inputs, teacher_logits)
607:        token_mask = self._build_distill_token_mask(inputs, logits)
698:                token_mask = self._build_distill_token_mask(inputs, logits)

$ export PYTHONPATH=. && pytest tests/test_e2e_dataset_mix.py::VAEE2ETrainerPromptKdMaskHelperTest tests/smoke/test_loss_pipeline_smoke.py tests/smoke/test_one_step_train_smoke.py -q
.........
9 passed in 6.67s
```

`rg` substring matches include the private method name and three `self._build_distill_token_mask(...)` call sites. Shared helper direct call: **1 import + 1 call inside private helper**; dense / CPU student KL / teacher gamma all go through the private helper.

## Concerns

- None for Task 7 scope. Dispatcher formulas and truncation/VAE/ckpt/eval pipelines were not touched.
