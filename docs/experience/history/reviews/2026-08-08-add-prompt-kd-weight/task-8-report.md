> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-8-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 8 Report: Update Experiment Scripts, Preserve Defaults

**Status:** DONE  
**Commits:** none

## What changed

- `scripts/catlora_distill_4gpu_res0.sh`: Added `--distill_prompt_kd_weight "default=0.0"` after `--distill_loss_alpha`, alongside other distill loss CLI flags. Literal on the python command line; no new shell variable.
- `compressed_e2e_fintuning/scripts/e2e_decoder.sh`: Added `--prompt_kd_weight 0.0` after `--distill_alpha` in both `PARALLEL_MODE=dp` (torchrun) and `PARALLEL_MODE=layer_mp` (python) branches.

## Verification

- Manual review: both scripts keep default weight at `0.0` (not 0.05/0.1); no `conda activate` or shell var wrapper for the new hyperparameter.
- No runtime test run: shell-only CLI wiring; behavior unchanged at default 0.0.

## Concerns

- None. Scripts now expose the new flags explicitly while preserving existing experiment behavior.
