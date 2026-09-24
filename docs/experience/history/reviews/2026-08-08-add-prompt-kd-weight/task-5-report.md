> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-5-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 5 Report: Integrate Prompt Weight into Every Category KD Branch

## Status

Done

## Commits

none

## Changes

### `train_utils/lora_training.py`
- Kept Task 4 `CustomSFTTrainer.__init__(prompt_kd_weight=0.0)` store/validate behavior.
- Added local `build_token_mask(reference_logits)` inside `compute_loss()` that calls shared `build_distill_token_mask(...)` with `full_inputs` labels/attention and `self.prompt_kd_weight`.
- Replaced all 16 per-branch `build_distill_token_mask(...)` call sites with `build_token_mask(logits)`.
- Covered: `rkl`, `dual_rkl`, `kl`, `r_kl_top*`, `dual_r_kl_top*`, `kl_top*`, `kd_top*`, `mse`, `kd`, `dual_kl`, `dual_kl_top*`, `dual_kd_top*`, `dual_kd`, `eakld_top*`, `eakld`, `eakld_kd`.
- SFT/origin unchanged; CE mixes for `kd`/`kd_top`/`dual_kd*`/`eakld_kd` still use response-only labels + original alpha; prompt weight only on KD term.
- Hidden / pre-MLP hidden alignment still uses attention mask only.

### `tests/test_cat_eval_adapter_match.py`
- `_build_pre_mlp_trainer` (`__new__` fixture) now sets `prompt_kd_weight=0.0`, required because it bypasses `__init__`.

## Verification

### rg
```text
rg -n "build_distill_token_mask" train_utils/lora_training.py
10:    build_distill_token_mask,
751:                return build_distill_token_mask(
```
Result: 1 import + 1 real call inside local helper; no leftover per-branch calls.

### pytest
```bash
conda activate bitvae
export PYTHONPATH=.
pytest tests/test_distill_losses.py tests/test_cat_eval_adapter_match.py -q
```
Result: `74 passed in 5.66s`

## Concerns

- Pre-MLP unit tests construct `CustomSFTTrainer` via `__new__`; any new trainer attribute used in `compute_loss` must be mirrored in that fixture (done for `prompt_kd_weight`).
- Smoke helper `tests/smoke/test_one_step_train_smoke.py` also uses `__new__`; not in this task’s pytest set, but may need the same attribute when that path hits KD mask building.
