> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-8-fix1-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 8 Fix Round 1 Review

**Finding:** Stale `VAEE2ETrainerPromptKdMaskHelperTest` used `build_distill_token_mask` + `prompt_kd_weight`.

## ADDRESSED
Yes. Test renamed to `test_private_helper_forwards_regions_without_prompt_kd_weight`; calls `_build_distill_token_regions`, patches `build_distill_token_regions`, returns `DistillTokenRegions`, asserts only `labels`/`attention_mask`/`reference_logits` forwarded, explicitly rejects `prompt_kd_weight` in kwargs. Matches production helper at `trainer.py:473-482`.

## New Breakage
None from this fix. Verified: `pytest tests/test_e2e_dataset_mix.py::VAEE2ETrainerPromptKdMaskHelperTest -q` → 1 passed.

## Overall
Fix is correct and minimal. Task 8 telemetry work unaffected. Remaining `test_e2e_dataset_mix.py` failures (`dummy.txt`, etc.) still pre-existing and out of scope.
