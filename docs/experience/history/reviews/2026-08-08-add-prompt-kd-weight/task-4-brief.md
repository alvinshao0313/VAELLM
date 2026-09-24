> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-4-brief.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## Task 4: Add Category-Distillation Parameter Chain

**Files:** `train_utils/cat_train_args.py`, `train_utils/cat_train_pipeline.py`, `train_utils/lora_utils.py`, `tests/test_cat_eval_adapter_match.py`

- [ ] 先加参数测试：默认 0.0；`default=0.05,after:q_proj=0.1` 正确解析；负值拒绝；2.0 可接受。
- [ ] `NormalizedCatArgs` 增加 `distill_prompt_kd_weight: OverrideTable[float]`，放在 distill loss 参数附近。
- [ ] `ResolvedDistillRuntimeConfig` 增加 `prompt_kd_weight: float`。
- [ ] 新增 `_DISTILL_PROMPT_KD_WEIGHT_SPEC`，复用现有 `_parse_nonnegative_float_text()`；允许 selector 与其他 after-category distill 参数相同；示例 `default=0.0,after:q_proj=0.05`。
- [ ] parser 新增 `--distill_prompt_kd_weight`，type=str，默认 `default=0.0`。
- [ ] `process_cat_train_args()` 将 raw string 解析成 OverrideTable。
- [ ] `resolve_distill_runtime_config()` 按 after_category resolve 为 float。
- [ ] `train_utils/cat_train_pipeline.py` 的 `distill_tables` 增加该表，保证错误 category selector 会被现有 validation 拒绝。
- [ ] `train_utils/lora_utils.py` 的 `_ResolvedDistillStageConfig` 增加 `prompt_kd_weight`，并在 `_resolve_distill_stage_config()` 透传。
- [ ] LoRA 蒸馏参数日志增加 resolved `prompt_kd_weight`，不要打印 raw OverrideTable。
- [ ] `_build_lora_trainer()` 传 `prompt_kd_weight=float(cfg.prompt_kd_weight)` 给 `CustomSFTTrainer`。
- [ ] 运行 `pytest tests/test_cat_eval_adapter_match.py -q`。

---

