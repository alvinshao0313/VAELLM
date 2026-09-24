> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-08-add-prompt-kd-weight/task-9-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 9 Report: Documentation

## Status

完成。

## Commits

none（按任务要求未提交）

## Files Changed

1. **`docs/cat_train_args.md`**
   - 在 §2.2 after-category override 列表中加入 `--distill_prompt_kd_weight` 及示例写法。
   - 在 §3 参数表新增一行：默认 `default=0.0`、范围 `>=0`、after-category override。
   - 新增 §6.11.1 Prompt KD weighting：语义、mask 规则、EAKLD 共用 weighted mask、0.05/0.1 仅为实验示例。

2. **`compressed_e2e_fintuning/README.md`**
   - 在 Hidden-state 对齐之后新增 **Prompt KD weighting** 小节，覆盖 `--prompt_kd_weight` 默认值、取值含义、与 CE/hidden loss 的边界、mask 规则、EAKLD 行为、`mcqa` 限制、实验示例免责声明。

## Verification

- 对照 `train_utils/distill_losses.py` 中 `build_distill_token_mask()` 与 `train_utils/cat_train_args.py` 中 `_DISTILL_PROMPT_KD_WEIGHT_SPEC` 确认参数名、默认值、override 语法与代码一致。
- 未运行测试（文档-only 变更）。

## Concerns

无。
