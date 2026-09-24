> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-10-region-normalized-prompt-kd-and-step-token-telemetry/task-6-review.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 6 Review: DistillTokenStatsAccumulator

## Spec: ✅

- 公共类型/API 与 brief 一致：`DistillWindowTokenStats`（frozen，三字段）+ `DistillTokenStatsAccumulator.update` / `consume_global`。
- 计数语义正确：`valid = attn≠0`（或全 True）；prompt=`valid∧label==-100`；response=`valid∧label≠-100`；无 causal shift；EOS 有监督即 response；`-100`+attn 0 不计入 prompt。
- `update` 累加 detached float32 3-vector，无 `.item()`/Python sync。
- `consume_global` 空本地仍 `zeros(3)` 参与 `accelerator.reduce(sum)`；仅 `global_samples==0` 返回 `None`；仅在 consume 后重置 `_accumulator`（未接入 trainer）。
- brief 所列 8 项测试均存在；`PYTHONPATH=. pytest tests/test_distill_token_stats.py -q` → **10 passed**。

## Quality: 良好

实现简洁、与现有 `train_utils` 风格一致；测试覆盖 brief 要点并含 `attention_mask=None` 补充用例。

## Findings

无阻塞问题。

- **Minor**：未单独测 `attention_mask` 非 rank-2（brief 仅要求 shape/device mismatch；labels rank 已测）。
- **Minor**：裸跑 `pytest tests/test_distill_token_stats.py -q` 需 `PYTHONPATH=.`（与同仓其它 `train_utils` 测试一致，非本 task 独有）。

## ⚠️

- consume 在 `global_samples==0` 时同样清零状态；与「第二次 consume 不得重复旧值」测试一致，符合 logging-window 语义。
- 报告写「reset after every consume」；与全局约束「successful consume 后 reset」在空窗 consume 上措辞略歧义，行为可接受。
