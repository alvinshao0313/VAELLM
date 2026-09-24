> 归档于 2026-09-23；类型：历史计划/审查；不等于当前已验证结论，也不构成操作授权。
> 原路径：`.superpowers/sdd/2026-08-18-down-proj-mmlu-layer-sensitivity-ablation/task-7-report.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# Task 7 Report

## Status
完成。`summarize_final()` 在写入 `final_summary.json` 后生成 `report.md` 与三张诊断图。

## 改动文件
- `experiments/down_layer_sensitivity/summarize.py` — 新增 `_plot_layer_sensitivity`、`_plot_nmse_vs_mmlu_sensitivity`、`_plot_cumulative_recovery`、`_render_report`、`_generate_report_and_plots`；接入 `summarize_final`
- `experiments/down_layer_sensitivity/README.md` — 新建，说明输出产物
- `experiments/down_layer_sensitivity/tests/test_summarize.py` — 成功路径断言 report/plots；失败路径断言不生成

## Commits
无（按约束未提交）

## 测试
`pytest -q experiments/down_layer_sensitivity/tests/test_summarize.py` — 17 passed

## Concerns
- Task 8 的 shell 用法尚未写入 README（Task 7 brief 未要求）
- 正式 run 未在本机执行；图表/报告仅在合成 fixture 上验证
