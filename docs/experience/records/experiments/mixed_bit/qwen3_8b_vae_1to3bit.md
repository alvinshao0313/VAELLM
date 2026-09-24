# Qwen3-8B 混合位宽历史结果整理

整理日期：2026-09-23；实际运行日期未由现有汇报确认。类型：历史结果二次整理，未重新运行。run_id：`qwen3_8b_vae_1to3bit`、`qwen3_8b_vae_1to3bit_no1bit`。结论状态：本配置尚不能支持混合位宽改善语言模型质量。

## 来源与验证范围

- [原始 HTML 汇报](historical_report.html)：方法、KL、PPL 和八任务分数；本次仅转录核心结果，没有重跑或独立复算下游指标。
- [原运行配置](../../../../../.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/resolved_run_config.json)。
- [含 1-bit 分配摘要](../../../../../.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/allocation/topk_k256/optimal_2bit_summary.md)、[分配 JSON](../../../../../.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/allocation/topk_k256/optimal_2bit.json)。
- [排除 1-bit 分配摘要](../../../../../.result/mix_bit/qwen3_8b/isolated/qwen3_8b_vae_1to3bit_no1bit/allocation/topk_k256/optimal_2bit_summary.md)、[分配 JSON](../../../../../.result/mix_bit/qwen3_8b/isolated/qwen3_8b_vae_1to3bit_no1bit/allocation/topk_k256/optimal_2bit.json)。
- [代价表摘要](../../../../../.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/costs/topk_k256/cost_table_summary.md)。

## 对照与结果

原汇报条件：Qwen3-8B，36×7=252 个 decoder Linear；35 个类别×位宽候选，两级 residual、dim32、MSE 10000 steps。校准 1000×2048，teacher top-k=256；基线为均匀 2-bit 学生，单层替换产生 ΔKL，以参数量加权码流平均位宽≤2.0 为分配约束；无蒸馏恢复。位宽不包含所有 decoder/保护参数与未压缩部分的整体存储成本。

| 指标 | 均匀 2-bit | 混合含 1-bit | 混合排除 1-bit |
| --- | ---: | ---: | ---: |
| 分配平均位宽 | 2.0 | 1.840882 | 2.0 |
| 校准 KL（汇报） | 8.264 | 6.961 | 7.358 |
| wiki-PPL（汇报） | 10307 | 447174 | 30262 |
| 八任务均值 %（汇报） | 35.76 | 36.51 | 36.62 |
| 可加目标 ΣΔKL（分配摘要） | 0 | -51.054835 | -35.796040 |

两种求解均记录全局最优；这只针对定义的离散分配代理目标。按基线 KL 加 ΣΔKL 得到的预测为负值，与真实混合模型的非负 KL 不符，说明该可加近似不能直接当最终模型 KL。含 1-bit 未用满预算，本身不表示求解失败，因为约束为≤。

校准 KL 降低同时 PPL 恶化，任务均值的小幅变化未经这里独立统计核验，不能据此宣称有效提升。排除 1-bit 改善了相对含 1-bit 的 PPL，但仍差于均匀基线。候选质量、代价估计噪声与跨层相互作用的贡献尚未独立消融；原汇报中“根因更像代价信号”应视为解释假设。

## 经验与后续使用

已提炼到 [Mixed-bit 经验：代理目标与最终质量](../../../lessons/mixed_bit_pipeline.md)。后续若继续这条路线，应检验代理可加性误差和完整模型质量，再决定是否扩大预算，不能只凭求解器 optimal 或较低校准 KL 推进。

原始证据保留原位，HTML 汇报迁入本主题目录，旧路径保留别名。本次未处理这两个历史运行的权重与缓存用途，未据此删除任何模型或数据。
