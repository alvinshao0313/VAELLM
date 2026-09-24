# VAELLM 经验库

远程主目录：`iaaccn74:/home/shaoyuantian/program/VAELLM/docs/experience/`。整理日期：2026-09-23。

[全项目入口](../README.md) · [文档管理规范](../MANAGEMENT.md)

## 先查经验，再设计实验

设计或修改实验前，先按模块、损失、错误信息和评估目标搜索本库，阅读命中的经验及来源。设计中简要写清：参考了哪条经验、本次如何规避、哪些条件不同、还需要验证什么。没有命中时如实记录；需要重做已失败方案时说明改变的条件或新假设。

```bash
# 从远程项目根目录执行；有 rg 时优先使用它。
rg -n -i '编码|STE|翻码|proxy' docs/experience/lessons
rg -n -i 'OOM|offload|显存|resume' docs/experience
# 服务器没有 rg 时，无需安装：
grep -RniE '编码|STE|翻码|proxy' docs/experience/lessons
```

## 经验与结果分开

- **lessons/**：可复用经验。每篇明确问题、证据、原因/解释、建议动作、适用边界和来源。
- **records/**：原始实验结果、实现验证、恢复报告和交接记录。配置、指标及复现证据归这里，不能只改名就称为经验。
- **history/**：历史方案和审查材料。计划、勾选项及“READY”不能代替实测；旧命令、授权、路径和“当前状态”仅代表当时。
- **[全项目目录](../INDEX.md)**：持续更新的分类与路径索引；[CATALOG.md](CATALOG.md) 保留首次归档的来源映射。旧路径仅保留跳转，不维护第二份正文。

实验原始 JSON、核心日志、配置、输入模型和正在运行的任务仍在各自结果目录；本文档整理不运行实验、不改变算法，也不更新任何实验的完成状态。

## 按问题查找

| 主题 | 经验文档 | 检索词 |
| --- | --- | --- |
| 基线、公平对照、评测口径 | [对照设计与结论边界](lessons/experiment_design.md) | baseline、seed、LR、step、全压、PPL |
| 输出感知目标和辅助损失尺度 | [输出损失与尺度](lessons/output_losses.md) | MSE、AMSE、Gram、block_output、BSQ |
| 残差 LoRA 与初始化/续训 | [残差 LoRA](lessons/residual_lora.md) | additive、replace、PEFT、rank |
| LiftQuant 式恢复与编码优化 | [编码坐标、STE 和更新预算](lessons/liftquant_recovery.md) | proxy、翻码、decoder、round、clamp |
| 打包、导出、重载、精确续训 | [Checkpoint 与数值路径](lessons/checkpoint_lifecycle.md) | v6、packed、BF16、TF32、finalization |
| 精度、显存、offload、多卡 | [资源与数值精度](lessons/memory_precision.md) | OOM、allocated、reserved、autocast |
| token mask、prompt KD、日志统计 | [KD 语义与数据边界](lessons/kd_and_data.md) | causal、padding、prompt、mask、telemetry |
| 码本、混合位宽、权重旋转 | [压缩配置与存储口径](lessons/compression_choices.md) | codebook、stage、bpw、Hadamard、保护通道 |
| Mixed-bit 候选、缓存与质量验证 | [Mixed-bit 工程及评测经验](lessons/mixed_bit_pipeline.md) | fingerprint、tokenizer、manifest、worker、KL、PPL |
| 旧配置、运行状态与收尾 | [复现及文档维护](lessons/reproducibility.md) | CLI、历史、清理、复现、日志 |

## 实验完成后的更新流程

分类、结果总结和经验更新采用 [轻量工作流程](WORKFLOW.md)。每次实验结束都检查经验增量；有新结论更新已有主题，没有新认识则注明原因。结果按研究主题/日期保存，经验按问题主题维护，二者双向引用。

## 维护规则

1. 新实验结果进入 `records/`；有可复用结论时，先更新对应 `lessons/`，新主题才新建经验文档，新主题更新入口，随后刷新全项目自动目录。
2. 经验写法：**问题/触发条件 → 原始证据 → 已证实原因或明确标注的解释 → 下次动作 → 适用边界 → 来源**。未验证的解释标为假设，不写成根因。
3. 不用训练 loss、单层 NMSE、机制冒烟或部分题目分数替代全量下游结论；不同模型、压缩范围、数据和预算不得直接排名。
4. 结论变化时补充日期和替代关系，保留必要证据。旧材料里的“待完成”不代表今天仍未完成；核实当前状态后另写更新。
5. 原始指标保留在结果目录；经验引用唯一来源，避免复制多份指标表、日志或模型。遵守 AGENTS.md 中的收尾授权。

## 最近专题

- [LiftQuant编码恢复经验](lessons/liftquant_recovery.md)：最新状态、证据和限制见正文。
- [硬码翻转贡献对照](records/experiments/liftquant_recovery/BIT_CONTRIBUTION_20260923.md)：最新状态、证据和限制见正文。

- [0910 checkpoint 端到端 rank8 搜索](records/e2e_0910/2026-09-24_rank8_search.md)：分阶段筛选方案、运行状态与晋级结果见正文。
