# VAELLM 文档入口

当前工作区：`cmcc:/home/cmcc/VAELLM/`（2026-10-30 实测及用户确认）；旧服务器用户为 `star`，旧记录中的 `/home/star/VAELLM/` 和 iaaccn74 路径仅作历史来源。先读项目规则，再按任务选主题。

## SC-CAT 项目级标准入口

SC-CAT新模型定制搜索与一阶段基线任务先读以下两份独立规范。通用规则在此维护，具体实验只记录配置、结果和证据。

| 文档 | 职责 |
| --- | --- |
| [SC-CAT压缩流程标准](sc_cat_protocol.md) | 固定目标、搜索表、选择规则、类别预算、逐类恢复与验收 |
| [SC-CAT执行与结果记录规范](sc_cat_execution.md) | LLM接手步骤、真实接口、三张结果表及其[存储命名规则](sc_cat_execution.md#result-storage)、正式链接续与交付 |

状态：离线与逐类蒸馏搜索规则、首类／剩余类别分批预算方式已确认，可交接实现适配；具体模型额度、执行对齐与整链验证尚未完成。实际批次、可用资源及授权仍以[当前计划](experience/records/w25_transfer/current_plan.md)为准。

## 按任务查找

| 要做什么 | 从这里开始 |
| --- | --- |
| 协作规则与代码习惯 | [项目规范](../AGENTS.md) |
| 推进当前低比特目标 | [唯一执行计划与阶段完成条件](experience/records/w25_transfer/current_plan.md) |
| 了解正式训练链路 | [项目概览](../README.md) |
| CAT 压缩与恢复 | [CAT 架构和参数](guides/cat_training.md)、[checkpoint 逐类别恢复](guides/catlora_distill_from_checkpoint.md) |
| 新模型定制搜索、类别排序与自动恢复选点 | [主协议：范围、计费与迁移](sc_cat_protocol.md)、[离线执行：状态、接口、决策与退出](sc_cat_execution.md) |
| 从 BF16 复现 Qwen3 8B 的 69% 压缩成品 | [一键脚本与完整操作说明](guides/qwen3_8b_69_reproduction.md) |
| 梳理当前目标、研究范围与迁移边界 | [当前决定与完整验收](experience/records/w25_transfer/current_plan.md)、[研究专题与历史证据入口](experience/records/w25_transfer/2026-10-04_mmlu65_llama31.md) |
| 并行改进压缩基座与基座后恢复 | [当前子任务与剩余预算](experience/records/w25_transfer/current_plan.md#当前子任务与剩余预算)、[新基座对齐恢复](../experiments/cat_aligned_recovery_20261028/README.md) |
| 迁移成品、检查新服务器及SCP测速 | [研究接续包、新服务器自检与传输测试](guides/research_handoff.md) |
| E2E 微调 | [E2E 模块说明](../compressed_e2e_fintuning/README.md) |
| 添加残差保护或残差 LoRA | [base checkpoint residual](guides/cat_residual_from_base.md)、[残差 LoRA 配置](guides/residual_lora.md) |
| 准备训练数据 | [EdgeRazor 接入](guides/edgerazor_dataset.md) |
| 设计实验、排查已知问题 | [按分类→专题→条目查经验](experience/README.md#按问题查找)，支持分类检索，再看命中的证据 |
| 理解VAELLM超参、敏感条件和调参方向 | [本体超参作用与故障诊断](experience/lessons/compression/fitting.md#compression-choices--超参作用与故障诊断2026-10-29)、[损失与尺度](experience/lessons/compression/losses.md) |
| 记录实验与提炼经验 | [实验工作流程](experience/WORKFLOW.md) |
| 混合位宽 | [模块与命令](../mix_bit/README.md)、[位宽分配经验](experience/lessons/compression/bit_allocation.md) |
| 向量自适应残差阶数原型 | [down 同预算配对、码流与验证](../experiments/adaptive_vector_rvq/README.md) |
| 自适应残差阶数训练与4bit初始化 | [第一阶段通用基线流程规范](adaptive_residual_stage1_protocol.md)、[门控设计与Qwen证据](../experiments/progressive_residual_bsq_20261030/README.md)、[历史计划](../experiments/progressive_residual_bsq_20261030/PLAN.md)、[文献事实](../experiments/progressive_residual_bsq_20261030/LITERATURE.md) |
| LiftQuant 式恢复 | [模块入口](../experiments/liftquant_recovery/README.md)、[编码优化经验](experience/lessons/post_recovery/codes.md)、[局部选码](experience/lessons/post_recovery/code_search.md) |
| 分组/完整输出损失实验 | [分组输出实验](../experiments/activation_loss_ab/README.md)、[完整输出实验](../experiments/linear_output/README.md)、[经验](experience/lessons/compression/losses.md) |
| down 层敏感度实验 | [运行说明](../experiments/down_layer_sensitivity/README.md) |
| 第三方算子 | [HiFloat4 上游说明](../HiFloat4/README.md)；安装/环境变更仍按项目授权 |
| 找全部文件或旧路径 | [自动分类目录](INDEX.md)、[首次经验归档来源](experience/CATALOG.md) |
| 新建、整理、归档文档 | [组织与既有长文整理规范](MANAGEMENT.md#阅读顺序与维护职责2026-10-30)、[条目模板与收尾](experience/WORKFLOW.md) |

## 快速检索

在远程项目根目录、已激活的 `bitvae` shell 中执行：

```bash
python tools/docs.py search 'OOM|resume'
python tools/docs.py categories
python tools/docs.py search '学习率|步数' --category category_recovery --topic optimization
python tools/docs.py search '翻码|proxy' --scope lessons
python tools/docs.py search 'mixed.bit|位宽' --scope records
python tools/docs.py search 'round_base' --scope history
```

默认只搜索规则、使用说明和经验，跳过历史、具体结果、工具交接和生成报告；每次最多返回 40 行，进一步缩小关键词再阅读全文。旧路径别名不重复搜索。

## 文档地图

- `sc_cat_protocol.md`、`sc_cat_execution.md`：项目级SC-CAT流程与执行规范，正文独立于实验记录。
- `guides/`：跨模块的当前使用说明。
- 模块旁的 `README.md`：入口、接口、最小用法；正文就近维护，统一从这里导航。
- `experience/lessons/<分类>/<专题>.md`：按阶段或领域、问题专题、经验条目三级组织的可复用经验。
- `experience/records/`：具体实验的设计、实际结果和证据索引。
- `experience/history/`：历史方案、审查和被替代说明，按需查。
- 原结果目录：原始指标、核心日志、配置，以及仍与产物绑定的说明和自动报告。
- `.ai-bridge/`：工具交接状态；不当长期规范或最新实验结果。

入口不复制实验指标和实时进度；有效状态以主题正文及其原始证据为准。
