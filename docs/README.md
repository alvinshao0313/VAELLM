# VAELLM 文档入口

文档主源：`iaaccn74:/home/shaoyuantian/program/VAELLM/`。先读项目规则，再按任务选一两个主题；无需加载整个目录。

| 要做什么 | 从这里开始 |
| --- | --- |
| 协作规则与代码习惯 | [项目规范](../AGENTS.md) |
| 了解正式训练链路 | [项目概览](../README.md) |
| CAT 压缩与恢复 | [CAT 架构和参数](guides/cat_training.md)、[checkpoint 逐类别恢复](guides/catlora_distill_from_checkpoint.md) |
| E2E 微调 | [E2E 模块说明](../compressed_e2e_fintuning/README.md) |
| 添加残差保护或残差 LoRA | [base checkpoint residual](guides/cat_residual_from_base.md)、[残差 LoRA 配置](guides/residual_lora.md) |
| 准备训练数据 | [EdgeRazor 接入](guides/edgerazor_dataset.md) |
| 设计实验、排查已知问题 | [按问题查经验](experience/README.md)，再看命中的证据 |
| 记录实验与提炼经验 | [实验工作流程](experience/WORKFLOW.md) |
| 混合位宽 | [模块与命令](../mix_bit/README.md)、[工程与评测经验](experience/lessons/mixed_bit_pipeline.md) |
| LiftQuant 式恢复 | [模块入口](../experiments/liftquant_recovery/README.md)、[编码优化经验](experience/lessons/liftquant_recovery.md) |
| 分组/完整输出损失实验 | [分组输出实验](../experiments/activation_loss_ab/README.md)、[完整输出实验](../experiments/linear_output/README.md)、[经验](experience/lessons/output_losses.md) |
| down 层敏感度实验 | [运行说明](../experiments/down_layer_sensitivity/README.md) |
| 第三方算子 | [HiFloat4 上游说明](../HiFloat4/README.md)；安装/环境变更仍按项目授权 |
| 找全部文件或旧路径 | [自动分类目录](INDEX.md)、[首次经验归档来源](experience/CATALOG.md) |
| 新建、修改、归档文档 | [文档管理规范](MANAGEMENT.md) |

## 快速检索

在远程项目根目录、已激活的 `bitvae` shell 中执行：

```bash
python tools/docs.py search 'OOM|resume'
python tools/docs.py search '翻码|proxy' --scope lessons
python tools/docs.py search 'mixed.bit|位宽' --scope records
python tools/docs.py search 'round_base' --scope history
```

默认只搜索规则、使用说明和经验，跳过历史、具体结果、工具交接和生成报告；每次最多返回 40 行，进一步缩小关键词再阅读全文。旧路径别名不重复搜索。

## 文档地图

- `guides/`：跨模块的当前使用说明。
- 模块旁的 `README.md`：入口、接口、最小用法；正文就近维护，统一从这里导航。
- `experience/lessons/`：按问题持续更新的可复用经验。
- `experience/records/`：具体实验的设计、实际结果和证据索引。
- `experience/history/`：历史方案、审查和被替代说明，按需查。
- 原结果目录：原始指标、核心日志、配置，以及仍与产物绑定的说明和自动报告。
- `.ai-bridge/`：工具交接状态；不当长期规范或最新实验结果。

入口不复制实验指标和实时进度；有效状态以主题正文及其原始证据为准。
