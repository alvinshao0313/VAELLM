# 10月28日七路 LongBench 中断续评

服务器在10月27日22:02重启，七条LongBench进程消失，原执行记录未及写出退出码。10月28日12:27已实测恢复：Linux5.15.0-198、8张A80080GB、NVIDIA590.48.01；`bitvae`仍为PyTorch2.6.0+cu124，真实CUDA运算通过。原驱动为570.86.10，本次差异写入恢复记录；模型、数据和评测协议不变。

当前状态（10月28日20:25）：七路独立恢复现均为`complete_existing_full8_plus_full21_nine`／`exit_code=0`，各`run.exit=0`。最后完成的[D05完整终态](../../result/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/qwen_recompression_d_recovery_20261027/README.md#10月28日2025d05续评完整终态)于20:25:18落盘，完整21项4750条、九项63.7130354690，质量`not_met`；本批最好仍是G09，历史最佳和目标均未改变。以下19:09表保留当时状态，由本段替代其D05仍运行的描述，不改写旧退出码。

## 已执行内容

只补原21任务／4750条的缺失预测，复用全部已完成的八项评测。原config逐字节一致；每个已有JSONL样本核对原归档身份，并绑定原文件字节前缀SHA。旧running／失败记录保留，恢复使用独立`recovery.json`。原生strict加载、logical≤2.6bit／rank0及252模块预热均已通过，每条均已实际追加预测。下表保留启动身份／行数，并在最后一列维护19:09终态索引：

| 任务 | GPU | 启动PID | 中断时行数 | 启动核实时行数 | 19:09状态／唯一结果正文 |
| --- | --- | --- | --- | --- | --- |
| C06 | 2 | 65231 | 1690 | 1699 | [已完成](../../result/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/README.md#10月28日1635c06续评完整终态) |
| G06 | 6 | 65914 | 1710 | 1712 | [已完成](../../result/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/README.md#10月28日1909新增完整终态汇总) |
| C09 | 1 | 65915 | 1698 | 1700 | [已完成](../../result/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/qwen_recompression_recovery_tail_20261027/README.md#10月28日1909两09续评完整终态) |
| G09 | 3 | 65917 | 1743 | 1746 | [已完成](../../result/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/qwen_recompression_recovery_tail_20261027/README.md#10月28日1909两09续评完整终态) |
| Gce10 | 0 | 65916 | 1768 | 1769 | [已完成](../../result/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/README.md#10月28日1909新增完整终态汇总) |
| Gce50 | 7 | 65918 | 1813 | 1823 | [已完成](../../result/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/README.md#10月28日1909新增完整终态汇总) |
| D05 | 5 | 65919 | 863 | 889 | [仍在续评，尚无完整九项](../../result/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/qwen_recompression_d_recovery_20261027/README.md#10月28日1909d-ce50补评完整终态) |

表中PID及行数为一次启动快照；完成状态取自各独立恢复终态和真实退出文件，不由启动行数推定。19:09核查时，六路均`complete_existing_full8_plus_full21_nine`／exit0且`run.exit=0`；D05仍`running`、无`run.exit`及完整九项。全部原预测字节前缀、原full8结果及旧执行报告均保持不变。实际PID/start_ticks、原生load audit和首条追加日志在`launch_verification.json`，源文件／模型／数据／配置及每份预测前缀绑定在`prepared.json`。

结果根：`result/qwen3_8b_mmlu65_20261004/qwen_overnight_resume_20261028/`。各任务独立目录含`launch.json`、`recovery.json`、`run.log`；tmux会话为`qwen_resume_<任务>_20261028`。已经启动，不能将一次性入口重复执行为第二个同目录任务。

## 实现与约束

- `common.py`固定七个原评测位置、UUID和原GPU，首先导入验证过的`source_snapshot_20261005_01`，再通过绝对路径载入原native LongBench wrapper。CPU实核`experiments`、`rotation.hadamard_utils`和`train_utils.v6_model_loader`均来自同一snapshot，避免4032维旋转依赖混用。
- `prepare.py`在CPU检查完整full8、原配置／归档／源码、checkpoint身份与已有预测，输出一次性绑定。
- `run.py --job <任务>`调用原`runner.run(..., resume=True)`。各任务只使用指定物理GPU；调用者先激活`bitvae`，实际环境和命令保存在各`launch.json`，脚本不切换conda环境。
- 保持BF16／FA2／32K输入、官方21项提示词与各自生成长度、原greedy与seed42。每路24小时预算，在样本之间检查，不截短正在生成的官方样本；无自动重试、占卡或监控守护。
- 完整4750条后核旧前缀与full8不变，再在原评测位置生成`scores.json`和`nine_task_acceptance.json`，独立恢复终态如实记录完成／失败。不能用部分任务均值替代LongBench整体。
- 七条实际code／decoder／保护／rotation开销及逻辑位宽沿各原审计；packed padding另报。各任务均rank0，G／D的packed大于2.6，不混称物理文件已≤2.6。

GPU4的D-CE50单donor补评已于17:44:34完成，独立report及外层退出均0，完整九项的唯一表与证据见[D CE50终态](../../result/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/qwen_recompression_d_recovery_20261027/README.md#10月28日1909d-ce50补评完整终态)。它不是七路续评之一；入口仍为`experiments/d_donor_longbench_20261028/`，原full8复用且保持，未重训。

本次仅核实已有终态及更新索引，不主动监控GPU，不创建监控自动化或清理模型；D05已自然完整结束。当前已验证全局最佳与目标仍沿主题正文；本批最好已更新为G09，不能把旧raw G稀疏继续列作本批最好。上述七路均为有效完整评测、质量`not_met`，不把程序完成写成目标达成。无新增算法经验。

主题记录：[重新压缩与恢复](../../result/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/README.md)、[恢复后半链](../../result/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/qwen_recompression_recovery_tail_20261027/README.md)、[D版本](../../result/qwen3_8b_mmlu65_20261004/qwen_recompression_followup_20261026/qwen_recompression_d_recovery_20261027/README.md)。
