# 全层恢复的两组 decoder 学习率对照（2026-09-24）

运行状态：已启动，运行中；已确认真实优化持续推进，停止主动监控。结论状态：证据不足。用户本轮明确授权 GPU 0、1 各一个设置并行；仅此两组，不自动扩大搜索队列。

## 问题与依据

以同一个初始 `result/linear_output/distill_init` 做完整 LiftQuant 式逐层恢复，判断是否改善相同下游任务。唯一训练变量是 decoder LR：`1.25e-5` 与 `6.25e-6`。前者有此前384步 joint 对照支持，后者是假设更保守更新可缓和 decoder 过冲的半值候选，尚无更优实测结论。

参考[编码恢复经验](../../../lessons/liftquant_recovery.md)及[工程等价性验证](2026-09-23_engineering_equivalence.md)。此前128-token单block局部收益不能外推到全层或下游，旧2e-4过冲说明不应盲目增大学习率。本次保持敏感度码坐标、所有code与decoder可学、官方实际调度序列、teacher align=1、whole-block MSE以及其余冻结约束。

## 固定配置和比较口径

- 初始 checkpoint ID `c9880629-6b54-4482-8086-857ba185d162`，Qwen3-8B，无旋转、单 residual stage，无 LoRA；192 compressed Linear + 60 dense。
- 训练 blocks：0、9至35，共28块。每块独立优化，teacher prefix 始终为原始FP模型；未训练的dense块仍参与teacher传播。
- 官方完整 RedPajama，固定数据 revision `4b6d76ca56b821e4f2110204943b7a67927b565c`，11原始Arrow共930,514行。数据来源及SHA见数据目录source_manifest。
- `nsamples=4096, holdout=128, seqlen=2048, batch_size=2, epochs=2, seed=42`，每块3968次更新，两组均从原始checkpoint新加载。
- code LR上限均为`2e-5`，实际仍取 `min(cap, proxy.std()/50)`；decoder LR是本次唯一变量。
- 物理 GPU 0/1，各64GiB allocator上限；bitvae。主机实测约1.4TiB available，双任务hidden+target共256GiB，另计模型/保存样例；磁盘约856GiB空闲，运行前再核对。
- payload为压缩目标每32权重64bit，即2bit/weight；不包括decoder等开销，不宣称全模型2bit。
- 评测沿用 `train_utils.eval_utils.run_lm_eval`：boolq/rte/winogrande/mmlu用acc，arc_easy/arc_challenge/openbookqa/piqa用acc_norm；0-shot、batch1、无limit、八任务等权均值。MMLU遵守既有按大小聚合。
- 初始模型与两个成品均用同一packed BF16全驻留路径。旧历史初始基线走whole-decoder fused，不能直接当本次同路径对照；GPU0完成训练后重评初始和成品，GPU1评自身成品，共用GPU0初始基线。

主指标为相对初始模型的八任务均值，辅看逐任务与逐block留出MSE；训练/留出MSE下降本身不等于下游提升。两组不是官方LiftQuant压缩表示的逐项复刻，code/decoder是既有VAELLM适配。

## 入口、验证和运行位置

复现入口 `experiments/liftquant_recovery/run_full_decoder_1p25e5.sh` 与 `run_full_decoder_6p25e6.sh`，须从已激活bitvae的项目根目录运行。薄Python工作流 `run_experiment.py` 串联原恢复入口与相同评測入口，失败即停止，无自动重试。

正式结果根目录 `.result/liftquant_recovery/full_layers_lr_20260924_01/`，两个子目录 `decoder_1p25e5`、`decoder_6p25e6`；每份 `run.json`记录实际命令/状态，`recovery/manifest.json`记录代码与初始payload身份、参数，`recovery/block_metrics.json`为原始逐块指标，`evaluation/`保留原始下游结果。

先执行正式形状短测：同初始模型、完整Arrow，blocks0/9，8样本、2留出、2048token、batch2、2epochs，每块6步。覆盖partial/full块、两轮数据、块间推进、原子boundary两次、完整native导出和严格重载。只缩短样本与步数，不能用128token测试替代。另核对全模型streamed/resident输出和真实评测数据入口。

短测结果根 `.result/liftquant_recovery/formal_shape_20260924_01/`。恢复入口的旧 `final_experiment_run` 字段仅区分API-page数据，因此Arrow短测会标true；本轮实际用途以外层 `run.json:purpose=short_validation` 和本记录为准，不当作正式训练。

当前适用的14decoder数学、7梯度分支、新旧训练轨迹和层边界精确恢复证据直接复用，不扩大回归矩阵。仅待正式形状与新驻留评测路径通过后启动两组；确认真实训练推进后交付并停止主动监控。

## 经验和产物

目前无新增质量经验，候选优劣等待正式结果。完整校准数据用于本次两组和具体复现；初始checkpoint始终保护。正式成品保留用于已安排的下游评测；完成后按评测结论与既有清理规则决定留存。短测权重仅用于链路验证，结束并记录后清理，保留配置、核心日志和原始指标。

## 本轮实测与正式启动

2026-09-24：两组正式形状短测均退出0。每组block0、9各6步、两轮数据，整体与留出MSE均下降，码位翻转0（此步数只验证链路）；两组token SHA完全相同。全部冻结状态SHA不变、hard→native与完整保存重载输出严格相等。训练allocated峰值8.0097GiB、reserved峰值9.7168GiB。八任务limit1全部读取并计算成功，这些少量样本分数不用于候选排名。

全驻留评测验证退出0：同一初始模型、真实2×2048输入，logits误差0、NLL差0、argmax一致率100%、192个BF16缓存未重建；allocated峰值19.5061GiB。单批逐层搬运6.646s、驻留准备加前向2.301s，非全量benchmark。新适用经验已回写上述经验页。

原始证据：短测目录的summary.json、合并后的validation.log、两个recovery/block_metrics.json及evaluation/B.json；驻留目录 `.result/liftquant_recovery/eval_residency_20260924_01/gpu/summary.json`；原数据source_manifest.json记录11份上游hash及4096×2048校准token SHA `6b5fa59d5e9dc29c0c907e3f5da127e34de28329e6055efaca8bb357a4084c02`。

正式启动标识：

| GPU | decoder LR | tmux session | 工作流PID | 成品目录 |
| --- | --- | --- | --- | --- |
| 0 | 1.25e-5 | lq_full_a_20260924 | 2831293 | decoder_1p25e5 |
| 1 | 6.25e-6 | lq_full_b_20260924 | 2831297 | decoder_6p25e6 |

日志分别为正式根目录a.log、b.log；工作流状态在各子目录run.json，最终进程退出码为a.exit/b.exit。实际调用为激活bitvae后 `bash experiments/liftquant_recovery/run_full_decoder_1p25e5.sh` 和 `bash experiments/liftquant_recovery/run_full_decoder_6p25e6.sh`。有限启动检查已确认GPU0至少完成block0第288步、GPU1至少第384步，loss有限；两组正常推进后立即结束启动检查，不等待完整训练。

短测全部已结束，无后续权重用途，删除两份完整短测模型及两个boundary共14,853,037,668字节（约13.83GiB），保留真实配置、核心日志、原始指标和用于驻留回归的token输入。初始模型、完整数据及正式任务不受影响。

用户补充允许在有依据时利用空闲显存增大batch。本次保留已验证的batch2：固定样本与epoch时，batch8会使每块更新从3968降至992，连同码阈值跨越和调度发生变化，现有证据不足以判断更优。两臂继续只比较decoder LR；该决定仅针对本次对照，不形成后续batch限制。

启动核验完成时间（UTC）：2026-09-24T01:37:26.703845+00:00。正式两组4096×2048实际输入SHA与准备阶段一致。一次性数据验证token副本已删除67110213字节，实际正式运行token仍保留；11原始Arrow与初始checkpoint受保护。短测CLI/退出码/清理结果已合并到summary及单份validation.log，已用完的一次性脚本和重复小文件已清理。本轮未提交Git、未改环境，未以短测指标声称下游改善。
