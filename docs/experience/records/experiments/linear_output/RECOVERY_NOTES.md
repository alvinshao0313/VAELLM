> 归档于 2026-09-23；类型：实验结果/验证/恢复记录；不是独立的通用经验。
> 原路径：`experiments/linear_output/RECOVERY_NOTES.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## 2026-09-23 当前状态：已停止并保存模型

用户已停止训练并取消 weight-MSE 对照，不再执行下文的自动对照流程。
最后可用的已组装模型已单独保存在 `/home/shaoyuantian/program/VAELLM/result/linear_output/output_alignment_v6_partial192`，共 192 个压缩 Linear、60 个原始 BF16 Linear；不是完整 W2。
八项全量 0-shot 简单平均 39.54%，结果与校验信息随 checkpoint 保存。
旧测试与失败启动、未完成恢复训练、MSE 对照数据已清理；正式输出对齐的分模块产物仍保留。
下文为历史配置和运行记录，不能作为仍在运行的任务状态。

# 2026-09-23 导出失败与恢复记录

## 实际状态

首次全量输出对齐实验 `full_output_w2_v5` 的 252 个 Linear 均完成 5000 次更新。

| 分片 | 已完成更新 | 已保存 packed Linear | 状态 |
|---|---:|---:|---|
| worker_0_9 | 5000 | 3 / 63 | 第四个 Linear 导出失败 |
| worker_9_18 | 5000 | 63 / 63 | COMPLETE |
| worker_18_27 | 5000 | 63 / 63 | COMPLETE |
| worker_27_36 | 5000 | 63 / 63 | COMPLETE |

没有完整 final_model，没有下游精度结果，weight_mse 对照尚未由旧 watcher 启动。
不能从单层训练损失或部分保存结果推断完整 W2 模型精度优于旧 CAT。
旧失败分片的 manifest 保留了过期的 TRAINING 状态；进程已退出。

## 原因与数据损失

`model.layers.0.self_attn.o_proj` 的 training-to-packed 相对 L2 为 0.0239107，超过旧 0.02 门限。
旧校验混合了不同数值路径：BF16 训练 decoder、FP32 参数融合，以及 CUDA native decoder 内的 TF32 运算。
这不是一个只检验保存/恢复正确性的同精度比较。

旧 runner 在开始导出前没有先保存全部 VAE 状态，故第 0–8 层后续 60 个 Linear 的最终训练状态已丢失，日志不能恢复。
其余三个完整分片和前三个成功 packed 模块都保留不动。

旧 watcher 另有 worker_* 会同时匹配 .log 文件、未指定 lm-eval tasks、后处理失败缺少严格检查等问题。
不再使用旧 watcher 的等待/合并逻辑。

## 本次修复

- 保留训练 encoder 实际选出的 bits，使用未融合 FP32 decoder 与 CPU FP32 packed decoder 做严格等价检查；门限 5e-6。
- BF16 到部署解码、CUDA 到 CPU FP32 的差异分别记录，不伪装成严格保存一致性。
- 保存后按同一部署路径重新加载，再验证 round-trip；导出不修改 live VAE 的参数或 train/eval 状态。
- 任何导出前，原子保存整个分片的 `final_training_state.pt`，含 VAE encoder/decoder、optimizer、scheduler、配置、步数和来源校验值。
- 每个成功导出立即持久化 manifest；失败明确记录 FAILED 和模块名。
- `--export_only` 从最终训练状态重新导出，不使用校准数据，不重新训练。它不能恢复本次旧版本未保存的 60 个 Linear。

## 恢复运行方式

入口：`python -m experiments.linear_output.recover_experiment --root result/linear_output/recovery_20260923`

必须先在 shell 中激活 bitvae。入口会拒绝覆盖既有结果目录。

1. GPU 4 以原始初始化顺序、seed=31、data_seed=31、5000 步和 batch 8 重跑第 0–8 层整个分片。
   重跑 63 个而不是仅 60 个，以保持原始 VAE 初始化次序；其余 189 个模块复用已完成的原分片。
2. GPU 5–7 同时训练 weight_mse 的第 9–35 层对照。
3. 输出模型满足全部 252 个模块后合并到 `result/linear_output/full_output_w2_v5/final_model/`，然后在 GPU 4 评测。
4. GPU 4 的评测退出后再运行 weight_mse 第 0–8 层，避免同卡竞争。
5. 对照全部完成后保存到恢复目录的 `weight/final_model/`，再评测。

评测明确使用 boolq、rte、winogrande、arc_easy、arc_challenge、openbookqa、piqa、mmlu，0-shot，完整任务集，不设样本上限。
`cat_eval --eval_linear_mse` 测量的是权重重构 MSE，不是校准激活上的输出 MSE。
本轮没有增加 LoRA 或 E2E 精度恢复，不能直接视作与包含这些步骤的历史 CAT 结果等预算比较。

恢复目录 `status.json`、`logs/` 保存各进程、命令、代码 hash、退出状态。
两组评测均完成且来源权重、初始化、步数、batch、阶段数一一对应后才生成 `COMPARISON.md`。
所有新运行在导出前保存最终 VAE 状态；旧版已结束分片的 encoder/optimizer 不存在，不能补生成。

## 验证

11 项 CPU/CUDA 测试通过：输出损失、梯度、VAE 恢复、打包/重载、导出中断、损坏工件和保存失败状态。
小型 CUDA 测试的严格 FP32 融合误差为 3.89e-7；部署数值差异另列，不用放宽一致性门限掩盖错误。
真实一层训练/导出和加载评测的 smoke 状态见对应 recovery_* 日志；smoke 不属于正式精度结果。

实际恢复启动：2026-09-23 01:12:29 UTC。真实一层七种投影 2-step smoke 已 COMPLETE（7/7），final_training_state.pt 已保存，FP32 融合相对误差均低于 4.2e-7、packed round-trip 最大差均为 0；已保存整模型的 BoolQ 2-sample 加载评测 smoke 通过。短跑结果被完整评测结果校验器明确拒绝，不计入正式精度。
