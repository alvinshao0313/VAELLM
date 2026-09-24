> 归档于 2026-09-23；类型：实验结果/验证/恢复记录；不是独立的通用经验。
> 原路径：`experiments/linear_output/EXPERIMENT_SUMMARY.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

## 当前交付：蒸馏初始化 checkpoint

训练已停止，MSE 对照已取消。可加载的 checkpoint 位于 `/home/shaoyuantian/program/VAELLM/result/linear_output/distill_init`。
已通过 CAT checkpoint 蒸馏、E2E 两条生产加载器，以及实际 decoder 反向传播验证。
模型范围为 192 个压缩 Linear 与 60 个原始 BF16 Linear。具体加载参数见该目录 README.md。
用户已明确确认删除，历史目录清理完成；result/linear_output/ 仅保留 distill_init/。

# W2 Linear Output 对齐实验说明

## 目的

验证一种独立于现有 CAT/E2E 流程的 W2 压缩训练目标：

- 保持 Linear 权重原始布局 [out_features, in_features]；
- 不做转置、旋转或 channel protection；
- 将输入通道轴按连续的 32 个权重划分为一个向量块；
- 每个目标 Linear 独立训练一个 VAE；
- 用 Linear 在校准激活上的输出误差作为主重构目标；
- 与相同 VAE、相同训练预算下的传统 weight-MSE 目标比较下游精度。

本实验只新增和使用 experiments/linear_output/ 下的代码，不修改现有 CAT/E2E 流程。

## W2 配置

W2 在本实验中指单阶段的：

- codebook_dim = 32
- codebook_bits = 64
- code payload = 64 / 32 = 2.0 bits per weight
- residual_stages = 1

不使用残差阶段。保存时同时记录包含 decoder 等参数的实际 Linear state 开销。

## 主目标

对一个权重矩阵 W、其重构矩阵 W_hat 和校准输入 X，优化：

L_out = 1 / (N_valid * d_out) * || X (W_hat - W)^T ||_F^2

实现先恢复 VAE 输出的原始权重尺度，再对所有输入块的贡献累加后计算完整输出误差。不会把每个 32 维块的误差分别计算后再相加，因此保留了不同输入块之间的交叉项。

原始 bias 不参与优化；因为 teacher 和重构 Linear 使用同一个 bias，bias 在误差中抵消。

## 辅助损失

BSQ 的 commitment、entropy 和其他原有辅助项保持不变，仍对完整权重块集合统一计算。解码可以分块以控制显存，但不会把全局熵项错误地改成分块熵的平均值。

## 校准数据和教师输入

使用冻结、未压缩的原始 Qwen3-8B：

--dataset_mix "edgerazor_ii_7m=0.614,edgerazor_ii_gen=0.121,edgerazor_tulu=0.050,edgerazor_am=0.115,vaellm_eval_task=0.100"
--dataset_task sft
--model_max_length 1024
--dynamic_padding true

每次 teacher forward 捕获目标 Linear 的输入激活，去除 padding token，只保留有效 token。第一轮实验使用 teacher-input 口径：所有目标 Linear 都使用原始教师模型产生的激活，不把前面已压缩层的误差传入当前 Linear。

后续如需研究误差补偿，再单独实现按网络顺序的 student-input / teacher-target 消融，不与本轮结果混合。

## 训练预算

- 目标 Linear：q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj；
- Qwen3-8B 共 36 层，目标数 36 × 7 = 252 个 Linear；
- 每个 Linear：5000 次 optimizer update；
- 校准序列 batch：8；
- 每次更新使用该 Linear 的全部 32 维权重块；
- 每个 Linear 只有一个 VAE、一个训练阶段；
- VAE 权重和优化器状态独立保存。

## 计算实现

output_kernel.py 提供 PyTorch 参考实现、CUDA/Triton 路径和宽矩阵 cuBLAS torch.mm 路径，均计算完整的 X (W_hat - W)^T 目标。自定义 autograd 梯度与该目标一致。

训练过程中权重块常驻 GPU，避免每个 step 重复从 CPU 搬运完整矩阵。VAE 解码使用显存分块；本轮 worker 使用 vae_chunk_vectors=262144。

## 当前输出目录

本轮输出目录：

result/linear_output/full_output_w2_v5/

四个 worker 的层段和 GPU：

| GPU | 层段 | 输出 |
|---|---:|---|
| 4 | 0–8 | worker_0_9/ |
| 5 | 9–17 | worker_9_18/ |
| 6 | 18–26 | worker_18_27/ |
| 7 | 27–35 | worker_27_36/ |

每个 worker 包含 manifest.json、逐步 training.jsonl、每个 Linear 的 record.json，以及 packed/ native v6 checkpoint。

四个 worker 使用同一套确定性数据配置和冻结教师模型。并行只划分目标 Linear，不共享或覆盖 VAE 参数。

## 完成后的自动流程

watch_full_experiment.sh 会等待四个 worker 全部达到 COMPLETE，然后：

1. 使用 merge_workers.py 合并 252 个 packed Linear；
2. 保存完整 v6 模型；
3. 运行 cat_eval.py 的 Linear MSE 和 lm-eval；
4. 用完全相同的配置运行 weight_mse 对照；
5. 合并并评测对照模型。

## 解释结果时需要保留的口径

本流程是逐 Linear 独立训练的对比实验，不等同于原 CAT 中按类别共享或并行训练的 VAE。比较时应同时报告：

- 输出对齐目标下的 Linear MSE；
- weight-MSE；
- code payload bpw；
- 包含 decoder 的实际存储开销；
- 完整模型的下游任务结果；
- 与现有 CAT baseline 的训练配置差异。

生成时间：2026-09-22
