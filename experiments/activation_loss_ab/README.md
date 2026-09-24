# VAE 分组输出损失：最小下游精度对照

这是独立实验，不修改 CAT/E2E 的 CLI、默认损失、checkpoint 格式或训练脚本。

## 对照目标

固定一个完整、不转置、不分片的 Linear 权重矩阵，每行沿输入维每 32 个权重一组。输入维必须能被 32 整除。校准激活统计为未中心化二阶矩 `H_g = E[x_g x_g^T]`。

`delta` 是经过生产路径 residual-stage 均值/标准差归一化后的权重重建误差：

- `mse`：生产 AutoEncoder 的普通逐元素 MSE。
- `amse`：生产 AutoEncoder 的 `mean(delta^2 * diag(H_g))`；通道统计仍按生产路径转换到训练 dtype。
- `block_output`：`mean_g(delta_g^T H_g delta_g) / 32`，二次型使用 FP32。除以 32 是为了在 H 为单位矩阵时与逐元素 MSE 一致，而不是额外放大重构项 32 倍。

BSQ 辅助损失保留原公式和权重。两个激活相关模式均使用未做尺度归一化的原始激活统计，没有按 trace 归一化，没有阻尼，也没有附加 MSE 正则。因而本实验回答的是“在本次统一的短程训练预算与原有辅助损失设置下直接替换”的效果，而不是各目标分别调到最优后的比较。

`training.py` 复用生产数据准备、MultiLayerVAE/BSQ、stage normalization、优化器和 packed payload 转换，仅实现当前实验需要的单矩阵训练循环。`mse`、`amse` 的两阶段训练在真实 Qwen 权重小块上与 `train_group_vae_payload` 做逐权重零误差验证；不是 mock 或随机小模型。

推理时先用生产 VAELinear 解码真实压缩 payload，再把解码权重以 BF16 回填完整 Qwen3-8B，测 LM-eval 下游精度。这不是压缩 kernel 的速度测试，也不是整个模型都量化到 2 bit。

## 历史运行与结果

2026-09-22 的配置、指标和解释集中在 [实验结果报告](../../docs/experience/records/.result/activation_loss_ab/20260922_input32_downstream_v2/REPORT_CN.md)，可复用结论见 [输出损失经验](../../docs/experience/lessons/output_losses.md)。原始证据仍在 `.result/activation_loss_ab/20260922_input32_downstream_v2/`。本页只维护方法、代码入口与重跑方式。

## 文件

- `objectives.py`：二阶矩及带真实 block 索引的分组输出损失。
- `calibration.py`：无 padding 的实际模型输入激活采集。
- `training.py`：受限、经 CAT 一致性验证的训练及真实压缩转换。
- `evaluation.py`：官方 LM-eval 题目/指标、逐题对齐、配对 bootstrap 和 McNemar 检验。
- `run.py`：prepare / variant / summarize 入口。
- `test_correctness.py`：8 项必要回归测试；历史验证结果见上方报告。

## 重跑

在已激活的 bitvae 环境中设置：

```bash
export CUDA_VISIBLE_DEVICES=4
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m pytest experiments/activation_loss_ab/test_correctness.py -q
python -m experiments.activation_loss_ab.run prepare --output-dir .result/activation_loss_ab/new_run
python -m experiments.activation_loss_ab.run variant --output-dir .result/activation_loss_ab/new_run --mode mse --seed 31
```

对 `mse / amse / block_output` 与 `31 / 47` 的六个组合分别执行 variant，再运行：

```bash
python -m experiments.activation_loss_ab.run summarize --output-dir .result/activation_loss_ab/new_run
```

输出目录和模式目录必须是新目录，不自动覆盖旧结果。首次 `_v2` 之前的目录只包含一次结果序列化失败时留下的校准/权重记录，不应作为完成实验使用。
