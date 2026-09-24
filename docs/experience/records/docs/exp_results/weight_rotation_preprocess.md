> 归档于 2026-09-23；类型：实验结果/验证/恢复记录；不是独立的通用经验。
> 原路径：`docs/exp_results/weight_rotation_preprocess.md`。原文中的“当前/本目录/进行中”保留当时语境；实验数据仍在原结果路径。本次仅归档，未复跑或更新实验状态。
> 可复用结论从 [经验库入口](../../../README.md) 查阅。

# 局部双侧权重预处理：实现与最小验证

日期：2026-09-22。

## 结论

已实现可选的 QuIP# 风格局部双侧随机 Hadamard 权重预处理，保留原 `rot_llm` 的含义和默认行为。采用现有 VAE 压缩，不包含 QuIP# 的 Hessian/BlockLDLQ/E8 码本算法，也不新增在线激活旋转 kernel。

最小真实权重对照中，全维度双侧变换的重建 NMSE 在三个类别、两个种子上均低于不旋转；Q 的收益较明显，gate/down 较小。32 维双侧分块并未稳定优于当前单侧方案。尚不能据此认定整模型 PPL、KL 或任务精度会提升，更不能认定充分训练后仍有相同增益。

## 使用

新参数默认关闭：

```text
--weight_rotation none|two_sided   # 默认 none
--weight_rotation_block_size 32    # 默认 32；0 表示每侧完整维度
```

在原脚本后附加参数即可启用，无需修改脚本原配置：

```bash
bash scripts/catlora_simple2.sh \
  --weight_rotation two_sided \
  --weight_rotation_block_size 32
```

需要使用仓库规定的 bitvae 环境。`tools/cat_train.py` 将已有 `--seed` 传递到预处理，旋转种子由运行 seed 和完整模块名稳定派生；使用独立 CPU Generator，不推进 VAE 的全局随机数状态。程序直接调用 `train_group_vae_payload` 时，应显式提供 `training_args.seed`，缺省为 0。

限制：

- 与 `--rot_llm` 互斥，防止无意叠加全局换基与局部预处理。
- 首版只接受所有目标类别使用 `recon_loss_type=mse`。原输入通道上的 wa_mse/amse 权重不能直接用于旋转后的输入坐标，因此显式拒绝，而不是悄悄改变目标。
- 正块大小必须整除移除保护通道后的输入、输出维度。
- `block_size=0` 要求两个剩余维度均有仓库支持的精确 Hadamard 分解。例如移除 32 个输入通道后，4096 变成 4064，此维度不被完整 Hadamard 支持，将明确报错。不会补零、截断或自动改成分块。
- 因此，不要直接把当前开启通道保护的完整训练命令改成 `block_size=0` 就假定能运行。无保护的 Qwen3-8B 原始维度与保护后的维度是不同条件。

## 坐标与执行顺序

对待压缩子矩阵 W_c，使用固定正交 U、V：

```text
W_tilde = U @ W_c @ V.T
W_c_hat = U.T @ decode(bits) @ V
```

压缩：先按原坐标选择、移除保护通道；对剩余矩阵双侧变换；再执行现有转置/分块、阶段归一化和多阶段 VAE 残差编码。每个 residual stage 使用同一坐标，不反复旋转残差。

解码：现有 decoder 反归一化与矩阵重组；合并阶段；双侧逆旋转；回填原坐标保护通道；添加原坐标补偿。LoRA 和受保护通道的含义不改变。Linear 外部仍使用原来的输入、输出坐标，bias 也保持原坐标。

逆旋转统一放在 VAELinear._finalize_decoded_weight_from_compressed，普通解码和分组预热均经过此处。缓存的是恢复原坐标后的完整权重。缓存命中后不再做激活旋转；decoder 训练或禁用缓存时仍需执行逆旋转，会增加计算和临时显存，并非所有模式零开销。

Hadamard 运算使用 float32 累加，float64 参考输入保持 float64；关闭外围 autocast 后执行变换，最后恢复输入 dtype。非 Sylvester 因子逆变换使用转置，不能假定 Hadamard 一定对称。

## 保存与恢复

每个 VAELinear 的 v6 metadata 记录变换类型、版本、块大小和种子；int8 正负号向量作为持久 buffer 保存。加载时重建结构并严格加载状态，检查符号合法性。

默认不旋转时不增加 state_dict 键，也不改变旧的 CAT 跨类别恢复身份信息。启用时把模式与块大小纳入恢复身份，避免用不同预处理配置静默续训。旧模型不会因启用新参数而自动得到新的压缩码；需要重新压缩。

主要改动：

| 文件 | 职责 |
|---|---|
| `rotation/weight_preprocess.py` | 双侧 Hadamard、逆变换、确定性符号、状态契约 |
| `train_utils/config/configs.py`、`train_utils/config/cli.py` | 新配置、CLI 和互斥/损失检查 |
| `train_utils/cat_runtime_adapter.py`、`tools/cat_train.py` | 参数与已有 seed 传递 |
| `train_utils/cat_train_pipeline.py` | 保护后旋转、VAE payload 和模块转换 |
| `litebsq/vae_linear.py` | 统一逆旋转解码入口 |
| `train_utils/checkpoint_v6.py`、`train_utils/cat_runtime_state_v6.py` | v6 状态和恢复身份 |
| `tests/test_weight_rotation.py` | 长期回归测试 |
| `experiments/weight_rotation_ab.py` | 最小真实权重 A/B，可复现入口 |

没有修改用户已有 `compressed_e2e_fintuning/scripts/e2e_decoder.sh` 工作区改动，没有切分支、commit 或 push，没有新增依赖。

## 正确性验证

在 bitvae 环境、物理 GPU 4（A800）上执行：

```bash
conda run -n bitvae env CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  python -m pytest \
  tests/test_weight_rotation.py \
  tests/test_training_stack_checkpoint_v6.py \
  tests/test_training_stack_config_cli.py \
  tests/test_cat_runtime_state_v6.py \
  tests/test_vae_recon_loss_contract.py \
  tests/test_e2e_runtime_v6_modes.py -q
```

结果：**139 passed**。其中新增旋转测试 20 项，CUDA 用例实际执行，没有跳过。

覆盖：CPU 双精度 dense 参考、正交性、非对称 Hadamard 因子的正确逆变换；CPU gradcheck；GPU FP32/BF16 正反变换与梯度；完整 4096×12288 MLP 维度的 GPU 往返；两阶段/多分块/转置布局；input/output 通道保护、LoRA、原始权重切换、缓存；分组预热；真实 CAT 两阶段训练到 int8 保护通道转换，再 v6 保存/重载和重载后 decoder 反向。

容差：CPU float64 参考为 2e-12；GPU float32 正反变换约 2e-6，梯度约 1e-5；两次 BF16 存储允许 relative L2 < 0.007；完整 MLP 维度 float32 往返 relative L2 < 2e-6。相同设备、相同 decoder 状态的 v6 解码结果做零容差一致性检查。

这不是整模型多卡蒸馏长程验收，也没有单独宣称完成 Sparse Bit、所有 offload 模式或旧版本 checkpoint 导出路径的组合验证。

## 真实权重最小 A/B

使用本地缓存中真实 Qwen/Qwen3-8B 第 0 层的 q_proj、gate_proj、down_proj。每个矩阵取前 512 行、1024 列。输入 SHA256、原矩阵形状和 checkpoint snapshot 曾记录于原始 results.json；按清理要求，原始 JSON 已删除，不再作为可查阅的归档文件。本文件保留实验设置、汇总结果、分析与复现命令。

每类两种子 31/37，四个配置，共 24 个压缩结果：

```text
none             不旋转
current_r1       当前共享 32 维块的单侧权重旋转
 two_sided_32     局部双侧，块大小 32
 two_sided_full   局部双侧，两侧分别覆盖完整测试子矩阵维度
```

current_r1 对 q/gate 右乘，对 down 左乘，使用仓库原 random_hadamard_matrix。这里复现的是该单个权重的旋转，不包含整模型 RMSNorm 融合或残差流运行，不能当成完整 rot_llm 模型实验。

所有配置使用相同 VAE 容量、相同随机初始化/数据种子和训练预算：BSQ 每向量 32 bits、向量维度 32、两个 residual stages，即编码主体 2 bpw；BF16；encoder hidden=128 / resblocks=0，symmetric decoder hidden=128 / resblocks=1；layer norm、swish、normalize_weight、new_quant；每阶段 200 步，batch=2048；AdamW lr=0.003，linear scheduler，无 warmup/weight decay。所有类别统一为编码主体 2 bpw，与当前脚本针对 down 额外增加 bit 的配置不同。

本对照不启用保护通道和蒸馏，以隔离预处理；保护通道的组合功能另由上面的训练/保存/加载测试验证。单个子矩阵独立训练 decoder，不是跨 36 层共享 decoder 的正式规模。

评估在恢复原坐标后进行：

```text
NMSE = ||W_hat - W||_F^2 / ||W||_F^2
```

### 两个种子均值，越低越好

| 类别 | 不旋转 | 当前单侧 R1 | 双侧 block32 | 双侧 full | full 相对不旋转降幅 | full 相对当前 R1 降幅 |
|---|---:|---:|---:|---:|---:|---:|
| q_proj | 0.153412 | 0.151660 | 0.152976 | 0.142099 | 7.37% | 6.30% |
| gate_proj | 0.150835 | 0.152701 | 0.147159 | 0.146780 | 2.69% | 3.88% |
| down_proj | 0.149041 | 0.146622 | 0.147651 | 0.146123 | 1.96% | 0.34% |

没有统计显著性结论。尤其 down 的 full 与当前单侧差距很小。block32 在 q/down 上平均反而略差于当前单侧，仅 gate 有相对稳定的改善。full 在每个测试的 category×seed 配对中均低于不旋转，但只有两个种子且训练较短，可能包含收敛速度差异。

### 存储口径

测试中所有配置编码主体均为 2 bpw，decoder 容量相同，但并非完全相同总存储预算。每个 512×1024 子矩阵的旋转符号额外占 1536 bytes，即 0.0234375 bpw。

包含独立 decoder 和 tensor state 后：不旋转/current_r1 为 2.525390625 bpw，新双侧为 2.548828125 bpw。未计 JSON 文件/文件系统开销。当前 R1 的整模型融合存储成本也不由这个单权重宿主模型模拟。正式跨层共享 decoder 的摊销与此不同，不能把上述数字当成完整 Qwen3-8B 平均位宽。

### 数据清理与复现

2026-09-23 按用户要求删除本次原始实验目录 `result/weight_rotation_ab_20260922/`。实验记录仅保留本文件；实现代码、可复现实验入口和长期回归测试保留，其他实验数据不在清理范围内。

已删除 q/gate/down 的 seed31、seed37 六组主对照，以及 `q_seed31_recheck` 和 `smoke_q_seed31` 两组验证目录，共 8 个子目录、16 个 JSON/日志文件。上表仍为六组主对照的两个种子均值；2 步 smoke 和重复验证均未计入。

历史验证说明：初次实验 logger 未写入 training.log，随后修复日志绑定并重复 Q、seed31 的完整四模式对照，NMSE 与首次逐项一致。该重复验证的原始 JSON 和日志也已清理；此处保留验证结论，不再提供已删除文件的查阅路径。

复现一个类别（重新生成结果；使用新的输出目录，已有目录会报错）：

```bash
conda run -n bitvae env CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  python -m experiments.weight_rotation_ab \
  --category q_proj --rows 512 --cols 1024 \
  --steps 200 --batch-size 2048 --seed 31 \
  --output-dir result/weight_rotation_ab_reproduce/q_seed31
```

替换 category 为 gate_proj/down_proj、seed 为 37 可复现其余配对。

## 当前判断

功能已接通并通过上述验证。小样本重建结果支持把全维度双侧作为后续实验分支，但不支持直接替换默认方案。下一阶段应比较更多层和完整权重矩阵、足够的 VAE 训练预算、真实激活加权误差以及相同蒸馏预算后的模型级 KL/PPL/任务指标；这些尚未执行。
