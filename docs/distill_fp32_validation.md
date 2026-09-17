# FP32 蒸馏参数验证记录

## 单模块显存对照

2026-09-17，在 `bitvae` 环境、A800 80GB 上测试真实 `VAELinear` / `Decoder` 模块，未启动完整 8B 训练。

代表形状取自 `Qwen_Qwen3-8B_20260910_180322/final_model` 的 MLP up_proj：压缩权重 `12288 × 4064`，两个 stage，每个 decoder 为 `32 → 128 → 32`，symmetric、LayerNorm、无 residual block。每 stage packed uint8 编码形状为 `[1560576, 1, 4]`。测试使用固定随机种子的编码、权重和 4 个 token 输入，开启 decoder checkpoint 重计算，使用 AdamW，连续运行 3 步，记录预热后第 3 步。

两组均使用 BF16 计算，初始参数数值相同，只有参数存储精度不同：

| 参数精度 | 峰值 allocated | 峰值 reserved | 梯度 / AdamW 动量 |
| --- | ---: | ---: | --- |
| BF16 | 4064.78 MiB | 4630 MiB | BF16 / BF16 |
| FP32 | 4065.11 MiB | 4654 MiB | FP32 / FP32 |
| 差值 | **+0.33 MiB** | +24 MiB | |

两组的 ParallelLinear、Normalize 输出及最终层输出均为 BF16，并命中 packed uint8 前向核。reserved 包含 CUDA allocator 的缓存分配，不能当作有效张量的额外开销。

这只是当前代表形状的单模块实测，不能直接推算整模峰值。该 checkpoint 的全部 decoder 共 4,633,344 个浮点参数；若全部训练，参数、梯度及两个 AdamW 动量各由 2 字节增至 4 字节，固定存储增量约 **35.35 MiB**。这不包括临时工作区、激活驻留、通信和 LoRA/norm/head；其他优化器也可能采用不同状态存储。

## 验证范围

`tests/test_distill_precision.py` 使用真实 Qwen3、PEFT、VAELinear 和 AdamW 模块，检查：

- CAT/E2E CLI、无效组合、未启用组件、冻结参数以及 head 的各种训练模式。
- FP32 小更新累积、梯度及 AdamW 动量精度、FP32 LoRA 初始化与融合。
- CAT/E2E 可变状态恢复、组件选择的恢复约束、三种导出精度、压缩码不变和下一阶段训练一致性。
- CUDA BF16/FP16 前向、反向、GradScaler 和 checkpoint 重计算；单/多 stage、linear/symmetric/asymmetric decoder、layer/rms/group/batch norm。
- decoder 与 Sparse Bit 联合梯度，以及 FP32 参数下的低精度批量预热。

`tests/test_e2e_decoder_sparse_exact_resume.py` 另外使用真实 Trainer、Sparse Bit 管理器和 v6 训练步断点，对比连续训练与中断恢复后的参数、优化器、调度器和 Sparse Bit 状态。

这些检查验证数值存储、计算路径与保存恢复行为，不证明下游任务精度一定提高。任务收益仍需要使用同一数据、种子和训练步数进行对照实验。

本次结果：新增精度检查 **50 项通过**；Trainer/Sparse Bit 断点恢复及融合核检查 **5 项通过**；相关 CLI、CAT/E2E、checkpoint 和参数快照回归 **284 项通过**。最终调整后重跑的 60 项相关检查也全部通过。两份实验脚本通过 `bash -n`，工作区通过 `git diff --check`。未新增依赖、mock 或 stub，未运行完整 8B 训练。

## 修改文件

| 文件 | 修改内容 |
| --- | --- |
| `train_utils/config/configs.py` | 组件解析、规范化和共享优化配置 |
| `train_utils/config/cli.py` | CAT/E2E CLI 与配置传递 |
| `train_utils/distill_precision.py` | 选择性 FP32 参数、计算边界、导出与加载精度 |
| `train_utils/model_level_trainables.py` | 将精度选择传入 LoRA 初始化 |
| `e2e_common/full_lora.py` | 保留 FP32 初始化精度，FP32 分块融合 |
| `e2e_common/post_norm_head.py` | head 融合直接写入目标精度 |
| `litebsq/parallel_layers.py` | linear、bias、norm 的混合 dtype 计算 |
| `litebsq/vae_linear.py` | decoder 参数与计算精度分开，覆盖重计算和无梯度路径 |
| `litebsq/vae_linear_prewarm.py` | 预热输出及缓存采用计算精度 |
| `litebsq/fused_multistage_decoder.py` | 预热融合核使用相同计算 dtype 的小权重副本 |
| `train_utils/cat_after_category_common.py` | CAT 当前/剩余类蒸馏接入，阶段模型同步转换 |
| `train_utils/cat_checkpoint_v6.py` | CAT 完整模型导出前应用保存精度 |
| `train_utils/cat_checkpoint_distill_v6.py` | CAT checkpoint 蒸馏模型导出精度 |
| `train_utils/checkpoint_v6.py` | 加载前恢复参数 dtype，避免先舍入再升精度 |
| `train_utils/cat_step_resume_v6.py` | CAT 组件选择进入精确恢复约束 |
| `compressed_e2e_fintuning/v6_runtime_state.py` | E2E 组件选择进入精确恢复约束 |
| `compressed_e2e_fintuning/runtime_v6_pipeline.py` | E2E 接入、导出后评估及独立精度误差记录 |
| `compressed_e2e_fintuning/scripts/e2e_decoder.sh` | 两个实验分支各增加默认 `none` CLI 行 |
| `scripts/catlora_simple2.sh` | 增加默认 `none` CLI 行 |
| `tests/test_distill_precision.py` | 新增上述 50 项精度检查 |
| `tests/test_cat_after_category_common.py` | 既有配置 fixture 补齐新字段默认值 |
| `tests/test_e2e_decoder_sparse_exact_resume.py` | 真实 Trainer 增加 FP32 decoder 的断点恢复对照 |
| `compressed_e2e_fintuning/README.md` | 参数范围、保存语义、续训规则与用法 |
| `docs/distill_fp32_validation.md` | 验证范围、结果和显存实测记录 |
