# Compressed E2E v6

正式入口是：

```bash
python -m compressed_e2e_fintuning.main \
  --student_checkpoint_dir /path/to/v6/final_model \
  --dataset_mix "openorca=1.0" \
  --train_mode decoder
```

入口只调用 `runtime_v6`。输入必须是完整 v6 `round_base`、`category_boundary` 或 `final_model`；legacy checkpoint 必须先用 `tools/migrate_checkpoint_v6.py` 转换。

## 训练模式

`--train_mode` 由 `decoder`、`lora`、`sparse_bit` 三种组件组合：

- `none`
- `decoder`
- `lora`
- `sparse_bit`
- `decoder_lora`
- `decoder_sparse_bit`
- `lora_sparse_bit`
- `decoder_lora_sparse_bit`

模型级 LoRA 只有 plain full-space 实现，参数为 `--lora_rank`、`--lora_alpha`、`--lora_dropout`。不支持 DoRA、RSLoRA、AdaLoRA 或 compressed-subspace LoRA。

`--target_layers` 接受 `all`、范围或显式层号；`--target_modules` 接受 `all` 或完整 projection 名集合。两者共同限定训练目标，普通未压缩 Linear 不会被当作 VAELinear decoder 目标。

## 数据与 loss

数据通过 `--dataset_mix` 或 `--train_file` 输入，`--dataset_task` 为 `sft` 或 `lm`。`--model_max_length` 是截断上限，`--dynamic_padding true` 按 micro-batch 动态 padding。

模型级 loss 只有：

```text
sft, kl, kl_top, kd, kd_top
```

`kl_top`/`kd_top` 的 K 用 `--top_k`；hidden 与 pre-MLP 对齐分别用 `--hidden_loss_weight`、`--pre_mlp_hidden_loss_weight`。

## 并行与正式脚本

`compressed_e2e_fintuning/scripts/e2e_decoder.sh` 只保留 shell 级 `dp`/`layer_mp` 分支：

- `dp`：`train_mode=decoder_sparse_bit`，`kl_top`，K=100。
- `layer_mp`：`train_mode=decoder_lora`，`kl_top`，K=1000。

`scripts/compressed_e2e_simple.sh` 是单入口示例。旧 `e2e_stage1_pretrain.sh`、`e2e_stage2_instruct.sh` 已删除。

## 保存与续训

`training_step` 用于精确恢复 optimizer/scheduler/RNG/组件状态，并通过 checkpoint id 绑定 `round_base`。稳定 `final_model` 可独立加载，所有临时 PEFT proxy 已 finalize，Sparse Bit score 已提交为硬 bit，`lora_config=null`。

## 按组件保留 FP32 训练参数

CAT 在线压缩后的蒸馏、CAT checkpoint 蒸馏和 E2E 共用 `--distill_fp32_components`，默认 `none`。例如在原实验命令中使用：

```bash
--distill_fp32_components decoder,norm,lm_head \
--bf16 true
```

可选组件为 `lora`、`decoder`、`norm`、`lm_head`，用逗号组合；重复值会去重，未知值以及 `none,norm` 这样的混用会报错。参数只调整已启用训练目标的精度，不会解冻新目标；启动日志列出指定组件、实际生效组件及参数量。

| 组件 | 范围 |
| --- | --- |
| `lora` | 主干普通层和压缩层的 LoRA |
| `decoder` | 当前训练的 VAE decoder，含内部 norm 和 bias |
| `norm` | `--norm_train_mode` 选中的主干 norm |
| `lm_head` | `--lm_head_train_mode` 选中的 linear、full 或 head LoRA |

选中参数及其梯度使用 FP32；普通 AdamW 的两个动量也为 FP32。`--bf16` / `--fp16` 继续决定计算精度，decoder 保留 packed uint8 路径，隐藏张量及重构权重保持对应的低精度。独立 VAE 重构训练、Sparse Bit 编码和 score 不受此参数控制。FP16 GradScaler 若仍遇到 FP16 可训练参数，会在创建 Trainer 前报出需要调整的组件。

训练步断点保留实际参数精度和优化器状态，不转换正在训练的参数。CAT 阶段模型、最终模型在 `bf16=true` 时导出 BF16，在 `fp16=true` 时导出 FP16；两者都未设置时保留实际精度。压缩码、整数索引和格式规定的编码数据保持原格式。CAT 内存中的阶段模型同步应用导出精度；E2E 最终评估也在转换后执行，融合误差和导出转换误差分别记录。

组件选择会进入配置快照和精确恢复约束。更改选择后可以从阶段/最终模型开始新实验，但不能作为同一个训练步断点的精确续训。默认 `none` 兼容旧断点的恢复约束。

显存实测及验证范围见 [FP32 蒸馏验证记录](../docs/distill_fp32_validation.md)。
