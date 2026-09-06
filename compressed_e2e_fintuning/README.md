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
