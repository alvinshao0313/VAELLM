# CAT 训练：架构与公共参数

## 当前架构

CAT online 与 checkpoint-distill 共用 `train_utils.cat_after_category_common`：同一套目标 inventory、数据、五种模型级 loss、plain full-space LoRA、decoder finalize 和 v6 exact-resume 逻辑。

在线 CAT 有六种 `after_category_mode`；checkpoint-distill 只允许 current-family 三种。两条入口必须显式声明调用来源，不再根据目标数量或模型拓扑猜测。

稳定 checkpoint 与训练 step 分离：

- `training_step` 保存 optimizer/scheduler/RNG、当前组件状态和 `round_base` 身份，供精确续训。
- `category_boundary`/`final_model` 保存 finalize 后完整模型，不携带活动 PEFT proxy，`lora_config=null`。

旧 parser、compressed-subspace、模型级 DoRA/RSLoRA/AdaLoRA、旧 loss family 和 block-distill 路径不再属于 active stack。正式脚本见 `scripts/catlora_simple2.sh`、`scripts/catlora_distill_from_checkpoint.sh`。

## 公共 CLI

`tools/cat_train.py` 使用 `train_utils.config.cli.parse_cat_cli`。正式脚本只使用这里的 public 参数；旧 CAT 参数名会直接报错。

## 最小示例

```bash
python tools/cat_train.py \
  --model_path Qwen/Qwen3-8B \
  --output_dir .result/catlora \
  --compression_categories "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --target_layers all \
  --vae_steps "default=10000" \
  --vae_batch_size 8192 \
  --vae_learning_rate 3e-3 \
  --after_category_mode none \
  --convert \
  --save_model
```

## 参数分组

- 压缩范围：`--compression_categories`、`--target_layers`、`--skip_layers`。
- VAE：`--vae_steps`、`--vae_batch_size`、`--vae_learning_rate`、`--vae_weight_decay`、`--vae_optim`、`--vae_lr_scheduler_type`，以及 codebook/decoder/recon 参数。
- channel protection：`--channel_protect_mode` 只支持 `none`/`channel`，配合 `--channel_scope`、`--channel_protect_count`、`--channel_rank_metric`、`--channel_axis`、`--channel_quant`；`--channel_refresh_after_category=true` 会在每类压缩/补偿完成后，以当前 student 状态刷新下一类别的保护统计（global scope 则对全部 remaining categories 重规划剩余预算）。
- 数据：`--dataset_mix`/`--train_file`、`--dataset_task`、`--model_max_length`、`--dynamic_padding`。
- 恢复训练：`--steps`、`--batch_size`、`--learning_rate`、`--decoder_lr`、`--weight_decay`、`--logging_steps`。
- LoRA/辅助参数：plain full-space `--lora_rank`、`--lora_alpha`、`--lora_dropout`，以及 `--norm_train_mode`、`--lm_head_train_mode`。

类别参数支持 `default=...` 与 `cat:<category>=...`；after-category 参数支持 `default=...` 与 `after:<category>=...`。

## Online 六种模式

`--after_category_mode` 可取：

| 模式 | 训练对象 |
|---|---|
| `current_decoder` | 当前新压缩类别的 decoder |
| `current_lora` | 当前新压缩类别的 full-space LoRA |
| `current_lora_decoder` | 当前类别 decoder + LoRA |
| `remaining_lora` | 尚未压缩的 dense projection LoRA |
| `remaining_lora_current_decoder` | remaining LoRA + 当前类别 decoder |
| `remaining_lora_prefix_decoder` | remaining LoRA + 已完成前缀 decoder |

非 `none` 模式必须提供数据源。模型级 loss 仅支持 `sft`、`kl`、`kl_top`、`kd`、`kd_top`；K 使用独立 `--top_k`。

## v6 保存

类别完成后可写 `category_boundary`，全部完成后写 `final_model`。`completed_categories` 必须是精确前缀。训练中的 LoRA/decoder 状态只属于 exact-resume `training_step`；稳定 checkpoint 已 finalize，per-target `low_rank_a/b` shape 是唯一 LoRA 拓扑真相，`lora_config=null`。

旧 checkpoint 请先运行 `tools/migrate_checkpoint_v6.py`。独立 block-VAE/block-distill 路径已删除。
