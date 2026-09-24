# VAELLM

当前训练栈只有三条正式链路：

- CAT 类别压缩：`tools/cat_train.py`
- 已压缩模型的端到端恢复训练：`python -m compressed_e2e_fintuning.main`
- 已完成 VAE 压缩后的逐类别恢复：`tools/cat_distill_from_vae_checkpoint.py`

三条链路共享 `train_utils.config` 的数据、loss、目标范围和运行时配置。模型级 loss 仅支持 `sft`、`kl`、`kl_top`、`kd`、`kd_top`；top-k 必须通过独立的 `--top_k` 指定。

## 推荐流程

```bash
bash scripts/catlora_simple2.sh
bash compressed_e2e_fintuning/scripts/e2e_decoder.sh
python tools/cat_eval.py --checkpoint_dir .result/example/final_model
```

CAT online 支持六种 `--after_category_mode`：

- `current_decoder`
- `current_lora`
- `current_lora_decoder`
- `remaining_lora`
- `remaining_lora_current_decoder`
- `remaining_lora_prefix_decoder`

checkpoint-distill 只支持前三种 current 模式。CAT/E2E 的模型级 LoRA 只支持 plain full-space LoRA。独立 block-VAE/block-distill 路径和 E2E stage1/stage2 脚本已经删除。

## 目标范围

- CAT 用 `--compression_categories`、`--target_layers` 和 `--skip_layers` 定义压缩集合。
- E2E 用 `--target_layers` 和 `--target_modules` 定义训练集合。
- `skip_layers` 只允许引用已发现的 CAT projection inventory；不会把普通 dense Linear 当作压缩目标。

## Checkpoint v6

所有正式读写统一使用 `vaellm_model_checkpoint_v6`，schema version 为 6：

- `training_step`：用于精确续训，引用不可变的 `round_base`，并保存 optimizer/scheduler/RNG 等可变状态。
- `round_base`：一轮训练的独立完整基座。
- `category_boundary`：CAT 完成一个精确类别前缀后的稳定完整模型。
- `final_model`：完成 finalize 后可独立加载的稳定模型。

稳定 checkpoint 已把 LoRA 写回每个目标的 `low_rank_a/b`；拓扑由各目标 payload shape 定义，metadata 中 `lora_config` 必须为 `null`。旧 checkpoint 不再被正式训练/评估入口直接加载，先执行：

```bash
python tools/migrate_checkpoint_v6.py \
  --source /path/to/legacy/checkpoint \
  --output_dir /path/to/new/v6/checkpoint \
  --dry_run false
```

`--dry_run true` 只检查可迁移性，不写文件。specialized bitpack/candidate 工具保留自己的紧凑 artifact schema，但完整模型输入输出仍走 v6。

详细参数见 [CAT 参数](docs/guides/cat_training.md)、[checkpoint-distill](docs/guides/catlora_distill_from_checkpoint.md) 和 [E2E](compressed_e2e_fintuning/README.md)。

## 项目文档

使用指南、模块说明、经验、实验结果和历史材料统一从 [文档入口](docs/README.md) 查找。维护方式见 [文档管理规范](docs/MANAGEMENT.md)。

## 经验与实验记录

实验设计前先检索 [经验库](docs/experience/README.md)，阅读相关经验及来源，避免重复踩坑。经验、原始结果、历史方案分别维护；完整来源见 [目录](docs/experience/CATALOG.md)。
