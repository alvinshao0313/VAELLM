# 从 v6 VAE checkpoint 做逐类别恢复

入口：

```bash
python tools/cat_distill_from_vae_checkpoint.py \
  --model_path Qwen/Qwen3-8B \
  --resume_from_checkpoint /path/to/v6/final_model \
  --output_dir .result/catlora_distill \
  --compression_categories "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --target_layers all \
  --after_category_mode current_lora \
  --dataset_mix "openorca=1.0" \
  --loss_type "default=kl_top" \
  --top_k "default=100"
```

该入口不重新训练 VAE，只重放现有 VAELinear 类别的恢复阶段。它只接受：

- `current_decoder`
- `current_lora`
- `current_lora_decoder`

remaining-family 只属于 online CAT。LoRA 仅为 plain full-space。

输入必须是可独立加载的 v6 full checkpoint。每类成功后写稳定 `category_boundary`；最终写 `final_model`。checkpoint-distill 的进度使用独立的 `checkpoint_distill_completed_categories`，不会修改 online CAT compression progress。

`--distill_reset_completed false` 按已完成类别跳过；`true` 只重置 checkpoint-distill 自己的完成列表。`--distill_independent_categories true` 每轮只激活当前类别，默认则使用前缀累积状态。

精确训练续跑使用 `training_step`，其中包含 `round_base_ref`、`round_base_checkpoint_id` 和可变训练状态。稳定 boundary/final checkpoint 已 finalize，`lora_config=null`，可脱离训练 sidecar 单独加载。

legacy checkpoint 必须先运行 `tools/migrate_checkpoint_v6.py`；不再自动识别旧 block/E2E/CAT 格式。
