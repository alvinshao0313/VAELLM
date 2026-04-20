#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,2}"

# 说明：
# - 这是 OpenOrca 蒸馏脚本，序列长度固定为 4096，本身显存压力就很高。
# - 当前脚本显式开启梯度检查点，优先压低训练显存。
# - proxy dense base 会在 fresh 建 proxy 后、resume 加载后自动 grouped materialize，优先走当前训练 device。
# - --prewarm_frozen_vae 现在只负责冻结层 VAELinear cache prewarm，不负责 proxy materialize。
# - 以下配置是有意为之，不是漏配：
#   --teacher_model_path 显式传基础 teacher
#   --save_strategy / --save_steps 显式写成 HF 默认值，保持当前行为
#   --unload_vae_original_weights_on_save true，优先导出更紧凑的最终模型
# - 以下参数当前为 None，不传空字符串以免改变语义：
#   --access_token --resume_from_checkpoint --dataset_config_name
#   --train_file --eval_file --max_train_samples --max_eval_samples

#   --gradient_checkpointing true \
#   --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
#   --ddp_find_unused_parameters true \

torchrun --standalone --nproc_per_node=2 -m e2e_fintuning.main \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --ddp_find_unused_parameters true \
  --report_to none \
  --student_checkpoint_dir .result/meta-llama_Llama-2-7b-hf_20260323_071142/final_model \
  --teacher_model_path meta-llama/Llama-2-7b-hf \
  --run_root_dir .result/e2e_vae_lora_openorca \
  --dataset_name Open-Orca/OpenOrca \
  --train_split train \
  --eval_split validation \
  --text_field text \
  --loss_type kd \
  --distill_temperature 1.0 \
  --distill_alpha 0.3 \
  --post_attn false \
  --model_max_length 4096 \
  --decoder_layers 0-31 \
  --target_modules all \
  --vae_lora_variant dora \
  --vae_lora_rank 8 \
  --vae_lora_alpha 16 \
  --vae_lora_dropout 0.0 \
  --vae_lora_tune_bias false \
  --tune_final_norm true \
  --use_post_norm_head_linear false \
  --vae_lora_init_mode zero \
  --vae_adalora_target_r 8 \
  --vae_adalora_init_r 12 \
  --vae_adalora_tinit 0 \
  --vae_adalora_tfinal 0 \
  --vae_adalora_delta_t 1 \
  --vae_adalora_beta1 0.85 \
  --vae_adalora_beta2 0.85 \
  --vae_adalora_orth_reg_weight 0.5 \
  --lora_hif4_act false \
  --eval_hif4_act false \
  --prewarm_frozen_vae true \
  --prewarm_log_every 32 \
  --prewarm_group_size 8 \
  --skip_ppl_eval false \
  --ppl_seqlen 2048 \
  --ppl_limit -1 \
  --save_tokenizer true \
  --unload_vae_original_weights_on_save true \
  --bf16 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 5000 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.0 \
  --weight_decay 0.0 \
  --seed 42 \
  --max_steps 10000 \
  "$@"
