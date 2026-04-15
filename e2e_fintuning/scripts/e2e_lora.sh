#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# 说明：
# - 这是单卡 Alpaca-GPT4 蒸馏基线。
# - e2e 蒸馏链路里，与 cat_train 的 --lora_post_attn / --lora_temperature / --lora_loss_alpha
#   对应的参数分别是 --post_attn / --distill_temperature / --distill_alpha。
# - e2e LoRA 训练里，与 cat_train 对齐的训练期开关是 --lora_hif4_act，最终 PPL 评测开关是 --eval_hif4_act。
# - 当前脚本保持 HF 默认，不额外启用梯度检查点。
# - 以下配置是有意为之，不是漏配：
#   --teacher_model_path 不显式传，默认从 student checkpoint meta 推断
#   --save_strategy no，默认只落最终导出
#   --unload_vae_original_weights_on_save false，保留原始权重便于后续检查

torchrun --standalone --nproc_per_node=1 -m e2e_fintuning.main \
  --student_checkpoint_dir .result/meta-llama_Llama-2-7b-hf_20260323_071142/final_model \
  --run_root_dir .result/e2e_vae_lora \
  --dataset_name vicgalle/alpaca-gpt4 \
  --train_split train \
  --eval_split validation \
  --text_field text \
  --loss_type kd \
  --distill_temperature 1.0 \
  --distill_alpha 0.3 \
  --post_attn false \
  --model_max_length 2048 \
  --decoder_layers 0-31 \
  --target_modules all \
  --vae_lora_variant plain \
  --vae_lora_rank 8 \
  --vae_lora_alpha 16 \
  --vae_lora_dropout 0.0 \
  --vae_lora_tune_bias true \
  --vae_lora_init_mode zero \
  --tune_final_norm true \
  --use_post_norm_head_linear true \
  --lora_hif4_act false \
  --eval_hif4_act false \
  --prewarm_frozen_vae true \
  --prewarm_log_every 32 \
  --skip_ppl_eval false \
  --ppl_seqlen 2048 \
  --ppl_limit -1 \
  --save_tokenizer true \
  --unload_vae_original_weights_on_save true \
  --bf16 true \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --logging_steps 1 \
  --save_strategy no \
  --max_steps 1500 \
  "$@"
