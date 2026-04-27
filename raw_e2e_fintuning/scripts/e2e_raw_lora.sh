#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# 说明：
# - 原模型 LoRA/DoRA/AdaLoRA 训练入口（非 VAE 轨）。
# - 该脚本只走 raw_e2e_fintuning，不读取 student_checkpoint_dir。
# - 默认使用 dataset_mix 单源比例 alpaca=1.0。

torchrun --standalone --nproc_per_node=1 -m raw_e2e_fintuning.main \
  --student_model_path meta-llama/Llama-2-7b-hf \
  --run_root_dir .result/e2e_raw_lora \
  --dataset_mix "alpaca=1.0" \
  --loss_type sft \
  --distill_temperature 1.0 \
  --distill_alpha 0.3 \
  --post_attn false \
  --model_max_length 4096 \
  --decoder_layers 0-31 \
  --target_modules all \
  --lora_variant plain \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.0 \
  --lora_tune_bias true \
  --lora_init_mode zero \
  --lora_smooth false \
  --tune_final_norm true \
  --use_post_norm_head_linear true \
  --lora_hif4_act false \
  --eval_hif4_act false \
  --skip_ppl_eval false \
  --ppl_seqlen 2048 \
  --ppl_limit -1 \
  --save_tokenizer true \
  --raw_merge_and_save false \
  --bf16 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --logging_steps 10 \
  --save_strategy no \
  --max_steps 2000 \
  "$@"
