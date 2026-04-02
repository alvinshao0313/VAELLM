#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4}"

torchrun --standalone --nproc_per_node=4 -m e2e_fintuning.main \
  --student_checkpoint_dir .result/meta-llama_Llama-2-7b-hf_20260323_071142/final_model \
  --teacher_model_path meta-llama/Llama-2-7b-hf \
  --run_root_dir .result/e2e_vae_lora_redpajama \
  --dataset_name ZengXiangyu/RedPajama-Data-1T-Sample \
  --train_split train \
  --eval_split validation \
  --text_field text \
  --loss_type sft \
  --distill_temperature 1.0 \
  --distill_alpha 0.3 \
  --post_attn false \
  --model_max_length 4096 \
  --decoder_layers 0-31 \
  --target_modules all \
  --vae_lora_rank 8 \
  --vae_lora_alpha 16 \
  --vae_lora_dropout 0.0 \
  --lora_embedding false \
  --lora_lm_head false \
  --lora_hif4_act false \
  --prewarm_frozen_vae true \
  --prewarm_log_every 32 \
  --skip_ppl_eval false \
  --ppl_seqlen 2048 \
  --ppl_limit -1 \
  --save_tokenizer true \
  --unload_vae_original_weights_on_save false \
  --bf16 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 10000 \
  --max_steps 50000 \
  "$@"
