#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
torchrun --standalone --nproc_per_node=1 -m e2e_fintuning.main \
  --student_checkpoint_dir .result/meta-llama_Llama-2-7b-hf_20260319_045042/final_model \
  --run_root_dir .result/e2e_vae_lora_from_meta-llama_Llama-2-7b-hf_20260319_045042 \
  --dataset_name Salesforce/wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --train_split train \
  --eval_split validation \
  --loss_type sft \
  --finetune_mode vae_lora \
  --model_max_length 2048 \
  --decoder_layers 0-31 \
  --vae_lora_rank 8 \
  --vae_lora_alpha 16 \
  --vae_lora_dropout 0.0 \
  --prewarm_frozen_vae true \
  --save_tokenizer true \
  --bf16 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --logging_steps 1 \
  --max_steps 100 \
  "$@"
