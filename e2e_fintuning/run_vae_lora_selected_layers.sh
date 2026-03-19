#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/tmp/bitvae_hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/bitvae_hf_datasets}"

torchrun --nnodes=1 --nproc_per_node=1 -m e2e_fintuning.main \
  --student_checkpoint_dir /path/to/lfq_checkpoint \
  --run_root_dir .result/e2e_tmp \
  --dataset_name Salesforce/wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --train_split train \
  --eval_split validation \
  --loss_type sft \
  --finetune_mode vae_lora \
  --decoder_layers 28-31 \
  --vae_lora_rank 8 \
  --vae_lora_alpha 16 \
  --vae_lora_dropout 0.05 \
  --prewarm_frozen_vae true \
  --save_tokenizer true \
  --bf16 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --logging_steps 1 \
  --max_steps 20 \
  "$@"
