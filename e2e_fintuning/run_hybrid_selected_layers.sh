#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/tmp/bitvae_hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/bitvae_hf_datasets}"

torchrun --nnodes=1 --nproc_per_node=1 -m e2e_fintuning.main \
  --student_checkpoint_dir /path/to/lfq_checkpoint \
  --run_root_dir .result/e2e_tmp \
  --train_file /path/to/train.jsonl \
  --eval_file /path/to/eval.jsonl \
  --text_field text \
  --loss_type kd \
  --finetune_mode hybrid \
  --decoder_layers 24-31 \
  --vae_lora_rank 8 \
  --vae_lora_alpha 16 \
  --vae_lora_dropout 0.05 \
  --train_protected_outliers true \
  --prewarm_frozen_vae true \
  --distill_alpha 0.5 \
  --distill_temperature 1.0 \
  --save_tokenizer true \
  --bf16 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --logging_steps 1 \
  --max_steps 20 \
  "$@"
