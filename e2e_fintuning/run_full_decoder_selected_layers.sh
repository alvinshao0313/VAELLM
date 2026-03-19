#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
# export HF_HOME="${HF_HOME:-/tmp/bitvae_hf_home}"
# export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/bitvae_hf_datasets}"
conda run --no-capture-output -n bitvae python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=1 \
  -m e2e_fintuning.main \
  --student_checkpoint_dir .result/meta-llama_Llama-2-7b-hf_20260319_045042/final_model \
  --run_root_dir .result/e2e_continue_from_meta-llama_Llama-2-7b-hf_20260319_045042 \
  --dataset_name Salesforce/wikitext \
  --dataset_config_name wikitext-2-raw-v1 \
  --train_split train \
  --eval_split validation \
  --loss_type sft \
  --finetune_mode full \
  --decoder_layers 29-31 \
  --train_protected_outliers false \
  --model_max_length 2048 \
  --prewarm_frozen_vae true \
  --save_tokenizer true \
  --skip_ppl_eval false \
  --bf16 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --save_strategy no \
  --logging_steps 1 \
  --max_steps 2000 \
  "$@"
