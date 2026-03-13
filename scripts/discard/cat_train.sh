#!/usr/bin/env bash
set -euo pipefail

# Example: category-wise weight-VAE training, then replace Linear.
#
# Notes:
# - By default, categories are trained in this order:
#   q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,others
# - Set `--category_order auto` to train all discovered categories in alphabetical order.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/cat_train.sh --model_path ... [extra args]

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
LOG_FILE="${LOG_FILE:-./logs/linear_by_category_$(date +%Y%m%d_%H%M%S).log}" \
PYTHONPATH="${PYTHONPATH:-.}:." \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
python tools/cat_train.py \
  --output_dir "${OUTPUT_DIR:-./output_linear_by_category}" \
  --steps_per_category "${STEPS_PER_CATEGORY:-5000}" \
  --batch_size "${BATCH_SIZE:-2048}" \
  --log_every "${LOG_EVERY:-50}" \
  --eval_every "${EVAL_EVERY:-100}" \
  --eval_blocks "${EVAL_BLOCKS:-256}" \
  --ppl_limit "${PPL_LIMIT:--1}" \
  ${LORA_AFTER_CATEGORY:+--lora_after_category} \
  --lora_rank "${LORA_RANK:-8}" \
  --lora_alpha "${LORA_ALPHA:-16.0}" \
  --lora_dropout "${LORA_DROPOUT:-0.0}" \
  --lora_steps "${LORA_STEPS:-50}" \
  --lora_batch_size "${LORA_BATCH_SIZE:-2}" \
  --lora_nsamples "${LORA_NSAMPLES:-128}" \
  --lora_lr "${LORA_LR:-1e-4}" \
  --lora_weight_decay "${LORA_WEIGHT_DECAY:-0.0}" \
  --lora_log_every "${LORA_LOG_EVERY:-10}" \
  --train_device "${TRAIN_DEVICE:-cuda}" \
  --convert \
  --convert_device "${CONVERT_DEVICE:-cuda}" \
  --save_model \
  --linear_group_size "${LINEAR_GROUP_SIZE:-32}" \
  --intra_parallel "${INTRA_PARALLEL:-1}" \
  --codebook_bits "${CODEBOOK_BITS:-16}" \
  --codebook_dim "${CODEBOOK_DIM:-8}" \
  --base_ch "${BASE_CH:-128}" \
  --num_res_blocks "${NUM_RES_BLOCKS:-1}" \
  --quantizer_type "${QUANTIZER_TYPE:-BSQ}" \
  --gamma0 "${GAMMA0:-1.0}" \
  --gamma "${GAMMA:-1.0}" \
  --zeta "${ZETA:-1.0}" \
  --inv_temperature "${INV_TEMPERATURE:-100.0}" \
  --norm_type "${NORM_TYPE:-group}" \
  --decoder_type "${DECODER_TYPE:-symmetric}" \
  --normalize_weight \
  --new_quant \
  --use_checkpoint \
  --recon_loss_type "${RECON_LOSS_TYPE:-mse}" \
  --l1_weight "${L1_WEIGHT:-1.0}" \
  --lfq_weight "${LFQ_WEIGHT:-4}" \
  --commitment_loss_weight "${COMMITMENT_LOSS_WEIGHT:-0.25}" \
  --entropy_loss_weight "${ENTROPY_LOSS_WEIGHT:-0.1}" \
  --diversity_gamma "${DIVERSITY_GAMMA:-1.0}" \
  --optimizer "${OPTIMIZER:-adamw}" \
  --lr "${LR:-1e-2}" \
  --beta1 "${BETA1:-0.9}" \
  --beta2 "${BETA2:-0.95}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --lr_scheduler "${LR_SCHEDULER:-none}" \
  --lr_warmup_steps "${LR_WARMUP_STEPS:-0}" \
  --bf16 "${BF16:-True}" \
  "$@"
