#!/usr/bin/env bash
set -euo pipefail

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
LOG_FILE="${LOG_FILE:-./logs/lbl_train_tools_$(date +%Y%m%d_%H%M%S).log}" \
PYTHONPATH="${PYTHONPATH:-.}:." \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python tools/lbl_train.py \
  --model_path "meta-llama/Llama-2-7b-hf" \
  --logging_dir "logs" \
  --model_max_length 2048 \
  --fp16 False \
  --bf16 True \
  --log_on_each_node False \
  --per_device_train_batch_size 1 \
  --logging_steps 50 \
  --learning_rate 1e-2 \
  --lr 1e-4 \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight_decay 0. \
  --optimizer adamw \
  --lr_scheduler none \
  --lr_warmup_steps 0 \
  --lr_scheduler_type "cosine" \
  --model_path "meta-llama/Llama-2-7b-hf" \
  --new_quant \
  --layer_checkpointing \
  --codebook_bits 16 \
  --codebook_dim 8 \
  --base_ch 128 \
  --num_res_blocks 1 \
  --quantizer_type BSQ \
  --gamma0 1.0 \
  --gamma 1.0 \
  --zeta 1.0 \
  --inv_temperature 100.0 \
  --norm_type group \
  --recon_loss_type mse \
  --distil_loss_type none \
  --distil_loss_weight 0.0 \
  --use_output_mse_loss \
  --output_mse_loss_weight 1.0 \
  --l1_weight 1.0 \
  --lfq_weight 4 \
  --commitment_loss_weight 0.25 \
  --entropy_loss_weight 0.1 \
  --diversity_gamma 1.0 \
  --decoder_type symmetric \
  --normalize_weight \
  --use_checkpoint \
  --num_train_epochs 10 \
  --nsamples 128 \
  --w_input_batches 2 \
  --parallel_layers 1 $@
