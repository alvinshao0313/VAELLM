#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=31
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

# Caller activates bitvae and selects one GPU. No model-level distillation.
# Down-only 3-bit variant: --codebook_bits 'default=32,cat:down_proj=48'.
# Evaluate the complete final_model separately; avoid seven prefix evaluations.
python tools/cat_train.py \
  --model_path /home/star/models/Qwen/Qwen3-8B \
  --output_dir result/qwen3_8b_w25_20260930/reconstruction_w2 \
  --compression_categories q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --target_layers all \
  --seed 31 --data_seed 31 --deterministic true \
  --train_device cuda --convert --save_model --convert_device cuda \
  --allow_tail_group true \
  --transpose_modules q_proj,v_proj,o_proj,down_proj \
  --linear_group_size 36 \
  --vae_steps default=10000 --vae_batch_size 8192 \
  --vae_learning_rate 3e-3 --vae_weight_decay 0 --vae_optim adamw \
  --vae_lr_scheduler_type linear --vae_warmup_ratio 0 \
  --codebook_bits default=32 --codebook_dim default=32 --residual_stages default=2 \
  --base_ch default=128 --num_res_blocks default=0 \
  --norm_type default=layer --activation_type default=swish --decoder_type default=symmetric \
  --recon_loss_type default=mse --quantizer_type BSQ \
  --gamma0 1 --gamma 1 --zeta 1 --inv_temperature 100 \
  --beta1 0.9 --beta2 0.95 \
  --l1_weight 1 --lfq_weight 2.5 --commitment_loss_weight 0.25 --entropy_loss_weight 0.01 \
  --normalize_weight --weight_rotation two_sided --weight_rotation_block_size 0 \
  --vae_decoder_checkpoint true --new_quant \
  --channel_protect_mode none --after_category_mode none \
  --log_every 100 --eval_every 0 --eval_blocks 256 \
  --skip_ppl_eval true --eval_tasks '' \
  --bf16 true --fp16 false \
  "$@"
