#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$1"
export PYTHONHASHSEED=31
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
shift

python tools/cat_train.py \
  --model_path "Qwen/Qwen3-8B" \
  --output_dir "./.result/catlora_codebook_ab" \
  --compression_categories q_proj \
  --target_layers all \
  --skip_layers "" \
  --seed 31 \
  --data_seed 31 \
  --deterministic true \
  --train_device cuda \
  --convert \
  --save_model \
  --convert_device cuda \
  --allow_tail_group true \
  --transpose_modules "q_proj,v_proj,o_proj,down_proj" \
  --linear_group_size 36 \
  --vae_steps "default=10000" \
  --vae_batch_size 8192 \
  --vae_learning_rate 3e-3 \
  --vae_weight_decay 0 \
  --vae_optim adamw \
  --vae_lr_scheduler_type linear \
  --vae_warmup_ratio 0 \
  --activation_calib_dataset "alpaca=1" \
  --activation_calib_nsamples 128 \
  --activation_calib_seqlen 8192 \
  --activation_calib_seed 31 \
  --codebook_bits "default=32" \
  --codebook_dim "default=32" \
  --residual_stages "default=2" \
  --base_ch "default=128" \
  --num_res_blocks "default=0" \
  --decoder_base_ch "default=128" \
  --decoder_num_res_blocks "default=1" \
  --norm_type "default=layer" \
  --activation_type "default=swish" \
  --decoder_type "default=symmetric" \
  --recon_loss_type "default=mse" \
  --quantizer_type BSQ \
  --gamma0 1 \
  --gamma 1 \
  --zeta 1 \
  --inv_temperature 100 \
  --beta1 0.9 \
  --beta2 0.95 \
  --l1_weight 1 \
  --lfq_weight 2.5 \
  --commitment_loss_weight 0.25 \
  --entropy_loss_weight 0.01 \
  --normalize_weight \
  --vae_decoder_checkpoint true \
  --new_quant \
  --log_every 100 \
  --eval_every 0 \
  --eval_blocks 256 \
  --skip_ppl_eval false \
  --eval_tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu" \
  --channel_protect_mode none \
  --channel_protect_count "default=0" \
  --after_category_mode none \
  --bf16 true \
  --fp16 false \
  "$@"
