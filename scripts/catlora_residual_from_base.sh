#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash $0 <CUDA_VISIBLE_DEVICES> [extra args...]"
  echo "Example: bash $0 0"
  echo "Example: bash $0 1"
  echo "Example: bash $0 0,1"
  exit 1
fi

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$1
shift

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

export PYTHONHASHSEED=31
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

  # --outlier_protect_mode "channel_residual_vae" \
  # --outlier_rank_metric "channel_residual_actmax_abs" \
  # --outlier_rank_metric "channel_residual_actmean_abs" \
  # --outlier_protect_axis "input" \
  # --outlier_channel_scope "category" \
  # --outlier_protect_count "64" \
  # --outlier_protect_min_per_layer "32" \
  # --outlier_residual_vae_decoder_share_scope "category" \
  # --outlier_residual_vae_batch_multiplier "16" \
  # --outlier_residual_vae_steps "2000" \
  # --outlier_residual_vae_lr "5e-3" \
  # --outlier_residual_vae_stages "2" \
  # --outlier_residual_vae_codebook_bits "64" \
  # --outlier_residual_vae_codebook_dim "32" \


  # --outlier_protect_mode "residual_sparse" \
  # --outlier_rank_metric "sparse_residual_abs" \
  # --outlier_rank_metric "sparse_residual_actmean_abs" \
  # --sparse_residual_ratio "0.01" \
  # --outlier_residual_min_abs "1e-6" \
  # --outlier_residual_codec "blocked_quantized" \
  # --outlier_residual_index_bits "8" \
  # --outlier_residual_value_bits "8" \

python tools/cat_residual_from_base.py \
  --model_path "Qwen/Qwen3-8B" \
  --base_vae_checkpoint ".result/catlora/no_outlier_protect_vae_only_Qwen_Qwen3-8B_20260618_075940" \
  --output_dir ".result/catlora_residual_from_base/down_proj" \
  --target_categories "down_proj" \
  --transpose_modules "q_proj,v_proj,o_proj,down_proj" \
  --outlier_protect_mode "channel_residual_vae" \
  --outlier_rank_metric "channel_residual_actmean_abs" \
  --outlier_protect_axis "input" \
  --outlier_channel_scope "layer" \
  --outlier_protect_count "128" \
  --outlier_protect_min_per_layer "96" \
  --outlier_residual_vae_decoder_share_scope "category" \
  --outlier_residual_vae_batch_multiplier "16" \
  --outlier_residual_vae_steps "2000" \
  --outlier_residual_vae_lr "5e-2" \
  --outlier_residual_vae_stages "2" \
  --outlier_residual_vae_codebook_bits "128" \
  --outlier_residual_vae_codebook_dim "32" \
  --base_batch_size "8192" \
  --activation_calib_dataset "alpaca=1" \
  --activation_calib_nsamples "128" \
  --activation_calib_seqlen "8192" \
  --activation_calib_seed "31" \
  --activation_calib_device "" \
  --activation_calib_log_every "0" \
  --seed "31" \
  --deterministic "true" \
  --train_device "cuda" \
  --convert_device "cpu" \
  --eval_ppl "false" \
  --eval_tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu" \
  --ppl_limit "-1" \
  --eval_hif4_act "false" \
  --eval_before_residual "false" \
  --eval_after_residual "true" \
  --log_every "100" \
  --bf16 "true" \
  --fp16 "false" \
  --codebook_bits "32" \
  --codebook_dim "32" \
  --base_ch "128" \
  --num_res_blocks "1" \
  --decoder_base_ch "128" \
  --decoder_num_res_blocks "1" \
  --norm_type "layer" \
  --decoder_type "symmetric" \
  --recon_loss_type "mse" \
  --quantizer_type "BSQ" \
  --gamma0 "1.0" \
  --gamma "1.0" \
  --zeta "1.0" \
  --inv_temperature "100.0" \
  --lr "3e-3" \
  --beta1 "0.9" \
  --beta2 "0.95" \
  --weight_decay "0.0" \
  --optimizer "adamw" \
  --lr_scheduler "linear" \
  --lr_warmup_steps "0" \
  --l1_weight "1.0" \
  --lfq_weight "2.5" \
  --commitment_loss_weight "0.25" \
  --entropy_loss_weight "0.01" \
  --diversity_gamma "1.0" \
  --normalize_weight \
  --vae_decoder_checkpoint "true" \
  --new_quant \
  "$@"
