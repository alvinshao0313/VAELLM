#!/usr/bin/env bash
set -euo pipefail

# down_proj channel-protect A/B trial：与 res0-bf16-protect-channel-vae 的 down 离群保护对齐。
# 用法: bash scripts/catlora_codebook_ab_down_channel_single.sh <GPU_ID> [extra args...]
# 示例:
#   bash scripts/catlora_codebook_ab_down_channel_single.sh 4 \
#     --target_categories down_proj \
#     --codebook_bits default=64 --codebook_dim default=32 --residual_stages default=1 \
#     --output_dir .result/catlora_codebook_ab/down_proj/b64d32s1_channel

if [ $# -lt 1 ]; then
  echo "Usage: bash $0 <CUDA_VISIBLE_DEVICES> [extra args...]"
  echo "Example: bash $0 4 --target_categories down_proj --codebook_bits default=64 --codebook_dim default=32 --residual_stages default=1 --output_dir .result/catlora_codebook_ab/down_proj/b64d32s1_channel"
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

python tools/cat_train.py \
  --model_path "Qwen/Qwen3-8B" \
  --output_dir "./.result/catlora_codebook_ab" \
  --seed "31" \
  --deterministic "true" \
  --train_device "cuda" \
  --convert \
  --save_model \
  --convert_device "cuda" \
  --allow_tail_group "true" \
  --target_categories "down_proj" \
  --transpose_modules "q_proj,v_proj,o_proj,down_proj" \
  --skip_layers "" \
  --linear_group_size "36" \
  --steps_per_category "default=10000" \
  --batch_size "8192" \
  --activation_calib_dataset "alpaca=1" \
  --activation_calib_nsamples "128" \
  --activation_calib_seqlen "8192" \
  --activation_calib_seed "31" \
  --activation_calib_device "" \
  --activation_calib_log_every "0" \
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
  --normalize_weight \
  --vae_decoder_checkpoint "true" \
  --new_quant \
  --log_every "100" \
  --eval_every "0" \
  --eval_blocks "256" \
  --eval_ppl "true" \
  --eval_tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu" \
  --ppl_limit "-1" \
  --outlier_protect_mode "channel" \
  --outlier_mlp_rank_metric "none" \
  --outlier_mlp_fuse_weights "1,1,1" \
  --outlier_channel_scope "layer" \
  --outlier_protect_channel_quant "none" \
  --outlier_residual_vae_decoder_share_scope "none" \
  --outlier_residual_vae_batch_multiplier "4" \
  --outlier_residual_vae_steps "1500" \
  --outlier_residual_vae_lr "1e-3" \
  --outlier_residual_vae_stages "default=1" \
  --outlier_residual_vae_codebook_bits "default=4" \
  --outlier_residual_vae_codebook_dim "default=8" \
  --outlier_protect_count "default=128" \
  --outlier_protect_min_per_layer "0" \
  --outlier_protect_axis "input" \
  --outlier_residual_top_p "default=0.0" \
  --outlier_rank_metric "channel_weight_actmean_abs" \
  --outlier_residual_min_abs "0.0" \
  --outlier_residual_codec "blocked_quantized" \
  --outlier_residual_index_bits "8" \
  --outlier_residual_value_bits "8" \
  --distill_after_category "none" \
  --distill_dataset "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009" \
  --lora_rank "default=128" \
  --lora_alpha "default=128" \
  --lora_dropout "default=0.03" \
  --distill_steps "default=5000" \
  --distill_batch_size "default=1" \
  --distill_lr "default=1e-4" \
  --distill_weight_decay "default=0.001" \
  --distill_log_every "default=100" \
  --distill_temperature "default=1.0" \
  --distill_loss_alpha "default=0.5" \
  --distill_loss_type "default=eakld" \
  --distill_eakld_confidence_k "16" \
  --distill_teacher_logits_cpu_staging "true" \
  --distill_hidden_loss_weight "default=0.01" \
  --distill_pre_mlp_hidden_loss_weight "default=0.0" \
  --distill_hidden_alignment_layer_weighting "linear_depth" \
  --lora_use_dora "default=false" \
  --distill_tune_final_norm "false" \
  --distill_use_post_norm_head_linear "false" \
  --distill_hif4_act "false" \
  --eval_hif4_act "false" \
  --distill_gradient_accumulation_steps "1" \
  --distill_gradient_checkpointing "true" \
  --distill_gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --distill_optim "adamw_torch" \
  --distill_max_grad_norm "1.3" \
  --distill_warmup_ratio "0.1" \
  --distill_group_by_length "true" \
  --distill_lr_scheduler_type "constant_with_warmup" \
  --distill_model_max_length "8192" \
  --fp16 "false" \
  --bf16 "true" \
  "$@"
