#!/usr/bin/env bash

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DISTILL_GPUS="${DISTILL_GPUS:-0,1,2,3,4,5,6,7}"
if [[ ! "${DISTILL_GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "DISTILL_GPUS must be non-negative integers separated by commas, without spaces (for example: 5,6,7,8). Got: ${DISTILL_GPUS@Q}" >&2
  exit 2
fi
IFS=',' read -r -a DISTILL_GPU_LIST <<< "${DISTILL_GPUS}"
declare -A DISTILL_GPU_SEEN=()
for DISTILL_GPU in "${DISTILL_GPU_LIST[@]}"; do
  if [[ -n "${DISTILL_GPU_SEEN[${DISTILL_GPU}]:-}" ]]; then
    echo "DISTILL_GPUS must not contain duplicate GPU ids. Got: ${DISTILL_GPUS}" >&2
    exit 2
  fi
  DISTILL_GPU_SEEN["${DISTILL_GPU}"]=1
done
NPROC_PER_NODE="${#DISTILL_GPU_LIST[@]}"
export CUDA_VISIBLE_DEVICES="${DISTILL_GPUS}"
export DISTILL_NCCL_TIMEOUT_SEC=10800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800
export PYTHONHASHSEED=31
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 可按需补充的可选参数：
# --access_token "hf_xxx"
# --resume_from_checkpoint "/path/to/last_run/final_model"
# --include_all_linears
# --rot_llm
# --allow_tail_group "true"
# 排序代码，已关闭；不要在实际 CLI 中传入以下旧参数：
#   --intra_part_sort_mode "default=none"
#   --sort_prep_workers "0"
# joint decoder 联合优化代码，已关闭；不要在实际 CLI 中传入以下旧参数：
#   --joint_decoder_steps "default=0"
#   --joint_decoder_lr "default=0.005"
#   --joint_decoder_group_size "default=36"
#   --joint_decoder_batch_size "default=524288"
# --outlier_residual_top_p "default=0.01,cat:down_proj=0.02"
# --outlier_rank_metric "sparse_residual_abs" / "sparse_residual_actmax_abs" / "sparse_residual_actmean_abs" / "sparse_weight_abs" / "sparse_weight_actmax_abs" / "sparse_weight_actmean_abs"
# --outlier_rank_metric "channel_weight_abs" / "channel_weight_actmax_abs" / "channel_weight_actmean_abs" / "channel_residual_abs" / "channel_residual_actmax_abs" / "channel_residual_actmean_abs" / "channel_residual_actrms_abs"
# --outlier_channel_scope "layer" / "category"
# --outlier_protect_channel_quant "none" / "fp8_e4m3" / "fp8_e5m2" / "int8"
# --outlier_protect_mode "channel_residual_vae"
# --outlier_residual_vae_stages "default=1,cat:q_proj=2"
# --outlier_residual_vae_decoder_share_scope "none" / "category"
# --outlier_residual_vae_batch_multiplier "32"
# --outlier_residual_vae_steps "1500"
# --outlier_residual_vae_lr "0.002"
# --outlier_residual_min_abs "1e-6"
#   当前待压缩 student dense weight 只用于决定保留/稀疏 residual 位置；
#   residual = current_student_dense_weight - VAE_reconstruction
# --outlier_residual_codec "blocked_quantized" or "coo_fp16"
# --outlier_residual_index_bits "8"   # 8 or 4 慎用 4 bits，可能导致结果不稳定
# --outlier_residual_value_bits "8"   # 8 or 4 
# --activation_calib_dataset "openorca=1.0"  # 使用 dense_e2e dataset_mix alias，格式 alias=weight,...
# 数据加载：EdgeRazor lazy Dataset + dataloader_num_workers（默认 16）；不再使用 CAT_DISTILL_DATASET_NUM_PROC
# --eval_ppl "true"                   # 是否跑类别后 PPL；默认 true
# --eval_tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu"       # 可选：类别后下游任务评估；空串表示不跑
# 蒸馏数据：先运行 bash scripts/prepare_vaellm_edgerazor_data.sh，见 docs/edgerazor_dataset.md
# 可把 --distill_after_category 改为 remaining_lora_decoder 或 remaining_lora_all_decoder 做 decoder 联合恢复消融

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" tools/cat_train.py \
  --model_path "Qwen/Qwen3-8B" \
  --output_dir "/root/data/ckpts/result/catlora" \
  --resume_from_checkpoint "/root/data/ckpts/result/catlora/Qwen_Qwen3-8B_20260825_122546/after_up_proj/" \
  --seed "31" \
  --deterministic "true" \
  --train_device "cuda" \
  --convert \
  --save_model \
  --convert_device "cuda" \
  --allow_tail_group "true" \
  --target_categories "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
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
  --codebook_bits "default=32,cat:down_proj=64" \
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
  --eval_ppl "false" \
  --eval_tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu" \
  --ppl_limit "-1" \
  --outlier_protect_mode "channel" \
  --outlier_rank_metric "channel_weight_actmean_abs" \
  --outlier_mlp_rank_metric "none" \
  --outlier_mlp_fuse_weights "1,1,1" \
  --outlier_channel_scope "layer" \
  --outlier_protect_min_per_layer "0" \
  --outlier_protect_channel_quant "int8" \
  --outlier_protect_axis "input" \
  --outlier_protect_count "default=32,cat:o_proj=64,cat:down_proj=256" \
  --outlier_residual_vae_decoder_share_scope "none" \
  --outlier_residual_vae_batch_multiplier "4" \
  --outlier_residual_vae_steps "1500" \
  --outlier_residual_vae_lr "1e-3" \
  --outlier_residual_vae_stages "default=1" \
  --outlier_residual_vae_codebook_bits "default=4" \
  --outlier_residual_vae_codebook_dim "default=8" \
  --outlier_residual_top_p "default=0.0" \
  --outlier_residual_min_abs "0.0" \
  --outlier_residual_codec "blocked_quantized" \
  --outlier_residual_index_bits "8" \
  --outlier_residual_value_bits "8" \
  --distill_after_category "remaining_lora_all_decoder" \
  --distill_dataset "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009" \
  --lora_rank "default=12" \
  --lora_alpha "default=24" \
  --lora_dropout "default=0.03" \
  --distill_steps "default=5000" \
  --distill_batch_size "default=4" \
  --distill_lr "default=1e-4" \
  --distill_decoder_lr "default=1e-5" \
  --distill_weight_decay "default=0.001" \
  --distill_log_every "default=100" \
  --distill_temperature "default=1.0" \
  --distill_loss_alpha "default=0.5" \
  --distill_loss_type "default=kl_top_100" \
  --distill_eakld_confidence_k "16" \
  --distill_teacher_logits_cpu_staging "true" \
  --distill_teacher_model_offload "none" \
  --distill_hidden_loss_weight "default=0.1" \
  --distill_pre_mlp_hidden_loss_weight "default=0.001" \
  --distill_hidden_alignment_layer_weighting "linear_depth" \
  --lora_use_dora "default=true" \
  --distill_tune_final_norm "true" \
  --distill_use_post_norm_head_linear "true" \
  --distill_hif4_act "false" \
  --eval_hif4_act "false" \
  --distill_gradient_accumulation_steps "1" \
  --distill_gradient_checkpointing "true" \
  --distill_gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --distill_optim "adamw_torch" \
  --distill_max_grad_norm "1.3" \
  --distill_warmup_ratio "0.1" \
  --distill_group_by_length "true" \
  --distill_dynamic_padding "true" \
  --distill_lr_scheduler_type "cosine" \
  --distill_model_max_length "1024" \
  --fp16 "false" \
  --bf16 "true" \
  "$@"
