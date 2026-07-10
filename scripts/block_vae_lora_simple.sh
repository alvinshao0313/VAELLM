#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
SEED="${SEED:-42}"
export PYTHONHASHSEED="${SEED}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CAT_LORA_DATASET_NUM_PROC="${CAT_LORA_DATASET_NUM_PROC:-16}"

if [[ "${FULL_DETERMINISM:-false}" == "true" ]]; then
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
fi

# 可选参数：
# --access_token "hf_xxx" 用于私有或 gated 模型。
# --block_resume_from_checkpoint 只支持 inline/distill 路径；pretrain/pretrain_distill 模式不支持。
# --vae_pretrained_checkpoint "/path/to/block_vae_cache" 第一轮 distill 输入。
# --block_init_checkpoint "/path/to/final_model" 从上一轮 block final_model 开始新一轮 distill。
# --vae_pretrained_checkpoint、--block_init_checkpoint、--block_resume_from_checkpoint 三者只能选一个。

python tools/block_vae_lora_train.py \
  --model_path "Qwen/Qwen3-8B" \
  --output_dir ".result" \
  --seed "${SEED}" \
  --deterministic "${FULL_DETERMINISM:-false}" \
  --train_device "cuda" \
  --convert_device "cuda" \
  --unload_vae_original_weights_on_final_save \
  --block_vae_pipeline_mode "distill" \
  --block_init_checkpoint ".result/Qwen_Qwen3-8B_20260612_080538/final_model" \
  --block_vae_pretrain_devices "cuda" \
  --block_vae_pretrain_workers "1" \
  --block_vae_linear_group_size "36" \
  --block_vae_allow_tail_group "true" \
  --block_vae_categories "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --transpose_modules "q_proj,v_proj,o_proj,down_proj" \
  --codebook_bits "default=32" \
  --codebook_dim "default=32" \
  --residual_stages "default=2" \
  --base_ch "default=128" \
  --num_res_blocks "default=1" \
  --decoder_base_ch "default=128" \
  --decoder_num_res_blocks "default=1" \
  --norm_type "default=layer" \
  --decoder_type "default=symmetric" \
  --recon_loss_type "default=mse" \
  --intra_parallel "default=1x1" \
  --vae_steps "default=10000" \
  --vae_batch_size "8192" \
  --vae_gpu_resident_data "true" \
  --vae_log_every 100 \
  --vae_eval_every 1000 \
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
  --entropy_loss_weight "1e-2" \
  --diversity_gamma "1.0" \
  --normalize_weight \
  --new_quant \
  --block_distill_dataset "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009" \
  --block_distill_steps "5000" \
  --block_distill_nsamples "5000" \
  --block_distill_seqlen "4096" \
  --block_distill_train_mode "both" \
  --block_lora_rank "32" \
  --block_lora_alpha "32" \
  --block_lora_lr "1e-5" \
  --block_lora_lr_scheduler "none" \
  --block_lora_warmup_steps "0" \
  --block_lora_variant "dora" \
  --block_lora_dropout "0.0" \
  --block_lora_bias "none" \
  --block_lora_hif4_act "false" \
  --block_adalora_init_rank "32" \
  --block_adalora_tinit "0" \
  --block_adalora_tfinal "0" \
  --block_adalora_delta_t "1" \
  --block_adalora_beta1 "0.85" \
  --block_adalora_beta2 "0.85" \
  --block_adalora_orth_reg_weight "0.5" \
  --block_loss_alpha "0." \
  --block_loss_beta "0.05" \
  --block_distill_entropy_aware_kl "true" \
  --block_distill_eakld_confidence_k "16" \
  --block_attn_query_chunk_size "4096" \
  --block_distill_log_every "10" \
  --block_decode_group_size "8" \
  --block_hidden_advance_batch_size "64" \
  --skip_layers "" \
  --block_layers "6" \
  --block_eval_after_each_layer "true" \
  --block_eval_tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu" \
  --block_eval_ppl "false" \
  --block_eval_ppl_limit "-1" \
  --block_eval_device "cuda" \
  --block_eval_hif4_act "false" \
  --block_keep_last_checkpoints "1" \
  --fp16 "false" \
  --bf16 "true" \
  "$@"
