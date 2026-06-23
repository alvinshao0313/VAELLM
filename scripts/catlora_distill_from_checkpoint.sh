#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=31
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export CAT_DISTILL_DATASET_NUM_PROC="${CAT_DISTILL_DATASET_NUM_PROC:-16}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1



python tools/cat_distill_from_vae_checkpoint.py \
  --model_path "Qwen/Qwen3-8B" \
  --resume_from_checkpoint ".result/catlora/no_outlier_protect_vae_only_Qwen_Qwen3-8B_20260618_075940" \
  --output_dir "./.result/catlora_distill" \
  --seed "31" \
  --deterministic "true" \
  --train_device "cuda" \
  --convert \
  --save_model \
  --convert_device "cuda" \
  --unload_vae_original_weights_on_final_save \
  --vae_decoder_checkpoint "true" \
  --target_categories "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --transpose_modules "q_proj,v_proj,o_proj,down_proj" \
  --skip_layers "" \
  --eval_ppl "false" \
  --eval_tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu" \
  --ppl_limit "-1" \
  --outlier_protect_mode "none" \
  --distill_after_category "compressed_lora" \
  --distill_dataset "openorca=0.2,fineweb_edu=0.18,race=0.24,sciq=0.14,alpaca=0.04,longalpaca=0.1,longalign=0.1" \
  --lora_rank "default=4" \
  --lora_alpha "default=512" \
  --lora_dropout "default=0.03" \
  --distill_steps "default=5000,after:q_proj=3000,after:k_proj=3000,after:v_proj=6000,after:o_proj=3000,after:gate_proj=3000,after:up_proj=3000,after:down_proj=3000" \
  --distill_batch_size "default=1" \
  --distill_nsamples "default=20000" \
  --distill_lr "default=1e-4" \
  --distill_weight_decay "default=0.001" \
  --distill_log_every "default=100" \
  --distill_post_attn "false" \
  --distill_temperature "default=1.0" \
  --distill_loss_alpha "default=0.5" \
  --distill_loss_type "default=kd_top_1000" \
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
  --distill_lr_scheduler_type "constant" \
  --distill_model_max_length "8192" \
  --fp16 "false" \
  --bf16 "true" \
  "$@"
