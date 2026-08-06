#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
export PYTHONHASHSEED=31
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 多卡蒸馏（DDP）：只改启动方式，CLI 参数与下方 python 命令一致。
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# torchrun --standalone --nproc_per_node=4 python tools/cat_distill_from_vae_checkpoint.py \
#   ... # 与下方相同参数
#
# 类别级续跑未完成类：--resume_from_checkpoint .../after_<category> + --distill_reset_completed false
# 在已蒸馏 ckpt 上再蒸一轮（含 LoRA 续训写回）：--distill_reset_completed true
# 详见 docs/catlora_distill_from_checkpoint.md

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
  --distill_after_category "compressed_lora" \
  --distill_dataset "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009" \
  --lora_rank "default=4" \
  --lora_alpha "default=4,after:k_proj=256" \
  --lora_dropout "default=0.03" \
  --distill_steps "default=5000,after:q_proj=5000,after:k_proj=8000,after:v_proj=8000,after:o_proj=3000,after:gate_proj=3000,after:up_proj=3000,after:down_proj=3000" \
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
  --distill_group_by_length "false" \
  --distill_lr_scheduler_type "constant_with_warmup" \
  --distill_model_max_length "8192" \
  --fp16 "false" \
  --bf16 "true" \
  "$@"
