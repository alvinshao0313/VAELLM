#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONHASHSEED=31
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Triton 编译 launcher 需要 -lcuda；本机只有 libcuda.so.1，用 CUDA stubs 供链接。
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}
# 覆盖分布式 lm_eval（尤其 mmlu）造成的 gather 等待；单位秒，默认 3 小时。
export DISTILL_NCCL_TIMEOUT_SEC=10800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800

# --distill_reset_completed false：按 completed_categories 跳过；true：在已有权重上再蒸一轮全部分类
# --distill_independent_categories false：前缀累积压缩状态；true：每类独立（已完成类恢复未压缩）
# edgerazor 配比：
  # --distill_dataset "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009" \
torchrun --standalone --nproc_per_node=4 tools/cat_distill_from_vae_checkpoint.py \
  --model_path "Qwen/Qwen3-8B" \
  --resume_from_checkpoint ".result/catlora/res0-bf16-protect-channel-vae/final_model" \
  --output_dir "./.result/catlora_distill/res0-bf16-protect-channel-vae" \
  --seed "33" \
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
  --distill_after_category "both" \
  --compressed_lora_scope "full" \
  --distill_reset_completed "true" \
  --distill_independent_categories "false" \
  --distill_dataset "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009" \
  --lora_rank "default=8,after:gate_proj=32,after:up_proj=32,after:down_proj=32" \
  --lora_alpha "default=8,after:gate_proj=32,after:up_proj=32,after:down_proj=32" \
  --lora_dropout "default=0.03" \
  --distill_steps "default=2000" \
  --distill_batch_size "default=4" \
  --distill_lr "default=2e-5" \
  --distill_weight_decay "default=0.001" \
  --distill_log_every "default=10" \
  --distill_temperature "default=1.0" \
  --distill_loss_alpha "default=0.5" \
  --distill_prompt_kd_weight "default=0.03" \
  --distill_loss_type "default=kl_top_100" \
  --distill_eakld_confidence_k "16" \
  --distill_teacher_logits_cpu_staging "true" \
  --distill_hidden_loss_weight "default=0.1" \
  --distill_pre_mlp_hidden_loss_weight "default=0.0" \
  --distill_hidden_alignment_layer_weighting "adaptive_top_3" \
  --lora_use_dora "default=false" \
  --distill_tune_final_norm "false" \
  --distill_use_post_norm_head_linear "false" \
  --distill_hif4_act "false" \
  --eval_hif4_act "false" \
  --distill_gradient_accumulation_steps "4" \
  --distill_gradient_checkpointing "true" \
  --distill_gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --distill_optim "adamw_torch" \
  --distill_max_grad_norm "1.5" \
  --distill_warmup_ratio "0.05" \
  --distill_group_by_length "false" \
  --distill_lr_scheduler_type "cosine" \
  --distill_model_max_length "1024" \
  --fp16 "false" \
  --bf16 "true" \
  "$@"
