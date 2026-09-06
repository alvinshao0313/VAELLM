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

python tools/cat_distill_from_vae_checkpoint.py \
  --model_path "Qwen/Qwen3-8B" \
  --resume_from_checkpoint ".result/catlora/no_outlier_protect_vae_only_Qwen_Qwen3-8B_20260618_075940" \
  --output_dir "./.result/catlora_distill" \
  --compression_categories "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --target_layers all \
  --skip_layers "" \
  --after_category_mode current_lora \
  --dataset_mix "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009" \
  --dataset_task sft \
  --seed 31 \
  --data_seed 31 \
  --model_max_length 8192 \
  --dynamic_padding true \
  --group_by_length false \
  --deterministic true \
  --train_device cuda \
  --convert \
  --save_model \
  --convert_device cuda \
  --vae_decoder_checkpoint true \
  --skip_ppl_eval true \
  --eval_tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu" \
  --lora_rank "default=4" \
  --lora_alpha "default=4,after:k_proj=256" \
  --lora_dropout "default=0.03" \
  --steps "default=5000,after:k_proj=8000,after:v_proj=8000,after:o_proj=3000,after:gate_proj=3000,after:up_proj=3000,after:down_proj=3000" \
  --batch_size "default=1" \
  --learning_rate "default=1e-4" \
  --weight_decay "default=0.001" \
  --logging_steps "default=100" \
  --loss_type "default=kl_top" \
  --top_k "default=100" \
  --temperature "default=1" \
  --alpha "default=0.5" \
  --prompt_loss_weight "default=0" \
  --hidden_loss_weight "default=0.01" \
  --pre_mlp_hidden_loss_weight "default=0" \
  --hidden_layer_weighting linear_depth \
  --teacher_output_offload cpu \
  --teacher_model_offload none \
  --norm_train_mode none \
  --lm_head_train_mode none \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --optim adamw_torch \
  --max_grad_norm 1.3 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type constant_with_warmup \
  --bf16 true \
  --fp16 false \
  "$@"
