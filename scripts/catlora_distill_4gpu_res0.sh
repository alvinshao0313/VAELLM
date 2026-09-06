#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONHASHSEED=31
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}
export DISTILL_NCCL_TIMEOUT_SEC=10800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800

torchrun --standalone --nproc_per_node=8 tools/cat_distill_from_vae_checkpoint.py \
  --model_path "Qwen/Qwen3-8B" \
  --resume_from_checkpoint "/root/data/ckpts/result/catlora/Qwen_Qwen3-8B_20260724_190531/final_model_down4bit_merged" \
  --output_dir "/root/data/ckpts/result/catlora_distill/res0-bf16-protect-channel-vae" \
  --compression_categories "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --target_layers all \
  --skip_layers "" \
  --after_category_mode current_lora_decoder \
  --distill_reset_completed true \
  --distill_independent_categories false \
  --dataset_mix "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009" \
  --dataset_task sft \
  --seed 33 \
  --data_seed 33 \
  --model_max_length 1024 \
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
  --lora_alpha "default=8" \
  --lora_dropout "default=0.03" \
  --steps "default=3000,after:down_proj=5000" \
  --batch_size "default=4" \
  --learning_rate "default=2e-5" \
  --weight_decay "default=0.001" \
  --logging_steps "default=10" \
  --loss_type "default=kl_top" \
  --top_k "default=100" \
  --temperature "default=1" \
  --alpha "default=0.5" \
  --prompt_loss_weight "default=0.001" \
  --hidden_loss_weight "default=0.3" \
  --pre_mlp_hidden_loss_weight "default=0" \
  --hidden_layer_weighting adaptive_top_3 \
  --teacher_output_offload cpu \
  --teacher_model_offload none \
  --norm_train_mode none \
  --lm_head_train_mode none \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --optim adamw_torch \
  --max_grad_norm 3.33 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --bf16 true \
  --fp16 false \
  "$@"
