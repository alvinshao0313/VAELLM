#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

STUDENT_CKPT="${STUDENT_CKPT:-/root/data/ckpts/result/catlora/Qwen_Qwen3-8B_20260828_183213/final_model/}"
PARALLEL_MODE="${PARALLEL_MODE:-dp}"   # dp | layer_mp

export DISTILL_NCCL_TIMEOUT_SEC="${DISTILL_NCCL_TIMEOUT_SEC:-10800}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ "${DISABLE_PROXY:-1}" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  unset all_proxy ALL_PROXY no_proxy NO_PROXY
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
fi

# 参数顺序：
# [Data]
# [Distill Loss]
# [VAE Compression]（本脚本无新增 VAE compression）
# [Channel Protection]（本脚本无 channel protection）
# [Distill Optimization]
# [Runtime / Distributed]
# [Evaluation]
# "$@" 位于最后，可覆盖本脚本中的默认值。

if [[ "${PARALLEL_MODE}" == "dp" ]]; then
  # 原正式 recipe：decoder + Sparse Bit；不启用 backbone LoRA。
  torchrun --standalone --nproc_per_node=8 -m compressed_e2e_fintuning.main \
    --student_checkpoint_dir "${STUDENT_CKPT}" \
    --run_root_dir /root/data/ckpts/result/compressed_e2e_fintuning \
    --train_mode decoder_sparse_bit \
    --seed 0 \
    --data_seed 0 \
    --dataset_mix "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009" \
    --dataset_task sft \
    --dynamic_padding true \
    --model_max_length 1024 \
    --group_by_length true \
    --target_layers 0-35 \
    --target_modules all \
    --loss_type kl_top \
    --top_k 100 \
    --temperature 1.0 \
    --alpha 0.5 \
    --prompt_loss_weight 0.0 \
    --hidden_loss_weight 0.1 \
    --pre_mlp_hidden_loss_weight 0.001 \
    --hidden_layer_weighting linear_depth \
    --selective_student_topk true \
    --selective_student_topk_chunk_rows 32 \
    --steps 5000 \
    --batch_size 8 \
    --gradient_accumulation_steps 1 \
    --decoder_lr 1e-5 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.001 \
    --max_grad_norm 1.5 \
    --logging_steps 10 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --lora_rank 12 \
    --lora_alpha 24 \
    --lora_dropout 0.03 \
    --norm_train_mode final \
    --lm_head_train_mode linear \
    --bit_active_ratio 0.03 \
    --bit_optimizer rms_sgd \
    --bit_lr auto \
    --bit_weight_decay 0.0 \
    --bit_round_steps auto \
    --parallel_mode dp \
    --layer_device_map auto \
    --teacher_output_offload cpu \
    --teacher_model_offload none \
    --teacher_output_pin_memory true \
    --teacher_output_chunk_tokens 8 \
    --vae_decoder_checkpoint true \
    --offload_mode none \
    --eval_after_save true \
    --eval_hif4_act false \
    --eval_tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu" \
    --eval_num_fewshot 0 \
    --eval_batch_size auto \
    --eval_device cuda \
    --eval_prewarm_group_size 8 \
    --skip_ppl_eval true \
    --ppl_seqlen 2048 \
    --ppl_limit -1 \
    --save_tokenizer true \
    --full_determinism false \
    --bf16 true \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 2 \
    "$@"
elif [[ "${PARALLEL_MODE}" == "layer_mp" ]]; then
  # 原正式 recipe：decoder + full-space LoRA；Sparse Bit 关闭。
  python -m compressed_e2e_fintuning.main \
    --student_checkpoint_dir "${STUDENT_CKPT}" \
    --run_root_dir /root/data/ckpts/result/compressed_e2e_fintuning \
    --train_mode decoder_lora \
    --seed 0 \
    --data_seed 0 \
    --dataset_mix "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009" \
    --dataset_task sft \
    --dynamic_padding true \
    --model_max_length 1024 \
    --group_by_length true \
    --target_layers 0-35 \
    --target_modules all \
    --loss_type kl_top \
    --top_k 1000 \
    --temperature 1.0 \
    --alpha 0.5 \
    --prompt_loss_weight 0.0 \
    --hidden_loss_weight 0.0 \
    --pre_mlp_hidden_loss_weight 0.0 \
    --hidden_layer_weighting adaptive_top_3 \
    --selective_student_topk true \
    --selective_student_topk_chunk_rows 32 \
    --steps 5000 \
    --batch_size 4 \
    --gradient_accumulation_steps 1 \
    --decoder_lr 3e-6 \
    --learning_rate 3e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.0001 \
    --max_grad_norm 15.0 \
    --logging_steps 10 \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --lora_rank 12 \
    --lora_alpha 24 \
    --lora_dropout 0.03 \
    --norm_train_mode none \
    --lm_head_train_mode none \
    --parallel_mode layer_mp \
    --layer_device_map auto \
    --teacher_output_offload cpu \
    --teacher_model_offload none \
    --teacher_output_pin_memory true \
    --teacher_output_chunk_tokens 8 \
    --vae_decoder_checkpoint true \
    --offload_mode none \
    --eval_after_save true \
    --eval_hif4_act false \
    --eval_tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu" \
    --eval_num_fewshot 0 \
    --eval_batch_size auto \
    --eval_device cuda \
    --eval_prewarm_group_size 8 \
    --skip_ppl_eval true \
    --ppl_seqlen 2048 \
    --ppl_limit -1 \
    --save_tokenizer true \
    --full_determinism false \
    --bf16 true \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 2 \
    "$@"
else
  echo "Unsupported PARALLEL_MODE=${PARALLEL_MODE}. Expected dp or layer_mp." >&2
  exit 1
fi
