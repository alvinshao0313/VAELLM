#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SEED="${SEED:-0}"
export PYTHONHASHSEED="${SEED}"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

FULL_DETERMINISM="${FULL_DETERMINISM:-false}"
MAX_STEPS="${MAX_STEPS:-5000}"
STUDENT_CKPT="${STUDENT_CKPT:-/root/data/ckpts/result/catlora/Qwen_Qwen3-8B_20260828_183213/final_model/}"
EVAL_TASKS="${EVAL_TASKS-boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
EVAL_PREWARM_GROUP_SIZE="${EVAL_PREWARM_GROUP_SIZE:-8}"
PARALLEL_MODE="${PARALLEL_MODE:-dp}"   # dp | layer_mp
NPROC="${NPROC:-8}"
EVAL_AFTER_SAVE="${EVAL_AFTER_SAVE:-true}"

export DISTILL_NCCL_TIMEOUT_SEC="${DISTILL_NCCL_TIMEOUT_SEC:-10800}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}"

if [[ "${FULL_DETERMINISM}" == "true" ]]; then
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
fi

if [[ "${DISABLE_PROXY:-1}" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  unset all_proxy ALL_PROXY no_proxy NO_PROXY
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
fi

# 说明：
# - 脚本只区分 dp / layer_mp；训练模式和其余参数都直接交给 Python 解析。
# - "$@" 位于最后，可覆盖脚本中的默认参数。
# - decoder / both 会忽略 PEFT LoRA 配置参数；compressed_lora 会使用 rank/alpha/dropout/scope。
# - compressed_lora 只训练 LoRA；脚本里 decoder 相关 trainable 开关即使存在也不会生效。
# - --decoder_lr 仅用于 decoder / both；不传时继承 --learning_rate，compressed_lora 会忽略它。
# - 无 low_rank 分支时 compressed_lora 新建 LoRA；已有 low_rank 分支时 rank/scope 从 checkpoint 推断。
# - compressed_lora 示例：
#   bash compressed_e2e_fintuning/scripts/e2e_decoder.sh --finetune_mode compressed_lora
# - subspace 示例：
#   bash compressed_e2e_fintuning/scripts/e2e_decoder.sh --finetune_mode compressed_lora --compressed_lora_scope compressed_subspace
# - Sparse Bit 参数直接写在下面的训练命令中；将 --sparse_bit_tuning 改为 true 即可开启。
# - "$@" 仍位于最后，因此也可以临时从命令行覆盖这些默认值。

if [[ "${PARALLEL_MODE}" == "dp" ]]; then
  torchrun --standalone --nproc_per_node="${NPROC}" -m compressed_e2e_fintuning.main \
    --seed "${SEED}" \
    --data_seed "${SEED}" \
    --full_determinism "${FULL_DETERMINISM}" \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --student_checkpoint_dir "${STUDENT_CKPT}" \
    --run_root_dir /root/data/ckpts/result/compressed_e2e_fintuning \
    --finetune_mode decoder \
    --parallel_mode dp \
    --dataset_mix "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009" \
    --dataset_task sft \
    --loss_type kl_top_100 \
    --eakld_confidence_k 16 \
    --distill_temperature 1.0 \
    --distill_alpha 0.5 \
    --prompt_kd_weight 0.0 \
    --hidden_loss_weight 0.1 \
    --distill_pre_mlp_hidden_loss_weight 0.001 \
    --hidden_layer_weighting linear_depth \
    --teacher_output_offload cpu \
    --distill_teacher_model_offload none \
    --teacher_output_pin_memory true \
    --teacher_output_chunk_tokens 8 \
    --selective_student_topk true \
    --selective_student_topk_chunk_rows 32 \
    --dynamic_padding true \
    --model_max_length 1024 \
    --decoder_layers 0-35 \
    --target_modules all \
    --sparse_bit_tuning true \
    --bit_active_ratio 0.03 \
    --bit_optimizer rms_sgd \
    --bit_lr auto \
    --bit_weight_decay 0.0 \
    --bit_round_steps auto \
    --lora_rank 12 \
    --lora_alpha 24 \
    --lora_dropout 0.03 \
    --compressed_lora_scope full \
    --parallel_stage_decode true \
    --vae_decoder_checkpoint true \
    --tune_final_norm true \
    --use_post_norm_head_linear true \
    --vae_tune_bias false \
    --offload_mode none \
    --eval_after_save "${EVAL_AFTER_SAVE}" \
    --eval_hif4_act false \
    --eval_tasks "${EVAL_TASKS}" \
    --eval_num_fewshot 0 \
    --eval_lm_batch_size auto \
    --eval_device "${EVAL_DEVICE}" \
    --eval_prewarm_group_size "${EVAL_PREWARM_GROUP_SIZE}" \
    --skip_ppl_eval true \
    --ppl_seqlen 2048 \
    --ppl_limit -1 \
    --save_tokenizer true \
    --bf16 true \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --decoder_lr 1e-5 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.001 \
    --max_grad_norm 1.5 \
    --logging_steps 10 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 2 \
    --max_steps "${MAX_STEPS}" \
    "$@"
elif [[ "${PARALLEL_MODE}" == "layer_mp" ]]; then
  python -m compressed_e2e_fintuning.main \
    --seed "${SEED}" \
    --data_seed "${SEED}" \
    --full_determinism "${FULL_DETERMINISM}" \
    --gradient_checkpointing true \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --student_checkpoint_dir "${STUDENT_CKPT}" \
    --run_root_dir /root/data/ckpts/result/compressed_e2e_fintuning \
    --finetune_mode both \
    --parallel_mode layer_mp \
    --dataset_mix "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009" \
    --dataset_task sft \
    --loss_type kl_top_1000 \
    --eakld_confidence_k 16 \
    --distill_temperature 1.0 \
    --distill_alpha 0.5 \
    --prompt_kd_weight 0.0 \
    --hidden_loss_weight 0.0 \
    --distill_pre_mlp_hidden_loss_weight 0.0 \
    --hidden_layer_weighting adaptive_top_3 \
    --teacher_output_offload cpu \
    --distill_teacher_model_offload none \
    --teacher_output_pin_memory true \
    --teacher_output_chunk_tokens 8 \
    --selective_student_topk true \
    --selective_student_topk_chunk_rows 32 \
    --dynamic_padding true \
    --model_max_length 1024 \
    --decoder_layers 0-35 \
    --target_modules all \
    --sparse_bit_tuning false \
    --bit_active_ratio 0.01 \
    --bit_optimizer rms_sgd \
    --bit_lr auto \
    --bit_weight_decay 0.0 \
    --bit_round_steps auto \
    --lora_rank 12 \
    --lora_alpha 24 \
    --lora_dropout 0.03 \
    --compressed_lora_scope full \
    --layer_device_map auto \
    --parallel_stage_decode true \
    --vae_decoder_checkpoint true \
    --tune_final_norm false \
    --use_post_norm_head_linear false \
    --vae_tune_bias false \
    --offload_mode none \
    --eval_after_save "${EVAL_AFTER_SAVE}" \
    --eval_hif4_act false \
    --eval_tasks "${EVAL_TASKS}" \
    --eval_num_fewshot 0 \
    --eval_lm_batch_size auto \
    --eval_device "${EVAL_DEVICE}" \
    --eval_prewarm_group_size "${EVAL_PREWARM_GROUP_SIZE}" \
    --skip_ppl_eval true \
    --ppl_seqlen 2048 \
    --ppl_limit -1 \
    --save_tokenizer true \
    --bf16 true \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --decoder_lr 3e-6 \
    --learning_rate 3e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.0001 \
    --max_grad_norm 15.0 \
    --logging_steps 10 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 2 \
    --max_steps "${MAX_STEPS}" \
    "$@"
else
  echo "Unsupported PARALLEL_MODE=${PARALLEL_MODE}. Expected dp or layer_mp." >&2
  exit 1
fi
