#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MAX_STEPS="${MAX_STEPS:-20000}"
STUDENT_CKPT="${STUDENT_CKPT:-.result/meta-llama_Llama-3.1-8B_20260421_113551/final_model}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-.result/dense_e2e_fintuning_alpaca}"
DATASET_MIX="${DATASET_MIX:-alpaca=1.0}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
DECODE_DEVICE="${DECODE_DEVICE:-auto}"
DECODE_GROUP_SIZE="${DECODE_GROUP_SIZE:-8}"

if [[ "${DISABLE_PROXY:-1}" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  unset all_proxy ALL_PROXY no_proxy NO_PROXY
fi

# 说明：
# - 压缩 checkpoint -> dense 重建 -> 标准 PEFT 训练。
# - 多卡时 `decode_device=auto` 会按当前 rank 的可见设备选卡，不再全部挤到 0 卡。
# - 数据预处理支持 `DATASET_NUM_PROC`，并且只让主进程先写 datasets cache，其余 rank 复用。
# - `--eval_strategy no` 时会直接跳过 eval 数据预处理。
# - 冒烟建议：MAX_STEPS=30 NPROC_PER_NODE=1 DATASET_NUM_PROC=1 bash dense_e2e_fintuning/scripts/e2e_dense_lora.sh

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m dense_e2e_fintuning.main \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --student_checkpoint_dir "${STUDENT_CKPT}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --dataset_mix "${DATASET_MIX}" \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --loss_type sft \
  --distill_temperature 1.0 \
  --distill_alpha 0.3 \
  --post_attn false \
  --model_max_length 8192 \
  --decoder_layers 0-31 \
  --target_modules all \
  --decode_device "${DECODE_DEVICE}" \
  --decode_group_size "${DECODE_GROUP_SIZE}" \
  --lora_variant dora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_tune_bias false \
  --lora_init_mode zero \
  --tune_final_norm true \
  --use_post_norm_head_linear false \
  --lora_hif4_act false \
  --eval_hif4_act false \
  --skip_ppl_eval false \
  --ppl_seqlen 2048 \
  --ppl_limit -1 \
  --save_tokenizer true \
  --bf16 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 6e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --logging_steps 10 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 1 \
  --max_steps "${MAX_STEPS}" \
  "$@"
