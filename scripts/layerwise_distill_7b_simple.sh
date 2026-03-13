#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}:."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STUDENT_CHECKPOINT_DIR="${STUDENT_CHECKPOINT_DIR:-.result/meta-llama_Llama-2-7b-hf_20260312_075405/final_model}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-meta-llama/Llama-2-7b-hf}"
OUTPUT_DIR="${OUTPUT_DIR:-.result}"

STUDENT_DEVICE="${STUDENT_DEVICE:-cuda:0}"

# 自动策略：有两张可见卡就把 teacher 放第二张，否则放 CPU
if [[ -z "${TEACHER_DEVICE:-}" ]]; then
  if [[ "${CUDA_VISIBLE_DEVICES}" == *","* ]]; then
    TEACHER_DEVICE="cuda:1"
  else
    TEACHER_DEVICE="cpu"
  fi
fi

python distill_utils/layerwise_distill.py \
  --student_checkpoint_dir "${STUDENT_CHECKPOINT_DIR}" \
  --teacher_model_path "${TEACHER_MODEL_PATH}" \
  --student_device "${STUDENT_DEVICE}" \
  --teacher_device "${TEACHER_DEVICE}" \
  --nsamples "${NSAMPLES:-128}" \
  --seqlen "${SEQLEN:-512}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --steps_per_layer "${STEPS_PER_LAYER:-100}" \
  --lr "${LR:-2e-5}" \
  --lambda_blk "${LAMBDA_BLK:-0.70}" \
  --lambda_res "${LAMBDA_RES:-0.25}" \
  --lambda_anchor "${LAMBDA_ANCHOR:-0.05}" \
  --log_every "${LOG_EVERY:-10}" \
  --output_dir "${OUTPUT_DIR}" \
  --save_model \
  --unload_vae_original_weights_on_save \
  "$@"
