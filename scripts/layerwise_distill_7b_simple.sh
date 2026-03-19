#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}:."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

STUDENT_CHECKPOINT_DIR="${STUDENT_CHECKPOINT_DIR:-.result/meta-llama_Llama-2-7b-hf_20260312_075405/final_model}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-meta-llama/Llama-2-7b-hf}"
OUTPUT_DIR="${OUTPUT_DIR:-.result}"
DISTILL_DEVICE="${DISTILL_DEVICE:-cuda:0}"

python distill_utils/layerwise_distill.py \
  --student_checkpoint_dir "${STUDENT_CHECKPOINT_DIR}" \
  --teacher_model_path "${TEACHER_MODEL_PATH}" \
  --distill_device "${DISTILL_DEVICE}" \
  --nsamples "${NSAMPLES:-1000}" \
  --seqlen "${SEQLEN:-512}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --steps_per_layer "${STEPS_PER_LAYER:-1000}" \
  --lr "${LR:-1e-5}" \
  --cache_dtype "${CACHE_DTYPE:-bfloat16}" \
  --teacher_label_dtype "${TEACHER_LABEL_DTYPE:-bfloat16}" \
  --memory_safety_factor "${MEMORY_SAFETY_FACTOR:-0.85}" \
  --lambda_blk "${LAMBDA_BLK:-0.0}" \
  --lambda_res "${LAMBDA_RES:-1.}" \
  --lambda_aug_loss "${LAMBDA_AUG_LOSS:-1e-3}" \
  --lambda_anchor "${LAMBDA_ANCHOR:-0.0}" \
  --lambda_norm "${LAMBDA_NORM:-0.0}" \
  --lambda_attn_map "${LAMBDA_ATTN_MAP:-0.0}" \
  --lambda_attn_block_mean "${LAMBDA_ATTN_BLOCK_MEAN:-0.0}" \
  --log_every "${LOG_EVERY:-50}" \
  --output_dir "${OUTPUT_DIR}" \
  --unload_vae_original_weights_on_save \
  --wandb_project "${WANDB_PROJECT:-Distill_7B}" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --wandb_name "${WANDB_NAME:-}" \
  --wandb_group "${WANDB_GROUP:-}" \
  --wandb_tags "${WANDB_TAGS:-}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  "$@"
