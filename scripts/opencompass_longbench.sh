#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

DENSE_HF_PATH="${DENSE_HF_PATH:-Qwen/Qwen3-8B}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-.result/final_model}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3-8B}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
MAX_OUT_LEN="${MAX_OUT_LEN:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
HF_NUM_GPUS="${HF_NUM_GPUS:-1}"
WORK_DIR="${WORK_DIR:-./eval_log/opencompass_longbench}"
COMPASS_DATA_CACHE="${COMPASS_DATA_CACHE:-${HOME}/.cache/opencompass}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
PREWARM_GROUP_SIZE="${PREWARM_GROUP_SIZE:-8}"
MAX_NUM_WORKERS="${MAX_NUM_WORKERS:-1}"
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-1}"
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-configs/models/opencompass_longbench_vaellm_$$.py}"
MODEL_CONFIG_NAME="$(basename "${MODEL_CONFIG_PATH}" .py)"

export DENSE_HF_PATH
export CHECKPOINT_DIR
export ADAPTER_DIR
export BASE_MODEL_PATH
export MAX_SEQ_LEN
export MAX_OUT_LEN
export BATCH_SIZE
export HF_NUM_GPUS
export COMPASS_DATA_CACHE
export EVAL_DEVICE
export PREWARM_GROUP_SIZE

if [[ -n "${CHECKPOINT_DIR}" && ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "CHECKPOINT_DIR does not exist: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

if [[ -n "${ADAPTER_DIR}" && ! -d "${ADAPTER_DIR}" ]]; then
  echo "ADAPTER_DIR does not exist: ${ADAPTER_DIR}" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}"
mkdir -p "$(dirname "${MODEL_CONFIG_PATH}")"

python scripts/build_opencompass_longbench_config.py --output "${MODEL_CONFIG_PATH}"
trap 'rm -f "${MODEL_CONFIG_PATH}"' EXIT

opencompass \
  --config-dir configs \
  --models "${MODEL_CONFIG_NAME}" \
  --datasets longbench \
  --summarizer longbench \
  -w "${WORK_DIR}" \
  --max-num-workers "${MAX_NUM_WORKERS}" \
  --max-workers-per-gpu "${MAX_WORKERS_PER_GPU}" \
  "$@"
