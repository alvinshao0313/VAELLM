#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH="${PYTHONPATH:-.}:." \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4} \
python tools/collect_activation_absmax.py \
  --model_path "${MODEL_PATH:-meta-llama/Llama-2-7b-hf}" \
  --access_token "${ACCESS_TOKEN:-}" \
  --device "${DEVICE:-cuda}" \
  --nsamples "${NSAMPLES:-512}" \
  --seqlen "${SEQLEN:-512}" \
  --projection_suffixes "${PROJECTION_SUFFIXES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}" \
  --output_dir "${OUTPUT_DIR:-./prepares}" \
  --log_every "${LOG_EVERY:-50}" \
  "$@"
