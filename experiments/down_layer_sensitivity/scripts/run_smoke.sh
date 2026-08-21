#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.

CHECKPOINT_DIR=".result/catlora/res0-bf16-protect-channel-vae/final_model"
OUTPUT_DIR=".result/experiments/down_layer_sensitivity"
GPUS="${GPUS:-0}"

python experiments/down_layer_sensitivity/run.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --gpus "${GPUS}" \
  --mode smoke
