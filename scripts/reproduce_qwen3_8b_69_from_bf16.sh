#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after activating bitvae; select two idle A800 80GB GPUs.
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export OMP_NUM_THREADS=4

MODEL_PATH="${MODEL_PATH:-/home/star/models/Qwen/Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-result/qwen3_8b_69_from_bf16_$(date +%Y%m%d_%H%M%S)}"

python -m experiments.qwen3_w25.reproduce_from_bf16 \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --gpus 0,1 \
  "$@"
