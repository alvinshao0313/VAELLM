#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_CONFIG="${ROOT_DIR}/mix_bit/configs/runs/qwen3_8b_vae_1to3bit.json"
INVENTORY="${ROOT_DIR}/.result/mix_bit/qwen3_8b/model_inventory.json"
DATASET="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/calibration/dataset.pt"
DATASET_MANIFEST="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/calibration/dataset_manifest.json"

python -m mix_bit.cli.build_teacher_cache \
  --run_config "${RUN_CONFIG}" \
  --inventory "${INVENTORY}" \
  --dataset "${DATASET}" \
  --dataset_manifest "${DATASET_MANIFEST}" \
  --teacher_topk 256 \
  --cache_prob_dtype bfloat16 \
  --chunk_samples 16 \
  --batch_size 1 \
  --device cuda
