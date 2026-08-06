#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_CONFIG="${ROOT_DIR}/mix_bit/configs/runs/qwen3_8b_vae_1to3bit.json"
INVENTORY="${ROOT_DIR}/.result/mix_bit/qwen3_8b/model_inventory.json"
POOL_MANIFEST="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/candidate_pool/candidate_manifest.json"
BASELINE_OVERLAY="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/baseline/b32d32s2/baseline_overlay.json"
DATASET="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/calibration/dataset.pt"
DATASET_MANIFEST="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/calibration/dataset_manifest.json"
TEACHER_CACHE="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/calibration/teacher_topk/k256"

# teacher_topk run (default). For exact_full_vocab, replace the kl_mode block with:
#   --kl_mode exact_full_vocab \
# and omit --teacher_topk / --teacher_cache.

python -m mix_bit.cli.compute_cost_table \
  --run_config "${RUN_CONFIG}" \
  --inventory "${INVENTORY}" \
  --pool_manifest "${POOL_MANIFEST}" \
  --baseline_overlay "${BASELINE_OVERLAY}" \
  --dataset "${DATASET}" \
  --dataset_manifest "${DATASET_MANIFEST}" \
  --kl_mode teacher_topk \
  --teacher_topk 256 \
  --teacher_cache "${TEACHER_CACHE}" \
  --gpus 4,5,6,7 \
  --batch_size 1
