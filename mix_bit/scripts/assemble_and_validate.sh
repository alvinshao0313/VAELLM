#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_CONFIG="${ROOT_DIR}/mix_bit/configs/runs/qwen3_8b_vae_1to3bit.json"
INVENTORY="${ROOT_DIR}/.result/mix_bit/qwen3_8b/model_inventory.json"
POOL_MANIFEST="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/candidate_pool/candidate_manifest.json"
COST_TABLE="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/costs/topk_k256/cost_table.jsonl"
COST_TABLE_META="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/costs/topk_k256/cost_table_meta.json"
ALLOCATION="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/allocation/topk_k256/optimal_2bit.json"
BASELINE_OVERLAY="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/baseline/b32d32s2/baseline_overlay.json"
MIXED_MODEL_DIR="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/mixed_model/topk_k256/optimal_2bit/final_model"
DATASET="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/calibration/dataset.pt"
DATASET_MANIFEST="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/calibration/dataset_manifest.json"
TEACHER_CACHE="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/calibration/teacher_topk/k256"

python -m mix_bit.cli.assemble_mixed_model \
  --run_config "${RUN_CONFIG}" \
  --inventory "${INVENTORY}" \
  --pool_manifest "${POOL_MANIFEST}" \
  --allocation "${ALLOCATION}" \
  --device cuda

python -m mix_bit.cli.validate_mixed_model \
  --run_config "${RUN_CONFIG}" \
  --inventory "${INVENTORY}" \
  --pool_manifest "${POOL_MANIFEST}" \
  --cost_table "${COST_TABLE}" \
  --cost_table_meta "${COST_TABLE_META}" \
  --allocation "${ALLOCATION}" \
  --baseline_overlay "${BASELINE_OVERLAY}" \
  --mixed_model_dir "${MIXED_MODEL_DIR}" \
  --dataset "${DATASET}" \
  --dataset_manifest "${DATASET_MANIFEST}" \
  --teacher_cache "${TEACHER_CACHE}" \
  --device cuda \
  --lm_batch_size auto
