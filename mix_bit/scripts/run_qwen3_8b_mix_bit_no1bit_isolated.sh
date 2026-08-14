#!/usr/bin/env bash
# Qwen3-8B mixed-bit：复用已有 cost_table，排除 1bit（b16d32s2），
# 只用 1.5/2/2.5/3bit 求解平均 2bit，并组装到隔离目录。
# 用法（必须已激活 bitvae）：
#   conda activate bitvae
#   bash mix_bit/scripts/run_qwen3_8b_mix_bit_no1bit_isolated.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_CONFIG="${ROOT_DIR}/mix_bit/configs/runs/qwen3_8b_vae_1to3bit.json"
INVENTORY="${ROOT_DIR}/.result/mix_bit/qwen3_8b/model_inventory.json"
POOL_MANIFEST="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/candidate_pool/candidate_manifest.json"
COST_TABLE="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/costs/topk_k256/cost_table.jsonl"
COST_TABLE_META="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/costs/topk_k256/cost_table_meta.json"
ISOLATED_ROOT="${ROOT_DIR}/.result/mix_bit/qwen3_8b/isolated/qwen3_8b_vae_1to3bit_no1bit"
ALLOCATION_DIR="${ISOLATED_ROOT}/allocation/topk_k256"
ALLOCATION="${ALLOCATION_DIR}/optimal_2bit.json"
MIXED_MODEL_DIR="${ISOLATED_ROOT}/mixed_model/topk_k256/optimal_2bit/final_model"

echo "[mix_bit] isolated solve_allocation exclude b16d32s2 (1bit)"
python -m mix_bit.cli.solve_allocation \
  --run_config "${RUN_CONFIG}" \
  --inventory "${INVENTORY}" \
  --pool_manifest "${POOL_MANIFEST}" \
  --cost_table "${COST_TABLE}" \
  --cost_table_meta "${COST_TABLE_META}" \
  --exclude_modes b16d32s2 \
  --output_dir "${ALLOCATION_DIR}"

echo "[mix_bit] isolated assemble_mixed_model"
python -m mix_bit.cli.assemble_mixed_model \
  --run_config "${RUN_CONFIG}" \
  --inventory "${INVENTORY}" \
  --pool_manifest "${POOL_MANIFEST}" \
  --allocation "${ALLOCATION}" \
  --output_dir "${MIXED_MODEL_DIR}" \
  --device cuda:0

echo "[mix_bit] done"
echo "isolated_root=${ISOLATED_ROOT}"
echo "allocation=${ALLOCATION}"
echo "mixed_model_dir=${MIXED_MODEL_DIR}"
