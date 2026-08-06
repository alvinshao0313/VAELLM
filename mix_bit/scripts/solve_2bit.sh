#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_CONFIG="${ROOT_DIR}/mix_bit/configs/runs/qwen3_8b_vae_1to3bit.json"
INVENTORY="${ROOT_DIR}/.result/mix_bit/qwen3_8b/model_inventory.json"
COST_TABLE="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/costs/topk_k256/cost_table.jsonl"
COST_TABLE_META="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/costs/topk_k256/cost_table_meta.json"

python -m mix_bit.cli.solve_allocation \
  --run_config "${RUN_CONFIG}" \
  --inventory "${INVENTORY}" \
  --cost_table "${COST_TABLE}" \
  --cost_table_meta "${COST_TABLE_META}"
