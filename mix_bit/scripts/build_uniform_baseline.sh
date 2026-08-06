#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_CONFIG="${ROOT_DIR}/mix_bit/configs/runs/qwen3_8b_vae_1to3bit.json"
INVENTORY="${ROOT_DIR}/.result/mix_bit/qwen3_8b/model_inventory.json"
POOL_MANIFEST="${ROOT_DIR}/.result/mix_bit/qwen3_8b/runs/qwen3_8b_vae_1to3bit/candidate_pool/candidate_manifest.json"

python -m mix_bit.cli.prepare_uniform_baseline \
  --run_config "${RUN_CONFIG}" \
  --inventory "${INVENTORY}" \
  --pool_manifest "${POOL_MANIFEST}" \
  --device cuda
