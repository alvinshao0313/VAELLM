#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_CONFIG="${ROOT_DIR}/mix_bit/configs/runs/qwen3_8b_vae_1to3bit.json"
OUTPUT_DIR="${ROOT_DIR}/.result/mix_bit/qwen3_8b"
OUTPUT="${OUTPUT_DIR}/model_inventory.json"

python -m mix_bit.cli.build_model_inventory \
  --run_config "${RUN_CONFIG}" \
  --output "${OUTPUT}"
