#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_CONFIG="${ROOT_DIR}/mix_bit/configs/runs/qwen3_8b_vae_1to3bit.json"
INVENTORY="${ROOT_DIR}/.result/mix_bit/qwen3_8b/model_inventory.json"

python -m mix_bit.cli.prepare_calibration \
  --run_config "${RUN_CONFIG}" \
  --inventory "${INVENTORY}"
