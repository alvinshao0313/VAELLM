#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/edgerazor_qwen3}"

export PYTHONPATH="${PROJECT_ROOT}"
unset HF_ENDPOINT
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE

echo "============================================"
echo " VAELLM EdgeRazor Data Preparation"
echo " Output: ${OUTPUT_DIR}"
echo "============================================"

bash "${SCRIPT_DIR}/download_distill_dataset.sh"
