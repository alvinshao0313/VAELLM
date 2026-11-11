#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

python -B -u -m experiments.sc_cat_recovery_cmcc_20261108.gpu_validation \
  --authorization artifacts/sc_cat/qwen--qwen3-8b/2026-11-08_172625_s31/authorization_import_fix.json
python -B -u -m experiments.sc_cat_recovery_cmcc_20261108.first_batch \
  --authorization artifacts/sc_cat/qwen--qwen3-8b/2026-11-08_172625_s31/authorization_import_fix.json \
  --technical-receipt artifacts/sc_cat/qwen--qwen3-8b/2026-11-08_172625_s31/technical_import_fix/validation.json
