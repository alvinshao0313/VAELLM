#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python -B -u -m experiments.sc_cat_recovery_cmcc_20261108.remaining --authorization artifacts/sc_cat/qwen--qwen3-8b/2026-11-08_172625_s31/authorization_second.json
