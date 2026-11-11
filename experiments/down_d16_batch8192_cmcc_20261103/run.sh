#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export PYTHONHASHSEED=31
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
python -B -u -m experiments.down_d16_batch8192_cmcc_20261103.sequence
