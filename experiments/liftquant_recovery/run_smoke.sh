#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=4
export PYTHONHASHSEED=31
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python experiments/liftquant_recovery/smoke.py "$@"
