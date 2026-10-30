#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=31
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
python -B -u /home/star/VAELLM/experiments/cat_structure_full_20261029/run.py --arm A --phase formal --gpu 3
