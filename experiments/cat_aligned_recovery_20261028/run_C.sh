#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/star/VAELLM/result/qwen3_8b_mmlu65_20261004/rotation_factor_probe/source_snapshot_20261005_01
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=1
export CUDA_VISIBLE_DEVICES=1
export VAELLM_ALIGNED_PROFILE=C
python -B -u /home/star/VAELLM/experiments/cat_aligned_recovery_20261028/run.py
