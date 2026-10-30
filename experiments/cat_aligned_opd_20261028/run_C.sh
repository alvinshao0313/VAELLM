#!/bin/bash
set -eu
export VAELLM_ALIGNED_PROFILE=C
export CUDA_VISIBLE_DEVICES=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/star/VAELLM/result/qwen3_8b_mmlu65_20261004/rotation_factor_probe/source_snapshot_20261005_01
python -B -u /home/star/VAELLM/experiments/cat_aligned_opd_20261028/run.py --prepared /data/star/VAELLM/cat_aligned_recovery_20261028/C/02_online_opd32/prepared.json --phase formal32 --gpu 1 --timeout-seconds 43200
