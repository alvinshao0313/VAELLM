#!/bin/bash
set -eu
export CUDA_VISIBLE_DEVICES=7
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/star/VAELLM/result/qwen3_8b_mmlu65_20261004/rotation_factor_probe/source_snapshot_20261005_01
python -B -u /home/star/VAELLM/experiments/g09_recovery_20261029/muon/run.py --phase formal32 --gpu 7 --timeout-seconds 43200
CUDA_VISIBLE_DEVICES='' python -B -u /home/star/VAELLM/experiments/g09_recovery_20261029/muon/export.py --report /data/star/VAELLM/g09_recovery_20261029/muon_opd32/formal32/report.json --output-dir /data/star/VAELLM/g09_recovery_20261029/muon_opd32/native_final/final_model
