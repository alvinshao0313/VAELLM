#!/bin/bash
set -eu
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=/home/star/VAELLM
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/star/VAELLM/result/qwen3_8b_mmlu65_20261004/rotation_factor_probe/source_snapshot_20261005_01
python -B -u /home/star/VAELLM/experiments/g09_hotpot_coverage_20261030/run.py --phase formal32 --gpu 4 --timeout-seconds 43200
python -B -u /home/star/VAELLM/experiments/g09_hotpot_coverage_20261030/export.py --report /data/star/VAELLM/g09_hotpot_coverage_20261030/formal32/report.json --output-dir /data/star/VAELLM/g09_hotpot_coverage_20261030/native_final/final_model
