#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/home/cmcc/VAELLM
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=4 python -B -u experiments/g09_hotpot_coverage_20261030/cmcc_export.py
printf '0\n' > /data1/cmcc/VAELLM/artifacts/g09_hotpot_coverage_cmcc_20261030/export.exit
CUDA_VISIBLE_DEVICES='' python -B -u experiments/g09_hotpot_coverage_20261030/cmcc_evaluate.py --gpus 5,6,7
printf '0\n' > /data1/cmcc/VAELLM/artifacts/g09_hotpot_coverage_cmcc_20261030/full9.exit
