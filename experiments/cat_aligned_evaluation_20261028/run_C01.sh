#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH=/home/star/VAELLM
python -B -u /home/star/VAELLM/experiments/cat_aligned_evaluation_20261028/evaluate.py --profile C --stage 01 --gpus 2,3
