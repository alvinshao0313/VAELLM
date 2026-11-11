#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=6
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -B -u -m experiments.muon_compact_cmcc_20261030.run --gpu 6 --timeout-seconds 7200
python -B -u -m experiments.muon_compact_cmcc_20261030.fresh
