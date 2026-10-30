#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python -B -m experiments.adaptive_sparse_pair_20261028.run \
  --output-dir /data/star/VAELLM/adaptive_sparse_pair_20261028/run
