#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${VAELLM_ALIGNED_GPU:?Set one newly assigned physical GPU}"
python -B -u /home/star/VAELLM/experiments/cat_aligned_protection_20261028/run.py \
  --prepared "/data/star/VAELLM/cat_aligned_recovery_20261028/${VAELLM_ALIGNED_PROFILE:?Choose C or G}/03_protection/prepared.json" \
  --arm scale32 --phase formal --gpu "$VAELLM_ALIGNED_GPU" --timeout-seconds 604800
