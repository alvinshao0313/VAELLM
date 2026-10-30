#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${VAELLM_ALIGNED_GPU:?Set one newly assigned physical GPU}"
python -B -u /home/star/VAELLM/experiments/cat_aligned_selected_20261028/selected_run.py \
  --stage B32 --prepared "/data/star/VAELLM/cat_aligned_recovery_20261028/${VAELLM_ALIGNED_PROFILE:?Choose C or G}/06_B32/prepared.json" \
  --phase formal --gpu "$VAELLM_ALIGNED_GPU" --timeout-seconds 604800
