#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
python -B -u -m experiments.s4_webmed_recovery_20261105.run \
  --plan /data1/cmcc/VAELLM/artifacts/progressive_residual_bsq_20261030/s4_from_bf16_20261102/skip_up_20261103/webmed_20261105/trial_20261105_1223/plan.json \
  --arm webmed
