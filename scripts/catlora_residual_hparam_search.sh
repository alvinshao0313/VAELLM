#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.

python tools/run_cat_residual_hparam_search.py \
  --gpus 0,1 \
  --categories up_proj,gate_proj \
  --search_root ".result/catlora_residual_from_base/hparam_search"
