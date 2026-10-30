#!/usr/bin/env bash
set -euo pipefail

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python -m experiments.adaptive_layer_transfer_20261028.run \
  --prepared-dir /data/star/VAELLM/adaptive_layer_transfer_20261028/prepared \
  "$@"
