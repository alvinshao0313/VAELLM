#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
python -B -u -m experiments.s4_qk_source128_20261107.run --arm qk
