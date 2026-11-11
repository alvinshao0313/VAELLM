#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
python -B -u -m experiments.cat_down_current_decoder_cmcc_20261104.sequence
