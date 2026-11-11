#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
python -B -u -m experiments.s4_qk_source_weights_20261107.run --arm web3
