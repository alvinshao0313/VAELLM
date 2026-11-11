#!/bin/bash
set -euo pipefail
cd /home/cmcc/VAELLM
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
python -B -u -m experiments.s4_attention128_20261106.run
