#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
python -B -u -m experiments.s4_scope_objective_20261107.run --arm A
