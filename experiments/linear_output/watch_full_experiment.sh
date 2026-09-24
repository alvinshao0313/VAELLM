#!/usr/bin/env bash
# Run from the VAELLM repository with bitvae activated.
# Reuses the three successful v5 shards; refuses to overwrite any results.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m experiments.linear_output.recover_experiment --root result/linear_output/recovery_20260923
