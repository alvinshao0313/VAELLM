#!/usr/bin/env bash
# Paired block-9 diagnostic only; activated bitvae shell, unique output path in $1.
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python -u -m experiments.liftquant_recovery.bit_ablation \
  --checkpoint result/linear_output/distill_init --output "$1" \
  --smoke-rows experiments/liftquant_recovery/redpajama_smoke_rows.json \
  --train-samples 16 --holdout-samples 16 --seqlen 128 --block 9 \
  --batch-size 2 --steps 384 --seed 42 --code-lr 2e-5 --decoder-lr 1.25e-5 --gpu-memory-gib 12
