#!/usr/bin/env bash
# Prepared full Stage B command; NOT launched. Requires cached RedPajama and GPU allocation.
# Run from VAELLM root in bitvae; $1 is a NEW output path, $2 the complete 11-shard Arrow directory.
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python -u -m experiments.liftquant_recovery.recover \
  --checkpoint result/linear_output/distill_init \
  --output "$1" \
  --redpajama-arrow-dir "$2" \
  --blocks all \
  --nsamples 4096 \
  --seqlen 2048 \
  --batch-size 2 \
  --epochs 2 \
  --seed 42 \
  --code-lr 2e-5 \
  --decoder-lr 1.25e-5 \
  --gpu-memory-gib 12
