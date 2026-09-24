#!/usr/bin/env bash
# Run from the VAELLM root in an activated bitvae shell. $1 is a NEW output path.
export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python -u -m experiments.liftquant_recovery.recover   --checkpoint result/linear_output/distill_init   --output "$1"   --blocks 9,10   --nsamples 4   --holdout 2   --batch-size 2   --seqlen 64   --epochs 16   --seed 42   --code-lr 2e-5   --decoder-lr 1.25e-5   --gpu-memory-gib 12   --smoke-rows experiments/liftquant_recovery/redpajama_smoke_rows.json
