#!/usr/bin/env bash
# From VAELLM root in activated bitvae. Output must be new.
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python -u -m experiments.liftquant_recovery.run_experiment \
  --checkpoint result/linear_output/distill_init \
  --output .result/liftquant_recovery/full_layers_lr_20260924_01/decoder_1p25e5 \
  --redpajama-arrow-dir data/redpajama_liftquant_4b6d76ca_20260924 \
  --blocks all \
  --nsamples 4096 \
  --holdout 128 \
  --seqlen 2048 \
  --batch-size 2 \
  --epochs 2 \
  --seed 42 \
  --code-lr 2e-5 \
  --decoder-lr 1.25e-5 \
  --gpu-memory-gib 64 \
  --evaluate-baseline
