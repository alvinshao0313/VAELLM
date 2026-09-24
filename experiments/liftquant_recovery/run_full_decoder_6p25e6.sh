#!/usr/bin/env bash
# From VAELLM root in activated bitvae. Initial baseline is evaluated by the GPU0 run.
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python -u -m experiments.liftquant_recovery.run_experiment \
  --checkpoint result/linear_output/distill_init \
  --output .result/liftquant_recovery/full_layers_lr_20260924_01/decoder_6p25e6 \
  --redpajama-arrow-dir data/redpajama_liftquant_4b6d76ca_20260924 \
  --blocks all \
  --nsamples 4096 \
  --holdout 128 \
  --seqlen 2048 \
  --batch-size 2 \
  --epochs 2 \
  --seed 42 \
  --code-lr 2e-5 \
  --decoder-lr 6.25e-6 \
  --gpu-memory-gib 64
