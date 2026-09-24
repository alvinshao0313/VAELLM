#!/usr/bin/env bash
# Historical 9-block recovery, superseded by resume_full_decoder_1p25e5_02.sh; its redundant boundary has been cleaned after validation.
# Caller activates bitvae and enters the verified restart_source_01 directory.
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/shaoyuantian/program/VAELLM/.result/liftquant_recovery/full_layers_lr_20260924_01/restart_source_01
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python -u -m experiments.liftquant_recovery.run_experiment \
  --checkpoint /home/shaoyuantian/program/VAELLM/result/linear_output/distill_init \
  --output /home/shaoyuantian/program/VAELLM/.result/liftquant_recovery/full_layers_lr_20260924_01/decoder_1p25e5_resume_01 \
  --resume /home/shaoyuantian/program/VAELLM/.result/liftquant_recovery/full_layers_lr_20260924_01/decoder_1p25e5/recovery/latest_boundary.pt \
  --redpajama-arrow-dir /home/shaoyuantian/program/VAELLM/data/redpajama_liftquant_4b6d76ca_20260924 \
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
