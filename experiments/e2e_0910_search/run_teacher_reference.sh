#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=4,5
export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DISTILL_NCCL_TIMEOUT_SEC=10800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800
python -m torch.distributed.run --standalone --nproc_per_node=2 -m experiments.e2e_0910_search.teacher_reference \
  --output result/compressed_e2e_fintuning/e2e_0910_search_20260924/teacher_reference
