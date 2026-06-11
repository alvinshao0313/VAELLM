#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "CHECKPOINT_DIR is required, e.g. CHECKPOINT_DIR=.result/<run>/final_model bash scripts/block_prefix_eval.sh" >&2
  exit 1
fi

# 可选参数：
# --access_token "hf_xxx"
# --lm_limit 100
# --eval_hif4_act "true"

python tools/block_prefix_eval.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu" \
  --eval_device "${EVAL_DEVICE}" \
  --lm_batch_size "auto" \
  --num_fewshot 0 \
  --prewarm_group_size 8 \
  --eval_hif4_act "false" \
  --eval_log_dir "./eval_log/block_prefix_eval" \
  "$@"
