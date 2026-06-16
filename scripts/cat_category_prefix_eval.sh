#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

CHECKPOINT_DIR="${CHECKPOINT_DIR:-.result/cat_train_final_model}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "CHECKPOINT_DIR is required, e.g. CHECKPOINT_DIR=.result/<run>/final_model bash scripts/cat_category_prefix_eval.sh" >&2
  exit 1
fi

# 可按需补充的可选参数：
# --base_model_path "Qwen/Qwen3-8B"
# --access_token "hf_xxx"
# --category_sweep "down_proj,gate_proj,up_proj"
# --tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu"
# --lm_limit 100
# --eval_hif4_act "true"

python tools/cat_category_prefix_eval.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --category_sweep "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --tasks "mmlu" \
  --eval_device "${EVAL_DEVICE}" \
  --lm_batch_size "auto" \
  --num_fewshot "0" \
  --prewarm_group_size "8" \
  --eval_hif4_act "false" \
  --eval_log_dir "./eval_log/cat_category_prefix_eval" \
  "$@"
