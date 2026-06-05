#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-.result/vae_e2e_fintuning/final_model_20260603_084628/final_model}"
ADAPTER_DIR="${ADAPTER_DIR:-}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "CHECKPOINT_DIR is required, e.g. CHECKPOINT_DIR=.result/<run>/final_model bash scripts/eval.sh" >&2
  exit 1
fi

# 可按需补充的可选参数：
# --access_token "hf_xxx"
# --tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu"
# --eval_hif4_act "true"
# - 带 proxy adapter 的 e2e checkpoint 会在 eval_device 上先 grouped materialize。
# - 非 proxy checkpoint 才走普通 VAELinear cache warmup。
# - 若设置 ADAPTER_DIR，会基于压缩 checkpoint 重建 dense 模型并挂载/merge adapter 后再评估。
if [[ -n "${ADAPTER_DIR}" ]]; then
  python tools/cat_eval.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --adapter_dir "${ADAPTER_DIR}" \
    --eval_ppl \
    --eval_lm_eval \
    --tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu" \
    --eval_device "${EVAL_DEVICE}" \
    --lm_batch_size "auto" \
    --num_fewshot "0" \
    --prewarm_group_size "8" \
    --ppl_seqlen "2048" \
    --ppl_limit "-1" \
    --eval_hif4_act "false" \
    --eval_log_dir "./eval_log" \
    "$@"
else
  python tools/cat_eval.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --eval_ppl \
    --eval_lm_eval \
    --tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu" \
    --eval_device "${EVAL_DEVICE}" \
    --lm_batch_size "auto" \
    --num_fewshot "0" \
    --prewarm_group_size "8" \
    --ppl_seqlen "2048" \
    --ppl_limit "-1" \
    --eval_hif4_act "false" \
    --eval_log_dir "./eval_log" \
    "$@"
fi
