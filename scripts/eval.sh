#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
ADAPTER_DIR="${ADAPTER_DIR:-}"

# 可按需补充的可选参数：
# --access_token "hf_xxx"
# --tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa"
# --eval_hif4_act "true"
# - 带 proxy adapter 的 e2e checkpoint 会在 eval_device 上先 grouped materialize。
# - 非 proxy checkpoint 才走普通 VAELinear cache warmup。
# - 若设置 ADAPTER_DIR，会基于压缩 checkpoint 重建 dense 模型并挂载/merge adapter 后再评估。

EXTRA_ARGS=()
if [[ -n "${ADAPTER_DIR}" ]]; then
  EXTRA_ARGS+=(--adapter_dir "${ADAPTER_DIR}")
fi

python tools/cat_eval.py \
  --checkpoint_dir "meta-llama/Llama-3.1-8B" \
  --eval_ppl \
  --eval_lm_eval \
  --tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa" \
  --eval_device "cuda" \
  --lm_batch_size "auto" \
  --num_fewshot "0" \
  --prewarm_group_size "8" \
  --ppl_seqlen "2048" \
  --ppl_limit "-1" \
  --eval_hif4_act "true" \
  --eval_log_dir "./eval_log" \
  "${EXTRA_ARGS[@]}" \
  "$@"
