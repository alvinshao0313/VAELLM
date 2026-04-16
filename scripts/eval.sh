#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=5

# 可按需补充的可选参数：
# --access_token "hf_xxx"
# --tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa"
# --eval_hif4_act "true"
# - 带 proxy adapter 的 e2e checkpoint 会在 eval_device 上先 grouped materialize。
# - 非 proxy checkpoint 才走普通 VAELinear cache warmup。

python tools/cat_eval.py \
  --checkpoint_dir ".result/e2e_vae_lora_openorca/final_model_20260415_123052/final_model" \
  --eval_ppl \
  --eval_lm_eval \
  --tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa" \
  --eval_device "cuda" \
  --lm_batch_size "auto" \
  --num_fewshot "0" \
  --prewarm_group_size "8" \
  --ppl_seqlen "2048" \
  --ppl_limit "-1" \
  --eval_hif4_act "false" \
  --eval_log_dir "./eval_log" \
  "$@"
