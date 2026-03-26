#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=4

# 可按需补充的可选参数：
# --access_token "hf_xxx"
# --tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa"

python tools/cat_eval.py \
  --checkpoint_dir ".result/e2e_vae_lora_redpajama/final_model_20260324_103747/final_model" \
  --eval_ppl \
  --eval_lm_eval \
  --tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa" \
  --eval_device "cuda" \
  --lm_batch_size "auto" \
  --num_fewshot "0" \
  --ppl_seqlen "2048" \
  --ppl_limit "-1" \
  --eval_log_dir "./eval_log" \
  "$@"
