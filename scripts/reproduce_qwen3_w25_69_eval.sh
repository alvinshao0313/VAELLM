#!/usr/bin/env bash
set -euo pipefail

# Run from /home/star/VAELLM; caller activates bitvae and selects idle GPU(s).
OUTPUT_DIR="${1:?Pass a new output directory as the first argument}"
mkdir -- "$OUTPUT_DIR"
trap 'printf "%s\n" "$?" > "$OUTPUT_DIR/run.exit"' EXIT

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export OMP_NUM_THREADS=4

CHECKPOINT_DIR=/home/star/VAELLM/result/qwen3_8b_w25_20260930/e2e_best_combo_down0_joint_s2000/step_1200/final_model

python tools/cat_eval.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --strict \
  --eval_lm_eval \
  --eval_device cuda:0 \
  --tasks mmlu \
  --num_fewshot 0 \
  --lm_batch_size 64 \
  --tokenizer_use_fast true \
  --prewarm_group_size 8 \
  --eval_log_dir "$OUTPUT_DIR/mmlu_logs" \
  --output_json "$OUTPUT_DIR/mmlu.json"

python tools/cat_eval.py \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --strict \
  --eval_lm_eval \
  --eval_device cuda:0 \
  --tasks boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa \
  --num_fewshot 0 \
  --lm_batch_size 32 \
  --tokenizer_use_fast true \
  --prewarm_group_size 8 \
  --eval_log_dir "$OUTPUT_DIR/seven_logs" \
  --output_json "$OUTPUT_DIR/seven.json"

python -m experiments.qwen3_w25.summarize_results \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --results "$OUTPUT_DIR/mmlu.json" "$OUTPUT_DIR/seven.json" \
  --output "$OUTPUT_DIR/acceptance.json"

