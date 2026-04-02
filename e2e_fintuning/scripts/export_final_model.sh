#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

# RESUME_CKPT 直接指向中间 checkpoint-* 根目录。
# 新的紧凑中间 ckpt 会在这个目录下直接保存 pytorch_model.bin + checkpoint_meta.json。
RESUME_CKPT=.result/e2e_vae_lora_redpajama/final_model_20260327_110717/trainer_state/checkpoint-80000
RUN_ROOT_DIR=${RESUME_CKPT}/.resume_export
OUT_FINAL_DIR=${RESUME_CKPT}/final_model

if [[ -e "$OUT_FINAL_DIR" ]]; then
  echo "Refusing to overwrite existing output: $OUT_FINAL_DIR" >&2
  exit 1
fi

extra_args=()
if [[ -n "${ACCESS_TOKEN:-}" ]]; then
  extra_args+=(--access_token "$ACCESS_TOKEN")
fi

eval "$(conda shell.bash hook)"
conda activate bitvae

python -m e2e_fintuning.main \
  --student_checkpoint_dir .result/meta-llama_Llama-2-7b-hf_20260323_071142/final_model \
  --run_root_dir "$RUN_ROOT_DIR" \
  --resume_from_checkpoint "$RESUME_CKPT" \
  --dataset_name ZengXiangyu/RedPajama-Data-1T-Sample \
  --train_split train \
  --eval_split validation \
  --text_field text \
  --loss_type sft \
  --distill_temperature 1.0 \
  --distill_alpha 0.3 \
  --post_attn false \
  --model_max_length 4096 \
  --decoder_layers 0-31 \
  --target_modules all \
  --vae_lora_rank 8 \
  --vae_lora_alpha 16 \
  --vae_lora_dropout 0.0 \
  --lora_embedding false \
  --lora_lm_head false \
  --lora_hif4_act false \
  --prewarm_frozen_vae true \
  --prewarm_log_every 32 \
  --skip_ppl_eval true \
  --ppl_seqlen 2048 \
  --ppl_limit -1 \
  --save_tokenizer true \
  --unload_vae_original_weights_on_save false \
  --bf16 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --logging_steps 10 \
  --num_train_epochs 0 \
  --max_steps -1 \
  "${extra_args[@]}"

shopt -s nullglob
runs=("$RUN_ROOT_DIR"/final_model_*)
latest_run="${runs[@]: -1}"
[[ -n "$latest_run" && -d "$latest_run/final_model" ]] || {
  echo "Missing export output under: $RUN_ROOT_DIR" >&2
  exit 1
}

mv "$latest_run/final_model" "$OUT_FINAL_DIR"

echo "Exported final model to: $OUT_FINAL_DIR"
echo "Resume export log dir: $latest_run"
