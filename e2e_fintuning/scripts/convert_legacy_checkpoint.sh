#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

OLD_CKPT="${OLD_CKPT:-.result/e2e_vae_lora_redpajama/final_model_20260330_085026/trainer_state/checkpoint-40000}"
STUDENT_CKPT="${STUDENT_CKPT:-.result/meta-llama_Llama-2-7b-hf_20260323_071142/final_model}"
OUT_CKPT="${OUT_CKPT:-${OLD_CKPT}/final_model}"

extra_args=()
if [[ -n "${ACCESS_TOKEN:-}" ]]; then
  extra_args+=(--access_token "$ACCESS_TOKEN")
fi

eval "$(conda shell.bash hook)"
conda activate bitvae

python -m e2e_fintuning.convert_legacy_checkpoint \
  --legacy_checkpoint_dir "$OLD_CKPT" \
  --student_checkpoint_dir "$STUDENT_CKPT" \
  --output_checkpoint_dir "$OUT_CKPT" \
  --decoder_layers 0-31 \
  --target_modules all \
  --vae_lora_rank 8 \
  --vae_lora_alpha 16 \
  --vae_lora_dropout 0.0 \
  --lora_embedding false \
  --lora_lm_head false \
  --loss_type sft \
  --post_attn false \
  --lora_hif4_act false \
  "${extra_args[@]}" \
  "$@"
