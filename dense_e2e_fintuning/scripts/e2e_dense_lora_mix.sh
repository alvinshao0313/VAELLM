#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MAX_STEPS="${MAX_STEPS:-20000}"
STUDENT_CKPT="${STUDENT_CKPT:-.result/meta-llama_Llama-2-7b-hf_20260323_071142/final_model}"
DECODE_DEVICE="${DECODE_DEVICE:-auto}"
DECODE_GROUP_SIZE="${DECODE_GROUP_SIZE:-8}"

if [[ "${DISABLE_PROXY:-0}" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  unset all_proxy ALL_PROXY no_proxy NO_PROXY
fi

# 说明：
# - 压缩 checkpoint -> dense 重建 -> 标准 PEFT 训练 -> 回写压缩 checkpoint。
# - 不依赖 raw_e2e_fintuning。
# - 冒烟建议：MAX_STEPS=30 bash dense_e2e_fintuning/scripts/e2e_dense_lora_mix.sh

torchrun --standalone --nproc_per_node=2 -m dense_e2e_fintuning.main \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --student_checkpoint_dir "${STUDENT_CKPT}" \
  --run_root_dir .result/dense_e2e_fintuning_mix \
  --dataset_mix "openorca=0.33,fineweb_edu=0.30,race=0.20,sciq=0.06,alpaca=0.02,longalpaca=0.05,longalign=0.04" \
  --loss_type sft \
  --distill_temperature 1.0 \
  --distill_alpha 0.3 \
  --post_attn false \
  --model_max_length 8192 \
  --decoder_layers 0-31 \
  --target_modules all \
  --decode_device "${DECODE_DEVICE}" \
  --decode_group_size "${DECODE_GROUP_SIZE}" \
  --lora_variant dora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_tune_bias false \
  --lora_init_mode zero \
  --tune_final_norm true \
  --use_post_norm_head_linear false \
  --lora_hif4_act false \
  --eval_hif4_act false \
  --skip_ppl_eval false \
  --ppl_seqlen 2048 \
  --ppl_limit -1 \
  --save_tokenizer true \
  --bf16 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 6e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --logging_steps 10 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 1 \
  --max_steps "${MAX_STEPS}" \
  "$@"
