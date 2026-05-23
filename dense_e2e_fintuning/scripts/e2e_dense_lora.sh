#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
SEED="${SEED:-0}"
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
MAX_STEPS="${MAX_STEPS:-1000}"
STUDENT_CKPT="${STUDENT_CKPT:-.result/Qwen_Qwen3-8B_20260422_070608/final_model}"

if [[ "${DISABLE_PROXY:-1}" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  unset all_proxy ALL_PROXY no_proxy NO_PROXY
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
fi

# 说明：
# - 压缩 checkpoint -> dense 重建 -> 标准 PEFT 训练 -> 回写压缩 checkpoint。
# - 不依赖 raw_e2e_fintuning。
# - 冒烟建议：MAX_STEPS=30 bash dense_e2e_fintuning/scripts/e2e_dense_lora.sh
#   --dataset_mix "openorca=0.33,fineweb_edu=0.30,race=0.20,sciq=0.06,alpaca=0.02,longalpaca=0.05,longalign=0.04" \

torchrun --standalone --nproc_per_node=4 -m dense_e2e_fintuning.main \
  --ddp_timeout 14400 \
  --seed "${SEED}" \
  --data_seed "${SEED}" \
  --full_determinism true \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --student_checkpoint_dir "${STUDENT_CKPT}" \
  --run_root_dir .result/dense_e2e_fintuning \
  --dataset_mix "openorca=0.20,fineweb_edu=0.18,race=0.24,sciq=0.14,alpaca=0.04,longalpaca=0.10,longalign=0.10" \
  --dataset_num_proc 64 \
  --loss_type kd_top_1000 \
  --distill_temperature 1.0 \
  --distill_alpha 0.5 \
  --post_attn false \
  --model_max_length 8192 \
  --decoder_layers 0-35 \
  --target_modules all \
  --decode_device "auto" \
  --decode_group_size 16 \
  --lora_variant dora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_tune_bias false \
  --lora_init_mode zero \
  --tune_final_norm true \
  --use_post_norm_head_linear true \
  --lora_hif4_act false \
  --eval_hif4_act false \
  --skip_ppl_eval false \
  --ppl_seqlen 2048 \
  --ppl_limit -1 \
  --save_tokenizer true \
  --bf16 true \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --logging_steps 10 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 10 \
  --max_steps "${MAX_STEPS}" \
  "$@"
