#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SEED="${SEED:-0}"
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

MAX_STEPS="${MAX_STEPS:-1000}"
STUDENT_CKPT="${STUDENT_CKPT:-.result/final_model}"
EVAL_TASKS="${EVAL_TASKS:-boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"

if [[ "${DISABLE_PROXY:-1}" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  unset all_proxy ALL_PROXY no_proxy NO_PROXY
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
fi

# 说明：
# - packed 压缩 checkpoint -> 直接端到端微调 VAELinear 多阶 decoder -> 保存新的 packed final_model。
# - 训练 VAELinear decoder，并默认额外训练 final norm / post-norm head linear。
# - vq_weight 和原模型其它参数保持冻结；VAELinear bias 默认不训练。
# - 这里是单进程多卡模型并行，不使用 torchrun。多卡切分由 --layer_device_map 控制。
# - 默认开启 streaming offload：Transformer layers 常驻 CPU，按需预取到 GPU，并卸载大 saved tensors。
# - auto 会按当前 CUDA_VISIBLE_DEVICES 内可见 GPU 均分 Transformer layers。
# - 训练保存 final_model 后会跑 lm-eval：${EVAL_TASKS}
# - 冒烟建议：
#   MAX_STEPS=30 bash vae_e2e_fintuning/scripts/e2e_vae_decoder.sh --skip_ppl_eval true --eval_tasks ""

python -m vae_e2e_fintuning.main \
  --seed "${SEED}" \
  --data_seed "${SEED}" \
  --full_determinism true \
  --gradient_checkpointing false \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --student_checkpoint_dir "${STUDENT_CKPT}" \
  --run_root_dir .result/vae_e2e_fintuning \
  --dataset_mix "openorca=0.20,fineweb_edu=0.18,race=0.24,sciq=0.14,alpaca=0.04,longalpaca=0.10,longalign=0.10" \
  --dataset_num_proc 64 \
  --loss_type kd_top_1000 \
  --distill_temperature 1.0 \
  --distill_alpha 0.5 \
  --post_attn false \
  --model_max_length 8192 \
  --decoder_layers 0-35 \
  --target_modules all \
  --layer_device_map auto \
  --parallel_stage_decode true \
  --tune_final_norm true \
  --use_post_norm_head_linear true \
  --vae_tune_bias false \
  --offload_mode none \
  --offload_checkpoint true \
  --offload_prefetch_distance 18 \
  --offload_min_tensor_bytes 16777216 \
  --offload_pin_memory true \
  --eval_hif4_act false \
  --eval_tasks "${EVAL_TASKS}" \
  --eval_num_fewshot 0 \
  --eval_lm_batch_size auto \
  --eval_device "${EVAL_DEVICE}" \
  --skip_ppl_eval false \
  --ppl_seqlen 2048 \
  --ppl_limit -1 \
  --save_tokenizer true \
  --bf16 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --logging_steps 10 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 10 \
  --max_steps "${MAX_STEPS}" \
  "$@"
