#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SEED="${SEED:-0}"
export PYTHONHASHSEED="${SEED}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

FULL_DETERMINISM="${FULL_DETERMINISM:-false}"
MAX_STEPS="${MAX_STEPS:-20000}"
STUDENT_CKPT="${STUDENT_CKPT:-.result/catlora_distill/res0-bf16-protect-channel-vae/cat_distill_best/final_model}"
# 用 ${VAR-default}：仅在未设置时填默认；允许 EVAL_TASKS="" 关闭最终 lm-eval。
EVAL_TASKS="${EVAL_TASKS-boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
EVAL_PREWARM_GROUP_SIZE="${EVAL_PREWARM_GROUP_SIZE:-8}"
PARALLEL_MODE="${PARALLEL_MODE:-layer_mp}"   # layer_mp | dp
NPROC="${NPROC:-4}"
EVAL_AFTER_SAVE="${EVAL_AFTER_SAVE:-false}"
# 分卡 lm-eval（尤其 mmlu）gather 等待；对齐 catlora_distill_4gpu_res0.sh
export DISTILL_NCCL_TIMEOUT_SEC="${DISTILL_NCCL_TIMEOUT_SEC:-10800}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}"

if [[ "${FULL_DETERMINISM}" == "true" ]]; then
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
fi

if [[ "${DISABLE_PROXY:-1}" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  unset all_proxy ALL_PROXY no_proxy NO_PROXY
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
fi

# 说明：
# - packed 压缩 checkpoint -> 直接端到端微调 VAELinear 多阶 decoder -> 保存新的 packed final_model。
# - 训练超参以 DP 分支为准；layer_mp 与 DP 同步（仅启动方式 / parallel_mode / layer_device_map 不同）。
# - 默认 --finetune_mode both；不训练 final norm / post-norm head linear；VAELinear bias 不训练。
# - PARALLEL_MODE=layer_mp：单进程层级模型并行，不用 torchrun；多卡切分由 --layer_device_map 控制。
# - PARALLEL_MODE=dp：torchrun 数据并行；每卡一份完整 student + teacher，忽略 --layer_device_map。
# - 当前默认 --offload_mode none（层常驻 GPU）。若改 streaming，只能配合 layer_mp。
#   --offload_mode 控制 student activation/layer offload，与 teacher-output offload 无关。
# - teacher 权重常驻 GPU，不做权重 offload。
# - CPU output 模式先 teacher forward，再把 logits 和必要 hidden targets 放到 CPU。
# - EAKLD 每次只回传 8 个 token 的完整词表 teacher logits。
# - adaptive_top_3 仅捕获 3 个 student hidden states。
# - auto（layer_mp）会按当前 CUDA_VISIBLE_DEVICES 内可见 GPU 均分 Transformer layers。
# - 训练保存 final_model 后会跑 lm-eval：${EVAL_TASKS}
# - 冒烟建议：
#   MAX_STEPS=30 bash compressed_e2e_fintuning/scripts/e2e_decoder.sh --skip_ppl_eval true --eval_tasks ""
# - 关闭 VAE decoder activation checkpoint（用显存换速度）：
#   bash compressed_e2e_fintuning/scripts/e2e_decoder.sh --vae_decoder_checkpoint false
# - DP 示例：
#   PARALLEL_MODE=dp NPROC=4 bash compressed_e2e_fintuning/scripts/e2e_decoder.sh
# - 保存中间 ckpt 后再分卡 lm-eval：
#   EVAL_AFTER_SAVE=true PARALLEL_MODE=dp NPROC=4 bash compressed_e2e_fintuning/scripts/e2e_decoder.sh --save_steps 1000

if [[ "${PARALLEL_MODE}" == "dp" ]]; then
  torchrun --standalone --nproc_per_node="${NPROC}" -m compressed_e2e_fintuning.main \
    --seed "${SEED}" \
    --data_seed "${SEED}" \
    --full_determinism "${FULL_DETERMINISM}" \
    --gradient_checkpointing true \
    --student_checkpoint_dir "${STUDENT_CKPT}" \
    --run_root_dir .result/compressed_e2e_fintuning \
    --finetune_mode both \
    --parallel_mode dp \
    --dataset_mix "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009" \
    --dataset_task sft \
    --loss_type eakld \
    --eakld_confidence_k 16 \
    --distill_temperature 1.0 \
    --distill_alpha 0.5 \
    --prompt_kd_weight 0.0 \
    --hidden_loss_weight 0.1 \
    --hidden_layer_weighting adaptive_top_3 \
    --teacher_output_offload cpu \
    --teacher_output_pin_memory true \
    --teacher_output_chunk_tokens 8 \
    --model_max_length "1024" \
    --decoder_layers 0-35 \
    --target_modules all \
    --parallel_stage_decode true \
    --vae_decoder_checkpoint true \
    --tune_final_norm false \
    --use_post_norm_head_linear false \
    --vae_tune_bias false \
    --offload_mode none \
    --eval_after_save "${EVAL_AFTER_SAVE}" \
    --eval_hif4_act false \
    --eval_tasks "${EVAL_TASKS}" \
    --eval_num_fewshot 0 \
    --eval_lm_batch_size auto \
    --eval_device "${EVAL_DEVICE}" \
    --eval_prewarm_group_size "${EVAL_PREWARM_GROUP_SIZE}" \
    --skip_ppl_eval true \
    --ppl_seqlen 2048 \
    --ppl_limit -1 \
    --save_tokenizer true \
    --bf16 true \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.0001 \
    --max_grad_norm 1.5 \
    --logging_steps 10 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 2 \
    --max_steps "${MAX_STEPS}" \
    "$@"
elif [[ "${PARALLEL_MODE}" == "layer_mp" ]]; then
  python -m compressed_e2e_fintuning.main \
    --seed "${SEED}" \
    --data_seed "${SEED}" \
    --full_determinism "${FULL_DETERMINISM}" \
    --gradient_checkpointing true \
    --student_checkpoint_dir "${STUDENT_CKPT}" \
    --run_root_dir .result/compressed_e2e_fintuning \
    --finetune_mode both \
    --parallel_mode layer_mp \
    --dataset_mix "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009" \
    --dataset_task sft \
    --loss_type eakld_top_100 \
    --eakld_confidence_k 16 \
    --distill_temperature 1.0 \
    --distill_alpha 0.5 \
    --prompt_kd_weight 0.0 \
    --hidden_loss_weight 0.01 \
    --hidden_layer_weighting adaptive_top_3 \
    --teacher_output_offload cpu \
    --teacher_output_pin_memory true \
    --teacher_output_chunk_tokens 8 \
    --model_max_length "1024" \
    --decoder_layers 0-35 \
    --target_modules all \
    --layer_device_map auto \
    --parallel_stage_decode true \
    --vae_decoder_checkpoint true \
    --tune_final_norm false \
    --use_post_norm_head_linear false \
    --vae_tune_bias false \
    --offload_mode none \
    --eval_after_save "${EVAL_AFTER_SAVE}" \
    --eval_hif4_act false \
    --eval_tasks "${EVAL_TASKS}" \
    --eval_num_fewshot 0 \
    --eval_lm_batch_size auto \
    --eval_device "${EVAL_DEVICE}" \
    --eval_prewarm_group_size "${EVAL_PREWARM_GROUP_SIZE}" \
    --skip_ppl_eval true \
    --ppl_seqlen 2048 \
    --ppl_limit -1 \
    --save_tokenizer true \
    --bf16 true \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.0001 \
    --max_grad_norm 1.5 \
    --logging_steps 10 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 2 \
    --max_steps "${MAX_STEPS}" \
    "$@"
else
  echo "Unsupported PARALLEL_MODE=${PARALLEL_MODE}. Expected layer_mp or dp." >&2
  exit 1
fi
