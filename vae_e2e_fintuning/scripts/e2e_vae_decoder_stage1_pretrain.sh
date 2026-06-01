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

CONDA_BIN="${CONDA_BIN:-/home/shaoyuantian/anaconda3/bin/conda}"
MAX_STEPS="${MAX_STEPS:-500}"
STUDENT_CKPT="${STUDENT_CKPT:-.result/final_model}"
PRETRAIN_DATASET_MIX="${PRETRAIN_DATASET_MIX:-fineweb_edu=1.0}"
EVAL_TASKS="${EVAL_TASKS:-boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
EVAL_PREWARM_GROUP_SIZE="${EVAL_PREWARM_GROUP_SIZE:-8}"
OUTLIER_RESIDUAL_TOP_P="${OUTLIER_RESIDUAL_TOP_P:-0.01}"
OUTLIER_RESIDUAL_MIN_ABS="${OUTLIER_RESIDUAL_MIN_ABS:-0.0}"
OUTLIER_RESIDUAL_CODEC="${OUTLIER_RESIDUAL_CODEC:-blocked_quantized}"
OUTLIER_RESIDUAL_INDEX_BITS="${OUTLIER_RESIDUAL_INDEX_BITS:-8}"
OUTLIER_RESIDUAL_VALUE_BITS="${OUTLIER_RESIDUAL_VALUE_BITS:-8}"
OUTLIER_RESIDUAL_BLOCK_SHAPE="${OUTLIER_RESIDUAL_BLOCK_SHAPE:-256,256}"

if [[ "${DISABLE_PROXY:-1}" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  unset all_proxy ALL_PROXY no_proxy NO_PROXY
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
fi

"${CONDA_BIN}" run -n bitvae python -m vae_e2e_fintuning.main \
  --seed "${SEED}" \
  --data_seed "${SEED}" \
  --full_determinism true \
  --gradient_checkpointing false \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --student_checkpoint_dir "${STUDENT_CKPT}" \
  --run_root_dir .result/vae_e2e_fintuning_stage1_pretrain \
  --dataset_task lm \
  --dataset_mix "${PRETRAIN_DATASET_MIX}" \
  --dataset_num_proc "${DATASET_NUM_PROC:-64}" \
  --loss_type kd_top_1000 \
  --distill_temperature 1.0 \
  --distill_alpha 0.5 \
  --post_attn false \
  --model_max_length "${MODEL_MAX_LENGTH:-8192}" \
  --decoder_layers "${DECODER_LAYERS:-0-35}" \
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
  --eval_prewarm_group_size "${EVAL_PREWARM_GROUP_SIZE}" \
  --skip_ppl_eval false \
  --ppl_seqlen 2048 \
  --ppl_limit -1 \
  --refresh_sparse_residual_after_train true \
  --refresh_sparse_residual_top_p "${OUTLIER_RESIDUAL_TOP_P}" \
  --refresh_sparse_residual_score abs \
  --refresh_sparse_residual_min_abs "${OUTLIER_RESIDUAL_MIN_ABS}" \
  --refresh_sparse_residual_codec "${OUTLIER_RESIDUAL_CODEC}" \
  --refresh_sparse_residual_index_bits "${OUTLIER_RESIDUAL_INDEX_BITS}" \
  --refresh_sparse_residual_value_bits "${OUTLIER_RESIDUAL_VALUE_BITS}" \
  --refresh_sparse_residual_block_shape "${OUTLIER_RESIDUAL_BLOCK_SHAPE}" \
  --save_tokenizer true \
  --bf16 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --logging_steps 10 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS:-500}" \
  --save_total_limit 10 \
  --max_steps "${MAX_STEPS}" \
  "$@"
