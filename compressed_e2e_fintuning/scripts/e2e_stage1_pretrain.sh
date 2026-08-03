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

MAX_STEPS="${MAX_STEPS:-500}"
STUDENT_CKPT="${STUDENT_CKPT:-.result/final_model}"
EVAL_TASKS="${EVAL_TASKS:-boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
EVAL_PREWARM_GROUP_SIZE="${EVAL_PREWARM_GROUP_SIZE:-8}"

if [[ "${DISABLE_PROXY:-1}" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  unset all_proxy ALL_PROXY no_proxy NO_PROXY
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
fi

python -m compressed_e2e_fintuning.main \
  --seed "${SEED}" \
  --data_seed "${SEED}" \
  --full_determinism true \
  --gradient_checkpointing false \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --student_checkpoint_dir "${STUDENT_CKPT}" \
  --run_root_dir .result/compressed_e2e_fintuning_stage1_pretrain \
  --finetune_mode decoder \
  --dataset_task sft \
  --dataset_mix "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009" \
  --parallel_mode layer_mp \
  --loss_type eakld_kd \
  --eakld_confidence_k 16 \
  --distill_temperature 1.0 \
  --distill_alpha 0.5 \
  --post_attn false \
  --model_max_length "8192" \
  --decoder_layers "${DECODER_LAYERS:-0-35}" \
  --target_modules all \
  --layer_device_map auto \
  --parallel_stage_decode true \
  --tune_final_norm true \
  --use_post_norm_head_linear true \
  --vae_tune_bias false \
  --offload_mode none \
  --eval_hif4_act false \
  --eval_tasks "${EVAL_TASKS}" \
  --eval_num_fewshot 0 \
  --eval_lm_batch_size auto \
  --eval_device "${EVAL_DEVICE}" \
  --eval_prewarm_group_size "${EVAL_PREWARM_GROUP_SIZE}" \
  --skip_ppl_eval false \
  --ppl_seqlen 2048 \
  --ppl_limit -1 \
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
