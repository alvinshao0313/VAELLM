#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

STUDENT_CKPT="${STUDENT_CKPT:-.result/cat_train_final_model}"

if [[ "${DISABLE_PROXY:-1}" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  unset all_proxy ALL_PROXY no_proxy NO_PROXY
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
fi

python -m compressed_e2e_fintuning.main \
  --seed 0 \
  --data_seed 0 \
  --full_determinism false \
  --gradient_checkpointing false \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --student_checkpoint_dir "${STUDENT_CKPT}" \
  --run_root_dir .result/compressed_e2e_fintuning_choice_kd \
  --finetune_mode decoder \
  --dataset_task mcqa \
  --dataset_mix "mmlu=0.5,race=0.2,sciq=0.15,arc=0.1,openbookqa=0.05" \
  --parallel_mode layer_mp \
  --loss_type choice_kd_ce \
  --distill_temperature 1.0 \
  --distill_alpha 0.8 \
  --hidden_loss_weight 0.0 \
  --hidden_layer_weighting uniform \
  --model_max_length 2048 \
  --decoder_layers 0-35 \
  --target_modules all \
  --layer_device_map auto \
  --parallel_stage_decode true \
  --vae_decoder_checkpoint true \
  --tune_final_norm true \
  --use_post_norm_head_linear true \
  --vae_tune_bias false \
  --offload_mode none \
  --eval_hif4_act false \
  --eval_tasks "boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu" \
  --eval_num_fewshot 0 \
  --eval_lm_batch_size auto \
  --eval_device cuda \
  --eval_prewarm_group_size 8 \
  --skip_ppl_eval false \
  --ppl_seqlen 2048 \
  --ppl_limit -1 \
  --save_tokenizer true \
  --bf16 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-6 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --max_grad_norm 1.5 \
  --logging_steps 10 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 2 \
  --max_steps 500 \
  "$@"
