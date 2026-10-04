#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

# Caller activates bitvae and selects four training GPUs; GPU 7 is reserved for export/evaluation.
STUDENT_CKPT="${STUDENT_CKPT:-/home/star/VAELLM/result/qwen3_8b_w25_20260930/reconstruction_block32_o3_down3/home_star_models_Qwen_Qwen3-8B_20261001_112401/final_model}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/home/star/VAELLM/result/qwen3_8b_w25_20260930/e2e_bits_block32_a100_mp4_r128_lr4e5}"

python -m experiments.qwen3_w25.train_with_step_exports \
  --export_cuda_visible_devices 7 \
  --student_checkpoint_dir "$STUDENT_CKPT" \
  --teacher_model_path /home/star/models/Qwen/Qwen3-8B \
  --run_root_dir "$RUN_ROOT_DIR" \
  --train_mode decoder_sparse_bit --distill_fp32_components decoder \
  --bit_proxy_coordinates decoder_sensitivity --bit_active_ratio 1 \
  --bit_optimizer adam --bit_lr 4e-5 --bit_round_steps 128 --bit_weight_decay 0 \
  --seed 0 --data_seed 0 \
  --dataset_mix 'edgerazor_ii_7m=0.341,edgerazor_ii_gen=0.067,edgerazor_tulu=0.028,edgerazor_am=0.064,vaellm_eval_task=0.500' \
  --dataset_task sft --dynamic_padding true --model_max_length 1024 --group_by_length true \
  --target_layers 0-35 --target_modules all \
  --loss_type kl_top_mass --top_k 100 --temperature 1 \
  --prompt_loss_weight 0.3 --hidden_loss_weight 0 --pre_mlp_hidden_loss_weight 0 \
  --selective_student_topk false \
  --steps 2000 --batch_size 4 --gradient_accumulation_steps 1 \
  --decoder_lr 1e-5 --learning_rate 1e-5 \
  --lr_scheduler_type cosine --warmup_ratio 0.01 --weight_decay 0 \
  --max_grad_norm 1.5 --logging_steps 10 \
  --gradient_checkpointing true --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --residual_lora_mode none --norm_train_mode none --lm_head_train_mode none \
  --parallel_mode layer_mp --layer_device_map auto \
  --teacher_output_offload cpu --teacher_model_offload none \
  --teacher_output_pin_memory true --teacher_output_chunk_tokens 8 \
  --vae_decoder_checkpoint true --offload_mode none \
  --eval_after_save false --eval_hif4_act false --eval_tasks '' \
  --eval_num_fewshot 0 --eval_batch_size auto --eval_device cuda --eval_prewarm_group_size 8 \
  --skip_ppl_eval true --save_tokenizer true --full_determinism false --bf16 true \
  --eval_strategy no --save_strategy steps --save_steps 400 --save_total_limit 1 \
  "$@"
