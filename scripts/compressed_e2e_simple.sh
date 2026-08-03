#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

SEED="${SEED:-42}"
export PYTHONHASHSEED="${SEED}"

# Required:
#   --student_checkpoint_dir ".result/your_cat_run/final_model"
#   --dataset_mix "wiki=1.0"  or  --train_file "/path/to/train.jsonl"
python -m compressed_e2e_fintuning.main \
  --run_root_dir ".result/compressed_e2e_fintuning" \
  --finetune_mode "decoder" \
  --seed "${SEED}" \
  --full_determinism "false" \
  --loss_type "sft" \
  --decoder_layers "all" \
  --target_modules "all" \
  --layer_device_map "auto" \
  --parallel_stage_decode "true" \
  --parallel_mode "layer_mp" \
  --offload_mode "streaming" \
  --offload_prefetch_distance "1" \
  --offload_min_tensor_bytes "1048576" \
  --offload_pin_memory "true" \
  --model_max_length "2048" \
  --per_device_train_batch_size "1" \
  --gradient_accumulation_steps "1" \
  --gradient_checkpointing "false" \
  --max_steps "1000" \
  --learning_rate "1e-5" \
  --weight_decay "0.0" \
  --warmup_ratio "0.03" \
  --lr_scheduler_type "cosine" \
  --logging_steps "1" \
  --save_steps "500" \
  --eval_strategy "no" \
  --save_total_limit "2" \
  --bf16 "true" \
  --fp16 "false" \
  --skip_ppl_eval "true" \
  "$@"
