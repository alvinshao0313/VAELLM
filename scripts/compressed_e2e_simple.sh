#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

export PYTHONHASHSEED=42

# Required:
#   --student_checkpoint_dir ".result/your_cat_run/final_model"
#   --dataset_mix "wiki=1.0"  or  --train_file "/path/to/train.jsonl"
# 参数顺序：
# [Data]
# [Distill Loss]
# [VAE Compression]（本脚本无新增 VAE compression）
# [Channel Protection]（本脚本无 channel protection）
# [Distill Optimization]
# [Runtime / Distributed]
# [Evaluation]
python -m compressed_e2e_fintuning.main \
  --run_root_dir ".result/compressed_e2e_fintuning" \
  --train_mode decoder \
  --seed 42 \
  --data_seed 42 \
  --model_max_length 2048 \
  --dynamic_padding true \
  --group_by_length true \
  --target_layers all \
  --target_modules all \
  --loss_type sft \
  --steps 1000 \
  --batch_size 1 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing false \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --parallel_mode layer_mp \
  --layer_device_map auto \
  --offload_mode streaming \
  --offload_prefetch_distance 1 \
  --offload_min_tensor_bytes 1048576 \
  --offload_pin_memory true \
  --skip_ppl_eval true \
  --full_determinism false \
  --bf16 true \
  --fp16 false \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 2 \
  "$@"
