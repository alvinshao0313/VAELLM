#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# 说明：
# - 这是单卡 mixed-dataset 蒸馏脚本，默认服务于通用推理/选择题类下游任务。
# - 当前混合池默认配比：
#   openorca=0.45,fineweb_edu=0.30,race=0.15,sciq=0.07,alpaca=0.03
# - 训练入口会对每个 source 分别做 load -> text -> tokenize -> pack -> resize，
#   然后再按概率 interleave，避免先混 raw text。
# - 以下配置是有意为之，不是漏配：
#   --teacher_model_path 不显式传，默认从 student checkpoint meta 推断
#   --save_strategy no，默认只落最终导出
#   --unload_vae_original_weights_on_save false，保留原始权重便于后续检查

torchrun --standalone --nproc_per_node=1 -m e2e_fintuning.main \
  --student_checkpoint_dir .result/meta-llama_Llama-2-7b-hf_20260323_071142/final_model \
  --run_root_dir .result/e2e_vae_lora_mix \
  --dataset_mix "openorca=0.45,fineweb_edu=0.30,race=0.15,sciq=0.07,alpaca=0.03" \
  --loss_type kd_top_1000 \
  --distill_temperature 1.0 \
  --distill_alpha 0.3 \
  --post_attn false \
  --model_max_length 4096 \
  --decoder_layers 0-31 \
  --target_modules all \
  --vae_lora_variant plain \
  --vae_lora_rank 8 \
  --vae_lora_alpha 16 \
  --vae_lora_dropout 0.0 \
  --vae_lora_tune_bias true \
  --vae_lora_init_mode zero \
  --tune_final_norm true \
  --use_post_norm_head_linear true \
  --lora_hif4_act false \
  --eval_hif4_act false \
  --prewarm_frozen_vae true \
  --prewarm_log_every 32 \
  --prewarm_group_size 8 \
  --skip_ppl_eval false \
  --ppl_seqlen 2048 \
  --ppl_limit -1 \
  --save_tokenizer true \
  --unload_vae_original_weights_on_save false \
  --bf16 true \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --logging_steps 10 \
  --save_strategy no \
  --max_steps 4000 \
  "$@"
