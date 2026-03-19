#!/usr/bin/env bash
set -euo pipefail

# 说明：
# - 默认开启 LoRA 微调（每个类别 VAE 完成后，对后续类别做 LoRA 并融合）。
# - 下面已尽量对齐你给的 LoRA 超参数。
# - 当前 tools/cat_train.py 未实现以下参数：gradient_accumulation_steps、
#   paged_adamw_8bit、warmup_ratio、group_by_length、dataset_text_field。
# - 可用 LORA_SCHEDULE 传入按类别覆盖的 JSON，例如：
#   {"q_proj":{"rank":8,"alpha":16,"steps":1000,"loss_type":"sft","use_dora":false},
#    "k_proj":{"rank":128,"alpha":256,"steps":2000,"loss_type":"r_kl_top_1000","use_dora":true}}
# - CODEBOOK_BITS / CODEBOOK_DIM 支持整数或按类别 JSON，例如：
#   CODEBOOK_BITS='{"default":32,"q_proj":24}'
#   CODEBOOK_DIM='{"default":16,"q_proj":8,"down_proj":32}'
# - 现在支持按 residual stage 传 JSON list（字符串类型选项需双引号）：
#   RESIDUAL_STAGES=2
#   CODEBOOK_BITS='[16,12]'
#   CODEBOOK_DIM='[16,8]'
#   BASE_CH='[128,96]'
#   NUM_RES_BLOCKS='[1,2]'
#   NORM_TYPE='["layer","group"]'
#   DECODER_TYPE='["symmetric","asymmetric"]'
#   DECODER_BASE_CH='[128,96]'
#   DECODER_NUM_RES_BLOCKS='[1,2]'
#   RECON_LOSS_TYPE='["wa_mse","mse"]'
#   STEPS_PER_CATEGORY='[5000,3000]'
#   INTRA_PART_SORT_MODE='["l2","l2"]'
# - NEW_QUANT 控制是否传入 --new_quant（默认开启）。

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="${PYTHONPATH:-.}:." \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5} \
python tools/cat_train.py \
  --output_dir "${OUTPUT_DIR:-.result}" \
  --save_model \
  --steps_per_category "${STEPS_PER_CATEGORY:-5000}" \
  --category_order "${CATEGORY_ORDER:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}" \
  --transpose_modules "${TRANSPOSE_MODULES:-v_proj,o_proj,gate_proj,up_proj,down_proj}" \
  --batch_size "${BATCH_SIZE:-2048}" \
  --log_every "${LOG_EVERY:-50}" \
  --eval_every "${EVAL_EVERY:-1000}" \
  --eval_blocks "${EVAL_BLOCKS:-256}" \
  --ppl_limit "${PPL_LIMIT:--1}" \
  --skip_layers "${SKIP_LAYERS:-1.down_proj}" \
  --lora_after_category \
  --lora_rank "${LORA_RANK:-8}" \
  --lora_alpha "${LORA_ALPHA:-16.0}" \
  --lora_dropout "${LORA_DROPOUT:-0.0}" \
  --lora_steps "${LORA_STEPS:-2000}" \
  --lora_batch_size "${LORA_BATCH_SIZE:-2}" \
  --lora_nsamples "${LORA_NSAMPLES:-10000000}" \
  --lora_lr "${LORA_LR:-1e-4}" \
  --lora_weight_decay "${LORA_WEIGHT_DECAY:-0.001}" \
  --lora_log_every "${LORA_LOG_EVERY:-2}" \
  --lora_loss_type "${LORA_LOSS_TYPE:-sft}" \
  --lora_use_dora "${LORA_USE_DORA:-false}" \
  --lora_tune_bias "${LORA_TUNE_BIAS:-false}" \
  --lora_tune_protected_outliers "${LORA_TUNE_PROTECTED_OUTLIERS:-false}" \
  --lora_bias_categories "${LORA_BIAS_CATEGORIES:-}" \
  --lora_schedule "${LORA_SCHEDULE:-}" \
  --train_device "${TRAIN_DEVICE:-cuda}" \
  --convert \
  --convert_device "${CONVERT_DEVICE:-cuda}" \
  --linear_group_size "${LINEAR_GROUP_SIZE:-32}" \
  --intra_parallel "${INTRA_PARALLEL:-1}" \
  --intra_part_sort_mode "${INTRA_PART_SORT_MODE:-none,none}" \
  --codebook_bits "${CODEBOOK_BITS:-32}" \
  --codebook_dim "${CODEBOOK_DIM:-32}" \
  --residual_stages "${RESIDUAL_STAGES:-2}" \
  --base_ch "${BASE_CH:-128}" \
  --num_res_blocks "${NUM_RES_BLOCKS:-1}" \
  --norm_type "${NORM_TYPE:-layer}" \
  --decoder_type "${DECODER_TYPE:-symmetric}" \
  --decoder_base_ch "${DECODER_BASE_CH:-128}" \
  --decoder_num_res_blocks "${DECODER_NUM_RES_BLOCKS:-1}" \
  --quantizer_type "${QUANTIZER_TYPE:-BSQ}" \
  --gamma0 "${GAMMA0:-1.0}" \
  --gamma "${GAMMA:-1.0}" \
  --zeta "${ZETA:-1.0}" \
  --inv_temperature "${INV_TEMPERATURE:-200.0}" \
  --normalize_weight \
  --use_checkpoint \
  --recon_loss_type "${RECON_LOSS_TYPE:-wa_mse}" \
  --outlier_protect_ratio "${OUTLIER_PROTECT_RATIO:-0.01}" \
  --outlier_protect_axis "${OUTLIER_PROTECT_AXIS:-input}" \
  --wa_mse_act_mode "${WA_MSE_ACT_MODE:-dynamic}" \
  --wa_mse_calib_dataset "${WA_MSE_CALIB_DATASET:-wikitext2}" \
  --wa_mse_calib_nsamples "${WA_MSE_CALIB_NSAMPLES:-512}" \
  --wa_mse_calib_seqlen "${WA_MSE_CALIB_SEQLEN:-512}" \
  --wa_mse_calib_seed "${WA_MSE_CALIB_SEED:-0}" \
  --wa_mse_calib_device "${WA_MSE_CALIB_DEVICE:-}" \
  --wa_mse_calib_log_every "${WA_MSE_CALIB_LOG_EVERY:-0}" \
  --l1_weight "${L1_WEIGHT:-1.0}" \
  --lfq_weight "${LFQ_WEIGHT:-5.0}" \
  --commitment_loss_weight "${COMMITMENT_LOSS_WEIGHT:-1e-1}" \
  --entropy_loss_weight "${ENTROPY_LOSS_WEIGHT:-1e-4}" \
  --diversity_gamma "${DIVERSITY_GAMMA:-1.0}" \
  --optimizer "${OPTIMIZER:-adamw}" \
  --new_quant \
  --lr "${LR:-1e-2}" \
  --beta1 "${BETA1:-0.9}" \
  --beta2 "${BETA2:-0.95}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --lr_scheduler "${LR_SCHEDULER:-linear}" \
  --lr_warmup_steps "${LR_WARMUP_STEPS:-0}" \
  --max_grad_norm "${MAX_GRAD_NORM:-0.3}" \
  --model_max_length "${MAX_SEQ_LENGTH:-2048}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
  --fp16 "${FP16:-False}" \
  --bf16 "${BF16:-True}" \
  "$@"
