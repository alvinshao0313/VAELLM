#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# 可按需补充的可选参数：
# --access_token "hf_xxx"
# --resume_from_checkpoint "/path/to/last_run/final_model"
# --include_all_linears
# --rot_llm
# --unload_vae_original_weights_on_final_save

conda run -n bitvae python tools/cat_train.py \
  --model_path "meta-llama/Llama-2-7b-hf" \
  --output_dir ".result" \
  --seed "0" \
  --train_device "cuda" \
  --convert \
  --convert_device "cuda" \
  --save_model \
  --allow_tail_group "true" \
  --category_order "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --transpose_modules "v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --projection_suffixes "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --skip_layers "1.down_proj" \
  --linear_group_size "32" \
  --steps_per_category "default=5000" \
  --steps_per_group "default=none" \
  --batch_size "2048" \
  --log_every "50" \
  --eval_every "1000" \
  --eval_blocks "256" \
  --ppl_limit "-1" \
  --intra_parallel "default=1x1" \
  --intra_part_sort_mode "default=none" \
  --outlier_protect_count "default=0" \
  --outlier_protect_axis "input" \
  --wa_mse_calib_dataset "wikitext2" \
  --wa_mse_calib_nsamples "512" \
  --wa_mse_calib_seqlen "512" \
  --wa_mse_calib_seed "0" \
  --wa_mse_calib_device "" \
  --wa_mse_calib_log_every "0" \
  --codebook_bits "default=32" \
  --codebook_dim "default=32" \
  --residual_stages "default=2" \
  --base_ch "default=128" \
  --num_res_blocks "default=1" \
  --decoder_base_ch "default=128" \
  --decoder_num_res_blocks "default=1" \
  --norm_type "default=layer" \
  --decoder_type "default=symmetric" \
  --recon_loss_type "default=wa_mse" \
  --quantizer_type "BSQ" \
  --gamma0 "1.0" \
  --gamma "1.0" \
  --zeta "1.0" \
  --inv_temperature "200.0" \
  --lr "1e-2" \
  --beta1 "0.9" \
  --beta2 "0.95" \
  --weight_decay "0.0" \
  --optimizer "adamw" \
  --lr_scheduler "linear" \
  --lr_warmup_steps "0" \
  --l1_weight "1.0" \
  --lfq_weight "5.0" \
  --commitment_loss_weight "0.1" \
  --entropy_loss_weight "1e-4" \
  --diversity_gamma "1.0" \
  --normalize_weight \
  --use_checkpoint \
  --new_quant \
  --lora_after_category \
  --lora_rank "default=8" \
  --lora_alpha "default=16.0" \
  --lora_dropout "default=0.0" \
  --lora_steps "default=2000" \
  --lora_batch_size "default=2" \
  --lora_nsamples "default=10000000" \
  --lora_lr "default=1e-4" \
  --lora_weight_decay "default=0.001" \
  --lora_log_every "default=2" \
  --lora_loss_type "default=sft" \
  --lora_use_dora "default=false" \
  --lora_gradient_accumulation_steps "1" \
  --lora_optim "adamw_torch" \
  --lora_max_grad_norm "0.3" \
  --lora_warmup_ratio "0.3" \
  --lora_group_by_length "true" \
  --lora_lr_scheduler_type "linear" \
  --lora_model_max_length "2048" \
  --fp16 "false" \
  --bf16 "true" \
  "$@"
