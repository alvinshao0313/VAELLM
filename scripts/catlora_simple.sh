#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=7
SEED="${SEED:-42}"
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export CAT_LORA_DATASET_NUM_PROC="${CAT_LORA_DATASET_NUM_PROC:-16}"

# 可按需补充的可选参数：
# --access_token "hf_xxx"
# --resume_from_checkpoint "/path/to/last_run/final_model"
# --include_all_linears
# --rot_llm
# --unload_vae_original_weights_on_final_save
# --allow_tail_group "true"
# --intra_part_sort_mode "default=none" / "default=spectral_cosine" / "default=act_spectral_cosine"
#   现在只支持单值模式，不再支持 row:...|col:...；排序只会发生在每个 part 内的列轴
# --sort_prep_workers "0"
#   排序预处理并行 worker 数：0=auto，1=串行，>1=显式 CPU 多进程；只影响 spectral_cosine / act_spectral_cosine
# --outlier_residual_top_p "default=0.01,cat:down_proj=0.02"
# --outlier_residual_score "abs" / "input_act_weighted_abs" / "original_weight_abs" / "input_act_weighted_original_weight_abs"
# --outlier_residual_min_abs "1e-6"
#   原始权重打分只决定保留哪些位置，真正保存的仍是这些位置上的 residual
#   若 |original-reconstructed| 小于该阈值，则该位置会从 top-p 中剔除，并继续往后补
# --outlier_residual_codec "blocked_quantized" or "coo_fp16"
# --outlier_residual_index_bits "8"   # 8 or 4 慎用 4 bits，可能导致结果不稳定
# --outlier_residual_value_bits "8"   # 8 or 4 
# --wa_mse_calib_dataset "openorca=1.0"  # 使用 dense_e2e dataset_mix alias，格式 alias=weight,...
# CAT_LORA_DATASET_NUM_PROC=16          # LoRA/校准数据 format 预处理并行进程数 位置在 lora_data.py
# --eval_ppl "true"                   # 是否跑类别后 PPL；默认 true
# --eval_tasks "boolq,rte,piqa"       # 可选：类别后下游任务评估；空串表示不跑
# --lora_after_category \ boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa

python tools/cat_train.py \
  --model_path "Qwen/Qwen3-8B" \
  --output_dir ".result" \
  --seed "${SEED}" \
  --deterministic "true" \
  --train_device "cuda" \
  --convert \
  --save_model \
  --convert_device "cuda" \
  --unload_vae_original_weights_on_final_save \
  --allow_tail_group "true" \
  --category_order "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --transpose_modules "q_proj,v_proj,o_proj,down_proj" \
  --projection_suffixes "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --skip_layers "" \
  --linear_group_size "36" \
  --steps_per_category "default=1000" \
  --joint_decoder_steps "default=1000" \
  --joint_decoder_lr "default=5e-3" \
  --joint_decoder_group_size "default=36" \
  --joint_decoder_batch_size "default=524288" \
  --batch_size "2048" \
  --log_every "100" \
  --eval_every "0" \
  --eval_blocks "256" \
  --eval_ppl "false" \
  --eval_tasks "" \
  --ppl_limit "-1" \
  --intra_parallel "default=1x1" \
  --intra_part_sort_mode "default=none" \
  --sort_prep_workers "0" \
  --outlier_protect_count "default=0" \
  --outlier_protect_axis "input" \
  --outlier_protect_mode "none" \
  --outlier_residual_top_p "default=0.01" \
  --outlier_residual_score "input_act_weighted_abs" \
  --outlier_residual_min_abs "0.0" \
  --outlier_residual_codec "blocked_quantized" \
  --outlier_residual_index_bits "8" \
  --outlier_residual_value_bits "8" \
  --wa_mse_calib_dataset "wiki=1.0" \
  --wa_mse_calib_nsamples "512" \
  --wa_mse_calib_seqlen "4096" \
  --wa_mse_calib_seed "${SEED}" \
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
  --recon_loss_type "default=mse" \
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
  --lora_dataset "openorca=0.20,fineweb_edu=0.18,race=0.24,sciq=0.14,alpaca=0.04,longalpaca=0.10,longalign=0.10" \
  --lora_rank "default=24" \
  --lora_alpha "default=48.0" \
  --lora_dropout "default=0.0" \
  --lora_steps "default=5000" \
  --lora_batch_size "default=1" \
  --lora_nsamples "default=20000" \
  --lora_lr "default=1e-4" \
  --lora_weight_decay "default=0.001" \
  --lora_log_every "default=2" \
  --lora_post_attn "false" \
  --lora_temperature "default=1.0" \
  --lora_loss_alpha "default=0.5" \
  --lora_loss_type "default=kd_top_1000" \
  --lora_use_dora "default=true" \
  --lora_tune_final_norm "true" \
  --lora_use_post_norm_head_linear "true" \
  --lora_hif4_act "false" \
  --eval_hif4_act "false" \
  --lora_gradient_accumulation_steps "1" \
  --lora_gradient_checkpointing "true" \
  --lora_gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --lora_optim "adamw_torch" \
  --lora_max_grad_norm "0.333" \
  --lora_warmup_ratio "0.3" \
  --lora_group_by_length "true" \
  --lora_lr_scheduler_type "cosine" \
  --lora_model_max_length "8192" \
  --fp16 "false" \
  --bf16 "true" \
  "$@"
