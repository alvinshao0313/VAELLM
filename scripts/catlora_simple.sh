#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

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
# --wa_mse_calib_dataset "wikitext2"  # 支持 wiki/wikitext2/fineweb_edu/openorca/redpajama/alpaca
# --eval_ppl "true"                   # 是否跑类别后 PPL；默认 true
# --eval_tasks "boolq,rte,piqa"       # 可选：类别后下游任务评估；空串表示不跑
# --lora_after_category \ boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa

python tools/cat_train.py \
  --model_path "Qwen/Qwen3-8B" \
  --output_dir ".result" \
  --seed "0" \
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
  --steps_per_category "default=20000" \
  --joint_decoder_steps "default=0" \
  --joint_decoder_lr "default=1e-2" \
  --joint_decoder_group_size "default=32,cat:down_proj=8" \
  --batch_size "2048" \
  --log_every "100" \
  --eval_every "0" \
  --eval_blocks "256" \
  --eval_ppl "true" \
  --eval_tasks "" \
  --ppl_limit "-1" \
  --intra_parallel "default=1x1" \
  --intra_part_sort_mode "default=none" \
  --sort_prep_workers "0" \
  --outlier_protect_count "default=0" \
  --outlier_protect_axis "input" \
  --outlier_protect_mode "residual_sparse" \
  --outlier_residual_top_p "default=0.01" \
  --outlier_residual_score "input_act_weighted_abs" \
  --outlier_residual_min_abs "0.0" \
  --outlier_residual_codec "blocked_quantized" \
  --outlier_residual_index_bits "8" \
  --outlier_residual_value_bits "8" \
  --wa_mse_calib_dataset "wikitext2" \
  --wa_mse_calib_nsamples "512" \
  --wa_mse_calib_seqlen "4096" \
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
  --lora_dataset "wiki" \
  --lora_rank "default=12" \
  --lora_alpha "default=24.0" \
  --lora_dropout "default=0.0" \
  --lora_steps "default=5000" \
  --lora_batch_size "default=2" \
  --lora_nsamples "default=10000000" \
  --lora_lr "default=8e-5" \
  --lora_weight_decay "default=0.001" \
  --lora_log_every "default=2" \
  --lora_post_attn "false" \
  --lora_temperature "default=1.0" \
  --lora_loss_alpha "default=0.5" \
  --lora_loss_type "default=kd_top_1000" \
  --lora_use_dora "default=false" \
  --lora_hif4_act "false" \
  --eval_hif4_act "false" \
  --lora_gradient_accumulation_steps "1" \
  --lora_optim "adamw_torch" \
  --lora_max_grad_norm "0.333" \
  --lora_warmup_ratio "0.3" \
  --lora_group_by_length "true" \
  --lora_lr_scheduler_type "linear" \
  --lora_model_max_length "4096" \
  --fp16 "false" \
  --bf16 "true" \
  "$@"
