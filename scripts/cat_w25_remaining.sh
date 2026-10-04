#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=31
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

# Caller activates bitvae and selects one GPU.
# Dense LoRA is merged after each category; compressed projections train only their decoders.
python tools/cat_train.py \
  --model_path /home/star/models/Qwen/Qwen3-8B \
  --output_dir result/qwen3_8b_w25_20260930/remaining_prefix_block32_s500 \
  --compression_categories q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --target_layers all --seed 31 --data_seed 31 --deterministic true \
  --train_device cuda --convert --save_model --convert_device cuda \
  --allow_tail_group true --transpose_modules q_proj,v_proj,o_proj,down_proj \
  --linear_group_size 36 \
  --vae_steps default=10000 --vae_batch_size 8192 \
  --vae_learning_rate 3e-3 --vae_weight_decay 0 --vae_optim adamw \
  --vae_lr_scheduler_type linear --vae_warmup_ratio 0 \
  --codebook_bits default=32,cat:o_proj=48,cat:down_proj=48 \
  --codebook_dim default=32 --residual_stages default=2 \
  --base_ch default=128 --num_res_blocks default=0 \
  --norm_type default=layer --activation_type default=swish --decoder_type default=symmetric \
  --recon_loss_type default=mse --quantizer_type BSQ \
  --gamma0 1 --gamma 1 --zeta 1 --inv_temperature 100 --beta1 0.9 --beta2 0.95 \
  --l1_weight 1 --lfq_weight 2.5 --commitment_loss_weight 0.25 --entropy_loss_weight 0.01 \
  --normalize_weight --weight_rotation two_sided --weight_rotation_block_size 32 \
  --vae_decoder_checkpoint true --new_quant \
  --channel_protect_mode channel --channel_scope layer --channel_axis input --channel_quant int8 \
  --channel_protect_count default=64,cat:down_proj=256 \
  --channel_rank_metric channel_weight_actmean_abs --channel_mlp_rank_metric none \
  --channel_refresh_after_category false \
  --activation_calib_dataset alpaca=1 --activation_calib_nsamples 512 \
  --activation_calib_seqlen 256 --activation_calib_seed 31 \
  --activation_calib_device cuda --activation_calib_log_every 64 \
  --after_category_mode remaining_lora_prefix_decoder --distill_fp32_components lora,decoder \
  --dataset_mix 'edgerazor_ii_7m=0.341,edgerazor_ii_gen=0.067,edgerazor_tulu=0.028,edgerazor_am=0.064,vaellm_eval_task=0.500' \
  --dataset_task sft --model_max_length 1024 --dynamic_padding true --group_by_length true \
  --lora_rank default=12 --lora_alpha default=24 --lora_dropout default=0.1 \
  --steps default=500 --batch_size default=4 \
  --learning_rate default=1e-4 --decoder_lr default=1e-5 --weight_decay default=0 \
  --loss_type default=kl_top_mass --top_k default=100 --temperature default=1 \
  --prompt_loss_weight default=0.3 --hidden_loss_weight default=0 --pre_mlp_hidden_loss_weight default=0 \
  --selective_student_topk false \
  --residual_lora_mode none --norm_train_mode none --lm_head_train_mode none \
  --teacher_output_offload cpu --teacher_model_offload none \
  --teacher_output_pin_memory true --teacher_output_chunk_tokens 8 \
  --gradient_accumulation_steps 1 --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --optim adamw_torch --max_grad_norm 1.5 --warmup_ratio 0.01 --lr_scheduler_type cosine \
  --logging_steps default=10 --log_every 100 --eval_every 0 --eval_blocks 256 \
  --skip_ppl_eval true --eval_tasks '' \
  --save_strategy steps --save_steps 250 --save_total_limit 1 \
  --bf16 true --fp16 false \
  "$@"
