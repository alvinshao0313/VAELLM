#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DISTILL_NCCL_TIMEOUT_SEC=10800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800
export PYTHONHASHSEED=31
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

torchrun --standalone --nproc_per_node=8 tools/cat_train.py \
  --model_path "Qwen/Qwen3-8B" \
  --output_dir "/root/data/ckpts/result/catlora" \
  --compression_categories "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --target_layers all \
  --skip_layers "" \
  --seed 31 \
  --data_seed 31 \
  --deterministic true \
  --train_device cuda \
  --convert \
  --save_model \
  --convert_device cuda \
  --allow_tail_group true \
  --transpose_modules "q_proj,v_proj,o_proj,down_proj" \
  --linear_group_size 36 \
  --vae_steps "default=10000" \
  --vae_batch_size 8192 \
  --vae_learning_rate 3e-3 \
  --vae_weight_decay 0 \
  --vae_optim adamw \
  --vae_lr_scheduler_type linear \
  --vae_warmup_ratio 0 \
  --activation_calib_dataset "alpaca=1" \
  --activation_calib_nsamples 128 \
  --activation_calib_seqlen 8192 \
  --activation_calib_seed 31 \
  --activation_calib_device "" \
  --activation_calib_log_every 0 \
  --codebook_bits "default=32,cat:down_proj=64" \
  --codebook_dim "default=32" \
  --residual_stages "default=2" \
  --base_ch "default=128" \
  --num_res_blocks "default=0" \
  --decoder_base_ch "default=128" \
  --decoder_num_res_blocks "default=1" \
  --norm_type "default=layer" \
  --activation_type "default=swish" \
  --decoder_type "default=symmetric" \
  --recon_loss_type "default=mse" \
  --quantizer_type BSQ \
  --gamma0 1 \
  --gamma 1 \
  --zeta 1 \
  --inv_temperature 100 \
  --beta1 0.9 \
  --beta2 0.95 \
  --l1_weight 1 \
  --lfq_weight 2.5 \
  --commitment_loss_weight 0.25 \
  --entropy_loss_weight 0.01 \
  --normalize_weight \
  --vae_decoder_checkpoint true \
  --new_quant \
  --log_every 100 \
  --eval_every 0 \
  --eval_blocks 256 \
  --skip_ppl_eval true \
  --channel_protect_mode channel \
  --channel_rank_metric channel_weight_actmean_abs \
  --channel_mlp_rank_metric none \
  --channel_mlp_fuse_weights "1,1,1" \
  --channel_scope layer \
  --channel_min_per_layer 0 \
  --channel_quant int8 \
  --channel_axis input \
  --channel_protect_count "default=32,cat:o_proj=64,cat:down_proj=256" \
  --after_category_mode remaining_lora_prefix_decoder \
  --dataset_mix "edgerazor_ii_7m=0.676,edgerazor_ii_gen=0.133,edgerazor_tulu=0.055,edgerazor_am=0.127,vaellm_eval_task=0.009" \
  --dataset_task sft \
  --model_max_length 1024 \
  --dynamic_padding true \
  --group_by_length true \
  --lora_rank "default=12" \
  --lora_alpha "default=24" \
  --lora_dropout "default=0.03" \
  --steps "default=5000" \
  --batch_size "default=4" \
  --learning_rate "default=1e-4" \
  --decoder_lr "default=1e-5" \
  --weight_decay "default=0.001" \
  --logging_steps "default=100" \
  --loss_type "default=kl_top" \
  --top_k "default=100" \
  --temperature "default=1" \
  --alpha "default=0.5" \
  --prompt_loss_weight "default=0" \
  --hidden_loss_weight "default=0.1" \
  --pre_mlp_hidden_loss_weight "default=0.001" \
  --hidden_layer_weighting linear_depth \
  --teacher_output_offload cpu \
  --teacher_model_offload none \
  --selective_student_topk true \
  --selective_student_topk_chunk_rows 32 \
  --norm_train_mode final \
  --lm_head_train_mode linear \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --optim adamw_torch \
  --max_grad_norm 1.3 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --bf16 true \
  --fp16 false \
  "$@"
