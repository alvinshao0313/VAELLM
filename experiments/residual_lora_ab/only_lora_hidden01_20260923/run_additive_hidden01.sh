#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH=.
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DISTILL_NCCL_TIMEOUT_SEC=10800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
# Match the original 5000-step cosine through step 2000; stop after step 2001.
python -m torch.distributed.run --standalone --nproc_per_node=4 -m compressed_e2e_fintuning.main \
  --student_checkpoint_dir /home/shaoyuantian/program/VAELLM/result/catlora/Qwen_Qwen3-8B_20260910_094022/final_model \
  --run_root_dir /home/shaoyuantian/program/VAELLM/result/compressed_e2e_fintuning/residual_lora_hidden01_20260923 \
  --train_mode lora \
  --target_layers 0-35 \
  --target_modules all \
  --bit_active_ratio 0.03 \
  --bit_lr auto \
  --bit_optimizer rms_sgd \
  --bit_round_steps auto \
  --bit_weight_decay 0.0 \
  --data_seed 0 \
  --dataset_mix edgerazor_ii_7m=0.341,edgerazor_ii_gen=0.067,edgerazor_tulu=0.028,edgerazor_am=0.064,vaellm_eval_task=0.5 \
  --dataset_task sft \
  --dynamic_padding true \
  --group_by_length true \
  --model_max_length 1024 \
  --seed 0 \
  --text_field text \
  --alpha 0.5 \
  --hidden_layer_weighting adaptive_top_3 \
  --hidden_loss_weight 0.1 \
  --loss_type kl_top_partial \
  --pre_mlp_hidden_loss_weight 0.1 \
  --prompt_loss_weight 0.3 \
  --selective_student_topk false \
  --selective_student_topk_chunk_rows 32 \
  --temperature 1.0 \
  --top_k 100 \
  --top_mse_weight 1.0 \
  --batch_size 8 \
  --decoder_lr 1e-05 \
  --distill_fp32_components lora,norm,lm_head,residual_lora \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant":false}' \
  --learning_rate 0.0001 \
  --logging_steps 10 \
  --lr_scheduler_type cosine \
  --max_grad_norm 1.5 \
  --optim adamw_torch \
  --steps 2001 \
  --warmup_ratio 0.03 \
  --weight_decay 0.001 \
  --lora_rank 8 \
  --lora_alpha 16.0 \
  --lora_dropout 0.1 \
  --lm_head_lr 0.0001 \
  --lm_head_train_mode lora \
  --norm_lr 0.0001 \
  --norm_train_mode all \
  --distill_hif4_act false \
  --layer_device_map auto \
  --offload_checkpoint true \
  --offload_min_tensor_bytes 1048576 \
  --offload_mode none \
  --offload_pin_memory true \
  --offload_prefetch_distance 1 \
  --parallel_mode dp \
  --teacher_model_offload none \
  --teacher_output_chunk_tokens 8 \
  --teacher_output_offload cpu \
  --teacher_output_pin_memory true \
  --vae_decoder_checkpoint true \
  --eval_after_save true \
  --eval_batch_size auto \
  --eval_device cuda \
  --eval_hif4_act false \
  --eval_num_fewshot 0 \
  --eval_prewarm_group_size 8 \
  --eval_tasks boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu \
  --ppl_limit -1 \
  --ppl_seqlen 2048 \
  --skip_ppl_eval true \
  --residual_lora_mode additive \
  --residual_lora_rank 8 \
  --residual_lora_alpha 16 \
  --residual_lora_dropout 0 \
  --residual_lora_lr 0.0001 \
  --save_tokenizer true \
  --full_determinism false \
  --bf16 true \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 3 \
  --warmup_steps 150 \
  --lr_scheduler_kwargs '{"num_cycles":0.19082474226804125}'
