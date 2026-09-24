#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=4,5
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
# Short correctness run: real model, batch, sequence length and two-rank accumulation.
python -m torch.distributed.run --standalone --nproc_per_node=2 -m compressed_e2e_fintuning.main \
  --student_checkpoint_dir /home/shaoyuantian/program/VAELLM/result/catlora/Qwen_Qwen3-8B_20260910_094022/final_model \
  --run_root_dir /home/shaoyuantian/program/VAELLM/result/compressed_e2e_fintuning/e2e_0910_search_20260924/verify_continuous \
  --train_mode decoder_lora \
  --target_layers 0-35 \
  --target_modules all \
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
  --hidden_loss_weight 0 \
  --loss_type kl_top_partial \
  --pre_mlp_hidden_loss_weight 0 \
  --prompt_loss_weight 0.3 \
  --selective_student_topk false \
  --selective_student_topk_chunk_rows 32 \
  --temperature 1.0 \
  --top_k 100 \
  --top_mse_weight 1.0 \
  --batch_size 8 \
  --decoder_lr 3e-6 \
  --distill_fp32_components lora,decoder,norm,lm_head \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant":false}' \
  --learning_rate 3e-4 \
  --logging_steps 1 \
  --lr_scheduler_type cosine \
  --max_grad_norm 1.5 \
  --optim adamw_torch \
  --steps 8 \
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
  --eval_limit 1 \
  --eval_device cuda \
  --eval_hif4_act false \
  --eval_num_fewshot 0 \
  --eval_prewarm_group_size 8 \
  --eval_tasks boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa,mmlu \
  --ppl_limit -1 \
  --ppl_seqlen 2048 \
  --skip_ppl_eval true \
  --residual_lora_mode none \
  --save_tokenizer true \
  --full_determinism false \
  --bf16 true \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 2 \
  --save_total_limit 1 \
  --warmup_steps 1 \
  --lr_scheduler_kwargs '{"num_cycles":0.0007216494845360825}'
