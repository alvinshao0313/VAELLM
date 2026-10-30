"""Explicit N fresh recipe; runtime never reads an old launch or checkpoint."""
from dataclasses import asdict
import shlex

import fresh_support as s

MIX = 'edgerazor_ii_7m=0.341,edgerazor_ii_gen=0.067,edgerazor_tulu=0.028,edgerazor_am=0.064,vaellm_eval_task=0.500'
FIXED = '''
--compression_categories q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
--target_layers all --seed 31 --data_seed 31 --deterministic true
--train_device cuda --convert --save_model --convert_device cuda
--allow_tail_group true --transpose_modules q_proj,v_proj,o_proj,down_proj --linear_group_size 36
--vae_steps default=10000 --vae_batch_size 8192
--vae_learning_rate 3e-3 --vae_weight_decay 0 --vae_optim adamw
--vae_lr_scheduler_type linear --vae_warmup_ratio 0
--residual_stages default=2 --base_ch default=128 --num_res_blocks default=0
--norm_type default=layer --activation_type default=swish
--decoder_type default=symmetric,cat:gate_proj=asymmetric
--decoder_base_ch default=none,cat:gate_proj=128 --decoder_num_res_blocks default=none,cat:gate_proj=0
--recon_loss_type default=mse --quantizer_type BSQ
--gamma0 1 --gamma 1 --zeta 1 --inv_temperature 100 --beta1 0.9 --beta2 0.95
--l1_weight 1 --lfq_weight 2.5 --commitment_loss_weight 0.25 --entropy_loss_weight 0.01
--normalize_weight --weight_rotation two_sided --weight_rotation_block_size 32
--weight_rotation_block_size_by_category cat:q_proj=0,cat:k_proj=0 --vae_decoder_checkpoint true --new_quant
--channel_protect_mode channel --channel_scope layer --channel_axis input --channel_quant int8
--channel_protect_count default=64,cat:down_proj=256
--channel_rank_metric channel_weight_actmean_abs --channel_mlp_rank_metric none --channel_refresh_after_category false
--activation_calib_dataset alpaca=1 --activation_calib_nsamples 512 --activation_calib_seqlen 256
--activation_calib_seed 31 --activation_calib_device cuda --activation_calib_log_every 64
--after_category_mode remaining_lora_prefix_decoder --distill_fp32_components lora,decoder
--dataset_task sft --model_max_length 1024 --dynamic_padding true --group_by_length true
--lora_rank default=12 --lora_alpha default=24 --lora_dropout default=0.1
--steps default=500 --batch_size default=4 --learning_rate default=1e-4 --decoder_lr default=1e-5 --weight_decay default=0
--loss_type default=kl_top_mass --top_k default=100 --temperature default=1 --prompt_loss_weight default=0.3
--hidden_loss_weight default=0 --pre_mlp_hidden_loss_weight default=0 --selective_student_topk false
--residual_lora_mode none --norm_train_mode none --lm_head_train_mode none
--teacher_output_offload cpu --teacher_model_offload none --teacher_output_pin_memory true --teacher_output_chunk_tokens 8
--gradient_accumulation_steps 1 --gradient_checkpointing true --gradient_checkpointing_kwargs '{"use_reentrant": false}'
--optim adamw_torch --max_grad_norm 1.5 --warmup_ratio 0.01 --lr_scheduler_type cosine
--logging_steps default=10 --log_every 100 --eval_every 0 --eval_blocks 256 --skip_ppl_eval true --eval_tasks ''
--save_strategy steps --save_steps 250 --save_total_limit 1 --bf16 true --fp16 false
'''


def argv(arm, base_model_path, output_dir, *, after_o=None):
    s.require(arm in ('A', 'B'), 'Select A or B.')
    bits = 'default=32,cat:q_proj=38,cat:k_proj=38,cat:o_proj=48,cat:down_proj=48'
    dim = 'default=32'
    if arm == 'B':
        bits += ',cat:gate_proj=64'
        dim += ',cat:gate_proj=64'
    values = ['--model_path', str(base_model_path), '--output_dir', str(output_dir),
              *shlex.split(FIXED), '--dataset_mix', MIX, '--codebook_bits', bits, '--codebook_dim', dim]
    if after_o is not None:
        values += ['--resume_from_checkpoint', str(after_o)]
    return values


def parse(arm, values):
    s.enable_snapshot()
    from train_utils.cat_category_runtime import resolve_category_runtime_configs
    from train_utils.cat_runtime_adapter import parse_cat_runtime_args
    from train_utils.cat_runtime_state_v6 import build_cat_cross_category_runtime_identity
    from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
    from train_utils.config.targets import parse_compression_categories, parse_skip_layers, parse_target_layers
    cat, hf, training, vae = parse_cat_runtime_args(values)
    training.seed = int(cat.seed)
    cfgs = resolve_category_runtime_configs(cat, vae, s.CATEGORIES)
    identity = s.with_policy(build_cat_cross_category_runtime_identity(cat_args=cat, vae_args=vae,
        resolved_category_cfgs=cfgs, compression_categories=parse_compression_categories(cat.compression_categories),
        target_layers=parse_target_layers(cat.target_layers), skip_layers=parse_skip_layers(cat.skip_layers),
        transpose_modules=cat.transpose_modules.split(',')), arm)
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae, training_args=training,
                                                    resolved_category_cfgs=cfgs)
    normalized['experiment_runtime'] = s.policy(arm)
    return cat, hf, training, vae, cfgs, identity, normalized


def plan(arm, base_model_path, output_dir):
    values = argv(arm, base_model_path, output_dir)
    cat, _, _, _, cfgs, identity, normalized = parse(arm, values)
    return dict(native_argv=values, runtime_identity=identity, normalized=normalized,
                category_configs={name: asdict(cfg) for name, cfg in cfgs.items()},
                recovery_configs={name: asdict(cat.resolve_after_category_config(name)) for name in s.CATEGORIES})
