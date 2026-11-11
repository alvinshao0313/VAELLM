"""The confirmed S4 recipe, with no category or future-weight optimization."""
from dataclasses import asdict
import shlex

from bindings import CATEGORIES, HF, s

FIXED = '''
--compression_categories q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
--target_layers all --seed 31 --data_seed 31 --deterministic true
--train_device cuda --convert --save_model --convert_device cuda
--allow_tail_group true --transpose_modules q_proj,v_proj,o_proj,down_proj --linear_group_size 36
--vae_steps default=10000 --vae_batch_size 8192
--vae_learning_rate 3e-3 --vae_weight_decay 0 --vae_optim adamw
--vae_lr_scheduler_type linear --vae_warmup_ratio 0
--residual_stages default=4 --base_ch default=128 --num_res_blocks default=0
--codebook_bits default=32,cat:gate_proj=64 --codebook_dim default=32,cat:gate_proj=64
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
--after_category_mode none --residual_lora_mode none --norm_train_mode none --lm_head_train_mode none
--log_every 100 --eval_every 0 --eval_blocks 256 --skip_ppl_eval true --eval_tasks ''
--save_strategy no --bf16 true --fp16 false
'''


def parse(output):
    s.enable_snapshot()
    from train_utils.cat_category_runtime import resolve_category_runtime_configs
    from train_utils.cat_runtime_adapter import parse_cat_runtime_args
    from train_utils.cat_runtime_state_v6 import build_cat_cross_category_runtime_identity
    from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
    from train_utils.config.targets import parse_compression_categories, parse_skip_layers, parse_target_layers
    values = ['--model_path', str(HF), '--output_dir', str(output), *shlex.split(FIXED)]
    cat, hf, training, vae = parse_cat_runtime_args(values)
    training.seed = int(cat.seed)
    cfgs = resolve_category_runtime_configs(cat, vae, CATEGORIES)
    identity = build_cat_cross_category_runtime_identity(cat_args=cat, vae_args=vae,
        resolved_category_cfgs=cfgs, compression_categories=parse_compression_categories(cat.compression_categories),
        target_layers=parse_target_layers(cat.target_layers), skip_layers=parse_skip_layers(cat.skip_layers),
        transpose_modules=cat.transpose_modules.split(','))
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae, training_args=training,
        resolved_category_cfgs=cfgs)
    return cat, hf, training, vae, cfgs, identity, normalized, values


def describe(output):
    _, _, _, _, cfgs, identity, normalized, values = parse(output)
    return dict(native_argv=values, category_configs={name: asdict(cfg) for name, cfg in cfgs.items()},
        runtime_identity=identity, normalized=normalized,
        vector_batches={name: 4096 if name == 'gate_proj' else 8192 for name in CATEGORIES},
        category_shuffle_seeds={name: 31 + index * 100000 for index, name in enumerate(CATEGORIES)},
        schedule_steps_per_stage=10000, fit_updates=280000, no_category_recovery=True,
        entropy='Original N CAT M36 pooled BSQ', source='Same original BF16 for each category')
