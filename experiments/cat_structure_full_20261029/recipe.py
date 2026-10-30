"""Explicit future-gate geometry and scalar-exposure contract for A and B."""
from copy import deepcopy

import common as c
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_runtime_state_v6 import build_cat_cross_category_runtime_identity
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload

ARMS = {'A': {'dim': 32, 'bits': 32}, 'B': {'dim': 64, 'bits': 64}}
FUTURE_GATE_FIELDS = ('codebook_bits', 'codebook_dim', 'decoder_type', 'decoder_base_ch', 'decoder_num_res_blocks')


def policy(arm):
    choice = ARMS[arm]
    return dict(schema='cat_structure_M36_scalar_exposure_v1', arm=arm,
        scalar_exposure_per_linear_update=c.SCALARS_PER_LINEAR_UPDATE,
        category_vector_batches={'gate_proj': c.SCALARS_PER_LINEAR_UPDATE // choice['dim'],
                                 'up_proj': 8192, 'down_proj': 8192},
        category_shuffle_seeds={'gate_proj': 400031, 'up_proj': 500031, 'down_proj': 600031},
        seed=31, group_size=36, entropy='Original CAT M36 pooled BSQ probabilities; no independent-layer loss.',
        continuation='New future-gate structure experiment, not exact continuation of the old future geometry; no inherited global RNG.')


def replace_option(argv, name, value):
    c.require(argv.count(name) == 1, 'Recipe requires exactly one ' + name)
    argv[argv.index(name) + 1] = str(value)


def training_argv(arm, old_row):
    argv = list(old_row['training_argv'][2:])
    dim, bits = ARMS[arm]['dim'], ARMS[arm]['bits']
    changes = {
        '--output_dir': c.OUTPUT / arm / 'native',
        '--resume_from_checkpoint': c.OUTPUT / arm / 'initial_after_o',
        '--codebook_bits': f'default=32,cat:q_proj=38,cat:k_proj=38,cat:o_proj=48,cat:gate_proj={bits},cat:up_proj=32,cat:down_proj=48',
        '--codebook_dim': f'default=32,cat:gate_proj={dim}',
        '--decoder_type': 'default=symmetric,cat:gate_proj=asymmetric',
        '--save_total_limit': 2,
    }
    for name, value in changes.items():
        replace_option(argv, name, value)
    # These category selectors are supported by the canonical CLI, and leave
    # the completed attention and future up/down decoder contracts unchanged.
    argv += ['--decoder_base_ch', 'default=none,cat:gate_proj=128',
             '--decoder_num_res_blocks', 'default=none,cat:gate_proj=0']
    return argv


def add_experiment_identity(identity, arm):
    value = deepcopy(identity)
    c.require('experiment_runtime' not in value, 'Do not overwrite an existing experiment runtime contract.')
    value['experiment_runtime'] = policy(arm)
    return value


def parse_recipe(arm, argv, saved_runtime):
    cat, hf, training, vae = parse_cat_runtime_args(argv)
    training.seed = int(cat.seed)
    cfgs = resolve_category_runtime_configs(cat, vae, c.CATEGORIES)
    identity = build_cat_cross_category_runtime_identity(cat_args=cat, vae_args=vae,
        resolved_category_cfgs=cfgs, compression_categories=c.CATEGORIES,
        target_layers=saved_runtime['runtime_identity']['target_layers'], skip_layers=(),
        transpose_modules=cat.transpose_modules.split(','))
    identity = add_experiment_identity(identity, arm)
    c.require(cat.seed == 31 and cat.linear_group_size == 36 and cat.batch_size == 8192
        and cat.target_layers == 'all' and not cat.skip_layers
        and cat.after_category_mode == 'remaining_lora_prefix_decoder', 'Changed original whole-category CAT scope.')
    for category in c.CATEGORIES[4:]:
        cfg = cfgs[category]
        expected_dim = ARMS[arm]['dim'] if category == 'gate_proj' else 32
        c.require(cfg.codebook_dim == expected_dim and cfg.residual_stages == 2 and cfg.steps == 10000,
            'Incorrect actual category dimension/stage/update budget.')
        recovery = cat.resolve_after_category_config(category)
        c.require(recovery.opt.steps == 500 and recovery.opt.batch_size == 4
            and recovery.data.model_max_length == 1024 and recovery.lora.rank == 12,
            'Keep original real B4/1024/500-step temporary rank12 recovery.')
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae, training_args=training,
                                                    resolved_category_cfgs=cfgs)
    normalized['experiment_runtime'] = policy(arm)
    return cat, hf, training, vae, cfgs, identity, normalized


def validate_fork(saved, current, arm):
    expected = deepcopy(saved)
    c.require(saved['compression_categories'] == current['compression_categories'] == list(c.CATEGORIES),
        'Keep every original category position.')
    changes = {}
    before = saved['resolved_category_runtime']['gate_proj']
    after = current['resolved_category_runtime']['gate_proj']
    for field in FUTURE_GATE_FIELDS:
        if before[field] != after[field]:
            expected['resolved_category_runtime']['gate_proj'][field] = after[field]
            changes['gate_proj.' + field] = {'source': before[field], 'fork': after[field]}
    expected['experiment_runtime'] = policy(arm)
    c.require(expected == current, 'Changed completed attention, up/down geometry, data-independent VAE settings or parent runtime outside the explicit future-gate contract.')
    return changes
