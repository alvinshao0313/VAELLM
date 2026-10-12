"""Fixed native K32/K34 gate comparison; padding is reported, not hidden."""
from copy import deepcopy
from pathlib import Path
import sys

from experiments.full_rht_gate_pair.common import ROOT, ORIGINAL, RESULT, check, read
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload

BASE = RESULT / 'llama_fineweb1600_down_d16_combo/final_model'
BASE_ID = 'e767c232-56fd-4b03-a4ae-dd0781b6e047'
DENOMINATOR = 6979321856
GROUPS = 1806336
LOGICAL_BITS = 17326653440
PACKED_BITS = 18020286464


def recipe(template, output, *, bits, steps):
    _require(bits in (32, 34) and steps in (2, 10000), 'Only the specified short/formal recipes are allowed.')
    argv = list(template['training_argv'][2:])
    changes = {'--output_dir': str(Path(output) / 'donor'), '--codebook_bits': f'default=32,cat:gate_proj={bits}',
               '--vae_steps': f'default={steps}', '--log_every': '1' if steps == 2 else '100'}
    for key, value in changes.items():
        _require(argv.count(key) == 1, 'Ambiguous native argument: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, ['gate_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        expected[section]['output_dir'] = str(Path(output) / 'donor')
    for section in ('canonical_config', 'vae_args'):
        expected[section]['codebook_bits'] = {'default': 32, 'by_category': {'gate_proj': bits}}
    expected['canonical_config']['vae_steps'] = {'default': steps}
    expected['canonical_config']['vae_opt_template']['log_every'] = 1 if steps == 2 else 100
    expected['cat_args']['log_every'] = 1 if steps == 2 else 100
    expected['resolved_category_runtime']['gate_proj'].update(codebook_bits=bits, steps=steps)
    _require(normalized == expected, 'Native recipe differs beyond K, steps/logging, and output.')
    cfg = resolved['gate_proj']
    _require(cat.seed == 31 and argv[argv.index('--data_seed') + 1] == '31' and cat.deterministic and cat.batch_size == 8192
             and cat.linear_group_size == 32 and cat.target_layers == 'all'
             and cat.after_category_mode == 'none' and not cat.resume_from_checkpoint
             and cat.transpose_modules == 'q_proj,v_proj,o_proj,down_proj'
             and not cat.channel_refresh_after_category and cat.activation_calib_dataset == 'alpaca=1'
             and cat.activation_calib_nsamples == 512 and cat.activation_calib_seqlen == 256
             and cat.activation_calib_seed == 31 and training.bf16 and not training.fp16
             and cfg.codebook_dim == 32 and cfg.residual_stages == 2 and cfg.base_ch == 128
             and cfg.num_res_blocks == 0 and cfg.channel_protect_count == 64,
             'Fixed fresh gate scientific conditions differ.')
    return [sys.executable, 'tools/cat_train.py', *argv], normalized
