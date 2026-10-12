"""One existing CAT option: matched K34True gate LayerNorm to RMSNorm."""
from copy import deepcopy
from pathlib import Path
import sys

from experiments.full_rht_gate_pair.common import ROOT, ORIGINAL, RESULT, check, read
from experiments.llama_gate_small_rate_pair.common import BASE, BASE_ID
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload

CONTROL = RESULT / 'llama_gate_k34_transpose/true'
CONTROL_ID = '98a80541-fd2b-4ce4-b88d-0809f82d4ef3'
OUTPUT = RESULT / 'llama_gate_rms'


def recipe(template, output, steps):
    _require(steps in (2, 10000), 'Only the native short or formal recipe is specified.')
    argv = list(template['training_argv'][2:])
    changes = {'--output_dir': str(output / 'donor'), '--norm_type': 'default=rms'}
    if steps == 2: changes.update({'--vae_steps': 'default=2', '--log_every': '1'})
    for key, value in changes.items():
        _require(argv.count(key) == 1, 'Ambiguous original option: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, ['gate_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    expected = deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'): expected[section]['output_dir'] = str(output / 'donor')
    for section in ('canonical_config', 'vae_args'): expected[section]['norm_type'] = {'default': 'rms'}
    expected['resolved_category_runtime']['gate_proj']['norm_type'] = 'rms'
    if steps == 2:
        expected['canonical_config']['vae_steps'] = {'default': 2}
        expected['canonical_config']['vae_opt_template']['log_every'] = expected['cat_args']['log_every'] = 1
        expected['resolved_category_runtime']['gate_proj']['steps'] = 2
    _require(normalized == expected, 'Recipe differs beyond norm/output and diagnostic steps/logging.')
    cfg = resolved['gate_proj']; opt = normalized['canonical_config']['vae_opt_template']
    _require(cat.compression_categories == 'gate_proj' and cat.target_layers == 'all'
        and cat.transpose_modules == 'q_proj,v_proj,o_proj,gate_proj,down_proj'
        and cat.batch_size == 8192 and cat.linear_group_size == 32 and cat.seed == 31
        and cat.after_category_mode == 'none' and not cat.resume_from_checkpoint
        and cat.activation_calib_dataset == 'alpaca=1' and cat.activation_calib_nsamples == 512
        and cat.activation_calib_seqlen == 256 and cat.activation_calib_seed == 31
        and cfg.steps == steps and cfg.codebook_bits == 34 and cfg.codebook_dim == 32 and cfg.residual_stages == 2
        and cfg.base_ch == 128 and cfg.num_res_blocks == 0 and cfg.decoder_type == 'symmetric'
        and cfg.norm_type == 'rms' and cfg.activation_type == 'swish' and cfg.channel_protect_count == 64
        and vae.weight_rotation == 'two_sided' and vae.weight_rotation_block_size == 32
        and opt['vae_learning_rate'] == .003 and opt['vae_lr_scheduler_type'] == 'linear'
        and opt['vae_warmup_ratio'] == 0 and training.bf16 and not training.fp16, 'Wrong matched RMS recipe.')
    return [sys.executable, 'tools/cat_train.py', *argv], normalized, vae, cfg


def rms_spec(spec):
    result = deepcopy(spec)
    for decoder in result['decoders'] + result['stage_decoders']:
        _require(decoder['norm_type'] == 'layer', 'Control decoder must use LayerNorm.')
        decoder['norm_type'] = 'rms'
    return result


def verify_short(record, directory):
    from experiments.longbench_v1.common import sha256
    launch, proof = read(directory / 'launch.json'), read(directory / 'verification.json')
    _require(launch['status'] == 'completed' and launch['exit_code'] == 0
        and launch['verification_sha256'] == sha256(directory / 'verification.json')
        and all((directory / (name + '.exit')).read_text().strip() == '0'
                for name in ('train', 'assembly', 'audit', 'dispatcher', 'outer')),
        'The genuine RMS short must complete before formal readiness.')
    _require(proof['status'] == 'verified_native_rms_short' and proof['base_checkpoint_id'] == record['base_checkpoint_id']
        and proof['source_union_save_fresh_tensors'] == record['expected_inventory']['serialized_tensors']
        and proof['rms_updated_stage_count'] == record['expected_inventory']['selected_stages']
        and proof['native_group8_prewarm']['warmed'] == record['architecture']['projection_count']
        and proof['integration_forward']['finite'] and proof['kernel_parity']['passed']
        and proof['bit_budget']['user_linear_bits'] == record['expected_user_linear_bits']
        and proof['bit_budget']['packed_linear_bits'] == record['expected_packed_linear_bits'], 'Incomplete new RMS path proof.')
    return proof
