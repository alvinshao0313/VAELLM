"""CPU preparation of a fixed native CAT recipe and isolated source-union simulation."""
import argparse
import ast
import os
from pathlib import Path
import sys

import torch

from experiments.qwen_attention_int4_pair.common import (ROOT, PUBLIC, RESULT, SOURCE_ID, CATEGORIES,
    BITS, DEPENDENCIES, read, check, _require)
from experiments.full_rht_gate_pair.common import check as check_template
from experiments.longbench_v1.common import sha256, write_json
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload


def prepare(arm, output):
    output = Path(output).resolve()
    _require(ROOT != PUBLIC and Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
             and not output.exists(), 'Prepare a fresh output in the reviewed CPU-only snapshot.')
    template_path = RESULT / 'fresh_o_bits_pair/control/launch.json'
    template = read(template_path); check_template(template)
    _require(template['status'] == 'completed' and template['exit_code'] == 0
             and template['base_checkpoint_id'] == SOURCE_ID, 'Require completed native O recipe provenance.')
    argv = list(template['training_argv'][2:])
    changes = {'--output_dir': str(output / 'donor'), '--compression_categories': ','.join(CATEGORIES),
               '--codebook_bits': 'default=32,' + ','.join('cat:' + k + '=' + str(v) for k, v in BITS[arm].items()),
               '--channel_protect_count': 'default=64', '--channel_refresh_after_category': 'false'}
    for key, value in changes.items():
        _require(argv.count(key) == 1, 'Ambiguous inherited CLI field: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, CATEGORIES)
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    _require(cat.compression_categories == ','.join(CATEGORIES) and cat.target_layers == 'all'
             and cat.seed == 31 and cat.deterministic and cat.batch_size == 8192 and cat.linear_group_size == 36
             and cat.after_category_mode == 'none' and not cat.channel_refresh_after_category
             and cat.activation_calib_nsamples == 512 and cat.activation_calib_seqlen == 256
             and cat.activation_calib_seed == 31 and training.bf16 and not training.fp16,
             'Native parser changed the fixed paired recipe.')
    for category, cfg in resolved.items():
        _require(cfg.steps == 10000 and cfg.codebook_bits == BITS[arm][category]
                 and cfg.codebook_dim == 32 and cfg.residual_stages == 2 and cfg.base_ch == 128
                 and cfg.num_res_blocks == 0, 'Category recipe changed: ' + category)
    base = Path(template['base_dir']); meta = read(base / 'checkpoint_meta.json')
    native = read(base.parent / 'acceptance.json')['bit_budget']
    scaled_dir = RESULT / 'protected_int4_scale_quality/qwen3'
    scaled = read(scaled_dir / 'launch.prepared.json'); review = read(scaled_dir / 'independent_review.json')
    _require(meta['checkpoint_id'] == native['checkpoint_id'] == scaled['source_checkpoint_id'] == SOURCE_ID
             and review['status'] == 'verified', 'Fixed source/budget/scale-control identity differs.')
    source_hashes = dict(template['source_sha256'])
    new_paths = [*sorted(Path(__file__).parent.glob('*.py')), *[ROOT / x for x in DEPENDENCIES]]
    for path in new_paths:
        ast.parse(path.read_text()); source_hashes[str(path)] = sha256(path)
        relative = path.relative_to(ROOT)
        _require(sha256(PUBLIC / relative) == sha256(path), 'New public/snapshot source differs: ' + str(relative))
    code_delta = decoder_delta = 0; per_category = {}
    for category in CATEGORIES:
        specs = [s for s in meta['converted_modules'] if s['name'].endswith('.' + category)]
        _require(len(specs) == 36, 'Incomplete source category.')
        change = BITS[arm][category] - BITS['control'][category]; delta_code = delta_decoder = 0
        for spec in specs:
            _require(spec['residual_stages'] == 2 and spec['stage_codebook_dims'] == [32, 32]
                     and spec['protected_input_indices'] == {'shape': [64], 'dtype': 'int64'}
                     and spec['weight_rotation']['block_size'] == 32, 'Unexpected source geometry.')
            for packed, decoder in zip(spec['stage_vq_weights'], spec['stage_decoders'], strict=True):
                groups, parts, old_k = packed['logical_shape']
                _require(parts == 1 and old_k == BITS['control'][category]
                         and packed['shape'] == [groups, 1, old_k // 8] and decoder['hidden_dim'] == 128
                         and decoder['num_res_blocks'] == 0 and decoder['param_dtype'] == 'bfloat16',
                         'Unexpected source packed/decoder layout.')
                delta_code += groups * change; delta_decoder += 128 * change * 16
        per_category[category] = delta_code + delta_decoder
        code_delta += delta_code; decoder_delta += delta_decoder
    delta = code_delta + decoder_delta; account = scaled['accounting']; denom = native['original_linear_weights']
    forecast = {'native_int8_union_bits': native['user_linear_bits'] + delta,
        'packed_sidecar_combination_bits': account['candidate_projected_linear_bits'] + delta,
        'expanded_simulation_bits': account['expanded_simulation_linear_bits'] + delta,
        'original_linear_weights': denom, 'code_delta_bits': code_delta, 'decoder_delta_bits': decoder_delta}
    for key in ('native_int8_union', 'packed_sidecar_combination', 'expanded_simulation'):
        forecast[key + '_bpw'] = forecast[key + '_bits'] / denom
    _require(forecast['packed_sidecar_combination_bits'] == {'control': 16615858176, 'candidate': 17363755008}[arm],
             'Expected exact integer candidate forecast.')
    hf = Path(meta['base_model_path']); dense_bytes = read(hf / 'model.safetensors.index.json')['metadata']['total_size']
    checkpoint_sizes = {}; remaining = dense_bytes
    for category in CATEGORIES:
        specs = [s for s in meta['converted_modules'] if s['name'].endswith('.' + category)]
        removed = sum(s['in_features'] * s['out_features'] * 2 for s in specs)
        compressed = sum(native['per_target'][s['name']]['budget_bits'] for s in specs) / 8 + per_category[category] / 8
        remaining += int(compressed) - removed; checkpoint_sizes['after_' + category] = remaining
    checkpoint_sizes['final_model'] = remaining
    evidence = dict(template['input_sha256'])
    extra = [template_path, base.parent / 'acceptance.json', scaled_dir / 'launch.prepared.json',
             scaled_dir / 'report.json', scaled_dir / 'independent_review.json',
             *[scaled_dir / ('lm_eval_results_' + s + '.json') for s in ('mmlu', 'seven')],
             RESULT / 'fresh_o_bits_pair/comparison.json',
             PUBLIC / 'result/qwen3_8b_w25_20260930/validation/mlp25_deployment.json',
             PUBLIC / 'result/qwen3_8b_w25_20260930/validation/mixed_k24_k56_d32_bf16/validation.json']
    evidence.update({str(path): sha256(path) for path in extra})
    record = {'schema': 'fresh_attention_int4_pair_v1', 'status': 'prepared_unbound', 'gpu': None,
        'queued': False, 'automatic_retry': False, 'cwd': str(ROOT), 'python_executable': sys.executable,
        'output_dir': str(output), 'arm': arm, 'categories': CATEGORIES, 'codebook_bits': BITS[arm],
        'base_dir': str(base), 'base_checkpoint_id': SOURCE_ID, 'hf_path': str(hf),
        'architecture': template['architecture'], 'model_policy': scaled['model_policy'],
        'source_native_mean_percent': scaled['control_mean_percent'], 'control_result_paths': scaled['control_result_paths'],
        'source_scale_control': {'mean_percent': review['mean_percent']['scale_search'],
            'raw_result_paths': {s: str(scaled_dir / ('lm_eval_results_' + s + '.json')) for s in ('mmlu', 'seven')}},
        'training_argv': [sys.executable, 'tools/cat_train.py', *argv], 'normalized_expected': normalized,
        'source_sha256': source_hashes, 'input_sha256': evidence, 'protected_stats': template['protected_stats'],
        'environment': template['environment'], 'package_versions': scaled['package_versions'],
        'forecast': forecast, 'timeout_seconds': 5400, 'termination_grace_seconds': 20,
        'protocol': scaled['protocol'], 'cpu_prepared': True,
        'seed_boundary': {'category_order': CATEGORIES, 'stage_shuffle_seeds': [[31 + i*100000 + j for j in (0, 1)] for i in range(4)],
            'initialization': 'One global seed31; K changes parameter RNG consumption. No bitwise paired trajectory claim.',
            'calibration': 'Original HF one-shot512x256 seed31 before compression, refresh false.'},
        'disk_estimate': {'checkpoint_tensor_bytes': checkpoint_sizes, 'five_native_checkpoints_bytes': sum(checkpoint_sizes.values()),
            'single_arm_provision_bytes': sum(checkpoint_sizes.values()) + 2*1024**3,
            'boundary': 'Estimate includes all native five saves; 2GiB allowance for containers/CAT state/tokenizers/sidecar. No automatic cleanup or queue.'},
        'comparison_boundary': 'Fresh candidate versus fresh control; replacing144 recovered attention projections loses prior recovery. Preserve f927108MLP and147 backbone tensors. Only BF16 expansion, no assembled checkpoint export.'}
    record['launcher_argv'] = [sys.executable, '-u', '-m', 'experiments.qwen_attention_int4_pair.run', '--launch-record', str(output / 'launch.json')]
    check(record); _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record); write_json(output / 'launch.prepared.json', record)
    print({'arm': arm, 'status': record['status'], 'forecast': forecast, 'disk': record['disk_estimate'],
           'source_pins': len(source_hashes), 'input_pins': len(evidence), 'prepared_sha256': sha256(output / 'launch.prepared.json')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--arm', choices=tuple(BITS), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); prepare(args.arm, args.output_dir)
