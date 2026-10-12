"""Prepare the existing native Llama OV bit-rate comparison without GPU work."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_gate_pair.common import EXPECTED_IDS, ORIGINAL, RESULT, ROOT, check, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import load_v6_meta

BITS = {'control': {'v_proj': 32, 'o_proj': 48}, 'candidate': {'v_proj': 40, 'o_proj': 56}}
CATEGORIES = ['v_proj', 'o_proj']
TEMPLATE = RESULT / 'full_rht_gate_pair/llama_block32/launch.json'
LLAMA_RESULT = ORIGINAL / 'result/llama31_8b_w25_20261004'


def recipe(template, output, bits):
    argv = list(template['training_argv'][2:])
    changes = {'--output_dir': str(output / 'donor'), '--compression_categories': ','.join(CATEGORIES),
               '--codebook_bits': f'default=32,cat:v_proj={bits["v_proj"]},cat:o_proj={bits["o_proj"]}'}
    for key, value in changes.items():
        _require(argv.count(key) == 1, 'Ambiguous inherited argument: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, CATEGORIES)
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    _require(cat.target_layers == 'all' and cat.batch_size == 8192 and cat.after_category_mode == 'none'
             and cat.linear_group_size == 32 and training.bf16 and not training.fp16,
             'Native parser changed the approved category recipe.')
    for category in CATEGORIES:
        cfg = resolved[category]
        _require(cfg.steps == 10000 and cfg.codebook_bits == bits[category],
                 'Native category override did not resolve: ' + category)
    return argv, normalized


def estimate(meta, bits, source_budget):
    details = {}; code_delta = decoder_delta = 0
    for category in CATEGORIES:
        specs = [s for s in meta['converted_modules'] if s['name'].endswith('.' + category)]
        _require(len(specs) == 32, 'Require all 32 native Llama projections: ' + category)
        category_code = category_decoder = 0
        for spec in specs:
            _require(spec['transpose'] and spec['residual_stages'] == 2
                     and spec['stage_codebook_dims'] == [32, 32]
                     and spec['protected_input_indices'] == {'shape': [64], 'dtype': 'int64'}
                     and spec['weight_rotation']['block_size'] == 32,
                     'Unexpected actual source geometry: ' + spec['name'])
            for packed, decoder in zip(spec['stage_vq_weights'], spec['stage_decoders'], strict=True):
                groups, parts, old_k = packed['logical_shape']
                _require(parts == 1 and old_k == BITS['control'][category]
                         and packed['shape'] == [groups, 1, old_k // 8]
                         and decoder['in_dim'] == old_k and decoder['hidden_dim'] == 128
                         and decoder['out_dim'] == 32 and decoder['num_res_blocks'] == 0
                         and decoder['param_dtype'] == 'bfloat16', 'Unsupported source bit geometry.')
                change = bits[category] - old_k
                category_code += groups * change
                category_decoder += 128 * change * 16
        details[category] = {'layers': 32, 'old_K': BITS['control'][category], 'new_K': bits[category],
                             'code_delta_bits': category_code, 'decoder_delta_bits': category_decoder}
        code_delta += category_code; decoder_delta += category_decoder
    source_bits = source_budget['user_linear_bits']; denominator = source_budget['original_linear_weights']
    _require(source_bits == 17100439552 and denominator == 6979321856, 'Source budget changed.')
    total = source_bits + code_delta + decoder_delta
    _require(total < denominator * 2.5, 'Estimated candidate must remain strictly below 2.5 bits.')
    return {'basis': 'Actual source metadata and prior measured source storage; candidate storage not yet measured.',
        'source_bits': source_bits, 'original_linear_weights': denominator, 'per_category': details,
        'code_delta_bits': code_delta, 'decoder_delta_bits': decoder_delta, 'estimated_total_bits': total,
        'estimated_bpw': total / denominator, 'remaining_bits_to_2_5': int(denominator * 2.5 - total),
        'unchanged_components': 'INT8 protection, indices/scales, rotation signs and all other source projections.'}


def prepare(arm, output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not output.exists(),
             'Use a new output and CPU-only snapshot invocation.')
    template = read(TEMPLATE); check(template)
    _require(template['model_type'] == 'llama' and template['block_size'] == 32
             and template['steps_per_stage'] == 10000 and template['full_evaluation'],
             'Expected the reviewed native Llama 10000-step block32 recipe.')
    base = Path(template['base_dir']); meta = load_v6_meta(str(base))
    _require(meta['checkpoint_id'] == template['base_checkpoint_id'] == EXPECTED_IDS['llama'],
             'Fresh source must be the original CAT3577, not a later recovery.')
    source_acceptance = LLAMA_RESULT / 'remaining_prefix_block32_s500_full8/acceptance.json'
    acceptance = read(source_acceptance)
    _require(acceptance['checkpoint_id'] == meta['checkpoint_id'], 'Source acceptance UUID differs.')
    forecast = estimate(meta, BITS[arm], acceptance['bit_budget'])
    _require(forecast['estimated_total_bits'] == (17100439552 if arm == 'control' else 17432838144),
             'Integer budget estimate changed.')
    argv, normalized = recipe(template, output, BITS[arm])
    sources = dict(template['source_sha256'])
    sources.update({str(p): sha256(p) for p in (ROOT / 'experiments/llama_ov_bits_pair').glob('*.py')})
    evidence = dict(template['input_sha256'])
    reuse = RESULT / 'best1200_fresh_gate_up_k40_24'
    reuse_paths = [reuse / name for name in ('launch.json', 'train.exit', 'assembly.exit', 'run.exit',
                   'final_model/assembly_report.json', 'eval/acceptance.json')]
    _require(read(reuse / 'launch.json')['status'] == 'evaluated'
             and all(int((reuse / name).read_text()) == 0 for name in ('train.exit', 'assembly.exit', 'run.exit')),
             'Prior native K40 full-chain evidence is not complete.')
    extra = [source_acceptance, LLAMA_RESULT / 'mmlu_category_attribution/comparison.json',
             LLAMA_RESULT / 'mmlu_category_attribution/independent_review.json',
             LLAMA_RESULT / 'remaining_prefix_stage_mmlu/comparison.json', *reuse_paths]
    evidence.update({str(p): sha256(p) for p in extra})
    record = {'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'model_type': 'llama', 'arm': arm, 'categories': CATEGORIES, 'codebook_bits': BITS[arm],
        'architecture': template['architecture'], 'base_dir': str(base), 'base_checkpoint_id': meta['checkpoint_id'],
        'control_result_paths': template['control_result_paths'], 'block_size': 32, 'steps_per_stage': 10000,
        'full_evaluation': True, 'expected_user_linear_bits': forecast['estimated_total_bits'],
        'budget_estimate': forecast, 'training_argv': [sys.executable, 'tools/cat_train.py', *argv],
        'normalized_expected': normalized, 'source_sha256': sources, 'input_sha256': evidence,
        'protected_stats': template['protected_stats'], 'snapshot_import_paths': template['snapshot_import_paths'],
        'environment': template['environment'], 'timeout_seconds': 2700, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_ov_bits_pair.run',
                          '--launch-record', str(output / 'launch.json')],
        'native_template': {'path': str(TEMPLATE), 'live_launch_not_pinned': True,
            'reuse': 'Immutable source/input identities and original argv copied; no template weights consumed.'},
        'validation_reuse': {'K40': [str(p) for p in reuse_paths],
            'K56': 'Existing tiled packed K implementation, seven bytes per vector with no padding; no new algorithm.',
            'no_new_gpu_short': True},
        'scope': 'Two fresh original-HF categories in order v_proj,o_proj; replace 64/224 projections, retain other160; no KD/LoRA.',
        'comparison_boundary': 'Fresh K32/48 is the matched control. CAT3577 comparison also loses its prior OV recovery; do not attribute that difference solely to rate.',
        'resource_estimate': {'seconds_limit': 2700, 'estimate_not_measurement': 'Two categories ~5-15min training plus loading/assembly/full8 ~10min; K56 throughput not yet measured.'},
        'protocol': template['protocol']}
    _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    record['cuda_initialized'] = False; check(record)
    output.mkdir(parents=True); write_json(output / 'launch.json', record)
    print({'arm': arm, 'status': record['status'], 'source_pins': len(sources), 'input_pins': len(evidence),
           'estimated_bpw': forecast['estimated_bpw'], 'launch_sha256': sha256(output / 'launch.json')}, flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=tuple(BITS), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.arm, args.output_dir)
