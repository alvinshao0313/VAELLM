"""CPU preparation of fresh Qwen O48/O56 native category compression."""
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

BITS = {'control': {'o_proj': 48}, 'candidate': {'o_proj': 56}}
CATEGORIES = ['o_proj']
TEMPLATE = RESULT / 'full_rht_gate_pair/qwen3_block32/launch.json'


def recipe(template, output, bits):
    argv = list(template['training_argv'][2:])
    changes = {'--output_dir': str(output / 'donor'), '--compression_categories': 'o_proj',
               '--codebook_bits': f'default=32,cat:o_proj={bits["o_proj"]}'}
    for key, value in changes.items():
        _require(argv.count(key) == 1, 'Ambiguous inherited argument: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, CATEGORIES)
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    _require(cat.target_layers == 'all' and cat.batch_size == 8192 and cat.after_category_mode == 'none'
             and cat.linear_group_size == 36 and training.bf16 and not training.fp16,
             'Native parser changed the approved category recipe.')
    _require(resolved['o_proj'].steps == 10000 and resolved['o_proj'].codebook_bits == bits['o_proj'],
             'Native O category override did not resolve.')
    return argv, normalized


def estimate(meta, bits, source_budget):
    specs = [s for s in meta['converted_modules'] if s['name'].endswith('.o_proj')]
    _require(len(specs) == 36, 'Require all 36 actual Qwen O projections.')
    code_delta = decoder_delta = 0
    for spec in specs:
        _require(spec['in_features'] == spec['out_features'] == 4096
                 and spec['compressed_in_features'] == 4032 and spec['compressed_out_features'] == 4096
                 and spec['transpose'] and spec['residual_stages'] == 2
                 and spec['stage_codebook_dims'] == [32, 32]
                 and spec['protected_input_indices'] == {'shape': [64], 'dtype': 'int64'}
                 and spec['weight_rotation']['block_size'] == 32, 'Unexpected source O geometry.')
        for packed, decoder in zip(spec['stage_vq_weights'], spec['stage_decoders'], strict=True):
            _require(packed['logical_shape'] == [516096, 1, 48] and packed['shape'] == [516096, 1, 6]
                     and decoder['in_dim'] == 48 and decoder['hidden_dim'] == 128
                     and decoder['out_dim'] == 32 and decoder['num_res_blocks'] == 0
                     and decoder['param_dtype'] == 'bfloat16', 'Unexpected actual packed/decoder geometry.')
            change = bits['o_proj'] - 48
            code_delta += 516096 * change; decoder_delta += 128 * change * 16
    source_bits = source_budget['user_linear_bits']; denominator = source_budget['original_linear_weights']
    _require(source_bits == 17065672704 and denominator == 6945767424, 'Original f927 storage changed.')
    total = source_bits + code_delta + decoder_delta
    _require(total <= denominator * 2.5, 'Estimated Qwen candidate exceeds 2.5 bits.')
    return {'basis': 'Actual f927 O geometry and measured source ledger; candidate storage not yet measured.',
        'source_bits': source_bits, 'original_linear_weights': denominator, 'o_layers': 36,
        'old_K': 48, 'new_K': bits['o_proj'], 'code_delta_bits': code_delta,
        'decoder_delta_bits': decoder_delta, 'estimated_total_bits': total,
        'estimated_bpw': total / denominator, 'remaining_bits_to_2_5': int(denominator * 2.5 - total),
        'unchanged_components': 'Protection, indices/scales, signs and all other216 projections; no added FP32 storage.'}


def prepare(arm, output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not output.exists(),
             'Use a new output and CPU-only snapshot invocation.')
    template = read(TEMPLATE); check(template)
    _require(template['model_type'] == 'qwen3' and template['block_size'] == 32
             and template['steps_per_stage'] == 10000 and template['full_evaluation']
             and template['status'] == 'completed' and template['exit_code'] == 0,
             'Expected the completed reviewed Qwen 10000-step block32 native recipe.')
    base = Path(template['base_dir']); meta = load_v6_meta(str(base))
    _require(meta['checkpoint_id'] == template['base_checkpoint_id'] == EXPECTED_IDS['qwen3'],
             'Fresh source must be f927, not a later recovery.')
    source_acceptance = RESULT / 'nine_task_baseline_1200/acceptance.json'; acceptance = read(source_acceptance)
    _require(acceptance['checkpoint_id'] == meta['checkpoint_id'], 'Source ledger UUID differs.')
    forecast = estimate(meta, BITS[arm], acceptance['bit_budget'])
    _require(forecast['estimated_total_bits'] == (17065672704 if arm == 'control' else 17364123648),
             'Integer budget estimate changed.')
    argv, normalized = recipe(template, output, BITS[arm])
    sources = dict(template['source_sha256'])
    sources.update({str(p): sha256(p) for p in (ROOT / 'experiments/qwen_o_bits_pair').glob('*.py')})
    evidence = dict(template['input_sha256'])
    reuse = RESULT / 'best1200_fresh_gate_up_k40_24'
    reuse_paths = [reuse / name for name in ('launch.json', 'train.exit', 'assembly.exit', 'run.exit',
                   'final_model/assembly_report.json', 'eval/acceptance.json')]
    _require(read(reuse / 'launch.json')['status'] == 'evaluated'
             and all(int((reuse / name).read_text()) == 0 for name in ('train.exit', 'assembly.exit', 'run.exit')),
             'Existing complete native K40 evidence is not verified.')
    lb = ORIGINAL / 'result/qwen3_8b_longbench_20261004'
    evidence.update({str(p): sha256(p) for p in [source_acceptance, TEMPLATE,
        Path(template['output_dir']) / 'audit.json', lb / 'comparison.json',
        lb / 'retrieval_attribution/ov/report.json', *reuse_paths]})
    record = {'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'model_type': 'qwen3', 'arm': arm, 'categories': CATEGORIES, 'codebook_bits': BITS[arm],
        'architecture': template['architecture'], 'base_dir': str(base), 'base_checkpoint_id': meta['checkpoint_id'],
        'control_result_paths': template['control_result_paths'], 'block_size': 32, 'steps_per_stage': 10000,
        'full_evaluation': True, 'expected_user_linear_bits': forecast['estimated_total_bits'],
        'budget_estimate': forecast, 'training_argv': [sys.executable, 'tools/cat_train.py', *argv],
        'normalized_expected': normalized, 'source_sha256': sources, 'input_sha256': evidence,
        'protected_stats': template['protected_stats'], 'snapshot_import_paths': template['snapshot_import_paths'],
        'environment': template['environment'], 'timeout_seconds': 2700, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.qwen_o_bits_pair.run',
                          '--launch-record', str(output / 'launch.json')],
        'validation_reuse': {'K40': [str(p) for p in reuse_paths],
            'K56': 'Existing generic seven-byte packed code pathway, also used by the separately scheduled native Llama OV experiment.',
            'no_new_gpu_short': True},
        'scope': 'Fresh original-HF o_proj only; replace36/252 and preserve other216; no KD/LoRA or FP32 deployment parameters.',
        'comparison_boundary': 'Fresh O48 is primary control; f927 is secondary. Fresh reconstruction replaces prior O recovery. Joint OV retrieval diagnosis does not identify O alone.',
        'resource_estimate': {'seconds_limit': 2700, 'estimate_not_measurement': 'One native category plus assembly and full8; K56 timing not yet measured.'},
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
