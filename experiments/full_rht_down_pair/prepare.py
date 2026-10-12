"""CPU preparation of the fixed native Qwen/Llama down block32/full pairs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_gate_pair.common import EXPECTED_BITS, EXPECTED_IDS, ORIGINAL, RESULT, ROOT, check, read
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require
from train_utils.cat_category_runtime import resolve_category_runtime_configs
from train_utils.cat_runtime_adapter import parse_cat_runtime_args
from train_utils.cat_train_runtime import build_cat_runtime_parameter_payload
from train_utils.checkpoint_v6 import load_v6_meta


def recipe(template, output, block_size):
    argv = list(template['training_argv'][2:])
    changes = {'--output_dir': str(output / 'donor'), '--compression_categories': 'down_proj',
               '--codebook_bits': 'default=32,cat:down_proj=48',
               '--channel_protect_count': 'default=256', '--weight_rotation_block_size': str(block_size)}
    for key, value in changes.items():
        _require(argv.count(key) == 1, 'Ambiguous inherited argument: ' + key)
        argv[argv.index(key) + 1] = value
    cat, _, training, vae = parse_cat_runtime_args(argv)
    resolved = resolve_category_runtime_configs(cat, vae, ['down_proj'])
    normalized = build_cat_runtime_parameter_payload(cat_args=cat, vae_args=vae,
        training_args=training, resolved_category_cfgs=resolved)
    _require(cat.target_layers == 'all' and cat.batch_size == 8192 and cat.after_category_mode == 'none'
             and cat.linear_group_size == template['architecture']['num_hidden_layers']
             and training.bf16 and not training.fp16
             and not cat.channel_refresh_after_category and cat.channel_axis == 'input',
             'Native parser changed the approved down category recipe.')
    _require(resolved['down_proj'].steps == 10000 and resolved['down_proj'].codebook_bits == 48,
             'Native down steps/K override did not resolve.')
    return argv, normalized


def source_geometry(meta, architecture):
    layers = architecture['num_hidden_layers']
    specs = [s for s in meta['converted_modules'] if s['name'].endswith('.down_proj')]
    _require(len(specs) == layers, 'Incomplete native source down category.')
    for spec in specs:
        shape = architecture['projection_shapes'][spec['name']]
        groups = shape[0] * (shape[1] - 256) // 32
        _require([spec['out_features'], spec['in_features']] == shape
                 and spec['compressed_in_features'] == shape[1] - 256
                 and spec['compressed_out_features'] == shape[0] and spec['transpose']
                 and spec['codebook_dim'] == 32 and spec['residual_stages'] == 2
                 and spec['stage_codebook_dims'] == [32, 32] and spec['parallel_parts'] == 1
                 and spec['protected_input_indices'] == {'shape': [256], 'dtype': 'int64'}
                 and spec['protected_input_qvalues'] == {'shape': [256, shape[0]], 'dtype': 'uint8'}
                 and spec['protected_input_scales'] == {'shape': [256], 'dtype': 'bfloat16'}
                 and spec['weight_rotation']['block_size'] == 32, 'Unexpected source down geometry.')
        for packed, decoder in zip(spec['stage_vq_weights'], spec['stage_decoders'], strict=True):
            _require(packed['logical_shape'] == [groups, 1, 48] and packed['shape'] == [groups, 1, 6]
                     and packed['storage_format'] == 'bitpack_u8' and decoder['in_dim'] == 48
                     and decoder['out_dim'] == 32 and decoder['hidden_dim'] == 128
                     and decoder['num_res_blocks'] == 0 and decoder['param_dtype'] == 'bfloat16',
                     'Source down packed-code/decoder storage differs from the approved recipe.')
    return {'down_count': layers, 'preserved_projection_count': architecture['projection_count'] - layers,
            'source_down_metadata_same_geometry_and_storage': True,
            'budget_boundary': 'Same code/decoder/protection/sign element counts as the measured source; actual assembled storage must still pass the runtime integer audit.'}


def prepare(model, block_size, output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and model in EXPECTED_IDS
             and block_size in (0, 32) and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
             and not output.exists(), 'Use a new CPU-only snapshot preparation for a fixed pair arm.')
    template_path = RESULT / f'full_rht_gate_pair/{model}_block32/launch.json'
    template = read(template_path); check(template)
    _require(template['status'] == 'completed' and template['exit_code'] == 0
             and template['model_type'] == model and template['block_size'] == 32
             and template['steps_per_stage'] == 10000 and template['full_evaluation'],
             'Require the completed native 10000-stage recipe for this actual model.')
    base = Path(template['base_dir']); meta = load_v6_meta(str(base))
    _require(meta['checkpoint_id'] == template['base_checkpoint_id'] == EXPECTED_IDS[model],
             'Use the original f927/3577 assembly base, not a newer recovered endpoint.')
    geometry = source_geometry(meta, template['architecture'])
    validation = RESULT / 'full_rht_gate_pair/native_validation_r1/verification.json'
    proof = read(validation); native = proof['down_projections'][model]
    _require(proof['status'] == 'verified' and proof['source_root'] == str(ROOT)
             and int((validation.parent / 'run.exit').read_text()) == 0
             and native['status'] == 'verified' and native['steps_per_stage'] == 2
             and native['batch_size'] == 8192 and native['seed'] == 31 and native['transpose']
             and native['protected_count'] == 256
             and native['input_factor'] == (188 if model == 'qwen3' else 220),
             'Existing native full-Hadamard down proof does not cover this geometry.')
    check_hashes(proof['source_sha256'])
    argv, normalized = recipe(template, output, block_size)
    sources = dict(template['source_sha256'])
    sources.update({str(p): sha256(p) for p in (ROOT / 'experiments/full_rht_down_pair').glob('*.py')})
    evidence = dict(template['input_sha256'])
    local_root = RESULT / 'rotation_factor_probe'
    evidence.update({str(p): sha256(p) for p in [template_path, validation, validation.parent / 'run.exit',
        Path(template['output_dir']) / 'audit.json', local_root / 'screen_s2000/screen.json',
        local_root / 'middle_screen_s2000/screen.json']})
    stages = template['stages']
    record = {'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'model_type': model, 'categories': ['down_proj'], 'codebook_bits': {'down_proj': 48},
        'architecture': template['architecture'], 'base_dir': str(base), 'base_checkpoint_id': meta['checkpoint_id'],
        'control_result_paths': template['control_result_paths'], 'block_size': block_size,
        'steps_per_stage': 10000, 'full_evaluation': True, 'expected_user_linear_bits': EXPECTED_BITS[model],
        'budget_estimate': {**geometry, 'estimated_total_bits': EXPECTED_BITS[model],
            'original_linear_weights': template['architecture']['original_linear_weights'],
            'estimated_bpw': EXPECTED_BITS[model] / template['architecture']['original_linear_weights']},
        'training_argv': [sys.executable, 'tools/cat_train.py', *argv], 'normalized_expected': normalized,
        'source_sha256': sources, 'input_sha256': evidence, 'protected_stats': template['protected_stats'],
        'snapshot_import_paths': template['snapshot_import_paths'], 'environment': template['environment'],
        'timeout_seconds': 2700, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.full_rht_down_pair.run',
                          '--launch-record', str(output / 'launch.json')],
        'validation_reuse': {'native_down_proof': str(validation), 'no_new_gpu_short': True,
            'boundary': 'Existing two-update full-Hadamard down training/decode proof plus native full-category CAT and assembly evidence; short weights never seed this run.'},
        'scope': 'Fresh original-HF down category only; fixed K48/D32/S2/H128R0/input256 INT8. No KD/LoRA or extra FP32 storage.',
        'comparison_boundary': 'Full minus fresh block32 is primary; original f927/3577 is secondary. Same-seed protection is selected afresh from the original HF and must match across the pair; original checkpoint indices are not imposed.',
        'resource_estimate': {'seconds_limit': 2700, 'estimated_full_chain_seconds': [1320, 2100],
            'same_model_gate_observed_seconds': {key: value['finished_unix'] - value['started_unix'] for key, value in stages.items()},
            'uncertainty': 'Down matrix elements are close to gate, but K48 and full-rotation execution cost for the whole down category are not yet measured.'},
        'protocol': template['protocol']}
    _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    record['cuda_initialized'] = False; check(record)
    output.mkdir(parents=True); write_json(output / 'launch.json', record)
    print({'model': model, 'block_size': block_size, 'status': record['status'],
           'source_pins': len(sources), 'input_pins': len(evidence),
           'estimated_bpw': record['budget_estimate']['estimated_bpw'],
           'launch_sha256': sha256(output / 'launch.json')}, flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('qwen3', 'llama'), required=True)
    parser.add_argument('--block-size', type=int, choices=(0, 32), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.model, args.block_size, args.output_dir)
