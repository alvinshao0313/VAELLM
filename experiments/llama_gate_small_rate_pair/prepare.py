"""CPU-only preparation of the K34 two-step native path; no formal arm is ready."""
import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.llama_gate_small_rate_pair.common import (ROOT, ORIGINAL, RESULT, BASE, BASE_ID,
    DENOMINATOR, GROUPS, LOGICAL_BITS, PACKED_BITS, check, read, recipe, _require)
from experiments.longbench_v1.common import file_stat, sha256, write_json
from train_utils.checkpoint_v6 import _build_decoder_from_spec


def prepare(output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and not output.exists()
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
             'Require a fresh CPU-only snapshot preparation.')
    template_dir = RESULT / 'full_rht_gate_pair/llama_block32'
    template = read(template_dir / 'launch.json')
    _require(template['status'] == 'completed' and template['exit_code'] == 0
             and all((template_dir / (s + '.exit')).read_text().strip() == '0'
                     for s in ('train', 'assembly', 'mmlu', 'seven', 'dispatcher', 'outer')),
             'Require the completed native fresh gate reference.')
    meta = read(BASE / 'checkpoint_meta.json'); review = read(BASE.parent / 'review.json')
    budget = read(BASE / 'assembly_report.json')['bit_budget']
    _require(meta['checkpoint_id'] == review['checkpoint_id'] == BASE_ID and review['status'] == 'verified'
             and meta['base_model_path'] == template['architecture']['base_model_path']
             and budget['user_linear_bits'] == budget['packed_linear_bits'] == 17095180288
             and budget['original_linear_weights'] == DENOMINATOR and len(meta['converted_modules']) == 224,
             'Wrong e767 source or native storage.')
    gates = [s for s in meta['converted_modules'] if s['name'].endswith('.gate_proj')]
    _require(len(gates) == 32, 'Require every gate in the source.')
    first = gates[0]['stage_decoders'][0]
    for spec in gates:
        _require(not spec['transpose'] and [spec['compressed_out_features'], spec['compressed_in_features']] == [14336, 4032]
                 and spec['stage_codebook_dims'] == [32, 32] and spec['stage_decoders'] == [first, first]
                 and spec['protected_input_indices'] == {'shape': [64], 'dtype': 'int64'}
                 and spec['weight_rotation']['block_size'] == 32
                 and all(p['logical_shape'] == [GROUPS, 1, 32] and p['shape'] == [GROUPS, 1, 4]
                         for p in spec['stage_vq_weights']), 'Unexpected source gate geometry.')
    with torch.random.fork_rng(devices=[]):
        old = sum(p.numel() for p in _build_decoder_from_spec(first).parameters())
        new = sum(p.numel() for p in _build_decoder_from_spec({**first, 'in_dim': 34}).parameters())
    components = deepcopy(budget['components'])
    components['code']['budget_bits'] += GROUPS * 2 * 64
    components['code']['storage_bits'] += GROUPS * 8 * 64
    for field in ('budget_bits', 'storage_bits'):
        components['decoder'][field] += (new - old) * 64 * 16
    _require(new - old == 256 and sum(v['budget_bits'] for v in components.values()) == LOGICAL_BITS
             and sum(v['storage_bits'] for v in components.values()) == PACKED_BITS
             and LOGICAL_BITS / DENOMINATOR < 2.5, 'Native constructor/budget estimate differs.')
    argv, normalized = recipe(template, output, bits=34, steps=2)
    drafts = {}
    for arm, k in (('control', 32), ('candidate', 34)):
        command, value = recipe(template, output.parent / arm, bits=k, steps=10000)
        drafts[arm] = {'status': 'recipe_only_pending_k34_validation', 'training_argv': command,
                       'normalized_expected': value, 'output_created': False}
    sources = dict(template['source_sha256']); inputs = dict(template['input_sha256'])
    for path in [*sorted(Path(__file__).parent.glob('*.py')), ROOT / 'experiments/llama_gate_up_rate_pair/audit.py',
                 ROOT / 'experiments/llama_gate_up_rate_pair/common.py', ROOT / 'experiments/full_rht_gate_pair/common.py']:
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
        _require(sha256(ORIGINAL / path.relative_to(ROOT)) == sha256(path), 'Public/snapshot differs.')
    for path in (template_dir / 'launch.json', template_dir / 'audit.json', BASE / 'checkpoint_meta.json',
                 BASE / 'config.json', BASE / 'assembly_report.json', BASE.parent / 'review.json',
                 ROOT / 'tests/test_fused_multistage_decoder_parity.py'):
        inputs[str(path)] = sha256(path)
    hf = Path(meta['base_model_path']); dense = read(hf / 'model.safetensors.index.json')['metadata']['total_size']
    gate_storage = sum(budget['per_target'][s['name']]['storage_bits'] for s in gates) + PACKED_BITS - budget['packed_linear_bits']
    donor_bytes = int(dense - 32 * 14336 * 4096 * 2 + gate_storage // 8)
    final_bytes = budget['full_model_tensor_bytes'] + (PACKED_BITS - budget['packed_linear_bits']) // 8
    provision = 2 * donor_bytes + final_bytes + 2 * 1024**3
    record = {'schema': 'llama_gate_k34_native_short_v1', 'status': 'prepared_unbound', 'gpu': None,
        'queued': False, 'automatic_retry': False, 'cwd': str(ROOT), 'python_executable': sys.executable,
        'output_dir': str(output), 'model_type': 'llama', 'base_dir': str(BASE), 'base_checkpoint_id': BASE_ID,
        'architecture': template['architecture'], 'categories': ['gate_proj'], 'codebook_bits': 34,
        'steps_per_stage': 2, 'full_evaluation': False, 'training_argv': argv, 'normalized_expected': normalized,
        'source_sha256': sources, 'input_sha256': inputs,
        'protected_stats': [*template['protected_stats'], file_stat(BASE / meta['state_dict_file'])],
        'environment': template['environment'], 'expected_budget_components': components,
        'budget_estimate': {'user_linear_bits': LOGICAL_BITS, 'user_linear_bpw': LOGICAL_BITS / DENOMINATOR,
            'packed_linear_bits': PACKED_BITS, 'packed_linear_bpw': PACKED_BITS / DENOMINATOR,
            'code_padding_bits': PACKED_BITS - LOGICAL_BITS, 'original_linear_weights': DENOMINATOR,
            'decoder_parameters_per_stage': [old, new], 'scope': 'Estimate; actual assembled tensor audit is required.'},
        'timeout_seconds': 1200, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_gate_small_rate_pair.run',
                          '--launch-record', str(output / 'launch.json')],
        'formal_recipes_not_ready': drafts,
        'validation_scope': 'Native all32 gate K34 2steps/stage with original512x256 calibration; real donor/e767 strict2979 assembly, logical/physical accounting, padding replay, BF16 group8 and four-token forward. No quality evaluation or formal readiness.',
        'comparison_boundary': 'Future primary fresh K34 minus fresh K32, same e767192 complement and HF gate target. Original e767 is secondary. Different K changes initialization/random consumption. No addition of independent gains, no up reduction, no INT4, no external LoRA.',
        'experience': 'K34 previously had budget calculations only. Completed K40/24 gate/up allocation was negative; this setting increases gate slightly without reducing up. Existing K16/24/32/48/64 GPU parity does not establish K34.',
        'resource_estimate': {'single_short_disk_provision_bytes': provision,
            'checkpoint_tensor_bytes': {'after_gate': donor_bytes, 'donor_final': donor_bytes, 'assembled': final_bytes},
            'filesystem_free_at_preparation': shutil.disk_usage(output.parent.parent).free,
            'seconds_limit': 1200, 'boundary': 'All32 and512x256 preserved; short-specific elapsed/peak must be measured.'},
        'cuda_initialized': False}
    _require(not torch.cuda.is_initialized() and shutil.disk_usage(output.parent.parent).free > provision,
             'CUDA initialized or insufficient output disk.')
    check(record); output.mkdir(parents=True)
    write_json(output / 'launch.prepared.json', record)
    print({'prepared': str(output / 'launch.prepared.json'), 'sha256': sha256(output / 'launch.prepared.json'),
           'source': len(sources), 'input': len(inputs), 'stat': len(record['protected_stats']),
           'budget': record['budget_estimate'], 'disk_provision_bytes': provision, 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)
