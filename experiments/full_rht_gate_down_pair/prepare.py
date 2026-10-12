"""Bind completed same-rotation gate/down categories for native composition."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_down_pair.compare import arm as down_arm
from experiments.full_rht_gate_pair.common import EXPECTED_BITS, RESULT, ROOT, check, read
from experiments.full_rht_gate_pair.compare import arm as gate_arm
from experiments.longbench_v1.common import file_stat, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require


BUDGET_FIELDS = ('original_linear_weights', 'user_linear_bits', 'packed_linear_bits',
                 'code_padding_bits', 'components', 'outside_linear_bits', 'full_model_tensor_bytes',
                 'per_target', 'storage_aliases', 'compressed_target_count')


def prepare(model, block_size, output):
    output = Path(output).resolve()
    _require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
             and ROOT != Path('/home/star/VAELLM') and Path.cwd() == ROOT
             and not output.exists(), 'Use a fresh CPU-only snapshot preparation.')
    label = model + ('_full' if block_size == 0 else '_block32')
    gate_dir = RESULT / 'full_rht_gate_pair' / label
    down_dir = RESULT / 'full_rht_down_pair' / label
    gate, ga, gs, _, ge = gate_arm(gate_dir, block_size)
    down, da, ds, _, de = down_arm(down_dir, block_size)
    for key in ('architecture', 'base_dir', 'base_checkpoint_id', 'control_result_paths',
                'protected_stats', 'environment', 'protocol'):
        _require(gate[key] == down[key], 'Source category contract differs: ' + key)
    _require(gate['model_type'] == down['model_type'] == model
             and all(ga['bit_budget'][key] == da['bit_budget'][key] for key in BUDGET_FIELDS),
             'Same-model source storage differs.')
    evidence = {**ge, **de}
    for directory in (gate_dir.parent, down_dir.parent):
        path = directory / (model + '_comparison.json')
        pair = read(path)
        _require(pair['status'] == 'verified_pair' and pair['model_type'] == model
                 and pair['pair_checks']['protection_tensor_records_exact']
                 and pair['pair_checks']['rotation_sign_records_exact'],
                 'Require the completed matched category-pair comparison.')
        selected = pair['arms']['full' if block_size == 0 else 'block32']
        expected = gs if directory == gate_dir.parent else ds
        _require(selected['checkpoint_id'] == expected['checkpoint_id']
                 and selected['task_metrics'] == expected['task_metrics'],
                 'Category-pair report names a different deployed endpoint.')
        evidence[str(path)] = sha256(path)
    sources = {**gate['source_sha256'], **down['source_sha256']}
    for path, digest in gate['source_sha256'].items():
        _require(sources[path] == digest, 'Conflicting shared source pin: ' + path)
    for path in [ROOT / 'experiments/full_rht_gate_pair/compare.py',
                 ROOT / 'experiments/full_rht_down_pair/compare.py',
                 *(ROOT / 'experiments/full_rht_gate_down_pair').glob('*.py')]:
        sources[str(path)] = sha256(path)
    checkpoints = {'gate': gate_dir / 'final_model', 'down': down_dir / 'final_model'}
    stats = list(down['protected_stats'])
    for directory in checkpoints.values():
        meta = read(directory / 'checkpoint_meta.json')
        evidence[str(directory / 'config.json')] = sha256(directory / 'config.json')
        stats.append(file_stat(directory / meta['state_dict_file']))
    record = {'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'model_type': model, 'block_size': block_size, 'architecture': down['architecture'],
        'original_base_dir': down['base_dir'], 'original_base_checkpoint_id': down['base_checkpoint_id'],
        'base_dir': str(checkpoints['down']), 'base_checkpoint_id': ds['checkpoint_id'],
        'donor_dir': str(checkpoints['gate']), 'donor_checkpoint_id': gs['checkpoint_id'],
        'control_result_paths': down['control_result_paths'],
        'category_result_paths': {key: [str(directory.parent / (name + '.json')) for name in ('mmlu', 'seven')]
                                  for key, directory in checkpoints.items()},
        'expected_user_linear_bits': EXPECTED_BITS[model], 'source_bit_budget': da['bit_budget'],
        'source_sha256': sources, 'input_sha256': evidence, 'protected_stats': stats,
        'environment': down['environment'], 'protocol': down['protocol'],
        'timeout_seconds': 1200, 'termination_grace_seconds': 20,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.full_rht_gate_down_pair.run',
                          '--launch-record', str(output / 'launch.json')],
        'scope': 'Compose existing same-rotation full gate/down categories; five remaining categories and backbone from the original compressed base. No new training or LoRA.',
        'comparison_boundary': 'Full/full versus block32/block32 is the paired comparison; original base and individual-category endpoints are secondary. Category improvements are not presumed additive.',
        'cuda_initialized': False}
    check(record)
    _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True)
    write_json(output / 'launch.json', record)
    print({'model_type': model, 'block_size': block_size, 'status': record['status'],
           'launch_sha256': sha256(output / 'launch.json')}, flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('qwen3', 'llama'), required=True)
    parser.add_argument('--block-size', type=int, choices=(0, 32), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.model, args.block_size, args.output_dir)
