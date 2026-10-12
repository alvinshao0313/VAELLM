"""Prepare the two formal arms only after the real K34 native short has passed."""
import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.llama_gate_small_rate_pair.common import (
    ROOT, ORIGINAL, RESULT, BASE, BASE_ID, DENOMINATOR, LOGICAL_BITS, PACKED_BITS, check, read, recipe, _require)
from experiments.longbench_v1.common import sha256, write_json


def prepare(output):
    output = Path(output).resolve(); short = output / 'validate_k34_s2'
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
             and not torch.cuda.is_initialized() and not any((output / a).exists() for a in ('control', 'candidate')),
             'Require CPU-only preparation of two new formal outputs.')
    previous = read(short / 'launch.json'); proof = read(short / 'verification.json'); check(previous)
    _require(previous['status'] == 'completed' and previous['exit_code'] == 0
             and previous['schema'] == 'llama_gate_k34_native_short_v1' and previous['steps_per_stage'] == 2
             and previous['verification_sha256'] == sha256(short / 'verification.json')
             and all((short / (s + '.exit')).read_text().strip() == '0'
                     for s in ('train', 'assembly', 'audit', 'dispatcher', 'outer')),
             'The real native short must naturally complete before formal preparation.')
    _require(proof['status'] == 'verified_native_short' and proof['base_checkpoint_id'] == BASE_ID
             and proof['zero_padding_and_pack_replay_banks'] == 64 and len(proof['actual_trained_kernel_parity']) == 8
             and proof['source_union_save_fresh_tensors'] == 2979 and proof['integration_forward']['finite']
             and proof['native_group8_prewarm']['warmed'] == 224
             and proof['bit_budget']['user_linear_bits'] == LOGICAL_BITS
             and proof['bit_budget']['packed_linear_bits'] == PACKED_BITS, 'Incomplete K34 native deployment proof.')
    template = read(RESULT / 'full_rht_gate_pair/llama_block32/launch.json')
    base_budget = read(BASE / 'assembly_report.json')['bit_budget']
    _require(read(BASE / 'checkpoint_meta.json')['checkpoint_id'] == BASE_ID, 'Wrong original source.')
    sources = dict(previous['source_sha256']); inputs = dict(previous['input_sha256'])
    for path in (Path(__file__), Path(__file__).with_name('formal_run.py'), ROOT / 'experiments/full_rht_gate_pair/compare.py'):
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
        _require(sha256(ORIGINAL / path.relative_to(ROOT)) == sha256(path), 'Public/snapshot differs.')
    for name in ('launch.prepared.json', 'launch.json', 'verification.json', 'train.exit', 'assembly.exit', 'audit.exit', 'dispatcher.exit', 'outer.exit'):
        inputs[str(short / name)] = sha256(short / name)
    provision = 2 * previous['resource_estimate']['single_short_disk_provision_bytes']
    _require(shutil.disk_usage(output).free > provision, 'Insufficient disk for the two formal arms.')
    records = {}
    for arm, k in (('control', 32), ('candidate', 34)):
        directory = output / arm; argv, normalized = recipe(template, directory, bits=k, steps=10000)
        draft = previous['formal_recipes_not_ready'][arm]
        _require(argv == draft['training_argv'] and normalized == draft['normalized_expected'], 'Formal recipe changed after short preparation.')
        budget = base_budget if k == 32 else proof['bit_budget']
        record = {'schema': 'llama_gate_small_rate_formal_v1', 'status': 'prepared_unbound', 'gpu': None,
            'queued': False, 'automatic_retry': False, 'cwd': str(ROOT), 'python_executable': sys.executable,
            'output_dir': str(directory), 'arm': arm, 'codebook_bits': k, 'steps_per_stage': 10000,
            'categories': ['gate_proj'], 'model_type': 'llama', 'full_evaluation': True,
            'base_dir': str(BASE), 'base_checkpoint_id': BASE_ID, 'architecture': previous['architecture'],
            'control_result_paths': [str(BASE.parent / (n + '.json')) for n in ('mmlu', 'seven')],
            'training_argv': argv, 'normalized_expected': normalized, 'source_sha256': sources, 'input_sha256': inputs,
            'protected_stats': previous['protected_stats'], 'environment': previous['environment'],
            'protocol': template['protocol'], 'expected_budget_components': deepcopy(budget['components']),
            'expected_user_linear_bits': budget['user_linear_bits'], 'expected_packed_linear_bits': budget['packed_linear_bits'],
            'expected_padding_bits': budget['code_padding_bits'], 'original_linear_weights': DENOMINATOR,
            'timeout_seconds': 2700, 'termination_grace_seconds': 20,
            'launcher_argv': [sys.executable, '-u', '-m', 'experiments.llama_gate_small_rate_pair.formal_run',
                              '--launch-record', str(directory / 'launch.json')],
            'validation_reuse': {'short_dir': str(short), 'verification_sha256': sha256(short / 'verification.json'),
                'no_short_weight_consumption': True, 'scope': 'K34 padding and BF16 packed/unpacked/group8 verified; do not repeat this kernel check in formal arms.'},
            'seed_boundary': 'Original native seed31, no extra category reset. K changes encoder/BSQ/decoder initialization and random consumption; same seed is not the same trajectory.',
            'comparison_boundary': previous['comparison_boundary'],
            'resource_estimate': {'disk_provision_bytes': previous['resource_estimate']['single_short_disk_provision_bytes'],
                'historical_full_gate_seconds': template['finished_unix'] - template['started_unix'],
                'seconds_limit': 2700, 'boundary': 'Prior full32 gate native training/assembly/full8; K34 full training time has not been measured.'},
            'cuda_initialized': False}
        check(record); records[arm] = record
    _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    for arm, record in records.items():
        directory = output / arm; directory.mkdir(); write_json(directory / 'launch.prepared.json', record)
        print({'arm': arm, 'prepared_sha256': sha256(directory / 'launch.prepared.json'),
               'source': len(sources), 'input': len(inputs), 'stat': len(record['protected_stats'])}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    prepare(parser.parse_args().output_dir)
