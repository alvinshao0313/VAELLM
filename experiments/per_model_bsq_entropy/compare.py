"""CPU comparison of a completed per-Linear entropy trial and its global control."""
from __future__ import annotations

import argparse
import copy
import inspect
import math
import os
from pathlib import Path

import torch

from experiments.full_rht_gate_pair.common import ORIGINAL, ROOT, RESULT, read
from experiments.full_rht_gate_pair.compare import arm, delta, score
from experiments.longbench_v1.common import sha256, write_json
from experiments.per_model_bsq_entropy.train import OBSERVED_CALLS
from experiments.qwen3_w25.assemble_categories import _require


ALGORITHM = 'isolated_mean_per_linear_marginal_entropy_v1'
TRAINER_SHA256 = 'b2542824f98e483ba924b9d028a54b694556bfcee5cb1642e4506af404ecbf42'


def normalized_recipe(record):
    value = copy.deepcopy(record['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        _require(value[section]['output_dir'] == str(Path(record['output_dir']) / 'donor'),
                 'Unexpected normalized output directory: ' + section)
        value[section]['output_dir'] = '<paired-output>'
    return value


def telemetry(directory, record):
    import json

    rows = []
    prefix = '[per_model_bsq_entropy] '
    for line in (Path(directory) / 'train.log').read_text().splitlines():
        if line.startswith(prefix + '{'):
            rows.append(json.loads(line[len(prefix):]))
    _require(rows, 'No actual per-model entropy observations in the native training log.')
    groups = {}
    entropy_fields = ('sample_entropy', 'global_marginal_entropy', 'mean_model_marginal_entropy',
        'global_minus_mean_model_entropy', 'model_marginal_entropy_min', 'model_marginal_entropy_max',
        'model_bit_probability_bias_abs_mean', 'model_bit_probability_bias_abs_max')
    settings = record['normalized_expected']['vae_args']
    for row in rows:
        shape = row['latent_shape']
        _require(len(shape) == 4 and 0 < shape[0] <= 8192
                 and shape[1:] == [record['architecture']['num_hidden_layers'], 1, 32]
                 and row['latent_dtype'] == 'torch.float32' and str(row['rank']) == '0',
                 'Unexpected native CAT latent shape, quantizer dtype or rank.')
        _require(all(math.isfinite(row[key]) for key in entropy_fields), 'Nonfinite entropy observation.')
        for key in ('entropy_loss_weight', 'inv_temperature', 'gamma0', 'gamma', 'zeta'):
            _require(row[key] == settings[key], 'Observed BSQ setting differs from the actual recipe: ' + key)
        _require(math.isclose(row['global_marginal_entropy'] - row['mean_model_marginal_entropy'],
                             row['global_minus_mean_model_entropy'], rel_tol=0, abs_tol=1e-5),
                 'Entropy gap does not match its observed components.')
        key = (str(row['rank']), row['instance'])
        groups.setdefault(key, []).append(row)
    _require(len(groups) == 2, 'Expected two distinct BSQ instances for this two-stage, one-group CAT run.')
    instances = []
    for (rank, instance), observations in sorted(groups.items()):
        _require([row['call'] for row in observations] == list(OBSERVED_CALLS),
                 'Missing, duplicate or reordered scheduled entropy observations.')
        instances.append({'rank': rank, 'instance': instance, 'observations': observations,
            'observed_ranges': {key: {'min': min(row[key] for row in observations),
                                      'max': max(row[key] for row in observations)} for key in entropy_fields}})
    return {'observation_count': len(rows), 'instances': instances,
        'boundary': 'Calls count BSQ.soft_entropy_loss invocations per instance, not optimizer steps. '
                    'No stage-number assignment or full-trajectory average is inferred. '
                    'Synthetic CPU evidence is not substituted for these real latent observations.'}


def compare(control_directory, candidate_directory, output):
    output = Path(output).resolve()
    _require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'Use CPU only.')
    _require(not output.exists() and not output.is_symlink(), 'Do not overwrite a comparison.')
    control_directory = Path(control_directory).resolve()
    candidate_directory = Path(candidate_directory).resolve()
    control, ca, cs, cr, ce = arm(control_directory, 32)
    candidate, na, ns, nr, ne = arm(candidate_directory, 32)
    _require(candidate['entropy_algorithm'] == ALGORITHM and 'entropy_algorithm' not in control,
             'Require the isolated per-model method and the original native global control.')
    _require(Path(candidate['historical_control_dir']).resolve() == control_directory
             and candidate['historical_control_checkpoint_id'] == cs['checkpoint_id']
             and Path(candidate['historical_control_audit_path']).resolve() == control_directory / 'audit.json'
             and list(map(Path, candidate['historical_control_result_paths']))
             == [control_directory / 'mmlu.json', control_directory / 'seven.json'],
             'Historical control binding differs from the requested comparison.')
    for key in ('model_type', 'architecture', 'base_dir', 'base_checkpoint_id', 'control_result_paths',
                'cwd', 'protected_stats', 'environment', 'protocol', 'block_size', 'steps_per_stage',
                'full_evaluation', 'expected_user_linear_bits'):
        _require(candidate[key] == control[key], 'Matched control differs: ' + key)
    _require(normalized_recipe(candidate) == normalized_recipe(control),
             'Native configuration differs beyond the output directory.')
    original_argv = control['training_argv']
    _require(len(original_argv) > 2 and original_argv[1] == 'tools/cat_train.py',
             'Historical control did not use the native CAT entry.')
    expected_argv = list(original_argv[2:])
    _require(expected_argv.count('--output_dir') == 1, 'Unexpected original output argument.')
    expected_argv[expected_argv.index('--output_dir') + 1] = str(candidate_directory / 'donor')
    _require(candidate['training_argv'] == [candidate['python_executable'], '-u', '-m',
             'experiments.per_model_bsq_entropy.train', *expected_argv], 'Candidate native argv changed.')
    for path, digest in control['source_sha256'].items():
        _require(candidate['source_sha256'].get(path) == digest, 'Native source pin changed: ' + path)
    allowed_extra = {str(ROOT / 'experiments/full_rht_gate_pair/compare.py'),
        *(str(ROOT / 'experiments/per_model_bsq_entropy' / name) for name in ('train.py', 'prepare.py', 'compare.py'))}
    _require(set(candidate['source_sha256']) - set(control['source_sha256']) <= allowed_extra,
             'Unexpected native source additions.')
    for path, digest in control['input_sha256'].items():
        _require(candidate['input_sha256'].get(path) == digest, 'Native input pin changed: ' + path)
    proof_path = RESULT / 'per_model_bsq_entropy/cpu_validation.json'; proof = read(proof_path)
    trainer = ROOT / 'experiments/per_model_bsq_entropy/train.py'
    _require(candidate['input_sha256'].get(str(proof_path)) == sha256(proof_path)
             and proof['status'] == 'verified' and proof['cuda_initialized'] is False
             and proof['telemetry_rng_loss_gradient_exact']
             and proof['source_sha256'][str(ORIGINAL / 'experiments/per_model_bsq_entropy/train.py')]
             == candidate['source_sha256'][str(trainer)] == sha256(trainer) == TRAINER_SHA256
             and proof['source_sha256'][str(ORIGINAL / 'litebsq/bsq.py')]
             == candidate['source_sha256'][str(ROOT / 'litebsq/bsq.py')],
             'CPU proof does not cover the exact isolated trainer and unchanged BSQ source.')
    _require(ca['protection_tensor_records'] == na['protection_tensor_records']
             and ca['rotation_sign_records'] == na['rotation_sign_records'], 'Protection/sign pairing changed.')
    budget_fields = ('original_linear_weights', 'user_linear_bits', 'packed_linear_bits',
                     'components', 'code_padding_bits')
    for key in budget_fields:
        _require(ca['bit_budget'][key] == na['bit_budget'][key], 'Storage changed: ' + key)
    base = Path(candidate['base_dir'])
    bs, br = score([Path(path) for path in candidate['control_result_paths']], candidate['architecture'], base)
    _require(bs['checkpoint_id'] == candidate['base_checkpoint_id'], 'Original compressed base identity changed.')
    for label, rows in (('candidate', nr), ('base', br)):
        for current, reference in zip(rows, cr, strict=True):
            for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                _require(current[key] == reference[key], label + ' raw task protocol changed: ' + key)
    observations = telemetry(candidate_directory, candidate)
    duration = {label: {'stages_seconds': {name: stage['finished_unix'] - stage['started_unix']
                   for name, stage in record['stages'].items()},
                   'dispatcher_wall_seconds': record['finished_unix'] - record['started_unix']}
                for label, record in (('historical_global', control), ('mean_per_linear', candidate))}
    report = {'status': 'verified_pair', 'model_type': candidate['model_type'],
        'primary_comparison': 'mean_per_linear_minus_historical_global', 'entropy_algorithm': ALGORITHM,
        'candidate_vs_control': delta(ns, cs),
        'secondary_vs_original_compressed_base': {'historical_global': delta(cs, bs), 'mean_per_linear': delta(ns, bs)},
        'arms': {'historical_global': cs, 'mean_per_linear': ns, 'base': bs},
        'pair_checks': {'same_normalized_recipe_except_output': True, 'same_native_source_and_inputs': True,
            'isolated_trainer_sha256': TRAINER_SHA256, 'cpu_proof_sha256': sha256(proof_path),
            'actual_steps_per_stage': 10000, 'stages': 2, 'complete_gate_count': na['gate_count'],
            'protection_and_rotation_signs_exact': True, 'actual_integer_budget_equal': True,
            'same_full_raw_protocol': True, 'leaf_count': 64, 'sample_count': 24742,
            'mmlu_subjects': 57, 'mmlu_samples': 14042, 'batch_sizes': {'mmlu': 64, 'seven': 32},
            'num_fewshot': 0, 'limit': None},
        'bit_budget': {key: na['bit_budget'][key] for key in (*budget_fields, 'user_linear_bpw')},
        'native_latent_telemetry': observations, 'wall_time': duration,
        'evidence_sha256': {**ce, **ne, str(proof_path): sha256(proof_path),
            str(base / 'checkpoint_meta.json'): sha256(base / 'checkpoint_meta.json'),
            **{path: sha256(path) for path in candidate['control_result_paths']}},
        'comparison_source_sha256': {str(path): sha256(path) for path in
             (Path(__file__).resolve(), Path(inspect.getfile(arm)), trainer)},
        'quality_boundary': 'Negative quality is a valid result. Entropy values and CPU gradient equivalence '
                            'do not establish downstream benefit; no MMLU or eight-task acceptance threshold.',
        'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated'}
    _require(not torch.cuda.is_initialized(), 'CPU comparison initialized CUDA.')
    write_json(output, report)
    print({'status': report['status'], 'output': str(output), 'sha256': sha256(output),
           'candidate_vs_control': report['candidate_vs_control']}, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.control, args.candidate, args.output)
