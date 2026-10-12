"""Prepare an isolated per-Linear entropy trial against a completed native gate control."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys

import torch

from experiments.full_rht_gate_pair.common import RESULT, ROOT, ORIGINAL, check, read, recipe
from experiments.full_rht_gate_pair.compare import arm
from experiments.longbench_v1.common import sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require


def prepare(model, output):
    output = Path(output).resolve()
    _require(ROOT != ORIGINAL and Path.cwd().resolve() == ROOT
             and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not output.exists(),
             'Use a fresh CPU-only output inside the reviewed snapshot.')
    control_dir = RESULT / 'full_rht_gate_pair' / (model + '_block32')
    template, audit, control, _, evidence = arm(control_dir, 32)
    _require(template['model_type'] == model, 'Historical control model mismatch.')
    argv, normalized = recipe(template['architecture']['base_model_path'], output / 'donor',
        layers=template['architecture']['num_hidden_layers'], block_size=32, steps=10000)
    expected = copy.deepcopy(template['normalized_expected'])
    for section in ('canonical_config', 'cat_args'):
        expected[section]['output_dir'] = str(output / 'donor')
    _require(normalized == expected, 'The native recipe changed beyond the output directory.')
    proof_path = RESULT / 'per_model_bsq_entropy/cpu_validation.json'
    proof = read(proof_path)
    _require(proof['status'] == 'verified' and proof['cuda_initialized'] is False,
             'Require completed independent-model loss/gradient CPU validation.')
    sources = dict(template['source_sha256'])
    for path in [ROOT / 'experiments/full_rht_gate_pair/compare.py',
                 *(ROOT / 'experiments/per_model_bsq_entropy').glob('*.py')]:
        sources[str(path)] = sha256(path)
    for relative in ('experiments/per_model_bsq_entropy/train.py', 'litebsq/bsq.py'):
        _require(proof['source_sha256'][str(ORIGINAL / relative)] == sha256(ROOT / relative),
                 'The CPU validation did not cover this isolated trainer/BSQ: ' + relative)
    evidence = {**template['input_sha256'], **evidence, str(proof_path): sha256(proof_path)}
    record = {key: copy.deepcopy(template[key]) for key in (
        'cwd', 'architecture', 'base_dir', 'base_checkpoint_id', 'control_result_paths',
        'block_size', 'steps_per_stage', 'full_evaluation', 'expected_user_linear_bits',
        'protected_stats', 'snapshot_import_paths', 'environment', 'protocol')}
    record.update(status='prepared_unbound', gpu=None, queued=False, automatic_retry=False,
        python_executable=sys.executable, output_dir=str(output), model_type=model,
        training_argv=[sys.executable, '-u', '-m', 'experiments.per_model_bsq_entropy.train', *argv],
        normalized_expected=normalized, source_sha256=sources, input_sha256=evidence,
        historical_control_dir=str(control_dir), historical_control_checkpoint_id=control['checkpoint_id'],
        historical_control_audit_path=str(control_dir / 'audit.json'),
        historical_control_result_paths=[str(control_dir / (name + '.json')) for name in ('mmlu', 'seven')],
        timeout_seconds=2700, termination_grace_seconds=20,
        launcher_argv=[sys.executable, '-u', '-m', 'experiments.full_rht_gate_pair.run',
                       '--launch-record', str(output / 'launch.json')],
        entropy_algorithm='isolated_mean_per_linear_marginal_entropy_v1',
        scope='Isolated new regularizer trial, not a main-framework change. Fresh full gate category; '
              'same native block32/K32/D32/S2/H128R0/BF16/input64/MSE10Kx2/B8192/seed31, no KD/LoRA. '
              'Only marginal entropy averages independent Linear entropies instead of pooling their probabilities.',
        comparison_boundary='Historical global-entropy control has the same native recipe, source, initialization '
              'and sampling settings. Actual native normalization and final protection/sign hashes must match. '
              'CPU semantics or lower auxiliary loss alone do not establish downstream benefit.',
        cuda_initialized=False)
    _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    check(record)
    output.mkdir(parents=True)
    write_json(output / 'launch.json', record)
    print({'status': record['status'], 'model': model, 'source_pins': len(sources),
           'input_pins': len(evidence), 'control_mean_percent': control['mean_percent'],
           'launch_sha256': sha256(output / 'launch.json')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('qwen3', 'llama'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.model, args.output_dir)
