"""Prepare matched Llama norm-only four-step and 400-step records on CPU, without scheduling."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import torch

from compressed_e2e_fintuning.args import parse_args
from experiments.llama_norm_recovery.dispatch import pins, require, sha
from experiments.llama_norm_recovery.validate import CONTROL, ROOT, configuration, read, training_arguments
from experiments.w25_eval.validate_norm_only_training import expected_inventory
from train_utils.runtime_snapshot import to_runtime_jsonable

RL = CONTROL.parent
DEFAULT_FORMAL = RL / 'final_norm_only_task50_lr1e5_s400'
DEFAULT_SHORT = RL / 'final_norm_only_task50_lr1e5_validate_s4'


def differences(left, right, prefix=''):
    if isinstance(left, dict) and isinstance(right, dict):
        result = {}
        for key in sorted(set(left) | set(right)):
            name = prefix + '.' + key if prefix else key
            if key not in left or key not in right:
                result[name] = {'control': left.get(key), 'candidate': right.get(key)}
            else:
                result.update(differences(left[key], right[key], name))
        return result
    return {} if left == right else {prefix: {'control': left, 'candidate': right}}


def prepare(formal, short):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and os.environ.get('ACCELERATE_GRADIENT_ACCUMULATION_STEPS', '1') == '1', 'CPU-only, GAS1 preparation required.')
    formal, short = Path(formal).resolve(), Path(short).resolve()
    require(formal != short and not any((p / 'launch.json').exists() for p in (formal, short)), 'Never overwrite an existing run.')
    control = read(CONTROL / 'launch.json')
    require(control['status'] == 'completed' and control['exit_code'] == 0
            and control['source_checkpoint_id'] == '35776715-2707-45a9-9fb4-2da1efef3f0e', 'Require the complete original pure-KL control.')
    pins(control)
    native = Path(control['native_final_model']).parent
    snapshot_path = native / 'normalized_e2e_runtime_args.json'
    actual = read(snapshot_path)
    old_cfg, old_hf, old_args = parse_args(control['argv'][3:])
    require(to_runtime_jsonable(old_cfg) == actual['canonical_config']
            and to_runtime_jsonable(old_hf) == actual['hf_args'], 'Current parser differs from the actual pure-KL control.')
    old_args_diff = differences(actual['training_args'], to_runtime_jsonable(old_args))
    require(set(old_args_diff) <= {'_n_gpu', 'logging_dir', 'output_dir'}, 'Unexpected control Trainer difference.')
    inventory = expected_inventory(control['source_checkpoint'], control['architecture'])
    require(control['architecture']['model_type'] == 'llama' and inventory['mutable_tensors'] == 65
            and inventory['norm_elements'] == 266240 and inventory['frozen_tensors'] == 2914
            and inventory['all_decoder_tensors'] == 1344 and inventory['all_code_banks'] == 448
            and inventory['serialized_tensors'] == 2979, 'This exact Llama source inventory differs.')
    sources = {str((ROOT / p).resolve()): value for p, value in control['source_sha256'].items()}
    qnorm = ROOT / 'result/qwen3_8b_mmlu65_20261004/best1200_norm_only_task50_s400/launch.json'
    for path, expected in read(qnorm)['source_sha256'].items():
        path = (ROOT / path).resolve()
        require(sha(path) == expected, 'Existing norm implementation changed: ' + str(path))
        sources[str(path)] = expected
    for path in sorted(Path(__file__).parent.glob('*.py')):
        ast.parse(path.read_text())
        sources[str(path.resolve())] = sha(path)
    require((Path(__file__).parent / 'dispatch.py').exists(), 'Dispatcher must be concrete before preparation.')
    inputs = deepcopy(control['input_evidence_sha256'])
    protocol = {}
    for name in ('mmlu', 'seven'):
        bf16 = list((RL / 'baseline_bf16' / name).glob('*/results_*.json'))
        require(len(bf16) == 1, 'Expected one complete BF16 protocol reference.')
        protocol[name] = [{'path': str(bf16[0]), 'format': 'hf'},
                         {'path': str(RL / 'remaining_prefix_block32_s500_full8' / (name + '.json')), 'format': 'cat_eval'},
                         {'path': str(CONTROL / 'step_0400' / (name + '.json')), 'format': 'cat_eval'}]
    evidence = [CONTROL / 'launch.json', snapshot_path, CONTROL / 'step_0400/eight_summary.json',
                CONTROL / 'step_0400/audit.json', CONTROL / 'step_0400/independent_review.json', qnorm]
    evidence += [Path(item['path']) for rows in protocol.values() for item in rows]
    for path in evidence:
        inputs[str(path)] = sha(path)
    records = []
    for output, kind in ((short, 'validate_s4'), (formal, 'formal_s400')):
        diagnostic = kind == 'validate_s4'
        argv = training_arguments(output, diagnostic=diagnostic)
        record = {'schema': 'llama_norm_only_matched_v1', 'kind': kind, 'status': 'prepared',
            'queued': False, 'automatic_retry': False, 'gpu': None, 'cwd': str(ROOT),
            'python_executable': sys.executable, 'output_dir': str(output), 'control_launch': str(CONTROL / 'launch.json'),
            'source_checkpoint': control['source_checkpoint'], 'source_checkpoint_id': control['source_checkpoint_id'],
            'source_bpw': control['source_bpw'], 'source_linear_bits': control['source_linear_bits'],
            'source_weight_stat': deepcopy(control['source_weight_stat']), 'hf_source': deepcopy(control['hf_source']),
            'input_data_stat': deepcopy(control['input_data_stat']), 'architecture': deepcopy(control['architecture']),
            'source_sha256': sources, 'input_evidence_sha256': inputs, 'expected_inventory': inventory,
            'training_argv': argv,
            'argv': ([sys.executable, '-m', 'experiments.llama_norm_recovery.validate', '--launch_record',
                      str(output / 'launch.json'), '--phase', 'train'] if diagnostic else
                     [sys.executable, '-m', 'experiments.w25_eval.train_norm_only', *argv]),
            'dispatch_argv': [sys.executable, '-u', '-m', 'experiments.llama_norm_recovery.dispatch',
                              '--launch_record', str(output / 'launch.json')],
            'environment': {'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONPATH': str(ROOT),
                'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'HF_HUB_OFFLINE': '1',
                'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'ACCELERATE_GRADIENT_ACCUMULATION_STEPS': '1'},
            'readiness': {'cpu_configuration': 'pending', 'resource_decision': 'not_scheduled',
                'norm_validation': {'path': str(short / 'verification.json'), 'sha256': None}},
            'schedule': {'max_steps': 2000, 'warmup_steps': 20, 'stop_step': 4 if diagnostic else 400,
                         'save_steps': 2 if diagnostic else 400, 'scheduler': 'cosine'},
            'protocol_references': protocol, 'cuda_initialized': False,
            'quality_evaluation': 'none' if diagnostic else 'complete_eight_only; LongBench/nine target not_evaluated',
            'comparison_boundary': 'Fresh original CAT3577. Only active component changes from all decoder to all native norms; FP32 master storage follows that selection. Norm LR inherits the same 1e-5. Same data recipe/seed/schedule; actual token stream has not yet been observed for this run.',
            'resource_note': 'Not scheduled. The root must bind an idle GPU and review the actual four-step resource/export proof before the formal run.'}
        cfg, hf, args, _architecture = configuration(record)
        diff = differences(actual['canonical_config'], to_runtime_jsonable(cfg))
        allowed = {'train_mode', 'aux.norm_train_mode', 'opt.distill_fp32_components', 'run_root_dir'}
        if diagnostic:
            allowed.update(('opt.logging_steps', 'remaining_argv'))
            remaining = list(actual['canonical_config']['remaining_argv'])
            require(remaining.count('--save_steps') == 1, 'Ambiguous saved Trainer argument.')
            remaining[remaining.index('--save_steps') + 1] = '2'
            require(to_runtime_jsonable(cfg)['remaining_argv'] == remaining,
                    'Only the diagnostic save interval may change in forwarded Trainer arguments.')
        require(set(diff) == allowed, 'Unexpected science change: ' + json.dumps(diff))
        require(to_runtime_jsonable(hf) == to_runtime_jsonable(old_hf), 'HF data/model arguments differ.')
        trainer_diff = differences(to_runtime_jsonable(old_args), to_runtime_jsonable(args))
        require(set(trainer_diff) <= ({'save_steps', 'logging_steps', 'logging_dir'} if diagnostic else {'logging_dir'}),
                'Unexpected Trainer change: ' + json.dumps(trainer_diff))
        record['cpu_configuration'] = {'canonical_config': to_runtime_jsonable(cfg), 'hf_args': to_runtime_jsonable(hf),
            'training_args': to_runtime_jsonable(args), 'candidate_vs_actual_control': diff,
            'candidate_vs_control_trainer': trainer_diff, 'control_actual_vs_cpu_trainer': old_args_diff,
            'resolved_norm_lr': cfg.opt.learning_rate if cfg.aux.norm_lr is None else cfg.aux.norm_lr}
        record['readiness']['cpu_configuration'] = 'passed'
        pins(record)
        records.append((output, record))
    require(not torch.cuda.is_initialized(), 'Preparation must never initialize CUDA.')
    for output, record in records:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'launch.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'cpu_ready_not_scheduled', 'cuda_initialized': False,
        'sources': len(sources), 'inputs': len(inputs), 'inventory': {k:v for k,v in inventory.items() if k != 'norm_shapes'},
        'records': {str(p / 'launch.json'): sha(p / 'launch.json') for p, _ in records}}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formal_output', type=Path, default=DEFAULT_FORMAL)
    parser.add_argument('--short_output', type=Path, default=DEFAULT_SHORT)
    args = parser.parse_args()
    prepare(args.formal_output, args.short_output)


if __name__ == '__main__':
    main()
