"""Prepare the preselected four-gate combination after reviewed local completion."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import sys

import torch

from experiments.gate_student_context_evaluation.prepare import (
    ROOT, SOURCE, SOURCE_ID, BANKS, budget_equal, read, require, require_state,
)
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.w25_eval.acceptance import collect_results
from experiments.w25_eval.model_inventory import model_inventory
from train_utils.checkpoint_bpw import audit_state_dict


LAYERS = (0, 9, 18, 27)
TARGETS = tuple(f'model.layers.{i}.mlp.gate_proj' for i in LAYERS)


def patched_state(source, endpoint):
    require(len(source) == 3423 and len(endpoint) == 52, 'Require original3423 state and exactly four complete13-tensor gate endpoints.')
    local = {key: value for key, value in source.items() if any(key.startswith(name + '.') for name in TARGETS)}
    require(set(local) == set(endpoint), 'The fixed four gate inventory differs; no layer selection is allowed.')
    changes = {}
    for key, original in local.items():
        value = endpoint[key]
        require(value.shape == original.shape and value.dtype == original.dtype, 'Endpoint topology or dtype changed: ' + key)
        if '._parallel_stage_decoder.' in key:
            require(value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all()), 'Require finite BF16 deployed decoder values.')
            changes[key] = int(torch.count_nonzero(value != original))
        else:
            require_state({key: value}, {key: original}, 'Frozen gate codes/rotation/protection')
    require(len(changes) == 24 and all(changes.values()), 'All selected decoder tensors must retain actual updates.')
    expected = {**source, **endpoint}
    banks = [key for key in source if key.rsplit('.', 1)[-1] in BANKS]
    require(len(banks) == 504 and all(torch.equal(source[key], expected[key]) for key in banks), 'All504 original packed banks must remain exact.')
    return expected, {'patched_local_tensor_count': 52, 'frozen_outside_target_tensors': 3371,
        'selected_decoder_tensors': 24, 'selected_targets': list(TARGETS), 'changed_decoder_elements': changes,
        'source_packed_bank_count': 504, 'all_source_packed_banks_exact': True, 'changed_code_bits': 0,
        'frozen_local_nonmutable_exact': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    base, output = args.base_root.resolve(), args.output_root.resolve()
    require(not output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Use a fresh CPU-only preparation directory.')
    torch.set_num_threads(4)
    proto = base / 'four_gate_student_context_probe'
    previous, report, review = (read(proto / name) for name in ('preparation.json', 'run/report.json', 'review.json'))
    require(report['status'] == 'completed' and (proto / 'run.exit').read_text().strip() == '0'
            and review['status'] == 'verified' and review['report_sha256'] == sha256(proto / 'run/report.json')
            and report['preparation_sha256'] == sha256(proto / 'preparation.json')
            and set(report['layers']) == {str(i) for i in LAYERS}
            and report['new_training_updates'] == 1500 and report['reused_layer_training_updates'] == 0
            and report['source_and_context_unchanged'] and report['capture']['l18_input_and_context_exact'],
            'Require natural local completion and independent review before binding any endpoint.')
    check_hashes(previous['source_sha256']); check_hashes(previous['input_sha256'])
    check_stat(previous['data']['source_weight_stat'])
    for item in previous['data']['hf_weight_files']:
        check_stat(item)
    meta = read(SOURCE / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == SOURCE_ID, 'Use fresh original f927.')
    state = torch.load(SOURCE / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    endpoint, endpoint_sources = {}, {}
    old = read(previous['data']['original_report'])
    for index, target in zip(LAYERS, TARGETS, strict=True):
        row = report['layers'][str(index)]
        path = Path(row['state_file'])
        require(sha256(path) == row['state_sha256'], 'A fixed local endpoint changed.')
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        require(len(payload) == 13, 'Every selected endpoint must contain its full native gate state.')
        if index == 18:
            require(row['status'] == 'reused_verified' and row['training_updates_this_run'] == 0
                    and row['old_native_values_and_metrics_exact']
                    and row['state_sha256'] == previous['data']['control_state_sha256'], 'L18 must be the already verified original-code control.')
            proof = old['arms']['original_codes']['training']['final_payload']
        else:
            require(row['status'] == 'trained' and row['training_updates_this_run'] == 500
                    and row['training']['steps'] == row['training']['schedule_steps'] == 500
                    and row['training']['initial_payload'] == previous['data']['per_layer'][str(index)]['expected_gate_state']
                    and row['fresh_strict_state_and_native_decode_exact'], 'Every new layer must have its completed fresh500 proof.')
            proof = row['training']['final_payload']
        require({key: tensor_record(value) for key, value in payload.items()} == proof, 'Saved endpoint differs from actual training evidence.')
        endpoint.update({target + '.' + key: value for key, value in payload.items()})
        endpoint_sources[str(index)] = {'target': target, 'path': str(path), 'sha256': row['state_sha256'], 'reused': index == 18}
    expected, patch = patched_state(state, endpoint)
    budget = audit_state_dict(meta, state, max_bpw=2.5, max_lora_rank=0)
    budget_equal(audit_state_dict(meta, expected, max_bpw=2.5, max_lora_rank=0), budget)
    inventory = model_inventory(meta['base_model_path'])
    control_paths = [SOURCE.parent / name for name in ('mmlu.json', 'seven.json')]
    control = collect_results(control_paths, inventory, checkpoint=(SOURCE, meta))
    gate_dir = base / 'gate_student_context_full8_r1/original_codes'
    gate_model = gate_dir / 'final_model'; gate_meta = read(gate_model / 'checkpoint_meta.json')
    gate_paths = [gate_dir / name for name in ('mmlu.json', 'seven.json')]
    require(read(gate_dir / 'launch.json')['status'] == 'completed' and (gate_dir / 'dispatcher.exit').read_text().strip() == '0',
            'Require the completed single-L18 full8 control.')
    gate_control = collect_results(gate_paths, inventory, checkpoint=(gate_model, gate_meta))
    sources, inputs = dict(previous['source_sha256']), dict(previous['input_sha256'])
    for relative in ('experiments/gate_student_context_evaluation/prepare.py', 'experiments/gate_student_context_evaluation/deploy.py',
                     'experiments/gate_student_context_evaluation/run.py', 'experiments/gate_down_student_context_evaluation/run.py',
                     'experiments/w25_eval/acceptance.py', 'experiments/w25_eval/model_inventory.py',
                     'experiments/qwen3_w25/summarize_results.py', 'experiments/qwen3_w25/step_export_callback.py',
                     'experiments/long_context_recovery/contract.py',
                     'train_utils/checkpoint_bpw.py'):
        path = ROOT / relative
        require(str(path) not in sources or sources[str(path)] == sha256(path), 'A reused source version changed.')
        sources[str(path)] = sha256(path)
    for path in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
    for path in (proto / 'preparation.json', proto / 'run/report.json', proto / 'review.json', proto / 'run.exit',
                 gate_dir / 'launch.json', gate_dir / 'dispatcher.exit', gate_dir / 'audit.json',
                 gate_dir / 'full8_summary.json', gate_model / 'checkpoint_meta.json', *control_paths, *gate_paths,
                 *(Path(row['path']) for row in endpoint_sources.values())):
        inputs[str(path)] = sha256(path)
    endpoint_path = output / 'four_gate_endpoint.pt'
    commands = [{'name': 'deploy', 'argv': [sys.executable, '-m', 'experiments.four_gate_student_context_evaluation.deploy',
        '--launch_record', str(output / 'launch.json')]}]
    for name, tasks, batch in [('mmlu', 'mmlu', '64'), ('seven', 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa', '32')]:
        commands.append({'name': name, 'argv': [sys.executable, 'tools/cat_eval.py', '--checkpoint_dir', str(output / 'final_model'),
            '--strict', '--eval_lm_eval', '--eval_device', 'cuda:0', '--tasks', tasks, '--num_fewshot', '0',
            '--lm_batch_size', batch, '--tokenizer_use_fast', 'true', '--prewarm_group_size', '8',
            '--eval_log_dir', str(output / (name + '_logs')), '--output_json', str(output / (name + '.json'))]})
    require(not torch.cuda.is_initialized(), 'Preparation must remain CPU-only.')
    output.mkdir(parents=True)
    torch.save(endpoint, endpoint_path)
    saved = torch.load(endpoint_path, map_location='cpu', weights_only=True, mmap=True)
    require_state(saved, endpoint, 'CPU composition of the preselected four local endpoints')
    inputs[str(endpoint_path)] = sha256(endpoint_path)
    record = {'status': 'prepared_unbound', 'arm': 'four_gates', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'source_checkpoint': str(SOURCE), 'source_checkpoint_id': SOURCE_ID,
        'source_weight_stat': previous['data']['source_weight_stat'], 'hf_weight_stats': previous['data']['hf_weight_files'],
        'architecture': inventory, 'source_bit_budget': budget, 'source_sha256': sources, 'input_sha256': inputs,
        'endpoint_path': str(endpoint_path), 'endpoint_sha256': inputs[str(endpoint_path)],
        'endpoint_state': {key: tensor_record(value) for key, value in saved.items()}, 'endpoint_sources': endpoint_sources,
        'patch_contract': patch, 'selected_targets': list(TARGETS), 'patched_local_tensor_count': 52,
        'frozen_outside_target_tensors': 3371, 'selected_decoder_tensors': 24,
        'source_native_policy': previous['data']['model_policy'], 'stage_commands': commands,
        'control_result_paths': [str(path) for path in control_paths], 'control_mean_percent': control['mean_percent'],
        'gate_only_control_results': [str(path) for path in gate_paths], 'gate_only_control_checkpoint': str(gate_model),
        'gate_only_control_checkpoint_id': gate_meta['checkpoint_id'], 'gate_only_control_mean_percent': gate_control['mean_percent'],
        'timeout_seconds': 1800, 'termination_grace_seconds': 20,
        'environment': {'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'},
        'protocol': {'fixed_layers': list(LAYERS), 'full8_leaves': 64, 'full8_samples': 24742, 'num_fewshot': 0,
            'mmlu_batch': 64, 'seven_batch': 32, 'limit': None, 'new_training': False, 'all_code_banks_frozen': 504,
            'longbench': 'not_evaluated', 'nine_task_goal': 'not_evaluated',
            'boundary': 'Preselected four-gate isolated full8 comparison; no NMSE-based layer selection or production integration.'},
        'launcher_argv': [sys.executable, '-m', 'experiments.four_gate_student_context_evaluation.run', '--launch_record', str(output / 'launch.json')]}
    write_json(output / 'launch.json', record)
    print(json.dumps({'status': 'prepared_unbound', 'launch_sha256': sha256(output / 'launch.json'), 'cuda_initialized': False,
        'source_pins': len(sources), 'input_pins': len(inputs), 'endpoint_tensor_count': len(saved),
        'source_bpw': budget['user_linear_bpw']}))


if __name__ == '__main__':
    main()
