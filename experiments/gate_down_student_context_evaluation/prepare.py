"""Prepare full8 evaluation of the two completed fixed-code gate/down endpoints."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import sys

import torch

from experiments.gate_student_context_evaluation.prepare import (
    ROOT, SOURCE, SOURCE_ID, TARGET, BANKS, budget_equal, read, require, require_state,
)
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.w25_eval.acceptance import collect_results
from experiments.w25_eval.model_inventory import model_inventory
from train_utils.checkpoint_bpw import audit_state_dict


ARMS = ('down_only', 'gate_down')
PREFIX = 'model.layers.18.mlp.'


def selected_projections(arm):
    require(arm in ARMS, 'Unknown completed fixed-code endpoint.')
    return ['down_proj'] if arm == 'down_only' else ['gate_proj', 'down_proj']


def local_endpoint(endpoint, arm):
    return {'down_proj.' + key: value for key, value in endpoint.items()} if arm == 'down_only' else endpoint


def patched_state(source, endpoint, arm):
    selected = selected_projections(arm)
    require(len(source) == 3423, 'Require the complete original f927 state.')
    local = {key[len(PREFIX):]: value for key, value in source.items()
             if any(key.startswith(PREFIX + kind + '.') for kind in selected)}
    replacement = local_endpoint(endpoint, arm)
    require(set(replacement) == set(local) and len(local) == 13 * len(selected), 'Incomplete native local endpoint.')
    changes = {}
    for key, original in local.items():
        value = replacement[key]
        require(value.shape == original.shape and value.dtype == original.dtype, 'Endpoint topology or dtype differs: ' + key)
        if '._parallel_stage_decoder.' in key:
            require(value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all()), 'Require finite native BF16 decoder values.')
            changes[key] = int(torch.count_nonzero(value != original))
        else:
            require_state({key: value}, {key: original}, 'Frozen local codes, rotation and protection')
    require(len(changes) == 6 * len(selected) and all(changes.values()), 'Selected decoder updates did not survive deployment.')
    expected = {**source, **{PREFIX + key: value for key, value in replacement.items()}}
    banks = [key for key in source if key.rsplit('.', 1)[-1] in BANKS]
    require(len(banks) == 504 and all(torch.equal(source[key], expected[key]) for key in banks), 'Every source packed bank must remain exact.')
    return expected, {'patched_local_tensor_count': len(local), 'frozen_outside_target_tensors': len(source) - len(local),
        'selected_decoder_tensors': len(changes), 'selected_projections': selected, 'changed_decoder_elements': changes,
        'source_packed_bank_count': len(banks), 'all_source_packed_banks_exact': True, 'changed_code_bits': 0,
        'frozen_local_nonmutable_exact': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    base, output = args.base_root.resolve(), args.output_root.resolve()
    require(not output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Use a new CPU-only preparation directory.')
    torch.set_num_threads(4)
    proto = base / 'gate_down_student_context_probe'
    previous, report, review = (read(proto / name) for name in ('preparation.json', 'run/report.json', 'review.json'))
    require(report['status'] == 'completed' and (proto / 'run.exit').read_text().strip() == '0'
            and sha256(proto / 'run/report.json') == '69bcd6ada1538d96a63c1386940a4d5287ec465aed3ae68841453ec28a5d67a7'
            and review['status'] == 'verified' and review['report_sha256'] == sha256(proto / 'run/report.json')
            and sha256(proto / 'review.json') == '3af628442a598ec4c25c56535a92577aa1e444de3e415f2a9864f6de29a101a3',
            'Require the reviewed terminal of the actual fixed-code gate/down probe.')
    check_hashes(previous['source_sha256']); check_hashes(previous['input_sha256'])
    check_stat(previous['source_weight_stat'])
    for item in previous['hf_weight_files']:
        check_stat(item)
    meta = read(SOURCE / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == SOURCE_ID, 'Use the original f927 checkpoint.')
    state = torch.load(SOURCE / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    budget = audit_state_dict(meta, state, max_bpw=2.5, max_lora_rank=0)
    budget_equal(budget, budget)
    inventory = model_inventory(meta['base_model_path'])
    control_paths = [SOURCE.parent / name for name in ('mmlu.json', 'seven.json')]
    control = collect_results(control_paths, inventory, checkpoint=(SOURCE, meta))
    gate_dir = base / 'gate_student_context_full8_r1/original_codes'
    gate_model = gate_dir / 'final_model'
    gate_meta = read(gate_model / 'checkpoint_meta.json')
    gate_paths = [gate_dir / name for name in ('mmlu.json', 'seven.json')]
    gate_launch = read(gate_dir / 'launch.json')
    require(gate_launch['status'] == 'completed' and (gate_dir / 'dispatcher.exit').read_text().strip() == '0', 'Require the completed gate-only quality control.')
    gate_control = collect_results(gate_paths, inventory, checkpoint=(gate_model, gate_meta))
    sources, inputs = dict(previous['source_sha256']), dict(previous['input_sha256'])
    for relative in ('experiments/gate_student_context_evaluation/prepare.py', 'experiments/gate_student_context_evaluation/deploy.py',
                     'experiments/gate_student_context_evaluation/run.py', 'experiments/w25_eval/acceptance.py',
                     'experiments/w25_eval/model_inventory.py', 'experiments/qwen3_w25/summarize_results.py',
                     'experiments/qwen3_w25/step_export_callback.py'):
        path = ROOT / relative
        require(str(path) not in sources or sources[str(path)] == sha256(path), 'An inherited source version changed.')
        sources[str(path)] = sha256(path)
    for path in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
    for path in (proto / 'preparation.json', proto / 'run/report.json', proto / 'review.json', proto / 'run.exit',
                 gate_dir / 'launch.json', gate_dir / 'dispatcher.exit', gate_dir / 'audit.json',
                 gate_dir / 'full8_summary.json', gate_model / 'checkpoint_meta.json', *control_paths, *gate_paths):
        inputs[str(path)] = sha256(path)
    records = {}
    for arm in ARMS:
        row = report['arms'][arm]
        path = Path(row['state_file'])
        require(sha256(path) == row['state_sha256'], 'The selected endpoint file changed.')
        endpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        expected, patch = patched_state(state, endpoint, arm)
        combined_local = {key[len(PREFIX):]: tensor_record(value) for key, value in expected.items()
                          if key.startswith((PREFIX + 'gate_proj.', PREFIX + 'down_proj.'))}
        require(combined_local == row['training']['final_payload'], 'Actual saved values differ from the completed training proof.')
        budget_equal(audit_state_dict(meta, expected, max_bpw=2.5, max_lora_rank=0), budget)
        arm_output = output / arm
        commands = [{'name': 'deploy', 'argv': [sys.executable, '-m', 'experiments.gate_down_student_context_evaluation.deploy',
            '--launch_record', str(arm_output / 'launch.json')]}]
        for name, tasks, batch in [('mmlu', 'mmlu', '64'), ('seven', 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa', '32')]:
            commands.append({'name': name, 'argv': [sys.executable, 'tools/cat_eval.py', '--checkpoint_dir', str(arm_output / 'final_model'),
                '--strict', '--eval_lm_eval', '--eval_device', 'cuda:0', '--tasks', tasks, '--num_fewshot', '0',
                '--lm_batch_size', batch, '--tokenizer_use_fast', 'true', '--prewarm_group_size', '8',
                '--eval_log_dir', str(arm_output / (name + '_logs')), '--output_json', str(arm_output / (name + '.json'))]})
        records[arm] = {'status': 'prepared_unbound', 'arm': arm, 'gpu': None, 'queued': False, 'automatic_retry': False,
            'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(arm_output),
            'source_checkpoint': str(SOURCE), 'source_checkpoint_id': SOURCE_ID,
            'source_weight_stat': previous['source_weight_stat'], 'hf_weight_stats': previous['hf_weight_files'],
            'architecture': inventory, 'source_bit_budget': budget, 'source_sha256': sources,
            'input_sha256': {**inputs, str(path): row['state_sha256']}, 'endpoint_path': str(path), 'endpoint_sha256': row['state_sha256'],
            'endpoint_state': {key: tensor_record(value) for key, value in endpoint.items()}, 'patch_contract': patch,
            'selected_projections': selected_projections(arm), 'patched_local_tensor_count': patch['patched_local_tensor_count'],
            'frozen_outside_target_tensors': patch['frozen_outside_target_tensors'], 'selected_decoder_tensors': patch['selected_decoder_tensors'],
            'source_native_policy': previous['data']['model_policy'], 'stage_commands': commands,
            'control_result_paths': [str(x) for x in control_paths], 'control_mean_percent': control['mean_percent'],
            'gate_only_control_results': [str(x) for x in gate_paths], 'gate_only_control_checkpoint': str(gate_model),
            'gate_only_control_checkpoint_id': gate_meta['checkpoint_id'], 'gate_only_control_mean_percent': gate_control['mean_percent'],
            'timeout_seconds': 1800, 'termination_grace_seconds': 20,
            'environment': {'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'},
            'protocol': {'full8_leaves': 64, 'full8_samples': 24742, 'num_fewshot': 0, 'mmlu_batch': 64,
                'seven_batch': 32, 'limit': None, 'single_layer_only': 18, 'new_training': False, 'all_code_banks_frozen': 504,
                'longbench': 'not_evaluated', 'nine_task_goal': 'not_evaluated',
                'boundary': 'Isolated fixed-code L18 scope quality comparison; no new production algorithm or all-layer rollout.'},
            'launcher_argv': [sys.executable, '-m', 'experiments.gate_down_student_context_evaluation.run', '--launch_record', str(arm_output / 'launch.json')]}
    require(not torch.cuda.is_initialized(), 'CPU preparation must not initialize CUDA.')
    output.mkdir(parents=True)
    for arm, record in records.items():
        path = output / arm; path.mkdir(); write_json(path / 'launch.json', record)
    print(json.dumps({'status': 'prepared_unbound', 'cuda_initialized': False,
        'arms': {a: sha256(output / a / 'launch.json') for a in ARMS}, 'source_pins': len(sources),
        'source_user_linear_bits': int(budget['user_linear_bits']), 'source_bpw': budget['user_linear_bpw']}))


if __name__ == '__main__':
    main()
