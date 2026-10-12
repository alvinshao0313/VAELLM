"""Prepare full8 for the reviewed group-accepted L18 gate plus decoder500."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import sys

import torch

from experiments.full_weight_bits_probe.core import popcounts
from experiments.gate_student_context_evaluation.prepare import (
    ROOT, SOURCE, SOURCE_ID, TARGET, BANKS, budget_equal, read, require, require_state,
)
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.w25_eval.acceptance import collect_results
from experiments.w25_eval.model_inventory import model_inventory
from train_utils.checkpoint_bpw import audit_state_dict


ARM = 'group_accept_decoder500'
EXPECTED_XOR = [194788, 516131]
REPORT_SHA = '301651b63204c6ca6ea1adbd00b6c1bc5e565c0a05207aacca3c5468e765fd07'
REVIEW_SHA = '7fc3172b96d72b76ba8db201fa287e0c5f8be51387d7e8a2a62b335063f80efe'
ENDPOINT_SHA = 'f88fd90b6b637abde81b6f6c6c96f7999d484e90e3232d9c526c413dbd3229fd'


def patched_state(source, endpoint, arm):
    require(arm == ARM and len(source) == 3423, 'Require the reviewed group-accepted endpoint and full f927 source.')
    local = {key[len(TARGET) + 1:]: value for key, value in source.items() if key.startswith(TARGET + '.')}
    require(set(local) == set(endpoint) and len(local) == 13, 'Exactly the thirteen native gate tensors are required.')
    changes, frozen = {}, 0
    for key, original in local.items():
        value = endpoint[key]
        require(value.shape == original.shape and value.dtype == original.dtype, 'Local topology/dtype changed: ' + key)
        if key.startswith('_parallel_stage_decoder.'):
            require(value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all()), 'Require finite deployed BF16 decoder.')
            changes[key] = int(torch.count_nonzero(value != original))
        elif key not in BANKS:
            require_state({key: value}, {key: original}, 'Frozen protection and rotation state')
            frozen += 1
    require(len(changes) == 6 and all(changes.values()) and frozen == 5,
            'Only six real decoder updates, two code banks and five frozen tensors are permitted.')
    xor = [popcounts(local[key], endpoint[key]) for key in BANKS]
    require(xor == EXPECTED_XOR, 'Actual packed codes do not match the reviewed accepted-group endpoint.')
    expected = {**source, **{TARGET + '.' + key: value for key, value in endpoint.items()}}
    banks = [key for key in source if key.rsplit('.', 1)[-1] in BANKS]
    outside = [key for key in banks if not key.startswith(TARGET + '.')]
    require(len(banks) == 504 and len(outside) == 502
            and all(expected[key] is source[key] for key in outside), 'Another projection code bank was patched.')
    return expected, {'patched_local_tensor_count': 13, 'frozen_outside_target_tensors': 3410,
        'selected_decoder_tensors': 6, 'selected_projections': ['gate_proj'], 'changed_decoder_elements': changes,
        'stage_changed_bits': xor, 'changed_code_bits': sum(xor), 'source_packed_bank_count': 504,
        'frozen_outside_target_packed_bank_count': 502, 'frozen_outside_target_packed_banks_exact': True,
        'frozen_local_nonmutable_tensors': frozen, 'frozen_local_nonmutable_exact': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    base, output = args.base_root.resolve(), args.output_root.resolve()
    require(not output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized(), 'Use a new CPU-only preparation directory.')
    torch.set_num_threads(4)
    proto = base / 'gate_group_decoder500'
    previous, report, review, launch = (read(proto / name) for name in
        ('preparation.json', 'run/report.json', 'review.json', 'launch.json'))
    require(report['status'] == launch['status'] == 'completed' and launch['exit_code'] == 0
            and (proto / 'run.exit').read_text().strip() == '0'
            and sha256(proto / 'run/report.json') == REPORT_SHA == review['report_sha256']
            and sha256(proto / 'review.json') == REVIEW_SHA and review['status'] == 'verified'
            and report['preparation_sha256'] == review['preparation_sha256'] == sha256(proto / 'preparation.json'),
            'Require the actual reviewed, naturally completed group-accepted decoder500 run.')
    training = report['training']
    require(training['status'] == 'completed' and training['steps'] == training['schedule_steps'] == 500
            and training['settings'] == previous['expected_training_settings']
            and training['initial_payload'] == previous['initial_state']
            and training['frozen_state_exact'] and training['frozen_objective_buffers_exact']
            and report['saved']['saved_state_exact'] and report['saved']['fixed_banks_exact']
            and review['initial_accepted_state_exact'] and review['seven_nondecoder_tensors_including_two_banks_exact'],
            'The completed fit did not preserve the reviewed code/init/training contract.')
    check_hashes(previous['source_sha256']); check_hashes(previous['input_sha256'])
    for item in previous['protected_stats']:
        check_stat(item)
    path = Path(report['saved']['state_path'])
    require(sha256(path) == report['saved']['state_sha256'] == ENDPOINT_SHA, 'The actual fitted endpoint changed.')
    endpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    require({k: tensor_record(v) for k, v in endpoint.items()} == training['final_payload'],
            'Actual saved tensors differ from the completed training proof.')
    meta = read(SOURCE / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == SOURCE_ID, 'Only fresh f927 may supply the complete deployment.')
    state = torch.load(SOURCE / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    budget = audit_state_dict(meta, state, max_bpw=2.5, max_lora_rank=0)
    budget_equal(budget, budget)
    expected, patch = patched_state(state, endpoint, ARM)
    budget_equal(audit_state_dict(meta, expected, max_bpw=2.5, max_lora_rank=0), budget)
    require(budget['user_linear_bits'] == 17065672704 and budget['original_linear_weights'] == 6945767424,
            'Original integer storage budget differs.')
    inventory = model_inventory(meta['base_model_path'])
    control_paths = [SOURCE.parent / name for name in ('mmlu.json', 'seven.json')]
    control = collect_results(control_paths, inventory, checkpoint=(SOURCE, meta))
    gate_dir = base / 'gate_student_context_full8_r1/original_codes'
    gate_model = gate_dir / 'final_model'; gate_meta = read(gate_model / 'checkpoint_meta.json')
    gate_paths = [gate_dir / name for name in ('mmlu.json', 'seven.json')]
    gate_launch = read(gate_dir / 'launch.json'); gate_audit = read(gate_dir / 'audit.json')
    require(gate_launch['status'] == 'completed' and gate_launch['exit_code'] == 0
            and (gate_dir / 'dispatcher.exit').read_text().strip() == '0'
            and gate_audit['status'] == 'verified' and gate_audit['checkpoint_id'] == gate_meta['checkpoint_id']
            and gate_audit['stage_changed_bits'] == [0, 0], 'Require the real completed original-code L18 quality control.')
    gate_control = collect_results(gate_paths, inventory, checkpoint=(gate_model, gate_meta))
    check_hashes(gate_launch['source_sha256']); check_hashes(gate_launch['input_sha256'])
    check_stat(gate_launch['source_weight_stat'])
    for item in gate_launch['hf_weight_stats']:
        check_stat(item)
    sources, inputs = dict(gate_launch['source_sha256']), dict(gate_launch['input_sha256'])
    for mapping, addition in ((sources, previous['source_sha256']), (inputs, previous['input_sha256'])):
        for key, digest in addition.items():
            require(key not in mapping or mapping[key] == digest, 'Inherited evidence version conflict: ' + key)
            mapping[key] = digest
    for relative in ('experiments/gate_student_context_evaluation/prepare.py',
                     'experiments/gate_student_context_evaluation/deploy.py', 'experiments/gate_student_context_evaluation/run.py',
                     'experiments/gate_down_student_context_evaluation/run.py'):
        p = ROOT / relative
        require(str(p) not in sources or sources[str(p)] == sha256(p), 'Existing helper source changed.')
        sources[str(p)] = sha256(p)
    for p in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(p.read_text()); sources[str(p)] = sha256(p)
    for p in (proto / 'preparation.json', proto / 'run/report.json', proto / 'review.json', proto / 'run.exit',
              proto / 'launch.json', path, gate_dir / 'launch.json', gate_dir / 'dispatcher.exit', gate_dir / 'audit.json',
              gate_dir / 'full8_summary.json', gate_model / 'checkpoint_meta.json', *control_paths, *gate_paths):
        inputs[str(p)] = sha256(p)
    commands = [{'name': 'deploy', 'argv': [sys.executable, '-m', 'experiments.gate_group_accept_evaluation.deploy',
        '--launch_record', str(output / 'launch.json')]}]
    for name, tasks, batch in [('mmlu', 'mmlu', '64'), ('seven', 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa', '32')]:
        commands.append({'name': name, 'argv': [sys.executable, 'tools/cat_eval.py', '--checkpoint_dir', str(output / 'final_model'),
            '--strict', '--eval_lm_eval', '--eval_device', 'cuda:0', '--tasks', tasks, '--num_fewshot', '0',
            '--lm_batch_size', batch, '--tokenizer_use_fast', 'true', '--prewarm_group_size', '8',
            '--eval_log_dir', str(output / (name + '_logs')), '--output_json', str(output / (name + '.json'))]})
    record = {'status': 'prepared_unbound', 'arm': ARM, 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'source_checkpoint': str(SOURCE), 'source_checkpoint_id': SOURCE_ID,
        'source_weight_stat': gate_launch['source_weight_stat'], 'hf_weight_stats': gate_launch['hf_weight_stats'],
        'architecture': inventory, 'source_bit_budget': budget, 'source_sha256': sources, 'input_sha256': inputs,
        'endpoint_path': str(path), 'endpoint_sha256': ENDPOINT_SHA, 'endpoint_state': training['final_payload'],
        'patch_contract': patch, 'selected_projections': ['gate_proj'], 'patched_local_tensor_count': 13,
        'frozen_outside_target_tensors': 3410, 'selected_decoder_tensors': 6,
        'source_native_policy': gate_launch['source_native_policy'], 'stage_commands': commands,
        'control_result_paths': [str(x) for x in control_paths], 'control_mean_percent': control['mean_percent'],
        'gate_only_control_results': [str(x) for x in gate_paths], 'gate_only_control_checkpoint': str(gate_model),
        'gate_only_control_checkpoint_id': gate_meta['checkpoint_id'], 'gate_only_control_mean_percent': gate_control['mean_percent'],
        'fit_report_sha256': REPORT_SHA, 'fit_review_sha256': REVIEW_SHA,
        'timeout_seconds': 1500, 'termination_grace_seconds': 20,
        'environment': {'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'},
        'protocol': {'full8_leaves': 64, 'full8_samples': 24742, 'num_fewshot': 0, 'mmlu_batch': 64,
            'seven_batch': 32, 'limit': None, 'single_layer_only': TARGET, 'new_training': False,
            'mutable_decoder_tensors': 6, 'mutable_packed_banks': 2, 'frozen_other_packed_banks': 502,
            'longbench': 'not_evaluated', 'nine_task_goal': 'not_evaluated',
            'boundary': 'Isolated quality verification of the actual group-accepted plus decoder500 endpoint. Extra search/acceptance compute differs from original-code500; no production integration or all-layer rollout.'},
        'launcher_argv': [sys.executable, '-m', 'experiments.gate_group_accept_evaluation.run',
                          '--launch_record', str(output / 'launch.json')], 'cuda_initialized': False}
    require(not torch.cuda.is_initialized(), 'CPU preparation must not initialize CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record)
    print(json.dumps({'status': record['status'], 'launch_sha256': sha256(output / 'launch.json'),
        'source_pins': len(sources), 'input_pins': len(inputs), 'stage_changed_bits': patch['stage_changed_bits'],
        'source_user_linear_bits': budget['user_linear_bits'], 'source_bpw': budget['user_linear_bpw'],
        'cuda_initialized': False}), flush=True)


if __name__ == '__main__':
    main()
