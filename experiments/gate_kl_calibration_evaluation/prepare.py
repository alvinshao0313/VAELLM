"""Bind both reviewed calibration128 code-only endpoints to independent full8 runs."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import sys

import torch

from experiments.full_weight_bits_probe.core import popcounts
from experiments.gate_group_accept_probe.core import apply_choices
from experiments.gate_final_kl_neighbor_transfer.core import choice_inventory
from experiments.gate_student_context_evaluation.prepare import (
    ROOT, SOURCE, SOURCE_ID, TARGET, BANKS, budget_equal, read, require, require_state,
)
from experiments.gate_student_context_evaluation.run import check
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.w25_eval.acceptance import collect_results
from train_utils.checkpoint_bpw import audit_state_dict

EXPECTED_XOR = {'alpaca128': [15483, 645], 'alpaca64_fineweb64': [15468, 660]}


def replay_endpoint(source, choices, arm):
    require(arm in EXPECTED_XOR, 'Unknown predeclared calibration arm.')
    inventory = choice_inventory(choices)
    require([x['group'] for x in inventory] == [0, 1, 2, 3]
            and [sum(x['stage_xor'][s] for x in inventory) for s in range(2)] == EXPECTED_XOR[arm],
            'Only the reviewed four-group compact choices may be replayed.')
    local = {key[len(TARGET) + 1:]: value for key, value in source.items() if key.startswith(TARGET + '.')}
    require(len(local) == 13, 'Unexpected native L18 gate inventory.')
    endpoint = dict(local)
    for stage, key in enumerate(BANKS):
        require(local[key].dtype == torch.uint8 and tuple(local[key].shape) == (1548288, 1, 4),
                'Require original Qwen L18 K32/D32/S2 packed bank geometry.')
        endpoint[key] = local[key].clone()
        apply_choices(endpoint[key][:choices.numel()], choices, stage)
    return endpoint


def patched_state(source, endpoint, arm):
    require(arm in EXPECTED_XOR and len(source) == 3423, 'Require a reviewed calibration arm on full fresh f927.')
    local = {key[len(TARGET) + 1:]: value for key, value in source.items() if key.startswith(TARGET + '.')}
    require(set(local) == set(endpoint) and len(local) == 13, 'Exactly thirteen native gate tensors are required.')
    frozen = decoder = 0
    for key, original in local.items():
        value = endpoint[key]
        require(value.shape == original.shape and value.dtype == original.dtype, 'Local topology/dtype changed: ' + key)
        if key not in BANKS:
            require_state({key: value}, {key: original}, 'Frozen decoder, protection and rotation state')
            frozen += 1
            decoder += int(key.startswith('_parallel_stage_decoder.'))
        else:
            require_state({key: value[16128:]}, {key: original[16128:]}, 'Codes outside groups0..3')
    require(frozen == 11 and decoder == 6, 'All local noncode tensors must remain exact.')
    xor = [popcounts(local[key], endpoint[key]) for key in BANKS]
    require(xor == EXPECTED_XOR[arm], 'Actual packed changes differ from the reviewed calibration proposal.')
    expected = {**source, **{TARGET + '.' + key: value for key, value in endpoint.items()}}
    outside = [key for key in source if not key.startswith(TARGET + '.')]
    banks = [key for key in source if key.rsplit('.', 1)[-1] in BANKS]
    require(len(outside) == 3410 and all(expected[key] is source[key] for key in outside)
            and len(banks) == 504 and sum(key in outside for key in banks) == 502, 'Another tensor was patched.')
    return expected, {'patched_local_tensor_count': 13, 'frozen_outside_target_tensors': 3410,
        'selected_decoder_tensors': 0, 'selected_projections': ['gate_proj'], 'changed_decoder_elements': {},
        'stage_changed_bits': xor, 'changed_code_bits': sum(xor), 'source_packed_bank_count': 504,
        'frozen_outside_target_packed_bank_count': 502, 'frozen_outside_target_packed_banks_exact': True,
        'frozen_local_nonmutable_tensors': 11, 'frozen_local_nonmutable_exact': True,
        'all_decoder_tensors_exact': True, 'frozen_complete_state_tensors': 3421}


def prepare(base, output, arm):
    base, output = Path(base).resolve(), Path(output).resolve()
    require(arm in EXPECTED_XOR and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized() and not output.exists(), 'Use a fresh CPU-only preparation directory.')
    torch.set_num_threads(4)
    proto, fixed = base / 'gate_kl_calibration_probe', base / 'gate_final_kl_neighbor4_full8'
    previous, report, review, terminal = (read(proto / name) for name in
        ('launch.prepared.json', 'run/report.json', 'review.json', 'launch.json'))
    require(report['status'] == terminal['status'] == 'completed' and terminal['exit_code'] == 0
            and all((proto / name).read_text().strip() == '0' for name in ('run.exit', 'outer.exit'))
            and review['status'] == 'verified' and review['report_sha256'] == terminal['report_sha256'] == sha256(proto / 'run/report.json')
            and review['prepared_sha256'] == sha256(proto / 'launch.prepared.json')
            and review['terminal_sha256'] == sha256(proto / 'launch.json')
            and report['source_checkpoint_id'] == review['source_checkpoint_id'] == SOURCE_ID,
            'Require the actually completed, independently reviewed calibration experiment.')
    experiment = report['experiment']
    require(experiment['status'] == 'completed' and experiment['source_checkpoint_id'] == SOURCE_ID
            and experiment['frozen_complement_count'] == 3410
            and all(experiment[key] for key in ('all13_final_bytes_exact', 'frozen_complement_versions_and_pointers_exact',
                'all_parameters_frozen_and_grad_none', 'original_cache_and_full_logits_restored_exact')),
            'Native source freezing and final restoration proof differs.')
    check_hashes(previous['source_sha256']); check_hashes(previous['input_sha256'])
    for stat in previous['protected_stats']: check_stat(stat)
    saved, proof = report['saved_choices'][arm], experiment['endpoint_records'][arm]
    choice_path = Path(saved['path'])
    require(choice_path == proto / 'run' / (arm + '_choices.pt') and sha256(choice_path) == saved['sha256']
            and saved['roundtrip_exact'] and proof['stage_xor'] == EXPECTED_XOR[arm]
            and proof['complete_bank_replay_exact'] and proof['local11_noncode_bytes_exact'], 'Saved proposal identity differs.')
    choices = torch.load(choice_path, map_location='cpu', weights_only=True)
    require(tensor_record(choices) == saved['tensor'] == experiment['generations'][arm]['choices']
            and choice_inventory(choices) == saved['inventory'] == proof['choice_inventory'], 'Compact proposal bytes differ.')
    template, fixed_review = read(fixed / 'launch.json'), read(fixed / 'review.json')
    require(template['status'] == 'completed' and template['exit_code'] == 0 and fixed_review['status'] == 'verified'
            and fixed_review['checkpoint_id'] == template['checkpoint_id']
            and all((fixed / (name + '.exit')).read_text().strip() == '0' for name in ('deploy', 'mmlu', 'seven', 'dispatcher', 'outer')),
            'Require the actual completed A32 fixed4 full8 reference, without its retired weights.')
    check(template)
    meta = read(SOURCE / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == SOURCE_ID, 'Only fresh f927 supplies the deployment.')
    source = torch.load(SOURCE / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    require({key[len(TARGET) + 1:]: tensor_record(value) for key, value in source.items() if key.startswith(TARGET + '.')}
            == experiment['initial_state'], 'Source thirteen tensors differ from the actual proposal initial state.')
    endpoint = replay_endpoint(source, choices, arm)
    endpoint_records = {key: tensor_record(value) for key, value in endpoint.items()}
    require(endpoint_records == proof['endpoint_state'] == experiment['generations'][arm]['applied_state']['endpoint_state'],
            'CPU complete-bank replay differs from the real native endpoint.')
    expected, patch = patched_state(source, endpoint, arm)
    budget = audit_state_dict(meta, source, max_bpw=2.5, max_lora_rank=0)
    budget_equal(budget, template['source_bit_budget'])
    budget_equal(audit_state_dict(meta, expected, max_bpw=2.5, max_lora_rank=0), budget)
    require(budget['user_linear_bits'] == 17065672704, 'Persistent integer budget changed.')
    keys = ('cwd', 'python_executable', 'source_checkpoint', 'source_checkpoint_id', 'source_weight_stat', 'hf_weight_stats',
        'architecture', 'source_native_policy', 'control_result_paths', 'control_mean_percent', 'gate_only_control_results',
        'gate_only_control_checkpoint', 'gate_only_control_checkpoint_id', 'gate_only_control_mean_percent', 'environment')
    record = {key: deepcopy(template[key]) for key in keys}
    fixed_paths = [fixed / (name + '.json') for name in ('mmlu', 'seven')]
    fixed_model = fixed / 'final_model'; fixed_meta = read(fixed_model / 'checkpoint_meta.json')
    control = collect_results(fixed_paths, record['architecture'], checkpoint=(fixed_model, fixed_meta))
    require(control['mean_percent'] == fixed_review['mean_percent'] == template['mean_percent'], 'Historical A32 raw score differs.')
    sources, inputs = dict(template['source_sha256']), dict(template['input_sha256'])
    for target, inherited in ((sources, previous['source_sha256']), (inputs, previous['input_sha256'])):
        for path, digest in inherited.items():
            require(path not in target or target[path] == digest, 'Inherited version conflict: ' + path)
            target[path] = digest
    for path in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
    for path in (proto / 'launch.prepared.json', proto / 'launch.json', proto / 'run/report.json', proto / 'review.json',
        proto / 'run.exit', proto / 'outer.exit', choice_path, fixed / 'launch.json', fixed / 'review.json', fixed / 'audit.json',
        fixed / 'full8_summary.json', fixed_model / 'checkpoint_meta.json', *fixed_paths,
        *(fixed / (name + '.exit') for name in ('deploy', 'mmlu', 'seven', 'dispatcher', 'outer'))):
        inputs[str(path)] = sha256(path)
    commands = deepcopy(template['stage_commands'])
    commands[0]['argv'] = [sys.executable, '-m', 'experiments.gate_kl_calibration_evaluation.deploy', '--launch_record', str(output / 'launch.json')]
    for stage in commands[1:]: stage['argv'] = [value.replace(str(fixed), str(output)) for value in stage['argv']]
    record.update(status='prepared_unbound', arm=arm, gpu=None, queued=False, automatic_retry=False,
        output_dir=str(output), source_bit_budget=budget, source_sha256=sources, input_sha256=inputs,
        endpoint_path=str(choice_path), endpoint_sha256=saved['sha256'], endpoint_state=endpoint_records,
        expected_stage_changed_bits=EXPECTED_XOR[arm], patch_contract=patch, selected_projections=['gate_proj'],
        patched_local_tensor_count=13, frozen_outside_target_tensors=3410, selected_decoder_tensors=0, stage_commands=commands,
        fixed4_control_results=list(map(str, fixed_paths)), fixed4_control_checkpoint=str(fixed_model),
        fixed4_control_checkpoint_id=fixed_meta['checkpoint_id'], fixed4_control_mean_percent=control['mean_percent'],
        calibration_report_sha256=sha256(proto / 'run/report.json'), calibration_review_sha256=sha256(proto / 'review.json'),
        endpoint_artifact_kind='compact_choices_replayed_on_fresh_source', groups=[0, 1, 2, 3],
        timeout_seconds=900, termination_grace_seconds=20,
        protocol={'full8_leaves': 64, 'full8_samples': 24742, 'num_fewshot': 0, 'mmlu_batch': 64, 'seven_batch': 32,
            'limit': None, 'single_layer_only': TARGET, 'new_training': False, 'mutable_decoder_tensors': 0,
            'mutable_packed_banks': 2, 'frozen_other_packed_banks': 502, 'longbench': 'not_evaluated', 'nine_task_goal': 'not_evaluated',
            'boundary': 'Both predeclared128-document proposals receive full8; no endpoint selection by proxy observations. Fixed groups0..3, code-only freshf927, no fit/search/LB/integration. f927 and historical A32 fixed4 are primary references; original-code gate500 is a different-method secondary reference.'},
        launcher_argv=[sys.executable, '-m', 'experiments.gate_kl_calibration_evaluation.run', '--launch_record', str(output / 'launch.json')],
        cuda_initialized=False)
    check(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record)
    print({'status': record['status'], 'arm': arm, 'launch_sha256': sha256(output / 'launch.json'),
        'source_pins': len(sources), 'input_pins': len(inputs), 'stage_xor': EXPECTED_XOR[arm],
        'actual_bpw': budget['user_linear_bpw'], 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--arm', choices=tuple(EXPECTED_XOR), required=True)
    args = parser.parse_args()
    prepare(args.base_root, args.output_root, args.arm)
