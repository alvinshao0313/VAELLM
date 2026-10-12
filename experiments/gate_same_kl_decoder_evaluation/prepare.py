"""Bind one reviewed final-mass-KL500 gate endpoint to a fresh native full8 run."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import sys

import torch

from experiments.full_weight_bits_probe.core import popcounts
from experiments.gate_final_kl_accept_probe.run import check as check_fit
from experiments.gate_student_context_evaluation.prepare import (
    ROOT, SOURCE, SOURCE_ID, TARGET, BANKS, budget_equal, read, require, require_state,
)
from experiments.gate_student_context_evaluation.run import check
from experiments.longbench_v1.common import sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.checkpoint_bpw import audit_state_dict

BASE = ROOT / 'result/qwen3_8b_mmlu65_20261004'
EXPECTED_XOR = {'original': [0, 0], 'alpaca128': [15483, 645], 'alpaca64_fineweb64': [15468, 660]}
DECODER = '_parallel_stage_decoder.'


def patched_state(source, endpoint, arm):
    require(arm in EXPECTED_XOR and len(source) == 3423, 'Require a predeclared sameKL500 arm on fresh f927.')
    local = {key[len(TARGET) + 1:]: value for key, value in source.items() if key.startswith(TARGET + '.')}
    require(set(local) == set(endpoint) and len(local) == 13, 'Exactly thirteen native gate tensors are required.')
    changes, frozen = {}, 0
    for key, original in local.items():
        value = endpoint[key]
        require(value.shape == original.shape and value.dtype == original.dtype, 'Local topology/dtype changed: ' + key)
        if key.startswith(DECODER):
            require(value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all()), 'Require finite deployed BF16 decoder.')
            changes[key] = int(torch.count_nonzero(value != original))
        elif key not in BANKS:
            require_state({key: value}, {key: original}, 'Frozen protection and rotation state'); frozen += 1
        else:
            require_state({key: value[16128:]}, {key: original[16128:]}, 'Codes outside groups0..3')
    require(len(changes) == 6 and frozen == 5, 'Only the six decoder tensors and the two predeclared banks may differ.')
    xor = [popcounts(local[key], endpoint[key]) for key in BANKS]
    require(xor == EXPECTED_XOR[arm], 'Packed changes differ from the fixed-code arm.')
    expected = {**source, **{TARGET + '.' + key: value for key, value in endpoint.items()}}
    banks = [key for key in source if key.rsplit('.', 1)[-1] in BANKS]
    outside = [key for key in banks if not key.startswith(TARGET + '.')]
    require(len(banks) == 504 and len(outside) == 502 and all(expected[key] is source[key] for key in outside),
            'Another projection bank was patched.')
    return expected, {'patched_local_tensor_count': 13, 'frozen_outside_target_tensors': 3410,
        'selected_decoder_tensors': 6, 'selected_projections': ['gate_proj'], 'changed_decoder_elements': changes,
        'actual_changed_decoder_tensors': sum(value > 0 for value in changes.values()),
        'stage_changed_bits': xor, 'changed_code_bits': sum(xor), 'source_packed_bank_count': 504,
        'frozen_outside_target_packed_bank_count': 502, 'frozen_outside_target_packed_banks_exact': True,
        'frozen_local_nonmutable_tensors': frozen, 'frozen_local_nonmutable_exact': True}


def prepare(fit, output, arm):
    fit, output = Path(fit).resolve(), Path(output).resolve()
    require(arm in EXPECTED_XOR and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and Path.cwd() == ROOT and not output.exists(), 'Use a fresh CPU-only output in the public root.')
    torch.set_num_threads(4)
    previous, terminal, report, review = (read(fit / name) for name in
        ('launch.prepared.json', 'launch.json', 'run/report.json', 'review.json'))
    require(previous['schema'] == 'gate_same_final_mass_kl_decoder_single500_v1'
            and previous['arm'] == terminal['arm'] == report['arm'] == arm
            and terminal['status'] == report['status'] == 'completed' and terminal['exit_code'] == 0
            and review['status'] == 'verified' and review['report_sha256'] == terminal['report_sha256']
            == sha256(fit / 'run/report.json') and terminal['prepared_sha256'] == sha256(fit / 'launch.prepared.json')
            and report['source_checkpoint_id'] == previous['source_checkpoint_id'] == SOURCE_ID
            and all((fit / name).read_text().strip() == '0' for name in ('run.exit', 'outer.exit')),
            'Require this arm actual500 terminal, unique reviewed report and successful outer exit.')
    check_fit(previous)
    require(tuple(report['experiment']['arms']) == (arm,) and previous['protocol']['updates_per_arm'] == 500
            and previous['protocol']['schedule_steps'] == 500, 'The two-update mechanism is not a formal endpoint.')
    result = report['experiment']['arms'][arm]; training, saved = result['training'], result['saved']
    mechanism = Path(previous['mechanism_review']).parent
    settings = read(mechanism / 'run/report.json')['experiment']['arms']['original']['training']['settings']
    source_data = previous['data']['source_data']
    initial = source_data['expected_gate_state'] if arm == 'original' else previous['arms'][arm]['initial_state']
    require(training['status'] == 'completed' and training['steps'] == training['schedule_steps'] == 500
            and len(training['trajectory']) == 500 and training['trajectory'][-1]['step'] == 500
            and training['actual_trainable_forward_calls'] == 16000 and training['all_six_gradients_finite']
            and all(value > 0 for value in training['stage_delta_l2_fp32'])
            and training['settings'] == settings and training['initial_payload'] == initial
            and training['frozen_local_tensors'] == 7 and training['two_banks_and_five_local_exact']
            and result['frozen_complement_count'] == 3410 and result['frozen_complement_versions_and_pointers_exact']
            and result['original_state_and_full_logits_restored_exact']
            and tuple(result['endpoint_observations']) == ('alpaca_held32', 'fineweb_rows500000_500031')
            and saved['tensor_count'] == 13 and saved['fresh_strict_local_state_exact']
            and saved['native_full_weight_reload_exact'], 'The actual final-mass-KL training/save contract differs.')
    path = Path(saved['path'])
    require(path == fit / 'run' / (arm + '_gate_state.pt') and sha256(path) == saved['sha256'], 'The saved endpoint changed.')
    endpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    require({key: tensor_record(value) for key, value in endpoint.items()} == training['final_payload']
            and all(training['final_payload'][key] == value for key, value in initial.items() if not key.startswith(DECODER)),
            'Saved thirteen tensors differ or fixed codes/protection/rotation changed during fitting.')
    template_dir = BASE / 'gate_student_context_full8_r1/original_codes'
    template = read(template_dir / 'launch.json'); check(template)
    meta = read(SOURCE / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == SOURCE_ID and template['source_checkpoint_id'] == SOURCE_ID, 'Fresh f927 identity differs.')
    source = torch.load(SOURCE / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    require({key[len(TARGET) + 1:]: tensor_record(value) for key, value in source.items() if key.startswith(TARGET + '.')}
            == source_data['expected_gate_state'], 'The source gate changed.')
    expected, patch = patched_state(source, endpoint, arm)
    budget = audit_state_dict(meta, source, max_bpw=2.5, max_lora_rank=0)
    budget_equal(budget, previous['source_bit_budget']); budget_equal(audit_state_dict(meta, expected, max_bpw=2.5, max_lora_rank=0), budget)
    require(budget['user_linear_bits'] == 17065672704, 'Original integer storage differs.')
    keys = ('cwd', 'python_executable', 'source_checkpoint', 'source_checkpoint_id', 'source_weight_stat', 'hf_weight_stats',
        'architecture', 'source_native_policy', 'control_result_paths', 'control_mean_percent', 'environment')
    record = {key: deepcopy(template[key]) for key in keys}
    sources, inputs = dict(previous['source_sha256']), dict(previous['input_sha256'])
    for inherited, dest in ((template['source_sha256'], sources), (template['input_sha256'], inputs)):
        for key, value in inherited.items():
            require(key not in dest or dest[key] == value, 'Source/evidence version conflict: ' + key); dest[key] = value
    helper_paths = [ROOT / relative for relative in ('experiments/full_rht_gate_pair/compare.py',
        'experiments/full_rht_gate_pair/audit.py', 'experiments/full_rht_gate_pair/common.py',
        'experiments/gate_down_student_context_evaluation/run.py')]
    for source_path in [*Path(__file__).resolve().parent.glob('*.py'), *helper_paths]:
        ast.parse(source_path.read_text()); digest = sha256(source_path)
        require(str(source_path) not in sources or sources[str(source_path)] == digest,
                'A reused evaluation helper version differs: ' + str(source_path))
        sources[str(source_path)] = digest
    evidence = [fit / name for name in ('launch.prepared.json', 'launch.bound.json', 'launch.json', 'run/report.json', 'review.json', 'run.exit', 'outer.exit')]
    evidence += [path, mechanism / 'run/report.json', template_dir / 'launch.json']
    inputs.update({str(item): sha256(item) for item in evidence})
    commands = deepcopy(template['stage_commands'])
    commands[0]['argv'] = [sys.executable, '-m', 'experiments.gate_same_kl_decoder_evaluation.deploy', '--launch_record', str(output / 'launch.json')]
    for stage in commands[1:]: stage['argv'] = [value.replace(str(template_dir), str(output)) for value in stage['argv']]
    record.update(status='prepared_unbound', gpu=None, arm=arm, queued=False, automatic_retry=False,
        output_dir=str(output), source_bit_budget=budget, source_sha256=sources, input_sha256=inputs,
        endpoint_path=str(path), endpoint_sha256=saved['sha256'], endpoint_state=training['final_payload'], patch_contract=patch,
        selected_projections=['gate_proj'], patched_local_tensor_count=13, frozen_outside_target_tensors=3410,
        selected_decoder_tensors=6, stage_commands=commands, fit_dir=str(fit), fit_report_sha256=sha256(fit / 'run/report.json'),
        fit_review_sha256=sha256(fit / 'review.json'), training_contract={'settings': settings,
            'train_tokens': previous['token_batches']['train'], 'teacher_caches': previous['teacher_caches'],
            'fresh_decoder_records': {key: value for key, value in initial.items() if key.startswith(DECODER)}},
        timeout_seconds=900, termination_grace_seconds=20, primary_control='f927_until_three_arm_comparison',
        paired_comparison='After all three complete, compare new-code arms against original sameKL500. No future UUID/raw is bound here.',
        protocol={'full8_leaves': 64, 'full8_samples': 24742, 'num_fewshot': 0, 'mmlu_batch': 64, 'seven_batch': 32,
            'limit': None, 'single_layer_only': TARGET, 'new_training': False, 'mutable_decoder_tensors': 6,
            'longbench': 'not_evaluated', 'nine_task_goal': 'not_evaluated',
            'boundary': 'All three fresh same-final-KL500 endpoints receive full8 regardless of development loss. Six decoder tensors may change; actual BF16 survival is recorded, not required6/6. Fixed original or reviewed128 codes. No new fitting/search/LB/integration.'},
        launcher_argv=[sys.executable, '-m', 'experiments.gate_same_kl_decoder_evaluation.run', '--launch_record', str(output / 'launch.json')],
        cuda_initialized=False)
    check(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record); write_json(output / 'launch.prepared.json', record)
    print({'status': record['status'], 'arm': arm, 'launch_sha256': sha256(output / 'launch.json'),
        'sources': len(sources), 'inputs': len(inputs), 'actual_bpw': budget['user_linear_bpw'], 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-dir', required=True, type=Path); parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=tuple(EXPECTED_XOR))
    args = parser.parse_args(); prepare(args.fit_dir, args.output_root, args.arm)
