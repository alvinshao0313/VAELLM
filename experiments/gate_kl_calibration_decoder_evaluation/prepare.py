"""Bind two reviewed calibration-code plus decoder500 endpoints to native full8."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import sys

import torch

from experiments.full_weight_bits_probe.core import popcounts
from experiments.gate_kl_calibration_decoder_probe.fit import check as check_fit
from experiments.gate_kl_calibration_evaluation.prepare import EXPECTED_XOR
from experiments.gate_student_context_evaluation.prepare import (
    ROOT, SOURCE, SOURCE_ID, TARGET, BANKS, budget_equal, read, require, require_state,
)
from experiments.gate_student_context_evaluation.run import check
from experiments.longbench_v1.common import sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.w25_eval.acceptance import collect_results
from train_utils.checkpoint_bpw import audit_state_dict


def patched_state(source, endpoint, arm):
    require(arm in EXPECTED_XOR and len(source) == 3423, 'Require a reviewed fitted arm on full fresh f927.')
    local = {key[len(TARGET) + 1:]: value for key, value in source.items() if key.startswith(TARGET + '.')}
    require(set(local) == set(endpoint) and len(local) == 13, 'Exactly thirteen native gate tensors are required.')
    changes, frozen = {}, 0
    for key, original in local.items():
        value = endpoint[key]
        require(value.shape == original.shape and value.dtype == original.dtype, 'Local topology/dtype changed: ' + key)
        if key.startswith('_parallel_stage_decoder.'):
            require(value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all()), 'Require finite deployed BF16 decoder.')
            changes[key] = int(torch.count_nonzero(value != original))
        elif key not in BANKS:
            require_state({key: value}, {key: original}, 'Frozen protection and rotation state'); frozen += 1
        else:
            require_state({key: value[16128:]}, {key: original[16128:]}, 'Codes outside groups0..3')
    require(len(changes) == 6 and all(changes.values()) and frozen == 5,
            'Only six decoder updates, two code banks and five frozen local tensors are permitted.')
    xor = [popcounts(local[key], endpoint[key]) for key in BANKS]
    require(xor == EXPECTED_XOR[arm], 'Packed changes differ from the reviewed calibration proposal.')
    expected = {**source, **{TARGET + '.' + key: value for key, value in endpoint.items()}}
    banks = [key for key in source if key.rsplit('.', 1)[-1] in BANKS]
    outside = [key for key in banks if not key.startswith(TARGET + '.')]
    require(len(banks) == 504 and len(outside) == 502 and all(expected[key] is source[key] for key in outside),
            'Another projection bank was patched.')
    return expected, {'patched_local_tensor_count': 13, 'frozen_outside_target_tensors': 3410,
        'selected_decoder_tensors': 6, 'selected_projections': ['gate_proj'], 'changed_decoder_elements': changes,
        'stage_changed_bits': xor, 'changed_code_bits': sum(xor), 'source_packed_bank_count': 504,
        'frozen_outside_target_packed_bank_count': 502, 'frozen_outside_target_packed_banks_exact': True,
        'frozen_local_nonmutable_tensors': frozen, 'frozen_local_nonmutable_exact': True}


def prepare(base, output, arm):
    base, output = Path(base).resolve(), Path(output).resolve()
    require(arm in EXPECTED_XOR and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized() and not output.exists(), 'Use a fresh CPU-only preparation directory.')
    torch.set_num_threads(4)
    proto = base / 'gate_kl_calibration_decoder500'
    previous, report, review, bound = (read(proto / name) for name in
        ('preparation.json', 'run/report.json', 'review.json', 'launch.json'))
    require(report['status'] == 'completed' and (proto / 'outer.exit').read_text().strip() == '0'
            and review['status'] == 'verified' and review['exit_code'] == 0 and review['worker_absent']
            and review['report_sha256'] == sha256(proto / 'run/report.json')
            and review['preparation_sha256'] == report['preparation_sha256'] == sha256(proto / 'preparation.json')
            and review['launch_sha256'] == sha256(proto / 'launch.json')
            and bound['preparation_sha256'] == review['preparation_sha256']
            and report['source_checkpoint_id'] == SOURCE_ID,
            'Require the completed report, real outer exit and unique reviewed fit; the immutable binding is not a terminal record.')
    check_fit(previous)
    result, reviewed = report['arms'][arm], review['arms'][arm]
    training, saved = result['training'], result['saved']
    require(training['status'] == 'completed' and training['steps'] == training['schedule_steps'] == 500
            and training['settings'] == previous['expected_training_settings']
            and training['initial_payload'] == previous['arms'][arm]['initial_state']
            and training['persistent_bits'] == previous['expected_persistent_bits']
            and training['frozen_state_exact'] and training['frozen_objective_buffers_exact']
            and saved['saved_state_exact'] and saved['fixed_banks_exact']
            and reviewed['local_native_tensors'] == 13 and reviewed['decoder_tensors_changed'] == 6
            and reviewed['nondecoder_tensors_exact'] == 7 and reviewed['training_settings_exact']
            and reviewed['original_state_plus_reviewed_choices_exact']
            and reviewed['stage_xor_vs_f927'] == EXPECTED_XOR[arm], 'The completed fixed-code fit contract differs.')
    path = Path(saved['state_path'])
    require(path == proto / 'run' / (arm + '_gate_state.pt')
            and sha256(path) == saved['state_sha256'] == reviewed['endpoint_sha256'], 'The actual fitted endpoint changed.')
    endpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    require({key: tensor_record(value) for key, value in endpoint.items()} == training['final_payload'],
            'Saved thirteen tensors differ from the measured fit endpoint.')

    unfitted = base / 'gate_kl_calibration_full8' / arm
    template, control_review = read(unfitted / 'launch.json'), read(unfitted / 'review.json')
    require(template['status'] == 'completed' and template['exit_code'] == 0 and control_review['status'] == 'verified'
            and all((unfitted / (name + '.exit')).read_text().strip() == '0'
                    for name in ('deploy', 'mmlu', 'seven', 'dispatcher', 'outer')),
            'Require the actual complete unfitted-code comparison, without loading its full model.')
    check(template)
    require(template['endpoint_state'] == training['initial_payload'], 'Fitted and unfitted arms do not share initial codes/decoder.')
    meta = read(SOURCE / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == SOURCE_ID, 'Only fresh f927 supplies the deployment complement.')
    source = torch.load(SOURCE / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    require({key[len(TARGET) + 1:]: tensor_record(value) for key, value in source.items() if key.startswith(TARGET + '.')}
            == previous['original_state'], 'The original f927 gate changed.')
    expected, patch = patched_state(source, endpoint, arm)
    budget = audit_state_dict(meta, source, max_bpw=2.5, max_lora_rank=0)
    budget_equal(budget, template['source_bit_budget'])
    budget_equal(audit_state_dict(meta, expected, max_bpw=2.5, max_lora_rank=0), budget)
    require(budget['user_linear_bits'] == 17065672704, 'Original integer storage budget differs.')
    model_path = unfitted / 'final_model'; control_meta = read(model_path / 'checkpoint_meta.json')
    require(control_meta['checkpoint_id'] == template['checkpoint_id'] == control_review['checkpoint_id'], 'Unfitted control UUID differs.')
    control_paths = [unfitted / (name + '.json') for name in ('mmlu', 'seven')]
    control = collect_results(control_paths, template['architecture'], checkpoint=(model_path, control_meta))
    require(control['mean_percent'] == template['mean_percent'] == control_review['mean_percent'], 'Unfitted raw score differs.')

    keys = ('cwd', 'python_executable', 'source_checkpoint', 'source_checkpoint_id', 'source_weight_stat', 'hf_weight_stats',
        'architecture', 'source_native_policy', 'control_result_paths', 'control_mean_percent', 'gate_only_control_results',
        'gate_only_control_checkpoint', 'gate_only_control_checkpoint_id', 'gate_only_control_mean_percent', 'environment')
    record = {key: deepcopy(template[key]) for key in keys}
    sources, inputs = dict(template['source_sha256']), dict(template['input_sha256'])
    for target, inherited in ((sources, previous['source_sha256']), (inputs, previous['input_sha256'])):
        for key, digest in inherited.items():
            require(key not in target or target[key] == digest, 'Inherited evidence version conflict: ' + key)
            target[key] = digest
    for source_path in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(source_path.read_text()); sources[str(source_path)] = sha256(source_path)
    for evidence in (proto / 'preparation.json', proto / 'run/report.json', proto / 'review.json', proto / 'launch.json',
        proto / 'outer.exit', path, unfitted / 'launch.json', unfitted / 'review.json', unfitted / 'audit.json',
        unfitted / 'full8_summary.json', model_path / 'checkpoint_meta.json', *control_paths,
        *(unfitted / (name + '.exit') for name in ('deploy', 'mmlu', 'seven', 'dispatcher', 'outer'))):
        inputs[str(evidence)] = sha256(evidence)
    commands = deepcopy(template['stage_commands'])
    commands[0]['argv'] = [sys.executable, '-m', 'experiments.gate_kl_calibration_decoder_evaluation.deploy',
        '--launch_record', str(output / 'launch.json')]
    for stage in commands[1:]: stage['argv'] = [value.replace(str(unfitted), str(output)) for value in stage['argv']]
    record.update(status='prepared_unbound', arm=arm, gpu=None, queued=False, automatic_retry=False,
        output_dir=str(output), source_bit_budget=budget, source_sha256=sources, input_sha256=inputs,
        endpoint_path=str(path), endpoint_sha256=saved['state_sha256'], endpoint_state=training['final_payload'],
        patch_contract=patch, selected_projections=['gate_proj'], patched_local_tensor_count=13,
        frozen_outside_target_tensors=3410, selected_decoder_tensors=6, stage_commands=commands,
        unfitted_control_results=list(map(str, control_paths)), unfitted_control_checkpoint=str(model_path),
        unfitted_control_checkpoint_id=control_meta['checkpoint_id'], unfitted_control_mean_percent=control['mean_percent'],
        fit_report_sha256=sha256(proto / 'run/report.json'), fit_review_sha256=sha256(proto / 'review.json'),
        timeout_seconds=900, termination_grace_seconds=20, primary_control='original_codes500',
        protocol={'full8_leaves': 64, 'full8_samples': 24742, 'num_fewshot': 0, 'mmlu_batch': 64, 'seven_batch': 32,
            'limit': None, 'single_layer_only': TARGET, 'new_training': False, 'mutable_decoder_tensors': 6,
            'mutable_packed_banks': 2, 'frozen_other_packed_banks': 502, 'longbench': 'not_evaluated', 'nine_task_goal': 'not_evaluated',
            'boundary': 'Both calibration128 proposals plus the same fresh decoder500 recipe receive full8. Primary original-code500; secondary each unfitted-code endpoint and f927. Extra final-KL proposal data/compute differs; conditional MLP fit does not optimize final KL. No new fitting/search, LB or integration.'},
        launcher_argv=[sys.executable, '-m', 'experiments.gate_kl_calibration_decoder_evaluation.run',
                      '--launch_record', str(output / 'launch.json')], cuda_initialized=False)
    check(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record)
    print({'status': record['status'], 'arm': arm, 'launch_sha256': sha256(output / 'launch.json'),
        'sources': len(sources), 'inputs': len(inputs), 'actual_bpw': budget['user_linear_bpw'], 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-root', required=True, type=Path); parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=tuple(EXPECTED_XOR))
    args = parser.parse_args(); prepare(args.base_root, args.output_root, args.arm)
