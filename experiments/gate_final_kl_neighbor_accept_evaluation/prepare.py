"""Bind the reviewed384-group final-KL accepted code endpoint to native full8."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import os
from pathlib import Path
import sys

import torch

from experiments.full_weight_bits_probe.core import popcounts
from experiments.gate_group_accept_probe.core import GROUPS, VECTORS_PER_GROUP, apply_choices, group_inventory
from experiments.gate_student_context_evaluation.prepare import (
    ROOT, SOURCE, SOURCE_ID, TARGET, BANKS, budget_equal, read, require, require_state,
)
from experiments.gate_student_context_evaluation.run import check
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.w25_eval.acceptance import collect_results
from train_utils.checkpoint_bpw import audit_state_dict

ARM = 'final_kl_gradient384_accepted_codes_only'
EXPECTED_XOR = [593465, 23431]


def replay_endpoint(source, choices, masks):
    group_inventory(choices)
    require(set(masks) == {'accepted_groups', 'effective_choices'}, 'Unexpected accepted-mask fields.')
    accepted, effective = masks['accepted_groups'], masks['effective_choices']
    require(accepted.dtype == torch.bool and tuple(accepted.shape) == (GROUPS,) and int(accepted.sum()) == 153
            and effective.dtype == torch.uint8 and effective.shape == choices.shape
            and torch.equal(effective, torch.where(accepted.repeat_interleave(VECTORS_PER_GROUP), choices, torch.zeros_like(choices))),
            'Effective codes must be exactly the complete384 proposal masked by actual153 accepts.')
    require([int(((effective > stage * 32) & (effective <= (stage + 1) * 32)).sum())
             for stage in range(2)] == EXPECTED_XOR, 'Accepted choices do not match the reviewed stage XOR.')
    local = {key[len(TARGET) + 1:]: value for key, value in source.items() if key.startswith(TARGET + '.')}
    require(len(local) == 13, 'Unexpected native gate inventory.')
    endpoint = dict(local)
    for stage, key in enumerate(BANKS):
        require(local[key].dtype == torch.uint8 and tuple(local[key].shape) == (1548288, 1, 4),
                'Require original D32/K32/S2 packed geometry.')
        endpoint[key] = local[key].clone()
        apply_choices(endpoint[key], effective, stage)
    return endpoint


def patched_state(source, endpoint, arm):
    require(arm == ARM and len(source) == 3423, 'Use the reviewed code-only endpoint on complete fresh f927.')
    local = {key[len(TARGET) + 1:]: value for key, value in source.items() if key.startswith(TARGET + '.')}
    require(set(endpoint) == set(local) and len(local) == 13, 'Exactly thirteen native gate tensors are required.')
    frozen = decoder = 0
    for key, original in local.items():
        require(endpoint[key].dtype == original.dtype and endpoint[key].shape == original.shape,
                'Local topology or dtype differs: ' + key)
        if key not in BANKS:
            require_state({key: endpoint[key]}, {key: original}, 'Frozen decoder/protection/rotation')
            frozen += 1
            decoder += int(key.startswith('_parallel_stage_decoder.'))
    require(frozen == 11 and decoder == 6, 'All decoder/protection/rotation tensors must remain unchanged.')
    xor = [popcounts(local[key], endpoint[key]) for key in BANKS]
    require(xor == EXPECTED_XOR, 'Only the actual reviewed accepted code changes may be deployed.')
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


def prepare(base, output):
    base, output = Path(base).resolve(), Path(output).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized() and not output.exists(),
            'Use a fresh CPU-only preparation directory.')
    torch.set_num_threads(4)
    proto, fixed = base / 'gate_final_kl_neighbor_accept_probe', base / 'gate_final_kl_neighbor4_full8'
    previous, report, review, terminal = (read(proto / name) for name in
        ('launch.prepared.json', 'run/report.json', 'review.json', 'launch.json'))
    require(report['status'] == terminal['status'] == 'completed' and terminal['exit_code'] == 0
            and all((proto / name).read_text().strip() == '0' for name in ('run.exit', 'outer.exit'))
            and review['status'] == 'verified' and review['report_sha256'] == terminal['report_sha256'] == sha256(proto / 'run/report.json')
            and review['prepared_sha256'] == sha256(proto / 'launch.prepared.json')
            and review['terminal_sha256'] == sha256(proto / 'launch.json')
            and review['all384_strict_decisions_and_fixed_order_recomputed']
            and report['source_checkpoint_id'] == review['source_checkpoint_id'] == SOURCE_ID,
            'Require the completed and independently reviewed actual384 search.')
    search = report['search']
    require(search['status'] == 'completed' and search['groups'] == 384 and search['order'] == list(range(384))
            and search['accepted_groups'] == review['accepted_groups'] == 153
            and search['stage_xor'] == review['actual_stage_xor'] == EXPECTED_XOR
            and search['frozen_complement_count'] == review['frozen_complement_count'] == 3410
            and review['changed_code_bits'] == sum(EXPECTED_XOR)
            and review['local11_noncode_bytes_exact'] and review['native_full_bank_mask_replay_exact']
            and all(search[key] for key in ('noncode_gate_frozen_exact', 'frozen_complement_versions_and_pointers_exact',
                'mask_replay_exact', 'final_train_score_repeated_exact', 'original_full_logits_restore_exact',
                'strict_final13_and_group8_weight_reload_exact')), 'Native search/frozen/replay evidence differs.')
    check_hashes(previous['source_sha256']); check_hashes(previous['input_sha256'])
    for stat in previous['protected_stats']: check_stat(stat)
    saved = report['saved']
    state_path, choice_path, mask_path = (Path(saved[key]) for key in ('state_path', 'choices_path', 'mask_path'))
    for path, key, review_key in ((state_path, 'state_sha256', 'saved_endpoint_state_sha256'),
            (choice_path, 'choices_sha256', 'generated_choices_sha256'), (mask_path, 'mask_sha256', 'saved_mask_sha256')):
        require(path.parent == proto / 'run' and sha256(path) == saved[key] == review[review_key], 'Saved search artifact changed: ' + str(path))
    endpoint = torch.load(state_path, map_location='cpu', weights_only=True, mmap=True)
    choices = torch.load(choice_path, map_location='cpu', weights_only=True)
    masks = torch.load(mask_path, map_location='cpu', weights_only=True)
    template, fixed_review = read(fixed / 'launch.json'), read(fixed / 'review.json')
    require(template['status'] == 'completed' and template['exit_code'] == 0 and fixed_review['status'] == 'verified'
            and fixed_review['checkpoint_id'] == template['checkpoint_id']
            and all((fixed / (name + '.exit')).read_text().strip() == '0' for name in ('deploy', 'mmlu', 'seven', 'dispatcher', 'outer')),
            'Require the completed actual fixed4 full8 reference.')
    check(template)
    meta = read(SOURCE / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == SOURCE_ID, 'Only fresh f927 may supply the deployment.')
    source = torch.load(SOURCE / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    local = {key[len(TARGET) + 1:]: tensor_record(value) for key, value in source.items() if key.startswith(TARGET + '.')}
    require(local == search['initial_state'], 'Source thirteen tensors differ from the actual search initial state.')
    replayed = replay_endpoint(source, choices, masks)
    require_state(endpoint, replayed, 'Saved13 versus complete CPU accepted-mask replay')
    endpoint_records = {key: tensor_record(value) for key, value in endpoint.items()}
    require(endpoint_records == search['final_state'], 'Saved endpoint differs from the real native final state.')
    expected, patch = patched_state(source, endpoint, ARM)
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
    require(control['mean_percent'] == fixed_review['mean_percent'] == template['mean_percent'], 'Fixed4 raw score differs.')
    sources, inputs = dict(template['source_sha256']), dict(template['input_sha256'])
    for path, digest in previous['source_sha256'].items():
        require(path not in sources or sources[path] == digest, 'Inherited source version conflict: ' + path)
        sources[path] = digest
    for path in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
    for path in (proto / 'launch.prepared.json', proto / 'launch.json', proto / 'run/report.json', proto / 'review.json',
        proto / 'run.exit', proto / 'outer.exit', state_path, choice_path, mask_path, fixed / 'launch.json',
        fixed / 'review.json', fixed / 'audit.json', fixed / 'full8_summary.json', fixed_model / 'checkpoint_meta.json',
        *fixed_paths, *(fixed / (name + '.exit') for name in ('deploy', 'mmlu', 'seven', 'dispatcher', 'outer'))):
        inputs[str(path)] = sha256(path)
    commands = deepcopy(template['stage_commands'])
    commands[0]['argv'] = [sys.executable, '-m', 'experiments.gate_final_kl_neighbor_accept_evaluation.deploy', '--launch_record', str(output / 'launch.json')]
    for stage in commands[1:]:
        stage['argv'] = [value.replace(str(fixed), str(output)) for value in stage['argv']]
    record.update(status='prepared_unbound', arm=ARM, gpu=None, queued=False, automatic_retry=False,
        output_dir=str(output), source_bit_budget=budget, source_sha256=sources, input_sha256=inputs,
        endpoint_path=str(state_path), endpoint_sha256=saved['state_sha256'], endpoint_state=endpoint_records,
        choices_path=str(choice_path), choices_sha256=saved['choices_sha256'], mask_path=str(mask_path), mask_sha256=saved['mask_sha256'],
        patch_contract=patch, selected_projections=['gate_proj'], patched_local_tensor_count=13,
        frozen_outside_target_tensors=3410, selected_decoder_tensors=0, stage_commands=commands,
        fixed4_control_results=list(map(str, fixed_paths)), fixed4_control_checkpoint=str(fixed_model),
        fixed4_control_checkpoint_id=fixed_meta['checkpoint_id'], fixed4_control_mean_percent=control['mean_percent'],
        search_report_sha256=sha256(proto / 'run/report.json'), search_review_sha256=sha256(proto / 'review.json'),
        endpoint_artifact_kind='native13_state_verified_by_fresh_source_choices_accepted_mask_replay',
        groups=384, accepted_groups=153, timeout_seconds=1200, termination_grace_seconds=20,
        protocol={'full8_leaves': 64, 'full8_samples': 24742, 'num_fewshot': 0, 'mmlu_batch': 64, 'seven_batch': 32,
            'limit': None, 'single_layer_only': TARGET, 'new_training': False, 'mutable_decoder_tensors': 0,
            'mutable_packed_banks': 2, 'frozen_other_packed_banks': 502, 'longbench': 'not_evaluated', 'nine_task_goal': 'not_evaluated',
            'boundary': 'Deploy only the saved153-of384 train32 final-KL accepts from original-gradient proposals. No decoder fit or extra search. f927 primary; fixed4 and original-code gate500 are secondary references with different local optimization budgets. Full8 judges transfer directly; no production integration or all-layer rollout.'},
        launcher_argv=[sys.executable, '-m', 'experiments.gate_final_kl_neighbor_accept_evaluation.run', '--launch_record', str(output / 'launch.json')],
        cuda_initialized=False)
    check(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record)
    print({'status': record['status'], 'launch_sha256': sha256(output / 'launch.json'), 'source_pins': len(sources),
        'input_pins': len(inputs), 'stage_xor': EXPECTED_XOR, 'accepted_groups': 153,
        'actual_bpw': budget['user_linear_bpw'], 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.base_root, args.output_root)
