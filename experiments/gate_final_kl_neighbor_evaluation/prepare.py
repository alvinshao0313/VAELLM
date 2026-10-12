"""Prepare full8 for the reviewed fixed four-group final-KL code-only endpoint."""
from __future__ import annotations

import argparse
import ast
import json
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
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.w25_eval.acceptance import collect_results
from experiments.w25_eval.model_inventory import model_inventory
from train_utils.checkpoint_bpw import audit_state_dict


ARM = 'final_kl_fixed4_codes_only'
EXPECTED_XOR = [15527, 601]
REPORT_SHA = '4891d4b642b23d44b078d0f483144f55a16fa904781d69b61f7e1264ba71afd7'
REVIEW_SHA = '241bec9e9e8de7395e428168a4ef76104fb9981701468cf38f924d889d9a6a8d'
ENDPOINT_SHA = '41b032ef565a1a7ed6719347f69df6eb70b9f54e398a2e2249f6dbdec81fd10e'


def replay_endpoint(source, choices):
    inventory = choice_inventory(choices)
    require([x['group'] for x in inventory] == [0, 1, 2, 3]
            and [sum(x['stage_xor'][s] for x in inventory) for s in range(2)] == EXPECTED_XOR,
            'Only the already fixed four-group choices may be replayed.')
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
    require(arm == ARM and len(source) == 3423, 'Require the reviewed code-only endpoint and full f927 source.')
    local = {key[len(TARGET) + 1:]: value for key, value in source.items() if key.startswith(TARGET + '.')}
    require(set(local) == set(endpoint) and len(local) == 13, 'Exactly the thirteen native gate tensors are required.')
    frozen, decoder = 0, 0
    for key, original in local.items():
        value = endpoint[key]
        require(value.shape == original.shape and value.dtype == original.dtype, 'Local topology/dtype changed: ' + key)
        if key not in BANKS:
            require_state({key: value}, {key: original}, 'Frozen decoder, protection and rotation state')
            frozen += 1
            decoder += int(key.startswith('_parallel_stage_decoder.'))
        else:
            require_state({key: value[16128:]}, {key: original[16128:]}, 'Codes outside fixed groups0..3')
    require(frozen == 11 and decoder == 6, 'All six decoder tensors and five protection/sign tensors must remain exact.')
    xor = [popcounts(local[key], endpoint[key]) for key in BANKS]
    require(xor == EXPECTED_XOR, 'Actual packed codes differ from the verified four-group proposal.')
    expected = {**source, **{TARGET + '.' + key: value for key, value in endpoint.items()}}
    outside = [key for key in source if not key.startswith(TARGET + '.')]
    banks = [key for key in source if key.rsplit('.', 1)[-1] in BANKS]
    outside_banks = [key for key in banks if key in outside]
    require(len(outside) == 3410 and all(expected[key] is source[key] for key in outside)
            and len(banks) == 504 and len(outside_banks) == 502, 'Another persistent tensor was patched.')
    return expected, {'patched_local_tensor_count': 13, 'frozen_outside_target_tensors': 3410,
        'selected_decoder_tensors': 0, 'selected_projections': ['gate_proj'], 'changed_decoder_elements': {},
        'stage_changed_bits': xor, 'changed_code_bits': sum(xor), 'source_packed_bank_count': 504,
        'frozen_outside_target_packed_bank_count': 502, 'frozen_outside_target_packed_banks_exact': True,
        'frozen_local_nonmutable_tensors': frozen, 'frozen_local_nonmutable_exact': True,
        'all_decoder_tensors_exact': True, 'frozen_complete_state_tensors': 3421}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    base, output = args.base_root.resolve(), args.output_root.resolve()
    require(not output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized(), 'Use a new CPU-only preparation directory.')
    torch.set_num_threads(4)
    proto = base / 'gate_final_kl_neighbor_transfer'
    previous, report, review, launch = (read(proto / name) for name in
        ('launch.prepared.json', 'run/report.json', 'review.json', 'launch.json'))
    require(report['status'] == launch['status'] == 'completed' and launch['exit_code'] == 0
            and all((proto / name).read_text().strip() == '0' for name in ('run.exit', 'outer.exit'))
            and sha256(proto / 'run/report.json') == REPORT_SHA == review['report_sha256']
            and sha256(proto / 'review.json') == REVIEW_SHA and review['status'] == 'verified'
            and review['prepared_sha256'] == sha256(proto / 'launch.prepared.json')
            and review['terminal_sha256'] == sha256(proto / 'launch.json'),
            'Require the actual reviewed, completed fixed four-group transfer observation.')
    transfer = report['transfer']
    require(report['source_checkpoint_id'] == transfer['source_checkpoint_id'] == SOURCE_ID
            and transfer['status'] == 'completed' and transfer['groups'] == review['group_order'] == [0, 1, 2, 3]
            and transfer['actual_stage_xor'] == review['actual_stage_xor'] == EXPECTED_XOR
            and all(transfer[key] for key in ('choices_complete_bank_replay_exact', 'noncode_gate_bytes_exact',
                'frozen_complement_versions_and_pointers_exact', 'all13_final_bytes_exact',
                'original_cache_and_three_full_logits_restored_exact', 'candidate_group8_repeated_exact')),
            'Require the verified native combined replay, frozen state and restoration proof.')
    check_hashes(previous['source_sha256'])
    for item in previous['protected_stats']:
        check_stat(item)
    path = Path(report['choices']['path'])
    require(path == base / 'gate_final_kl_neighbor_probe/run/choices.pt'
            and sha256(path) == report['choices']['sha256'] == review['saved_choices']['sha256'] == ENDPOINT_SHA,
            'Use only the actual original17KB compact choices, without a fitted decoder or new search.')
    choices = torch.load(path, map_location='cpu', weights_only=True)
    require(tensor_record(choices) == report['choices']['tensor'] == review['saved_choices']['tensor']
            and choice_inventory(choices) == previous['choice_inventory'] == transfer['choice_inventory'],
            'Actual saved choices differ from the reviewed fixed groups.')
    meta = read(SOURCE / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == SOURCE_ID, 'Only fresh f927 may supply the complete deployment.')
    state = torch.load(SOURCE / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    budget = audit_state_dict(meta, state, max_bpw=2.5, max_lora_rank=0)
    budget_equal(budget, budget)
    require({key[len(TARGET) + 1:]: tensor_record(value) for key, value in state.items()
             if key.startswith(TARGET + '.')} == transfer['initial_state'],
            'Actual source gate differs from the already verified native initial state.')
    endpoint = replay_endpoint(state, choices)
    require({k: tensor_record(v) for k, v in endpoint.items()} == transfer['endpoint_state'],
            'CPU complete-bank replay differs from the actual native transfer endpoint.')
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
    for key, digest in previous['source_sha256'].items():
        require(key not in sources or sources[key] == digest, 'Inherited source version conflict: ' + key)
        sources[key] = digest
    for relative in ('experiments/gate_student_context_evaluation/prepare.py',
                     'experiments/gate_student_context_evaluation/deploy.py', 'experiments/gate_student_context_evaluation/run.py',
                     'experiments/gate_down_student_context_evaluation/run.py'):
        p = ROOT / relative
        require(str(p) not in sources or sources[str(p)] == sha256(p), 'Existing helper source changed.')
        sources[str(p)] = sha256(p)
    for p in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(p.read_text()); sources[str(p)] = sha256(p)
    for p in (proto / 'launch.prepared.json', proto / 'run/report.json', proto / 'review.json', proto / 'run.exit', proto / 'outer.exit',
              proto / 'launch.json', path, gate_dir / 'launch.json', gate_dir / 'dispatcher.exit', gate_dir / 'audit.json',
              base / 'gate_final_kl_neighbor_probe/run/report.json', base / 'gate_final_kl_neighbor_probe/review.json',
              base / 'gate_final_kl_neighbor_probe/launch.prepared.json', gate_dir / 'full8_summary.json', gate_model / 'checkpoint_meta.json', *control_paths, *gate_paths):
        digest = sha256(p)
        require(str(p) not in previous['input_sha256'] or previous['input_sha256'][str(p)] == digest,
                'Previously bound replay evidence changed: ' + str(p))
        inputs[str(p)] = digest
    check_hashes(sources); check_hashes(inputs)
    commands = [{'name': 'deploy', 'argv': [sys.executable, '-m', 'experiments.gate_final_kl_neighbor_evaluation.deploy',
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
        'endpoint_path': str(path), 'endpoint_sha256': ENDPOINT_SHA, 'endpoint_state': transfer['endpoint_state'],
        'patch_contract': patch, 'selected_projections': ['gate_proj'], 'patched_local_tensor_count': 13,
        'frozen_outside_target_tensors': 3410, 'selected_decoder_tensors': 0,
        'source_native_policy': gate_launch['source_native_policy'], 'stage_commands': commands,
        'control_result_paths': [str(x) for x in control_paths], 'control_mean_percent': control['mean_percent'],
        'gate_only_control_results': [str(x) for x in gate_paths], 'gate_only_control_checkpoint': str(gate_model),
        'gate_only_control_checkpoint_id': gate_meta['checkpoint_id'], 'gate_only_control_mean_percent': gate_control['mean_percent'],
        'transfer_report_sha256': REPORT_SHA, 'transfer_review_sha256': REVIEW_SHA,
        'endpoint_artifact_kind': 'compact_choices_replayed_on_fresh_source', 'groups': [0, 1, 2, 3],
        'timeout_seconds': 1500, 'termination_grace_seconds': 20,
        'environment': {'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'},
        'protocol': {'full8_leaves': 64, 'full8_samples': 24742, 'num_fewshot': 0, 'mmlu_batch': 64,
            'seven_batch': 32, 'limit': None, 'single_layer_only': TARGET, 'new_training': False,
            'mutable_decoder_tensors': 0, 'mutable_packed_banks': 2, 'frozen_other_packed_banks': 502,
            'longbench': 'not_evaluated', 'nine_task_goal': 'not_evaluated',
            'boundary': 'Fixed four-group final-KL-gradient choices replayed simultaneously on fresh f927; no decoder fitting, reranking, acceptance or additional training. f927 is primary; original-code gate500 is a different-method secondary reference. No production integration or all-layer rollout.'},
        'launcher_argv': [sys.executable, '-m', 'experiments.gate_final_kl_neighbor_evaluation.run',
                          '--launch_record', str(output / 'launch.json')], 'cuda_initialized': False}
    require(not torch.cuda.is_initialized(), 'CPU preparation must not initialize CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record)
    print(json.dumps({'status': record['status'], 'launch_sha256': sha256(output / 'launch.json'),
        'source_pins': len(sources), 'input_pins': len(inputs), 'stage_changed_bits': patch['stage_changed_bits'],
        'source_user_linear_bits': budget['user_linear_bits'], 'source_bpw': budget['user_linear_bpw'],
        'cuda_initialized': False}), flush=True)


if __name__ == '__main__':
    main()
