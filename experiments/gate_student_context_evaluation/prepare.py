"""CPU contracts for full-eight-task evaluation of two isolated gate endpoints."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

import torch

from experiments.full_weight_bits_probe.core import popcounts
from experiments.gate_student_context_probe.data import SOURCE, SOURCE_ID, TARGET
from experiments.joint_reconstruction_probe.core import require
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.w25_eval.acceptance import collect_results
from experiments.w25_eval.model_inventory import model_inventory
from train_utils.checkpoint_bpw import audit_state_dict


ROOT = Path(__file__).resolve().parents[2]
ARMS = ('original_codes', 'exact64_codes')
BANKS = ('vq_weight', 'vq_weight_s1')
BUDGET_KEYS = ('original_linear_weights', 'user_linear_bits', 'packed_linear_bits', 'user_linear_bpw',
               'packed_linear_bpw', 'components', 'outside_linear_bits')


def read(path):
    return json.loads(Path(path).read_text())


def require_state(actual, expected, label):
    require(set(actual) == set(expected), label + ': persistent keys differ.')
    for key, value in expected.items():
        other = actual[key].detach().cpu()
        require(other.shape == value.shape and other.dtype == value.dtype
                and torch.equal(other.reshape(-1).contiguous().view(torch.uint8),
                                value.detach().cpu().reshape(-1).contiguous().view(torch.uint8)),
                label + ': persistent bytes differ at ' + key)
    return len(expected)


def patched_state(source, endpoint, arm):
    require(arm in ARMS and len(source) == 3423, 'Require an approved endpoint and complete original source.')
    local = {key[len(TARGET) + 1:]: value for key, value in source.items() if key.startswith(TARGET + '.')}
    require(set(local) == set(endpoint) and len(local) == 13, 'Exactly the real thirteen gate tensors are required.')
    changes = {}
    for key, original in local.items():
        value = endpoint[key]
        require(value.shape == original.shape and value.dtype == original.dtype, 'Endpoint topology or dtype changed: ' + key)
        allowed = key.startswith('_parallel_stage_decoder.') or (arm == 'exact64_codes' and key in BANKS)
        if not allowed:
            require_state({key: value}, {key: original}, 'Frozen local gate state')
        if key.startswith('_parallel_stage_decoder.'):
            require(value.dtype == torch.bfloat16 and bool(torch.isfinite(value).all()), 'Require finite deployed BF16 decoder.')
            changes[key] = int((value != original).sum())
    require(len(changes) == 6 and all(value > 0 for value in changes.values()), 'Every selected decoder tensor must retain a real deployed update.')
    xor = [popcounts(local[k], endpoint[k]) for k in BANKS]
    require((xor == [0, 0]) if arm == 'original_codes' else (xor == [371363, 986696]), 'Endpoint banks differ from the verified current-f927 search.')
    expected = dict(source)
    expected.update({TARGET + '.' + key: value for key, value in endpoint.items()})
    return expected, {'patched_local_tensor_count': 13, 'frozen_outside_target_tensors': len(source) - 13,
        'changed_decoder_elements': changes, 'stage_changed_bits': xor, 'changed_code_bits': sum(xor),
        'frozen_local_nonmutable_exact': True}


def budget_equal(actual, expected):
    require(actual['budget_passed'] and not actual['contract_violations'],
            'Checkpoint violates the rank-zero bit budget.')
    require(all(actual[k] == expected[k] for k in BUDGET_KEYS), 'Replacing the gate changed the original integer storage budget.')
    require(actual['user_linear_bits'] == int(actual['user_linear_bits'])
            and actual['packed_linear_bits'] == int(actual['packed_linear_bits'])
            and not any(actual['components'].get(k, {}).get('storage_bits', 0) for k in ('lora', 'residual_lora', 'dense_weights')),
            'Unexpected noninteger accounting or extra projection branch.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    base, output = args.base_root.resolve(), args.output_root.resolve()
    require(not output.exists(), 'Use a new independent full8 output root.')
    torch.set_num_threads(4)
    proto = base / 'gate_student_context_probe'
    previous, report, review = (read(proto / name) for name in ('preparation.json', 'run/report.json', 'review.json'))
    require(report['status'] == 'completed' and (proto / 'run.exit').read_text().strip() == '0'
            and sha256(proto / 'run/report.json') == '889c0c6e40ad9ae7f678a78a0962f6f6659aea7d5e1582021474e62212e39158'
            and review['status'] == 'verified' and review['report_sha256'] == sha256(proto / 'run/report.json'),
            'Require the reviewed, completed real-student prototype.')
    check_hashes(previous['source_sha256']); check_hashes(previous['input_sha256'])
    check_stat(previous['data']['source_weight_stat'])
    meta = read(SOURCE / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == SOURCE_ID, 'Wrong fresh source UUID.')
    state = torch.load(SOURCE / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    budget = audit_state_dict(meta, state, max_bpw=2.5, max_lora_rank=0)
    budget_equal(budget, budget)
    inventory = model_inventory(meta['base_model_path'])
    control_paths = [SOURCE.parent / name for name in ('mmlu.json', 'seven.json')]
    control = collect_results(control_paths, inventory, checkpoint=(SOURCE, meta))
    sources, inputs = dict(previous['source_sha256']), dict(previous['input_sha256'])
    for relative in ('experiments/w25_eval/acceptance.py', 'experiments/w25_eval/model_inventory.py',
                     'experiments/qwen3_w25/summarize_results.py', 'experiments/qwen3_w25/step_export_callback.py'):
        p = ROOT / relative
        require(str(p) not in sources or sources[str(p)] == sha256(p), 'Existing source version changed.')
        sources[str(p)] = sha256(p)
    for p in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(p.read_text()); sources[str(p)] = sha256(p)
    for p in (proto / 'preparation.json', proto / 'run/report.json', proto / 'review.json', proto / 'run.exit', *control_paths):
        inputs[str(p)] = sha256(p)
    records = {}
    for arm in ARMS:
        row = report['arms'][arm]
        path = Path(row['state_file'])
        require(sha256(path) == row['state_sha256'], 'The selected endpoint file changed.')
        endpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        require({k: tensor_record(v) for k, v in endpoint.items()} == row['training']['final_payload'], 'Endpoint no longer matches its training proof.')
        expected, patch = patched_state(state, endpoint, arm)
        budget_equal(audit_state_dict(meta, expected, max_bpw=2.5, max_lora_rank=0), budget)
        arm_output = output / arm
        commands = [{'name': 'deploy', 'argv': [sys.executable, '-m', 'experiments.gate_student_context_evaluation.deploy',
            '--launch_record', str(arm_output / 'launch.json')]}]
        for name, tasks, batch in [('mmlu', 'mmlu', '64'), ('seven', 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa', '32')]:
            commands.append({'name': name, 'argv': [sys.executable, 'tools/cat_eval.py', '--checkpoint_dir', str(arm_output / 'final_model'),
                '--strict', '--eval_lm_eval', '--eval_device', 'cuda:0', '--tasks', tasks, '--num_fewshot', '0',
                '--lm_batch_size', batch, '--tokenizer_use_fast', 'true', '--prewarm_group_size', '8',
                '--eval_log_dir', str(arm_output / (name + '_logs')), '--output_json', str(arm_output / (name + '.json'))]})
        records[arm] = {'status': 'prepared_unbound', 'arm': arm, 'gpu': None, 'queued': False, 'automatic_retry': False,
            'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(arm_output),
            'source_checkpoint': str(SOURCE), 'source_checkpoint_id': SOURCE_ID,
            'source_weight_stat': previous['data']['source_weight_stat'], 'hf_weight_stats': previous['data']['hf_weight_files'],
            'architecture': inventory, 'source_bit_budget': budget, 'source_sha256': sources,
            'input_sha256': {**inputs, str(path): row['state_sha256']}, 'endpoint_path': str(path), 'endpoint_sha256': row['state_sha256'],
            'endpoint_state': row['training']['final_payload'], 'patch_contract': patch,
            'source_native_policy': previous['data']['model_policy'], 'stage_commands': commands,
            'control_result_paths': [str(x) for x in control_paths], 'control_mean_percent': control['mean_percent'],
            'timeout_seconds': 3600, 'termination_grace_seconds': 20,
            'environment': {'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'},
            'protocol': {'full8_leaves': 64, 'full8_samples': 24742, 'num_fewshot': 0, 'mmlu_batch': 64,
                'seven_batch': 32, 'limit': None, 'single_layer_only': TARGET, 'new_training': False,
                'longbench': 'not_evaluated', 'nine_task_goal': 'not_evaluated',
                'boundary': 'Isolated single-layer quality verification; no new production algorithm or all-layer rollout.'},
            'launcher_argv': [sys.executable, '-m', 'experiments.gate_student_context_evaluation.run', '--launch_record', str(arm_output / 'launch.json')]}
    require(not torch.cuda.is_initialized(), 'CPU preparation must not initialize CUDA.')
    output.mkdir(parents=True)
    for arm, record in records.items():
        path = output / arm; path.mkdir(); write_json(path / 'launch.json', record)
    print(json.dumps({'status': 'prepared_unbound', 'cuda_initialized': False,
        'arms': {a: sha256(output / a / 'launch.json') for a in ARMS}, 'source_pins': len(sources),
        'source_user_linear_bits': int(budget['user_linear_bits']), 'source_bpw': budget['user_linear_bpw']}))


if __name__ == '__main__':
    main()
