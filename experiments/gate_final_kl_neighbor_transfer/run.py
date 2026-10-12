"""Prepare or execute the fixed four-group, three-population transfer observation."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from experiments.gate_final_kl_accept_probe.run import BASE, check, data_and_inputs, read, validate_teacher_cache
from experiments.gate_final_kl_neighbor_transfer.core import DOMAINS, GROUPS, choice_inventory, measure
from experiments.gate_group_accept_probe.core import BANKS, apply_choices
from experiments.gate_student_context_probe.data import ROOT, SOURCE_ID, TARGET
from experiments.gate_tail_objective_probe.core import teacher_pass
from experiments.longbench_v1.common import sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.utils import configure_deterministic_mode, set_seed


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and Path.cwd() == ROOT and not output.exists(), 'Use a fresh CPU-only preparation directory.')
    torch.set_num_threads(4)
    previous = BASE / 'gate_final_kl_neighbor_probe'
    old = read(previous / 'launch.prepared.json'); check(old)
    report, review, terminal = (read(previous / name) for name in ('run/report.json', 'review.json', 'launch.json'))
    require(report['status'] == terminal['status'] == 'completed' and terminal['exit_code'] == 0
            and all((previous / name).read_text().strip() == '0' for name in ('run.exit', 'outer.exit', 'review.exit'))
            and review['status'] == 'verified' and review['report_sha256'] == sha256(previous / 'run/report.json')
            and report['source_checkpoint_id'] == SOURCE_ID and old['groups'] == list(GROUPS), 'Require the completed and reviewed fixed four-group proposal.')
    choices_path = Path(report['saved_choices']['path'])
    require(sha256(choices_path) == report['saved_choices']['sha256'], 'Saved proposal bytes changed.')
    choices = torch.load(choices_path, map_location='cpu', weights_only=True)
    inventory = choice_inventory(choices)
    require(tensor_record(choices) == report['saved_choices']['tensor']
            and all(all(item[k] == actual[k] for k in ('group', 'selected_vectors', 'stage_xor', 'choice_histogram'))
                    for item, actual in zip(inventory, report['diagnostic']['groups'])), 'Reuse exactly the four existing proposals, regardless of subsequent observation scores.')
    data, batches, _ = data_and_inputs()
    require(data == old['data'] and tuple(batches) == DOMAINS, 'Original data/source changed.')
    full = BASE / 'gate_final_kl_accept_probe/full384'
    baseline_report, baseline_review = read(full / 'run/report.json'), read(full / 'review.json')
    require(baseline_report['status'] == 'completed' and baseline_report['source_checkpoint_id'] == SOURCE_ID
            and baseline_review['status'] == 'verified' and baseline_review['report_sha256'] == sha256(full / 'run/report.json')
            and all((full / name).read_text().strip() == '0' for name in ('run.exit', 'outer.exit'))
            and baseline_report['search']['before']['train'] == report['diagnostic']['before'], 'Bind only the verified original three-domain baseline, never its endpoint weight.')
    tail = BASE / 'gate_tail_objective_probe_r1/run'
    old_teacher = torch.load(tail / 'teacher_top100.pt', map_location='cpu', weights_only=True, mmap=True)
    validate_teacher_cache(old_teacher, batches, read(tail / 'teacher.json'))
    # Actual original packed banks: prove the compact proposal applies and undoes without a model or CUDA.
    checkpoint = Path(data['source_data']['source_checkpoint']); meta = read(checkpoint / 'checkpoint_meta.json')
    state = torch.load(checkpoint / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    cpu_banks = []
    for stage in range(2):
        key = f'{TARGET}.{BANKS[stage]}'
        require(key in state, 'Original stage bank key differs.')
        before = state[key][:len(choices)].clone(); after = before.clone()
        apply_choices(after, choices, stage); selected = tensor_record(after)
        apply_choices(after, choices, stage)
        require(torch.equal(before, after), 'CPU actual original-bank proposal apply/undo failed.')
        cpu_banks.append({'stage': stage, 'original': tensor_record(before), 'selected': selected, 'apply_undo_exact': True})
    sources, inputs = dict(old['source_sha256']), dict(old['input_sha256'])
    for path in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
    for path in [choices_path, *(previous / name for name in ('launch.prepared.json', 'launch.json', 'run/report.json', 'review.json', 'run.exit', 'outer.exit', 'review.exit')),
                 *(full / name for name in ('run/report.json', 'review.json', 'run.exit', 'outer.exit')),
                 tail / 'teacher_top100.pt', tail / 'teacher.json']:
        inputs[str(path)] = sha256(path)
    record = {'schema': 'gate_final_kl_neighbor_transfer_v1', 'status': 'prepared_unbound', 'gpu': None,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output), 'source_checkpoint_id': SOURCE_ID,
        'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': old['protected_stats'], 'data': data,
        'token_batches': {d: {k: tensor_record(v) for k, v in b.items()} for d, b in batches.items()},
        'choices_path': str(choices_path), 'choices_sha256': sha256(choices_path), 'choice_inventory': inventory,
        'proposal_report': str(previous / 'run/report.json'), 'original_baseline_report': str(full / 'run/report.json'),
        'old_teacher_path': str(tail / 'teacher_top100.pt'), 'old_teacher_report': str(tail / 'teacher.json'),
        'cpu_actual_bank_roundtrip': cpu_banks, 'source_bit_budget': old['source_bit_budget'],
        'environment': old['environment'], 'timeout_seconds': 600, 'termination_grace_seconds': 20,
        'queued': False, 'automatic_retry': False, 'groups': list(GROUPS), 'cuda_initialized': False,
        'protocol': {'arms': ['original_f927', 'fixed_groups0_1_2_3'],
            'proposal': 'Apply all four saved final-KL-gradient choices simultaneously to fresh f927. No re-ranking, acceptance, decoder fitting or subsequent tuning.',
            'data': 'All original train32, old Alpaca held32, and fixed FineWeb physical rows500000..500031, native micro1/max256. The latter two populations are previously explored diagnostic observations, not new unseen validation.',
            'objective': 'Native final top100+tail massKL T1 and NLL, FP32 per-token losses then fixed-order per-document FP64 sums; preserve original token/label semantics.',
            'teacher': 'Regenerate actual train32 HF probabilities and check against the verified proposal. Reuse byte-pinned and per-tensor checked teacher probabilities for the two observation domains.',
            'native': 'Strict f927 load, BF16/nativeFA2/FP32RoPE/globalgroup8; refresh actual packed plan and36gate category after mutation/restoration. Only two packed banks may differ.',
            'save': 'Use the existing compact choices as the saved replay artifact; save only small report/trajectory. No new model, gate endpoint or gradient.',
            'boundary': 'Observation of a fixed combined proposal. Independent group improvements are not assumed additive; no full8/LongBench or formal framework-integration claim.'},
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.gate_final_kl_neighbor_transfer.run', '--launch-record', str(output / 'launch.json')]}
    check(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record); write_json(output / 'launch.prepared.json', record)
    print(json.dumps({'status': record['status'], 'source_count': len(sources), 'input_count': len(inputs),
        'causal_tokens': {d: int(b['attention_mask'][:, 1:].sum()) for d, b in batches.items()},
        'cuda_initialized': False, 'launch_sha256': sha256(output / 'launch.json')}), flush=True)


def run(path):
    record = read(path); output = Path(record['output_dir']); code = 1
    require(record['status'] == 'bound_ready' and type(record['gpu']) is int and not record['queued']
            and not record['automatic_retry'] and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
            and Path.cwd() == Path(record['cwd']) and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
            and all(os.environ.get(k) == v for k, v in record['environment'].items())
            and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE'))
            and torch.cuda.device_count() == 1 and not (output / 'run').exists(), 'Require a fresh explicit one-shot binding.')
    check(record); work = output / 'run'; work.mkdir()
    record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(),
                  worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
    write_json(path, record); deadline = record['started_unix'] + record['timeout_seconds']
    try:
        torch.set_num_threads(4); torch.cuda.set_device(0); configure_deterministic_mode(True); set_seed(137)
        data, batches, archive = data_and_inputs()
        require(data == record['data'] and {d: {k: tensor_record(v) for k, v in b.items()} for d, b in batches.items()}
                == record['token_batches'], 'Runtime source or fixed tokens changed.')
        teacher, hf_mlp, teacher_report = teacher_pass(data, {'train': batches['train']}, archive, device=torch.device('cuda:0'), deadline=deadline, observer=lambda *args: None)
        del hf_mlp
        require(teacher_report['documents'] == read(record['proposal_report'])['teacher_train']['documents'], 'Regenerated train teacher changed.')
        old_teacher = torch.load(record['old_teacher_path'], map_location='cpu', weights_only=True, mmap=True)
        validate_teacher_cache(old_teacher, batches, read(record['old_teacher_report'])); teacher.update(old_teacher)
        choices = torch.load(record['choices_path'], map_location='cpu', weights_only=True)
        require(choice_inventory(choices) == record['choice_inventory'], 'Fixed combined choices changed.')
        with (work / 'trajectory.jsonl').open('x') as stream:
            def observer(row):
                stream.write(json.dumps(row) + '\n'); stream.flush(); print(json.dumps(row), flush=True)
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                result = measure(data, batches, teacher, choices, read(record['original_baseline_report'])['search']['before'], deadline=deadline, observer=observer)
        report = {'status': 'completed', 'launch_record': str(path), 'source_checkpoint_id': SOURCE_ID,
            'protocol': record['protocol'], 'teacher_train': teacher_report, 'transfer': result,
            'choices': {'path': record['choices_path'], 'sha256': record['choices_sha256'], 'tensor': tensor_record(choices)},
            'trajectory_sha256': sha256(work / 'trajectory.jsonl'), 'elapsed_seconds': time.time() - record['started_unix'],
            'saved_model_or_gradient': False, 'full8_and_longbench': 'not_evaluated'}
        check(record); write_json(work / 'report.json', report)
        record.update(status='completed', report_sha256=sha256(work / 'report.json')); code = 0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc()); print(record['traceback'], flush=True)
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'run.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path); group.add_argument('--launch-record', type=Path)
    args = parser.parse_args()
    if args.prepare: prepare(args.prepare.resolve())
    else: sys.exit(run(args.launch_record.resolve()))
