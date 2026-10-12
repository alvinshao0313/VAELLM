"""Bind fixed train32/group0..7 evidence, then run one independent gradient diagnostic."""
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

from experiments.gate_final_kl_accept_probe.run import BASE, check, data_and_inputs, read
from experiments.gate_final_kl_gradient_probe.core import ALPHAS, dot64, probe
from experiments.gate_group_accept_probe.core import group_inventory
from experiments.gate_student_context_probe.data import ROOT, SOURCE_ID
from experiments.gate_tail_objective_probe.core import teacher_pass
from experiments.longbench_v1.common import sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.utils import configure_deterministic_mode, set_seed


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and Path.cwd() == ROOT and not output.exists(), 'Prepare a fresh CPU-only directory in the real public root.')
    torch.set_num_threads(4)
    previous = BASE / 'gate_final_kl_accept_probe/validate8'
    old = read(previous / 'launch.prepared.json'); check(old)
    report, review = read(previous / 'run/report.json'), read(previous / 'review.json')
    terminal = read(previous / 'launch.json')
    require(report['status'] == terminal['status'] == 'completed' and terminal['exit_code'] == 0
            and (previous / 'run.exit').read_text().strip() == '0' and review['status'] == 'verified'
            and review['report_sha256'] == sha256(previous / 'run/report.json')
            and report['search']['groups'] == 8 and report['source_checkpoint_id'] == SOURCE_ID
            and report['search']['original_full_logits_restore_exact'], 'Require the actual completed native eight-group proof.')
    data, batches, _ = data_and_inputs(); batch = batches['train']
    require(data == old['data'] and {k: tensor_record(v) for k, v in batch.items()} == old['token_batches']['train']
            and int(batch['attention_mask'][:, 1:].sum()) == 4257, 'Use the identical original train32, not observation documents.')
    choices = torch.load(old['choices_path'], map_location='cpu', weights_only=True, mmap=True)
    inventory = group_inventory(choices)
    require(inventory == old['group_inventory'], 'Original proposal inventory changed.')
    sources, inputs = dict(old['source_sha256']), dict(old['input_sha256'])
    for path in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
    for name in ('launch.prepared.json', 'launch.json', 'run/report.json', 'review.json', 'run.exit'):
        path = previous / name; inputs[str(path)] = sha256(path)
    # An exact small CPU arithmetic check, not a claim about full-model CUDA backward.
    left = torch.tensor([[1.0, -2.0], [0.25, 4.0]], dtype=torch.float32)
    right = torch.tensor([[0.5, 2.0], [-4.0, 0.125]], dtype=torch.float32)
    require(dot64(left, right) == -4.0, 'FP64 contraction interface differs.')
    record = {'schema': 'gate_final_kl_gradient_v1', 'status': 'prepared_unbound', 'gpu': None,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'source_checkpoint_id': SOURCE_ID, 'source_sha256': sources, 'input_sha256': inputs,
        'protected_stats': old['protected_stats'], 'data': data,
        'token_batch': {k: tensor_record(v) for k, v in batch.items()},
        'choices_path': old['choices_path'], 'choices_sha256': old['choices_sha256'],
        'group_inventory': inventory[:8], 'native_validation_report': str(previous / 'run/report.json'),
        'source_bit_budget': old['source_bit_budget'], 'environment': old['environment'],
        'timeout_seconds': 600, 'termination_grace_seconds': 20, 'queued': False, 'automatic_retry': False,
        'protocol': {'fresh': 'Original f927; every group independently restores original codes/cache, no greedy acceptance.',
            'groups': list(range(8)), 'alphas': list(ALPHAS), 'population': 'Original train32 only; 4257 causal tokens at fixed256, micro1.',
            'gradient': 'Only native group8 BF16 L18 cached weight is an autograd leaf. Each document autograd.grad(KL.double().sum(), leaf), BF16 gradients accumulated in FP32; all persistent model parameters require_grad=False.',
            'objective': 'Existing native top100-plus-tail mass KL at T1; original teacher generated on these exact tokens and released before student.',
            'delta': 'FP32 subtraction of the two native original-coordinate BF16 group8 endpoints, with unchanged INT8 protection.',
            'interpolation': 'alpha1 uses native endpoint directly; alpha.25/.5 cast W0.float()+alpha*delta to BF16. These two weights are non-deployable diagnostic interpolations, not fractional bit codes.',
            'derivative': 'Report both alpha*<g,delta> and <g,actual_BF16_interpolated_displacement>, using FP64 inner products.',
            'observation': 'All fixed eight groups/three alphas, no heldout, no selection, no updates and no saved model/gradient tensors.',
            'boundary': 'New full-model backward/cache-leaf path has not run on GPU. CPU checks do not establish memory fit, timing or quality.'},
        'cpu_arithmetic': {'dot64': -4.0, 'full_model_backward_tested': False}, 'cuda_initialized': False,
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.gate_final_kl_gradient_probe.run',
                          '--launch-record', str(output / 'launch.json')]}
    check(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record); write_json(output / 'launch.prepared.json', record)
    print(json.dumps({'status': record['status'], 'source_count': len(sources), 'input_count': len(inputs),
                      'cuda_initialized': False, 'launch_sha256': sha256(output / 'launch.json')}), flush=True)


def run(path):
    record = read(path); output = Path(record['output_dir']); code = 1
    require(record['status'] == 'bound_ready' and type(record['gpu']) is int and not record['queued']
            and not record['automatic_retry'] and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
            and Path.cwd() == Path(record['cwd']) and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
            and all(os.environ.get(k) == v for k, v in record['environment'].items())
            and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE'))
            and torch.cuda.device_count() == 1 and not (output / 'run').exists(), 'Require an explicitly bound one-shot execution.')
    check(record); work = output / 'run'; work.mkdir()
    record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(),
                  worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
    write_json(path, record); deadline = record['started_unix'] + record['timeout_seconds']
    try:
        torch.set_num_threads(4); torch.cuda.set_device(0); configure_deterministic_mode(True); set_seed(137)
        data, batches, archive = data_and_inputs(); batch = batches['train']
        require(data == record['data'] and {k: tensor_record(v) for k, v in batch.items()} == record['token_batch'], 'Runtime inputs changed.')
        teacher, hf_mlp, teacher_report = teacher_pass(data, {'train': batch}, archive, device=torch.device('cuda:0'),
            deadline=deadline, observer=lambda *args: None)
        del hf_mlp
        previous = read(record['native_validation_report'])
        require(teacher_report['documents'] == previous['teacher_train']['documents'], 'Actual teacher top100/labels differ from the original native proof.')
        choices = torch.load(record['choices_path'], map_location='cpu', weights_only=True, mmap=True)
        with (work / 'trajectory.jsonl').open('x') as stream:
            def observer(kind, row):
                stream.write(json.dumps({'kind': kind, **row}) + '\n'); stream.flush()
                print(json.dumps({'kind': kind, 'index': row.get('index'), 'group': row.get('group'),
                                  'seconds': row.get('seconds')}), flush=True)
            with torch.autocast('cuda', enabled=False):
                result = probe(data, batch, teacher['train'], choices, deadline=deadline, observer=observer)
        require(result['before'] == previous['search']['before']['train'], 'Fresh baseline differs from previous native32 proof.')
        report = {'status': 'completed', 'launch_record': str(path), 'source_checkpoint_id': SOURCE_ID,
            'protocol': record['protocol'], 'teacher_train': teacher_report, 'diagnostic': result,
            'trajectory_sha256': sha256(work / 'trajectory.jsonl'), 'elapsed_seconds': time.time() - record['started_unix'],
            'full8_and_longbench': 'not_evaluated', 'saved_model_or_gradient': False}
        check(record); write_json(work / 'report.json', report)
        record.update(status='completed', report_sha256=sha256(work / 'report.json')); code = 0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc()); print(record['traceback'], flush=True)
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'run.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path); group.add_argument('--launch-record', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare.resolve())
    else:
        sys.exit(run(args.launch_record.resolve()))
