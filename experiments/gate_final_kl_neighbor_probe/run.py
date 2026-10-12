"""One-shot CPU preparation/execution of four fixed final-KL neighbor proposals."""
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
from experiments.gate_final_kl_neighbor_probe.core import GROUPS, SINGLE_CHECKS, adjoint_check, probe
from experiments.gate_group_accept_probe.core import VECTORS_PER_GROUP
from experiments.gate_student_context_probe.data import ROOT, SOURCE_ID, TARGET
from experiments.gate_tail_objective_probe.core import teacher_pass
from experiments.longbench_v1.common import sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from rotation.weight_preprocess import TwoSidedHadamard
from train_utils.utils import configure_deterministic_mode, set_seed


def cpu_adjoint(source):
    """Use the actual signs from four f92732x32 tiles, without decoding/loading a model."""
    checkpoint = Path(source['source_checkpoint']); meta = read(checkpoint / 'checkpoint_meta.json')
    spec = next(value for value in meta['converted_modules'] if value['name'] == TARGET)
    state = torch.load(checkpoint / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    output_signs = state[TARGET + '.weight_rotation.output_signs']
    input_signs = state[TARGET + '.weight_rotation.input_signs']
    require(spec['weight_rotation']['block_size'] == 32 and output_signs.shape == (12288,) and input_signs.shape == (4032,),
            'Require actual original block32 signs and geometry.')
    rows = []
    for item in SINGLE_CHECKS:
        start = item['group'] * 32; column = (item['vector'] % 126) * 32
        rotation = TwoSidedHadamard(32, 32, spec['weight_rotation'])
        rotation.output_signs.copy_(output_signs[start:start + 32]); rotation.input_signs.copy_(input_signs[column:column + 32])
        rotation.validate_state()
        index = torch.arange(1024, dtype=torch.float32).reshape(32, 32)
        g = (index.remainder(37) - 18) / 19
        delta = torch.zeros_like(g)
        delta[(item['vector'] // 126) % 32] = (torch.arange(32, dtype=torch.float32).remainder(11) - 5) / 17
        # Fixed scale variations exercise absolute-error scaling, not candidate selection.
        g *= 2. ** (item['group'] * 3 - 3); delta *= 2. ** (3 - item['group'] * 2)
        rows.append({**item, 'output_signs': tensor_record(rotation.output_signs),
                     'input_signs': tensor_record(rotation.input_signs), **adjoint_check(rotation, g, delta)})
    return {'status': 'verified', 'actual_source_signs': True, 'shape_per_case': [32, 32], 'cases': rows,
            'boundary': 'Independent CPU FP32 mathematical adjoint check; does not include native BF16 inverse rounding or GPU kernels.'}


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and Path.cwd() == ROOT and not output.exists(), 'Prepare a fresh CPU-only output.')
    torch.set_num_threads(4)
    previous = BASE / 'gate_final_kl_gradient_probe'
    old = read(previous / 'launch.prepared.json'); check(old)
    report, review, terminal = (read(previous / name) for name in ('run/report.json', 'review.json', 'launch.json'))
    require(report['status'] == terminal['status'] == 'completed' and terminal['exit_code'] == 0
            and all((previous / name).read_text().strip() == '0' for name in ('run.exit', 'outer.exit'))
            and review['status'] == 'verified' and review['report_sha256'] == sha256(previous / 'run/report.json')
            and report['source_checkpoint_id'] == SOURCE_ID and report['diagnostic']['gradient']['actual_cache_hits'] == 32
            and report['diagnostic']['final_full_train_scores_exact'], 'Require completed actual cached-leaf backward/restoration evidence.')
    data, batches, _ = data_and_inputs(); batch = batches['train']
    require(data == old['data'] and {k: tensor_record(v) for k, v in batch.items()} == old['token_batch']
            and int(batch['attention_mask'][:, 1:].sum()) == 4257, 'Use the original train32 only.')
    math_proof = cpu_adjoint(data['source_data'])
    sources, inputs = dict(old['source_sha256']), dict(old['input_sha256'])
    for path in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
    for name in ('launch.prepared.json', 'launch.json', 'run/report.json', 'review.json', 'run.exit', 'outer.exit'):
        path = previous / name; inputs[str(path)] = sha256(path)
    record = {'schema': 'gate_final_kl_neighbor_v1', 'status': 'prepared_unbound', 'gpu': None,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output), 'source_checkpoint_id': SOURCE_ID,
        'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': old['protected_stats'],
        'data': data, 'token_batch': old['token_batch'], 'gradient_validation_report': str(previous / 'run/report.json'),
        'source_bit_budget': old['source_bit_budget'], 'environment': old['environment'], 'cpu_adjoint': math_proof,
        'timeout_seconds': 600, 'termination_grace_seconds': 20, 'queued': False, 'automatic_retry': False,
        'groups': list(GROUPS), 'single_bit_checks': list(SINGLE_CHECKS),
        'protocol': {'fresh': 'Original f927. Fixed group0..3 independently from original banks; no accepted/greedy endpoint inputs.',
            'population': 'Only original train32,4257causal tokens, micro1 fixed256. No heldout/replay scoring or selection.',
            'gradient': 'Actual native BF16 L18 cached-weight leaf; per-doc autograd.grad(KL.double().sum()), BF16 return accumulatedFP32. All persistent parameters frozen.',
            'rotation': 'Take original gradient columns excluding protected64, in ascending original order; apply actual stored-sign FP32 forward rotation R(g). Gate has no transpose, D32/S2/one part.',
            'neighbors': 'Each of16128 vectors enumerates stage0bit0..31 then stage1bit0..31 using native packed BF16 decoder outputs and BF16 stage sum. No repeated qscale/mean/std fusion or decoder promotion.',
            'selection': 'Minimize FP64 dot of R(g) and each same-chunk BF16 summed-output difference. Keep original0 unless strictly negative; fixed order resolves ties; at most one bit per vector.',
            'validation': 'Four predetermined single-bit cases check both stages/bit endpoints and true32x32 original-coordinate support. Four full group endpoints use native categorygroup8 and complete train32 forward.',
            'report': 'Rotated per-vector prediction sum; actual native BF16 original-coordinate displacement dot; observed complete KL/NLL. Chunk/single-layer/group8 discrepancies are observations, not silently treated as exact equivalence.',
            'save': 'Only compact choices16128 and small trajectory/report. No full model, decoder endpoint, gradient or optimizer.',
            'boundary': 'New isolated proposal proxy. FP32 adjoint identity does not make native BF16 inverse/addition linear; no downstream quality or framework-integration claim.'},
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.gate_final_kl_neighbor_probe.run', '--launch-record', str(output / 'launch.json')],
        'cuda_initialized': False}
    check(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record); write_json(output / 'launch.prepared.json', record)
    print(json.dumps({'status': record['status'], 'source_count': len(sources), 'input_count': len(inputs),
        'cuda_initialized': False, 'cpu_adjoint_status': math_proof['status'], 'launch_sha256': sha256(output / 'launch.json')}), flush=True)


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
        data, batches, archive = data_and_inputs(); batch = batches['train']
        require(data == record['data'] and {k: tensor_record(v) for k, v in batch.items()} == record['token_batch'], 'Runtime inputs changed.')
        teacher, hf_mlp, teacher_report = teacher_pass(data, {'train': batch}, archive, device=torch.device('cuda:0'), deadline=deadline, observer=lambda *args: None)
        del hf_mlp
        previous = read(record['gradient_validation_report'])
        require(teacher_report['documents'] == previous['teacher_train']['documents'], 'Actual teacher probabilities/labels changed.')
        with (work / 'trajectory.jsonl').open('x') as stream:
            def observer(kind, row):
                stream.write(json.dumps({'kind': kind, **row}) + '\n'); stream.flush()
                print(json.dumps({'kind': kind, 'index': row.get('index'), 'group': row.get('group'),
                                  'observed_kl_delta_sum': row.get('observed_kl_delta_sum'), 'seconds': row.get('seconds')}), flush=True)
            with torch.autocast('cuda', enabled=False):
                choices, result = probe(data, batch, teacher['train'], previous, deadline=deadline, observer=observer)
        require(choices.dtype == torch.uint8 and choices.shape == (4 * VECTORS_PER_GROUP,) and int(choices.max()) <= 64, 'Compact choices changed layout.')
        choice_path = work / 'choices.pt'; torch.save(choices, choice_path)
        require(torch.equal(choices, torch.load(choice_path, map_location='cpu', weights_only=True)), 'Saved choices differ.')
        report = {'status': 'completed', 'launch_record': str(path), 'source_checkpoint_id': SOURCE_ID,
            'protocol': record['protocol'], 'teacher_train': teacher_report, 'diagnostic': result,
            'saved_choices': {'path': str(choice_path), 'sha256': sha256(choice_path), 'tensor': tensor_record(choices), 'roundtrip_exact': True},
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
    parser = argparse.ArgumentParser(description=__doc__); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path); group.add_argument('--launch-record', type=Path)
    args = parser.parse_args()
    if args.prepare: prepare(args.prepare.resolve())
    else: sys.exit(run(args.launch_record.resolve()))
