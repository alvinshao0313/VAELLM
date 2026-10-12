"""CPU preparation and one-shot execution of the independent8/384-group KL probes."""
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

from experiments.gate_final_kl_accept_probe.core import search
from experiments.gate_group_accept_probe.core import group_inventory
from experiments.gate_student_context_probe.data import ROOT, SOURCE_ID, prepare_data as student_data
from experiments.gate_tail_objective_probe.core import teacher_pass
from experiments.gate_tail_objective_probe.data import prepare_data as tail_data
from experiments.gate_tail_objective_probe.run import cpu_mass_kl_check
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.utils import configure_deterministic_mode, set_seed

BASE = ROOT / 'result/qwen3_8b_mmlu65_20261004'


def read(path):
    return json.loads(Path(path).read_text())


def check(record):
    check_hashes(record['source_sha256']); check_hashes(record['input_sha256'])
    for item in record['protected_stats']:
        check_stat(item)


def data_and_inputs():
    data, observations, archive = tail_data(BASE)
    source, original, _ = student_data(BASE)
    require(source == data['source_data'], 'Train and endpoint populations must share the same original native source.')
    batches = {'train': original['train'], **observations}
    require(int(batches['train']['attention_mask'].sum()) == 4289
            and {k: tensor_record(v) for k, v in batches['train'].items()} == source['token_batches']['train'],
            'Use the original32 training documents, with exact tokens and right padding.')
    return data, batches, archive


def validate_teacher_cache(cache, batches, previous):
    require(set(cache) == set(previous['documents']), 'Historical observation teacher domains differ.')
    for domain, items in cache.items():
        require(len(items) == 32, 'Require all32 historical observation documents.')
        for index, item in enumerate(items):
            valid = batches[domain]['attention_mask'][index, 1:].bool()
            labels = batches[domain]['input_ids'][index, 1:][valid]
            require(torch.equal(labels, item['labels']) and {k: tensor_record(v) for k, v in item.items()}
                    == previous['documents'][domain][index]['statistics'],
                    'Cached observation teacher probabilities/labels changed.')


def prepare(output, groups, validation):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and Path.cwd() == ROOT and not output.exists(), 'Prepare a fresh output in the public root with CUDA hidden.')
    torch.set_num_threads(4)
    tail_root, group_root = BASE / 'gate_tail_objective_probe_r1', BASE / 'gate_group_accept_probe'
    old = read(tail_root / 'launch.prepared.json')
    tail_report, tail_review = read(tail_root / 'run/report.json'), read(tail_root / 'review.json')
    group, group_report, group_review = (read(group_root / name) for name in ('preparation.json', 'run/report.json', 'review.json'))
    require(tail_report['status'] == group_report['status'] == 'completed'
            and tail_review['status'] == group_review['status'] == 'verified'
            and tail_review['report_sha256'] == sha256(tail_root / 'run/report.json')
            and group_review['report_sha256'] == sha256(group_root / 'run/report.json')
            and all((path / 'run.exit').read_text().strip() == '0' for path in (tail_root, group_root)),
            'Require completed native tail and original proposal/group proofs.')
    check(old); check(group)
    data, batches, _ = data_and_inputs()
    require(data == old['data'] and data['source_data']['expected_gate_state'] == group['initial_state'],
            'Original source or fixed document data changed.')
    choices_path = Path(group['paths']['choices'])
    choices = torch.load(choices_path, map_location='cpu', weights_only=True, mmap=True)
    require(group_inventory(choices) == group['group_inventory'], 'Use every original fixed group proposal, not the SSE acceptance mask.')
    old_teacher_path = tail_root / 'run/teacher_top100.pt'
    require(sha256(old_teacher_path) == tail_report['teacher_cache_sha256'], 'Historical observation teacher cache changed.')
    old_teacher = torch.load(old_teacher_path, map_location='cpu', weights_only=True, mmap=True)
    validate_teacher_cache(old_teacher, batches, tail_report['teacher'])
    sources, inputs = dict(old['source_sha256']), dict(old['input_sha256'])
    for path, digest in group['source_sha256'].items():
        require(path not in sources or sources[path] == digest, 'Inherited source versions differ.'); sources[path] = digest
    inputs.update(group['input_sha256'])
    for path in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
    for path in [choices_path, old_teacher_path, *(tail_root / name for name in ('launch.prepared.json', 'run/report.json', 'review.json', 'run/teacher.json', 'run.exit')),
                 *(group_root / name for name in ('preparation.json', 'run/report.json', 'review.json', 'run.exit'))]:
        inputs[str(path)] = sha256(path)
    proof = None
    if groups == 384:
        require(validation is not None, 'The full384 pass requires a completed separate8-group verification first.')
        proof = read(validation)
        require(proof['status'] == 'completed' and proof['search']['groups'] == 8
                and proof['search']['rollback_mechanism']['restored_original_score_exact']
                and proof['search']['mask_replay_exact'] and proof['search']['noncode_gate_frozen_exact']
                and proof['search']['frozen_complement_versions_and_pointers_exact']
                and proof['saved']['state_and_mask_roundtrip_exact'], 'Mechanism proof is incomplete.')
        previous = read(Path(proof['launch_record']).parent / 'launch.prepared.json')
        terminal = read(proof['launch_record'])
        require(terminal['status'] == 'completed' and terminal['exit_code'] == 0
                and terminal['report_sha256'] == sha256(validation)
                and proof['source_checkpoint_id'] == SOURCE_ID,
                'Bind the actual successful mechanism terminal and report identity.')
        require(previous['source_sha256'] == sources and previous['choices_path'] == str(choices_path)
                and previous['token_batches'] == {d: {k: tensor_record(v) for k, v in b.items()} for d, b in batches.items()},
                'Mechanism source/proposals/inputs differ; never continue from its learned state.')
        require((Path(proof['launch_record']).parent / 'run.exit').read_text().strip() == '0', 'Mechanism did not exit successfully.')
        inputs[str(validation)] = sha256(validation)
        for name in ('launch.json', 'launch.prepared.json', 'run.exit'):
            evidence = Path(proof['launch_record']).parent / name
            inputs[str(evidence)] = sha256(evidence)
    else:
        require(groups == 8 and validation is None, 'The independent mechanism probe uses exactly group0..7.')
    record = {'schema': 'gate_final_mass_kl_accept_v1', 'status': 'prepared_unbound', 'gpu': None,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output), 'groups': groups,
        'source_checkpoint_id': SOURCE_ID, 'source_sha256': sources, 'input_sha256': inputs,
        'protected_stats': old['protected_stats'], 'data': data, 'choices_path': str(choices_path),
        'choices_sha256': sha256(choices_path), 'group_inventory': group['group_inventory'],
        'old_teacher_path': str(old_teacher_path), 'old_teacher_report': str(tail_root / 'run/teacher.json'),
        'token_batches': {d: {k: tensor_record(v) for k, v in b.items()} for d, b in batches.items()},
        'mechanism_proof': None if validation is None else str(validation),
        'cpu_native_loss_parity': cpu_mass_kl_check(), 'source_bit_budget': old['source_bit_budget'],
        'timeout_seconds': 900 if groups == 8 else 3600, 'termination_grace_seconds': 20,
        'environment': {**old['environment'], 'PYTHONPATH': str(ROOT)}, 'queued': False, 'automatic_retry': False,
        'protocol': {'order': list(range(groups)), 'fresh': 'Original f92713 gate tensors on every independent run, no decoder fitting.',
            'proposal': 'Original saved weight-neighbor choices; no proposal search, truncation, or local-SSE mask preselection.',
            'acceptance': 'All originaltrain32, native full[1,256] forward; top100+tail mass KL atT1, FP32token losses then fixed row-order FP64 sums; accept only strict decrease.',
            'cache': 'Rebuild actual packed plan and entire36gate category using native group8 after apply or restore.',
            'observation': 'Oldheld32 and fixedFineWeb32 are evaluated only after all decisions; never select, stop, or reorder proposals.',
            'teacher': 'Generate real train32 teacher top100/full logZ once, release full teacher, then one frozen student; reuse checked historical observation teacher cache.',
            'mechanism': '8group run additionally applies/restores group0 and repeats the original score before its decisions; this is not a fabricated rejection.',
            'save': 'Only a thirteen-tensor gate state, accepted mask/effective choices, and JSON evidence. No whole-model export/optimizer.',
            'boundary': 'Isolated new objective prototype; lower calibration KL does not establish full8, LongBench, or deployment benefit.'},
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.gate_final_kl_accept_probe.run', '--launch-record', str(output / 'launch.json')],
        'cuda_initialized': torch.cuda.is_initialized()}
    check(record); require(not record['cuda_initialized'], 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record)
    write_json(output / 'launch.prepared.json', record)
    print(json.dumps({'status': record['status'], 'groups': groups, 'sources': len(sources), 'inputs': len(inputs),
                      'cuda_initialized': False, 'launch_sha256': sha256(output / 'launch.json')}), flush=True)


def run(path):
    record = read(path); output = Path(record['output_dir']); code = 1
    require(record['status'] == 'bound_ready' and type(record['gpu']) is int and not record['queued']
            and not record['automatic_retry'] and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
            and Path.cwd() == Path(record['cwd']) and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
            and all(os.environ.get(k) == v for k, v in record['environment'].items())
            and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE'))
            and torch.cuda.device_count() == 1 and not (output / 'run').exists(), 'Require an explicitly bound one-shot fresh execution.')
    check(record); work = output / 'run'; work.mkdir()
    record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(),
                  worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
    write_json(path, record); deadline = record['started_unix'] + record['timeout_seconds']
    try:
        torch.set_num_threads(4); torch.cuda.set_device(0); configure_deterministic_mode(True); set_seed(137)
        data, batches, archive = data_and_inputs()
        require(data == record['data'] and {d: {k: tensor_record(v) for k, v in b.items()} for d, b in batches.items()}
                == record['token_batches'], 'Runtime data changed.')
        teacher, hf_mlp, teacher_report = teacher_pass(data, {'train': batches['train']}, archive, device=torch.device('cuda:0'),
            deadline=deadline, observer=lambda *args: None)
        del hf_mlp
        old_teacher = torch.load(record['old_teacher_path'], map_location='cpu', weights_only=True, mmap=True)
        validate_teacher_cache(old_teacher, batches, read(record['old_teacher_report'])); teacher.update(old_teacher)
        choices = torch.load(record['choices_path'], map_location='cpu', weights_only=True, mmap=True)
        with (work / 'trajectory.jsonl').open('x') as stream:
            def observer(row):
                stream.write(json.dumps(row) + '\n'); stream.flush()
                print(json.dumps({k: row[k] for k in ('group', 'accepted', 'delta_kl_sum', 'seconds')}), flush=True)
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                state, masks, result = search(data, batches, teacher, choices, groups=record['groups'], deadline=deadline, observer=observer)
        state_path, mask_path = work / 'gate_state.pt', work / 'accepted_mask.pt'
        torch.save(state, state_path); torch.save(masks, mask_path)
        saved = torch.load(state_path, map_location='cpu', weights_only=True, mmap=True)
        saved_masks = torch.load(mask_path, map_location='cpu', weights_only=True, mmap=True)
        require({k: tensor_record(v) for k, v in saved.items()} == result['final_state']
                and all(torch.equal(v, saved_masks[k]) for k, v in masks.items()), 'Saved endpoint or mask changed.')
        report = {'status': 'completed', 'launch_record': str(path), 'teacher_train': teacher_report, 'search': result,
            'source_checkpoint_id': SOURCE_ID, 'protocol': record['protocol'], 'saved': {'state_path': str(state_path),
                'state_sha256': sha256(state_path), 'mask_path': str(mask_path), 'mask_sha256': sha256(mask_path),
                'state_and_mask_roundtrip_exact': True}, 'trajectory_sha256': sha256(work / 'trajectory.jsonl'),
            'elapsed_seconds': time.time() - record['started_unix'], 'full8_and_longbench': 'not_evaluated'}
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
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--groups', type=int, choices=(8, 384), default=8)
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--launch-record', type=Path)
    args = parser.parse_args()
    if args.prepare:
        require(args.launch_record is None, 'Choose prepare or run.')
        prepare(args.prepare.resolve(), args.groups, None if args.validation is None else args.validation.resolve())
    else:
        require(args.launch_record is not None, 'A launch record is required.')
        sys.exit(run(args.launch_record.resolve()))
