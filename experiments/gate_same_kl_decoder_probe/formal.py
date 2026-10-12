"""Prepare or run one independent fixed-code500 arm on native final mass KL."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from experiments.gate_final_kl_accept_probe.run import BASE, check, data_and_inputs, read, validate_teacher_cache
from experiments.gate_same_kl_decoder_probe.core import ARMS, HORIZON, STEPS, TOKENS, probe
from experiments.gate_same_kl_decoder_probe.run import source_states
from experiments.gate_student_context_probe.data import ROOT, SOURCE_ID
from experiments.gate_tail_objective_probe.data import DOMAINS
from experiments.longbench_v1.common import sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.utils import configure_deterministic_mode, set_seed

MECHANISM = BASE / 'gate_same_kl_decoder_probe_s2'
TAIL = BASE / 'gate_tail_objective_probe_r1'
CORE = Path(__file__).resolve().with_name('core.py')
OLD_CORE = MECHANISM / 'core_s2.py'
OLD_CORE_SHA = 'f5f042212c8d689c73fb97a31329afd1ad051ffbc463be9b548e7726b4b08c67'


def cached_teachers(record, batches):
    teachers = {}
    for entry in record['teacher_caches']:
        require(sha256(entry['path']) == entry['sha256'], 'Historical CPU teacher cache changed.')
        cache = torch.load(entry['path'], map_location='cpu', weights_only=True, mmap=True)
        previous = read(entry['report'])[entry['report_field']]
        validate_teacher_cache(cache, batches, previous)
        require(tuple(cache) == tuple(entry['domains']) and not set(cache).intersection(teachers),
                'Teacher domains must be disjoint and retain their original document order.')
        teachers.update(cache)
    require(tuple(teachers) == ('train', *DOMAINS), 'Require exactly train32 and the two old observation domains.')
    return teachers


def prepare(output, arm):
    require(arm in ARMS and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and Path.cwd() == ROOT and not output.exists(), 'Prepare one fresh CPU-only arm in the public root.')
    torch.set_num_threads(4)
    old = read(MECHANISM / 'launch.prepared.json')
    report, review, terminal = (read(MECHANISM / name) for name in ('run/report.json', 'review.json', 'launch.json'))
    require(report['status'] == terminal['status'] == 'completed' and terminal['exit_code'] == 0
            and review['status'] == 'verified' and review['updates_per_arm'] == STEPS
            and review['schedule_horizon'] == HORIZON and review['all6_native_FP32_gradient_and_Adam_evidence']
            and report['source_checkpoint_id'] == review['source_checkpoint_id'] == SOURCE_ID
            and review['prepared_sha256'] == sha256(MECHANISM / 'launch.prepared.json')
            and review['report_sha256'] == terminal['report_sha256'] == sha256(MECHANISM / 'run/report.json')
            and review['teacher_cache_sha256'] == report['teacher_cache_sha256'], 'Require the completed native two-update proof.')
    sources = dict(old['source_sha256'])
    require(sources[str(CORE)] == sha256(OLD_CORE) == OLD_CORE_SHA, 'The original mechanism core backup differs.')
    for path, digest in sources.items():
        require(sha256(OLD_CORE if path == str(CORE) else path) == digest,
                'Applicable mechanism source changed without a declared historical mapping: ' + path)
    sources[str(CORE)] = sha256(CORE)
    tail = read(TAIL / 'launch.prepared.json')
    tail_report, tail_review = (read(TAIL / name) for name in ('run/report.json', 'review.json'))
    require(tail_report['status'] == 'completed' and tail_review['status'] == 'verified'
            and tail_review['report_sha256'] == sha256(TAIL / 'run/report.json')
            and tail_review['cached_labels_native_causal_exact'] and tail_review['cached_tensor_records_exact'],
            'Require the completed original held32/FineWeb teacher-cache evidence.')
    for directory in (MECHANISM, TAIL):
        require(all((directory / name).read_text().strip() == '0' for name in ('run.exit', 'outer.exit')),
                'A required historical run did not finish successfully.')
    data, batches, archive = data_and_inputs()
    tokens = {domain: {key: tensor_record(value) for key, value in batch.items()} for domain, batch in batches.items()}
    require(data == tail['data'] and data['source_data'] == old['data']['source_data']
            and tokens['train'] == old['token_batch']
            and {name: tokens[name] for name in DOMAINS} == data['token_batches']
            and int(batches['train']['attention_mask'][:, 1:].sum()) == TOKENS,
            'Use the exact original train32 and previously observed two domains.')
    states = source_states(data['source_data'], old['arms'])
    require(tuple(states) == ARMS, 'All initial states must replay from original f927, never a fitted endpoint.')
    for inherited in (data['source_sha256'], tail['source_sha256']):
        for path, digest in inherited.items():
            if path in sources:
                require(sources[path] == digest, 'Native data/teacher source versions differ: ' + path)
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.suffix == '.py' and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / 'result'):
                digest = sha256(path)
                require(str(path) not in sources or sources[str(path)] == digest, 'A loaded source differs from the inherited proof.')
                sources[str(path)] = digest
    for path in (CORE, Path(__file__).resolve()):
        ast.parse(path.read_text()); sources[str(path)] = sha256(path)
    inputs = dict(old['input_sha256']); inputs.update(data['input_sha256'])
    evidence = [OLD_CORE]
    for directory in (MECHANISM, TAIL):
        evidence += [directory / name for name in ('launch.prepared.json', 'launch.json', 'run/report.json',
                      'review.json', 'run.exit', 'outer.exit', 'run/teacher_top100.pt')]
    inputs.update({str(path): sha256(path) for path in evidence})
    stats = {item['path']: item for item in old['protected_stats']}
    for item in tail['protected_stats']:
        require(item['path'] not in stats or stats[item['path']] == item, 'Source/data stat identities differ.')
        stats[item['path']] = item
    teacher_caches = [
        {'path': str(MECHANISM / 'run/teacher_top100.pt'), 'sha256': report['teacher_cache_sha256'],
         'report': str(MECHANISM / 'run/report.json'), 'report_field': 'teacher_train', 'domains': ['train']},
        {'path': str(TAIL / 'run/teacher_top100.pt'), 'sha256': tail_report['teacher_cache_sha256'],
         'report': str(TAIL / 'run/report.json'), 'report_field': 'teacher', 'domains': list(DOMAINS)}]
    step_seconds = [row['seconds'] for item in report['experiment']['arms'].values() for row in item['training']['trajectory']]
    protocol = deepcopy(old['protocol'])
    protocol.update(arms=[arm], updates_per_arm=HORIZON, schedule_steps=HORIZON,
        population='Exactly the old train32/4257 causal tokens per update; oldheld32/FineWeb32 only after the fixed BF16 endpoint.',
        updates='Six FP32 decoder masters/Adam, BF16 native live packed graph; LR1e-5/betas(.9,.95)/eps1e-8/wd0/linear500/warm0,500 updates.',
        teacher='Reuse the two reviewed CPU top100 caches with exact labels/statistics. No new teacher forward or cache copy.',
        boundary='One independent fresh500 arm, not continuation of s2 or conditional-MLP500; no code updates, heldout selection, full8 or framework integration.')
    record = {'schema': 'gate_same_final_mass_kl_decoder_single500_v1', 'status': 'prepared_unbound', 'gpu': None,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output), 'arm': arm,
        'source_checkpoint_id': SOURCE_ID, 'source_sha256': sources, 'input_sha256': inputs,
        'protected_stats': list(stats.values()), 'data': data, 'token_batches': tokens, 'arms': old['arms'],
        'initial_state_order': list(ARMS), 'teacher_caches': teacher_caches, 'source_bit_budget': old['source_bit_budget'],
        'historical_core': {'original_path': str(CORE), 'backup_path': str(OLD_CORE), 'original_sha256': OLD_CORE_SHA,
            'current_sha256': sources[str(CORE)], 'change': 'Explicit steps2|500, optional single arm, endpoint-only observations; no change to the native update algorithm.'},
        'mechanism_review': str(MECHANISM / 'review.json'), 'environment': old['environment'], 'protocol': protocol,
        'timeout_seconds': 4800, 'termination_grace_seconds': 20, 'queued': False, 'automatic_retry': False,
        'resource_estimate': {'measured_s2_step_seconds': [min(step_seconds), max(step_seconds)],
            'extrapolated_500_update_seconds': HORIZON * max(step_seconds),
            'measured_training_peak_allocated_bytes': max(item['training']['peak']['allocated_bytes'] for item in report['experiment']['arms'].values()),
            'boundary': 'Extrapolation from the same real32-document graph; full500 duration remains unmeasured.4800+20 outer budget, no retry or shortened run.'},
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.gate_same_kl_decoder_probe.formal',
                          '--launch-record', str(output / 'launch.json')], 'cuda_initialized': False}
    cached_teachers(record, batches)
    del states, archive
    check(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.json', record); write_json(output / 'launch.prepared.json', record)
    print(json.dumps({'status': record['status'], 'arm': arm, 'launch_sha256': sha256(output / 'launch.json'),
        'sources': len(sources), 'inputs': len(inputs), 'stats': len(stats), 'cuda_initialized': False}), flush=True)


def run(path):
    record = read(path); output = Path(record['output_dir']); code = 1
    require(record['status'] == 'bound_ready' and record['arm'] in ARMS and type(record['gpu']) is int
            and not record['queued'] and not record['automatic_retry']
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
            and Path.cwd() == Path(record['cwd']) and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
            and all(os.environ.get(k) == v for k, v in record['environment'].items())
            and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE'))
            and torch.cuda.device_count() == 1 and not (output / 'run').exists(), 'Bind one fresh independent500 arm without retry.')
    check(record); work = output / 'run'; work.mkdir()
    record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(),
        worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
    write_json(path, record); deadline = record['started_unix'] + record['timeout_seconds']
    try:
        torch.set_num_threads(4); torch.cuda.set_device(0); configure_deterministic_mode(True); set_seed(137)
        data, batches, archive = data_and_inputs()
        require(data == record['data'] and {d: {k: tensor_record(v) for k, v in b.items()} for d, b in batches.items()}
                == record['token_batches'], 'Runtime source or token populations changed.')
        teachers = cached_teachers(record, batches)
        arm_states = source_states(data['source_data'], record['arms'])
        del archive
        gc.collect()
        observations = {name: (batches[name], teachers[name]) for name in DOMAINS}
        with (work / 'trajectory.jsonl').open('x') as stream:
            def observer(arm, row):
                stream.write(json.dumps({'arm': arm, **row}) + '\n'); stream.flush()
                print(json.dumps({'arm': arm, 'step': row['step'], 'kl_mean': row['kl_mean'], 'seconds': row['seconds']}), flush=True)
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                result = probe(data, batches['train'], teachers['train'], arm_states, work=work, deadline=deadline,
                    observer=observer, steps=HORIZON, arm=record['arm'], observations=observations)
        require(result['status'] == 'completed' and tuple(result['arms']) == (record['arm'],)
                and result['arms'][record['arm']]['training']['steps'] == HORIZON
                and result['arms'][record['arm']]['training']['actual_trainable_forward_calls'] == 32 * HORIZON,
                'The predeclared full500 population must finish without truncation.')
        report = {'status': 'completed', 'launch_record': str(path), 'source_checkpoint_id': SOURCE_ID,
            'arm': record['arm'], 'protocol': record['protocol'], 'teacher_caches_reused': record['teacher_caches'],
            'historical_core': record['historical_core'], 'experiment': result,
            'trajectory_sha256': sha256(work / 'trajectory.jsonl'),
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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', type=Path); mode.add_argument('--launch-record', type=Path)
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.prepare:
        if args.arm is None:
            parser.error('--prepare requires --arm')
        prepare(args.prepare.resolve(), args.arm)
    else:
        if args.arm is not None:
            parser.error('--arm is fixed by the prepared record')
        sys.exit(run(args.launch_record.resolve()))
