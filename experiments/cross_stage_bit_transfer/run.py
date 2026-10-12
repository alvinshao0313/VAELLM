"""Prepare or run one fixed1024 three-arm development observation, without training."""
from __future__ import annotations

import argparse
import ast
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from experiments.cross_stage_bit_geometry.run import BASE, ROOT, SOURCE_ID, TARGET, check, load, read
from experiments.cross_stage_bit_transfer.search import ARMS, COUNT, chosen_packed, search, support
from experiments.cross_stage_bit_transfer.evaluate import evaluate
from experiments.gate_final_kl_accept_probe.run import validate_teacher_cache
from experiments.gate_student_context_probe.core import ready, records
from experiments.gate_tail_objective_probe.data import DOMAINS, prepare_data
from experiments.longbench_v1.common import sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.utils import configure_deterministic_mode, set_seed


def prepare(output):
    require(not output.exists() and Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized(), 'Fresh CPU-only preparation required.')
    torch.set_num_threads(4)
    previous = BASE / 'cross_stage_bit_geometry_f927_l18_s32'
    old, report, review, terminal = (read(previous / n) for n in ('launch.prepared.json', 'run/report.json', 'independent_review.json', 'launch.json'))
    require(report['status'] == terminal['status'] == 'completed' and terminal['exit_code'] == 0
            and review['status'] == 'independent_cpu_verified'
            and all((previous / n).read_text().strip() == '0' for n in ('run.exit', 'outer.exit'))
            and review['report_sha256'] == sha256(previous / 'run/report.json')
            and terminal['report_sha256'] == sha256(previous / 'run/report.json'), 'Completed verified geometry required.')
    check(old); layer, target, _, _ = load(old['paths']); cases = support()
    require(cases[:32] == old['cases'] == report['diagnostic']['cases'], 'Original32 support changed.')
    data, batches, _ = prepare_data(BASE)
    require(data['source_data']['expected_gate_state'] == old['initial_state'], 'Observation source differs from geometry source.')
    tail = BASE / 'gate_tail_objective_probe_r1/run'; teacher_path = tail / 'teacher_top100.pt'
    teacher_report = read(tail / 'teacher.json'); teacher = torch.load(teacher_path, map_location='cpu', weights_only=True, mmap=True)
    require(sha256(teacher_path) == read(tail / 'report.json')['teacher_cache_sha256'], 'Existing teacher cache hash differs.')
    validate_teacher_cache(teacher, batches, teacher_report)
    historical = BASE / 'gate_final_kl_neighbor_transfer'
    history, history_review = read(historical / 'run/report.json'), read(historical / 'review.json')
    require(history['status'] == 'completed' and history_review['status'] == 'verified'
            and history_review['report_sha256'] == sha256(historical / 'run/report.json'), 'Verified original whole-model baseline missing.')
    expected_before = {d: history['transfer']['before'][d] for d in DOMAINS}
    indices = torch.tensor([c['vector'] for c in cases])
    original = torch.stack([layer.get_stage_part_vq_storage(s, 0)[indices, 0] for s in range(2)], dim=1)
    probe = torch.linspace(0, 1088, COUNT).round().to(torch.int16)
    proposed = chosen_packed(original, probe)
    require(chosen_packed(proposed, probe).equal(original), 'CPU compact choice apply/undo failed.')
    sources, inputs = dict(data['source_sha256']), dict(data['input_sha256'])
    sources.update(old['source_sha256']); inputs.update(old['input_sha256'])
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            p = Path(filename).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(ROOT) and not p.is_relative_to(ROOT / 'result'):
                value = sha256(p); require(str(p) not in sources or sources[str(p)] == value, 'Inherited source pin changed.')
                sources[str(p)] = value
    for p in Path(__file__).parent.glob('*.py'):
        ast.parse(p.read_text()); sources[str(p.resolve())] = sha256(p)
    evidence = [previous / n for n in ('launch.prepared.json', 'launch.json', 'run/report.json', 'independent_review.json', 'run.exit', 'outer.exit')]
    evidence += [teacher_path, tail / 'teacher.json', tail / 'report.json', historical / 'run/report.json', historical / 'review.json']
    for p in evidence: inputs[str(p)] = sha256(p)
    record = {'schema': 'cross_stage_bit_transfer_v1', 'status': 'prepared_unbound', 'gpu': None,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output), 'paths': old['paths'],
        'source_checkpoint_id': SOURCE_ID, 'target': TARGET, 'source_sha256': sources, 'input_sha256': inputs,
        'protected_stats': old['protected_stats'] + data['input_data_stat'], 'data': data, 'cases': cases,
        'token_batches': {d: {k: tensor_record(v) for k, v in b.items()} for d, b in batches.items()},
        'teacher_path': str(teacher_path), 'teacher_report': str(tail / 'teacher.json'), 'expected_before': expected_before,
        'initial_state': records(layer), 'reference_weight': tensor_record(target), 'environment': old['environment'],
        'timeout_seconds': 600, 'termination_grace_seconds': 20, 'queued': False, 'automatic_retry': False,
        'cpu_compact_apply_undo_exact': True, 'cuda_initialized': False,
        'protocol': {'arms': list(ARMS), 'vectors': COUNT,
            'support': 'Retain original seed31 first32 cases verbatim; separate CPU generator seed31 randperm48384 RHT tiles, discard those32, take first992, then randint32 row offsets. Distinct tile pairs, not necessarily distinct output groups; no SSE/text filtering.',
            'search': 'Original+64single versus original+64single+1024crossstage pairs; each candidate full native BF16 projection decode/inverse/protection. Same FP64 original-coordinate HF weight SSE and original-first/earlier-candidate ties. Both endpoints independently start at f927.',
            'selection': 'No textual loss, gradient, KL acceptance or heldout selection; no decoder optimizer. Search budget and maximum flips differ explicitly.',
            'observation': 'Only existing Alpaca held32 and FineWeb physical500000..500031, native B1 fixed256,3714+7573causal positions per arm. Previously explored development data, not fresh blind validation.',
            'loss': 'Existing score/probability_metrics top100+tail mass KL T1 plus actual NLL; native BF16/FA2/FP32RoPE, globalgroup8, per-tokenFP32 then per-documentFP64 sums. Reuse actual label/hash-pinned teacher cache; no teacher regeneration.',
            'path_boundary': 'Search uses complete single-projection native decode; observation uses existing actual whole-model group8 cache. Record per-arm group8 HF SSE and differences to searched BF16 tiles; do not assume bitwise interchangeability.',
            'frozen': 'Original3423 state; only two gate codebanks change per arm,11noncode gate tensors and3410complement frozen. Restore13 and cache plus both-domain first full logits after single and after pair.',
            'save': 'Compact choices, candidate bit mapping, per-tile SSE and actual XOR, per-document KL/NLL/report only. No decoder/gate/full-model checkpoint, gradient or optimizer.',
            'boundary': 'One fixed coverage and seed, no adaptively expanded supports or early stopping. Geometry/output transfer diagnosis, not eight-task/LongBench accuracy or formal algorithm integration.',
            'resource_estimate': 'One A800,120-180seconds estimated,600+20hard limit,75GiB reservation,<100MB output.'},
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.cross_stage_bit_transfer.run', '--launch-record', str(output / 'launch.json')]}
    check(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.prepared.json', record)
    print(json.dumps({'status': record['status'], 'sha256': sha256(output / 'launch.prepared.json'),
                      'sources': len(sources), 'inputs': len(inputs), 'stats': len(record['protected_stats']),
                      'cases': len(cases), 'causal_tokens': data['causal_tokens']}), flush=True)


def run(path):
    record = read(path); output = Path(record['output_dir']); code = 1
    require(record['status'] == 'bound_ready' and type(record['gpu']) is int
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
            and all(os.environ.get(k) == v for k, v in record['environment'].items())
            and Path.cwd() == ROOT and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
            and not record['queued'] and not record['automatic_retry'] and torch.cuda.device_count() == 1
            and not (output / 'run').exists(), 'Use one fresh explicit binding.')
    check(record); work = output / 'run'; work.mkdir()
    record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(),
        worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19])); write_json(path, record)
    try:
        torch.set_num_threads(4); torch.cuda.set_device(0); configure_deterministic_mode(True); set_seed(31)
        deadline = record['started_unix'] + record['timeout_seconds']
        data, batches, _ = prepare_data(BASE)
        require(data == record['data'] and {d: {k: tensor_record(v) for k, v in b.items()} for d, b in batches.items()}
                == record['token_batches'], 'Runtime data/source differs.')
        teacher = torch.load(record['teacher_path'], map_location='cpu', weights_only=True, mmap=True)
        validate_teacher_cache(teacher, batches, read(record['teacher_report']))
        layer, target, _, _ = load(record['paths']); require(records(layer) == record['initial_state'], 'Source gate differs.')
        layer.to('cuda:0'); ready(layer); torch.cuda.reset_peak_memory_stats()
        observer = lambda row: print(json.dumps(row), flush=True)
        with torch.autocast('cuda', enabled=False):
            raw, projection_tiles, geometry = search(layer, target, record['cases'], deadline=deadline, observer=observer)
        del layer; gc.collect(); torch.cuda.empty_cache()
        with torch.autocast('cuda', enabled=False):
            transfer = evaluate(data, batches, teacher, record['cases'], raw['choices'], projection_tiles, target,
                                record['expected_before'], deadline=deadline, observer=observer)
        raw['actual_stage_xor'] = torch.tensor([transfer['actual_stage_xor'][a] for a in ARMS], dtype=torch.int64)
        raw_path = work / 'choices.pt'; torch.save(raw, raw_path)
        reread = torch.load(raw_path, map_location='cpu', weights_only=True)
        require(all(torch.equal(reread[k], v) for k, v in raw.items()), 'Compact artifact roundtrip failed.')
        check(record)
        report = {'status': 'completed', 'source_checkpoint_id': SOURCE_ID, 'protocol': record['protocol'],
            'search': geometry, 'transfer': transfer, 'raw_path': str(raw_path), 'raw_sha256': sha256(raw_path),
            'raw_tensors': {k: tensor_record(v) for k, v in raw.items()}, 'saved_model_or_optimizer': False,
            'full8_and_longbench': 'not_evaluated', 'elapsed_seconds': time.time() - record['started_unix']}
        write_json(work / 'report.json', report); record.update(status='completed', report_sha256=sha256(work / 'report.json')); code = 0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc()); print(record['traceback'], flush=True)
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'run.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', type=Path); modes.add_argument('--launch-record', type=Path)
    args = parser.parse_args()
    if args.prepare: prepare(args.prepare.resolve())
    else: sys.exit(run(args.launch_record.resolve()))
