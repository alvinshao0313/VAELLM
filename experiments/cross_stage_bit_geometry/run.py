"""CPU preparation and explicitly bound single-GPU geometry measurement."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from torch import nn

from experiments.cross_stage_bit_geometry.core import CANDIDATES, COUNT, SEED, measure, packed_candidate, support
from experiments.gate_student_context_probe.core import ready, records
from experiments.gate_student_context_probe.data import ROOT, SOURCE, SOURCE_ID, TARGET
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.checkpoint_v6 import _rebuild_converted_modules, refresh_vae_linear_runtime_after_state_load
from train_utils.utils import configure_deterministic_mode, set_seed

BASE = ROOT / 'result/qwen3_8b_mmlu65_20261004'


def read(path):
    return json.loads(Path(path).read_text())


def paths():
    previous = BASE / 'gate_student_context_probe'
    archive = BASE / 'rotation_factor_probe/middle_data/qwen'
    return {'capture': previous / 'run/student_capture/data_manifest.json',
        'prior_report': previous / 'run/report.json', 'prior_review': previous / 'review.json',
        'prior_exit': previous / 'run.exit', 'archive_manifest': archive / 'data_manifest.json',
        'archive': archive / 'inputs.pt', 'meta': SOURCE / 'checkpoint_meta.json'}


def load(p):
    capture, archive_manifest, meta = (read(p[k]) for k in ('capture', 'archive_manifest', 'meta'))
    require(capture['status'] == archive_manifest['status'] == 'captured'
            and capture['checkpoint_id'] == archive_manifest['checkpoint_id'] == meta['checkpoint_id'] == SOURCE_ID,
            'Use the existing original f927 capture/reference identities.')
    spec = next(s for s in meta['converted_modules'] if s['name'] == TARGET)
    require(spec == capture['target_spec'] and spec['weight_rotation']['block_size'] == 32
            and spec['stage_codebook_dims'] == [32, 32] and not spec['transpose'], 'Target native spec changed.')
    check_stat(capture['source_weight_stat'])
    require((SOURCE / meta['state_dict_file']).resolve() == Path(capture['source_weight_stat']['path']).resolve(), 'Wrong mmap source.')
    whole = torch.load(SOURCE / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    state = {k[len(TARGET) + 1:]: v for k, v in whole.items() if k.startswith(TARGET + '.')}
    require(len(state) == 13 and {k: tensor_record(v) for k, v in state.items()} == capture['expected_gate_state'],
            'Original thirteen gate tensors changed.')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        holder = nn.Module(); holder.projection = nn.Linear(4096, 12288, bias=False, dtype=torch.bfloat16)
        local_spec = deepcopy(spec); local_spec['name'] = 'projection'
        _rebuild_converted_modules(holder, [local_spec])
        layer = holder.projection
    layer.load_state_dict(state, strict=True); refresh_vae_linear_runtime_after_state_load(holder)
    layer.eval().requires_grad_(False); ready(layer)
    require(records(layer) == capture['expected_gate_state']
            and all(x.dtype == torch.bfloat16 for x in layer.parameters()), 'Strict native reconstruction changed values/dtypes.')
    archive = torch.load(p['archive'], map_location='cpu', weights_only=True, mmap=True)
    target = archive['targets'][TARGET]['reference_weight']
    require(archive['checkpoint_id'] == SOURCE_ID and archive['model_type'] == 'qwen3'
            and archive['layer_index'] == 18 and target.dtype == torch.bfloat16
            and tensor_record(target) == archive_manifest['tensors'][TARGET]['reference_weight']
            and torch.equal(layer.protected_input_indices, archive['targets'][TARGET]['protected_input_indices']),
            'Original HF weight or protection indices changed.')
    return layer, target, capture, archive_manifest


def check(record):
    check_hashes(record['source_sha256']); check_hashes(record['input_sha256'])
    for stat in record['protected_stats']:
        check_stat(stat)


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and Path.cwd() == ROOT and not output.exists(), 'Require fresh CPU-only preparation.')
    torch.set_num_threads(4); p = paths()
    prior, review = read(p['prior_report']), read(p['prior_review'])
    require(prior['status'] == 'completed' and review['status'] == 'verified'
            and review['report_sha256'] == sha256(p['prior_report'])
            and p['prior_exit'].read_text().strip() == '0', 'Existing source probe terminal evidence missing.')
    layer, target, capture, archive = load(p)
    require(sha256(p['archive']) == archive['inputs_sha256'], 'Reference archive hash changed.')
    cases = support(); indices = torch.tensor([c['vector'] for c in cases])
    original = torch.stack([layer.get_stage_part_vq_storage(s, 0)[indices, 0] for s in range(2)], dim=1)
    for bits in CANDIDATES:
        proposed = packed_candidate(original, bits)
        for s in range(2):
            expected = int(bits[s] >= 0)
            for row in (original[:, s] ^ proposed[:, s]).tolist():
                require(sum(int(v).bit_count() for v in row) == expected, 'CPU candidate XOR mismatch.')
    sources = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.suffix == '.py' and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / 'result'):
                sources[str(path)] = sha256(path)
    for path in Path(__file__).parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path.resolve())] = sha256(path)
    record = {'schema': 'cross_stage_bit_geometry_v1', 'status': 'prepared_unbound', 'gpu': None,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'paths': {k: str(v) for k, v in p.items()}, 'source_checkpoint_id': SOURCE_ID, 'target': TARGET,
        'source_sha256': sources, 'input_sha256': {str(v): sha256(v) for k, v in p.items() if k != 'archive'},
        'protected_stats': [capture['source_weight_stat'], file_stat(p['archive'])],
        'archive_hash_verified': archive['inputs_sha256'], 'initial_state': records(layer),
        'reference_weight': tensor_record(target), 'cases': cases, 'cpu_all1089_selected_xor_verified': True,
        'timeout_seconds': 600, 'termination_grace_seconds': 20, 'queued': False, 'automatic_retry': False,
        'environment': {'PYTHONPATH': str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4'},
        'protocol': {'seed': SEED, 'vectors': COUNT,
            'selection': 'CPU seed31: randperm384 first32 output RHT groups, randint32 row offsets, randint126 input blocks. No error/gradient/activation-based selection.',
            'source': 'Fresh f927 L18 gate only; native strict13 tensors, all frozen. No full causal model loading.',
            'candidates': 'Original, stage0bit0..31, stage1bit0..31, then lexicographic32x32 cross-stage pairs. Every candidate starts from original packed state.',
            'native_path': 'For each candidate simultaneously edit32 disjoint output-block supports, then complete original-shape BF16 _decode_weight including stage sum, inverse RHT and protected INT8 columns. No chunk-shape decode or additive-delta proxy.',
            'objective': 'FP64 SSE between actual BF16 candidate weights and existing BF16 HF reference, on each complete original-coordinate32x32 support. Both arms use the same target/reduction. Outside contributions are constant.',
            'curvature': 'Exact finite-step decomposition 2<baseline-HF,delta>+||delta||^2, reducedFP64; report actual pair-minus-summed-singles discrepancy but never use it for scoring.',
            'comparison': 'Original wins ties, then earlier candidate. Record both singly-strictly-worse and singly-nonbeneficial/pair-beneficial counts, per-vector best neighborhoods, actual XOR and weight displacement. Larger neighborhood/extra flips are explicit, not equal search/update cost.',
            'validation': 'Full-weight same-shape baseline repeat; five predefined candidates verify outside-support/protection, four also verify isolated first case. Full persistent state and original decoded weight restored.',
            'boundary': 'Only sampled neighborhood geometry, not an optimizer/PV reproduction, whole-model KL/NLL, eight tasks, LongBench or integration evidence.',
            'resource_estimate': 'One A800, expected2-5min, hard600+20s, reserve10GiB GPU and<10MB output. No large model export.'},
        'launcher_argv': [sys.executable, '-u', '-m', 'experiments.cross_stage_bit_geometry.run',
                          '--launch-record', str(output / 'launch.json')], 'cuda_initialized': False}
    check(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.prepared.json', record)
    print(json.dumps({'status': record['status'], 'sha256': sha256(output / 'launch.prepared.json'),
                      'sources': len(sources), 'inputs': len(record['input_sha256']), 'stats': 2, 'cases': cases}), flush=True)


def run(path):
    record = read(path); output = Path(record['output_dir']); code = 1
    require(record['status'] == 'bound_ready' and type(record['gpu']) is int
            and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
            and all(os.environ.get(k) == v for k, v in record['environment'].items())
            and Path.cwd() == Path(record['cwd']) and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
            and not record['queued'] and not record['automatic_retry'] and torch.cuda.device_count() == 1
            and not (output / 'run').exists(), 'Require a fresh explicit GPU binding.')
    check(record); work = output / 'run'; work.mkdir()
    record.update(status='running', worker_pid=os.getpid(), started_unix=time.time(),
        worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]))
    write_json(path, record)
    try:
        torch.set_num_threads(4); torch.cuda.set_device(0); configure_deterministic_mode(True); set_seed(SEED)
        layer, target, _, _ = load(record['paths'])
        require(records(layer) == record['initial_state'] and support() == record['cases'], 'Prepared state/support changed.')
        layer.to('cuda:0'); ready(layer); torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda', enabled=False):
            raw, diagnostic = measure(layer, target, record['cases'],
                deadline=record['started_unix'] + record['timeout_seconds'], observer=lambda row: print(json.dumps(row), flush=True))
        raw_path = work / 'geometry.pt'; torch.save(raw, raw_path)
        saved = torch.load(raw_path, map_location='cpu', weights_only=True)
        require(all(torch.equal(saved[k], v) for k, v in raw.items()), 'Small geometry artifact failed exact roundtrip.')
        check(record)
        report = {'status': 'completed', 'source_checkpoint_id': SOURCE_ID, 'target': TARGET,
            'protocol': record['protocol'], 'diagnostic': diagnostic, 'raw_path': str(raw_path),
            'raw_sha256': sha256(raw_path), 'raw_tensors': {k: tensor_record(v) for k, v in raw.items()},
            'metric_columns': ['sse_delta', 'cross_term', 'displacement_squared_norm', 'sse', 'pair_nonadditivity_squared_norm'],
            'saved_model': False, 'all_quality_evaluation': 'not_performed',
            'elapsed_seconds': time.time() - record['started_unix']}
        write_json(work / 'report.json', report); record.update(status='completed', report_sha256=sha256(work / 'report.json')); code = 0
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        print(record['traceback'], flush=True)
    finally:
        record.update(exit_code=code, finished_unix=time.time()); write_json(path, record)
        (output / 'run.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', type=Path); modes.add_argument('--launch-record', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare.resolve())
    else:
        sys.exit(run(args.launch_record.resolve()))
