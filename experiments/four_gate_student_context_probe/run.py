"""Prepare or run three fresh gate fits and reuse the independently verified L18 fit."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time

import torch

from experiments.four_gate_student_context_probe.data import (
    SOURCE_ID, LAYERS, TRAIN_LAYERS, prepare_data, capture, cache_records, target_name,
)
from experiments.gate_student_context_probe.run import merge, paired, read
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record
from train_utils.utils import configure_deterministic_mode, set_seed


def check(record):
    check_hashes(record['source_sha256']); check_hashes(record['input_sha256'])
    check_stat(record['data']['source_weight_stat'])
    for item in record['data']['hf_weight_files']:
        check_stat(item)


def prepare(args):
    require(args.base_root is not None and args.output_root is not None and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'Provide both directories with CUDA hidden for CPU preparation.')
    base, output = args.base_root.resolve(), args.output_root.resolve()
    require(not output.exists(), 'Use a new isolated output directory.')
    torch.set_num_threads(4)
    data, batches, archives = prepare_data(base)
    control_path = base / 'gate_student_context_full8_r1/review.json'
    control = read(control_path)
    require(control['status'] == 'verified' and control['source_checkpoint_id'] == SOURCE_ID
            and control['arms']['original_codes']['mean_percent'] == 69.49172020132849,
            'Bind the actual single-gate full8 control without claiming cross-layer additivity.')
    sources, pins = dict(data['source_sha256']), dict(data['input_sha256'])
    for relative in ('experiments/gate_student_context_probe/core.py', 'experiments/gate_student_context_probe/run.py',
                     'experiments/joint_reconstruction_probe/core.py', 'experiments/full_weight_bits_probe/core.py',
                     'litebsq/vae_linear_prewarm.py'):
        dependency = Path(__file__).resolve().parents[2] / relative
        merge(sources, {str(dependency): sha256(dependency)})
    for path in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(path.read_text()); merge(sources, {str(path): sha256(path)})
    merge(pins, {str(control_path): sha256(control_path)})
    record = {'status': 'cpu_ready', 'base_root': str(base), 'output_root': str(output), 'python': sys.executable,
        'source_checkpoint_id': SOURCE_ID, 'data': data, 'source_sha256': sources, 'input_sha256': pins,
        'cuda_initialized': torch.cuda.is_initialized(), 'full8_control_review': str(control_path),
        'protocol': {'layers': list(LAYERS), 'train_layers': list(TRAIN_LAYERS), 'reuse_layer': 18,
            'capture': 'One frozen original-f927 forward per document through four preselected gate hooks; stop at L27 gate.',
            'l18_reuse': 'Require exact original three HF matrices, all student inputs and frozen native up/down; no retraining.',
            'fit': 'Independent fresh original gate-only500 at each of0/9/27, fixed original codes; no rolling prefix or selection.',
            'optimizer': 'AdamW FP32 masters/moments, LR1e-5, betas(.9,.95), eps1e-8,wd0,linear500,warm0',
            'compute': 'Native BF16 decode/rotation/protection/MLP, FP32 MSE with fixed per-layer HF output energy.',
            'population': 'Same already-used train32/held32/replay32; each update uses all4289 train tokens; endpoints only for other splits.',
            'wall_budget_seconds': 900, 'kill_grace_seconds': 20, 'extra_mechanism_updates': 0,
            'expected_capture_peak_gib_reference': 65.7087, 'expected_fit_peak_gib_reference': 4.51,
            'estimate': 'Prior L18 capture37.56s and gate50044.35s; deeper capture and new stages need actual measurement.',
            'boundary': 'Only small local endpoints and output metrics; no assembled full model, full8, LongBench or production integration.'},
        'command': [sys.executable, '-m', 'experiments.four_gate_student_context_probe.run', '--preparation', str(output / 'preparation.json')]}
    require(not record['cuda_initialized'] and set(archives) == set(LAYERS), 'CPU preparation must not initialize CUDA.')
    check(record)
    output.mkdir(parents=True)
    write_json(output / 'preparation.json', record)
    print(json.dumps({'status': 'cpu_ready', 'preparation_sha256': sha256(output / 'preparation.json'),
        'source_pins': len(sources), 'input_pins': len(pins), 'layers': list(LAYERS), 'train_layers': list(TRAIN_LAYERS),
        'tokens': {s: sum(v) for s, v in data['document_lengths'].items()}, 'cuda_initialized': False}), flush=True)


def fresh(initial, device, *, payload=None):
    from experiments.four_gate_student_context_probe.core import ready
    layer = deepcopy(initial)
    if payload is not None:
        loaded = layer.load_state_dict(payload, strict=True)
        require(not loaded.missing_keys and not loaded.unexpected_keys, 'Incomplete saved local native state.')
    layer.to(device).eval().requires_grad_(False)
    ready(layer)
    return layer


def run(path):
    from experiments.four_gate_student_context_probe.core import measure, records, train
    p = read(path)
    require(p['status'] == 'cpu_ready' and Path(sys.executable).resolve() == Path(p['python']).resolve()
            and torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Use the prepared bitvae entry and one assigned GPU.')
    check(p)
    output = Path(p['output_root']) / 'run'
    require(not output.exists(), 'No overwrite, queue, resume or automatic retry.')
    output.mkdir()
    started = time.time(); deadline = started + p['protocol']['wall_budget_seconds']
    device = torch.device('cuda:0'); torch.cuda.set_device(device); torch.set_num_threads(4)
    configure_deterministic_mode(True); set_seed(31)
    report = {'status': 'running', 'pid': os.getpid(), 'argv': sys.argv,
        'process_start_ticks': int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'started_unix': started, 'deadline_unix': deadline, 'preparation_sha256': sha256(path),
        'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'protocol': p['protocol'], 'layers': {}}
    report_path = output / 'report.json'; write_json(report_path, report)
    try:
        data, batches, archives = prepare_data(p['base_root'])
        require(data == p['data'], 'Runtime source, tokens or per-layer HF matrices changed after preparation.')
        initial, caches, capture_manifest = capture(data, batches, archives, output / 'student_capture', device)
        gc.collect(); torch.cuda.empty_cache()
        frozen_cache = {i: cache_records(value) for i, value in caches.items()}
        report['capture'] = {'manifest_path': str(output / 'student_capture/manifest.json'),
            'manifest_sha256': sha256(output / 'student_capture/manifest.json'),
            'elapsed_seconds': capture_manifest['finished_unix'] - capture_manifest['started_unix'],
            'peak_allocated_bytes': capture_manifest['peak_allocated_bytes'], 'peak_reserved_bytes': capture_manifest['peak_reserved_bytes'],
            'l18_input_and_context_exact': capture_manifest['l18_input_and_context_exact'],
            'source_persistent_state_unchanged': capture_manifest['source_persistent_state_unchanged']}
        old = read(data['original_report'])
        layer = fresh(initial[18], device)
        baseline = measure(layer, archives[18], caches[18], target_name(18), device)
        require(baseline == old['before_original_codes'], 'New multi-hook L18 baseline does not reproduce every old metric.')
        del layer; gc.collect(); torch.cuda.empty_cache()
        control_state = torch.load(data['control_state_path'], map_location='cpu', weights_only=True, mmap=True)
        layer = fresh(initial[18], device, payload=control_state)
        require(records(layer) == old['arms']['original_codes']['training']['final_payload'], 'The retained L18 fit differs from its training proof.')
        endpoint = measure(layer, archives[18], caches[18], target_name(18), device)
        require(endpoint == data['control_endpoint'], 'Retained L18 endpoint no longer reproduces all old metrics.')
        report['layers']['18'] = {'status': 'reused_verified', 'training_updates_this_run': 0,
            'before': baseline, 'endpoint': endpoint, 'versus_original': paired(baseline, endpoint),
            'state_file': data['control_state_path'], 'state_sha256': data['control_state_sha256'],
            'original_training_report': data['original_report'], 'old_native_values_and_metrics_exact': True}
        del layer, control_state; gc.collect(); torch.cuda.empty_cache()
        report['resource_projection'] = {'remaining_seconds_before_fits': deadline - time.time(),
            'three_fits_prior_observed_seconds': 3 * 44.35352986998623,
            'estimate_only': True, 'capture_finished_without_mutating_source': True}
        write_json(report_path, report)
        for index in TRAIN_LAYERS:
            require(time.time() < deadline, 'The total approved deadline expired before a fresh arm.')
            layer = fresh(initial[index], device)
            baseline = measure(layer, archives[index], caches[index], target_name(index), device)
            trained = train(layer, archives[index], caches[index], target_name(index), deadline=deadline,
                observer=lambda row, i=index: logging.info('layer%s %s', i, row))
            require(trained['initial_payload'] == data['per_layer'][str(index)]['expected_gate_state']
                    and trained['steps'] == trained['schedule_steps'] == 500, 'A layer reused a fitted state or changed its budget.')
            endpoint = measure(layer, archives[index], caches[index], target_name(index), device)
            state = {name: value.detach().cpu() for name, value in layer.state_dict().items()}
            destination = output / f'f927_layer{index}_gate_decoder500.pt'
            torch.save(state, destination)
            saved = torch.load(destination, map_location='cpu', weights_only=True, mmap=True)
            rebuilt = fresh(initial[index], device, payload=saved)
            require(records(rebuilt) == trained['final_payload'] and len(saved) == 13, 'Saved endpoint failed fresh strict state reconstruction.')
            with torch.no_grad():
                require(torch.equal(rebuilt._decode_weight(torch.bfloat16), layer._decode_weight(torch.bfloat16)),
                        'Fresh native BF16 full gate decode differs from the trained deployment.')
            report['layers'][str(index)] = {'status': 'trained', 'training_updates_this_run': 500,
                'training': trained, 'before': baseline, 'endpoint': endpoint, 'versus_original': paired(baseline, endpoint),
                'state_file': str(destination), 'state_sha256': sha256(destination),
                'fresh_strict_state_and_native_decode_exact': True}
            write_json(report_path, report)
            del layer, rebuilt, saved, state; gc.collect(); torch.cuda.empty_cache()
        require(all(records(initial[i]) == data['per_layer'][str(i)]['expected_gate_state']
                    and cache_records(caches[i]) == frozen_cache[i] for i in LAYERS), 'An original per-layer state or fixed context was mutated.')
        check(p)
        report.update(status='completed', finished_unix=time.time(), source_and_context_unchanged=True,
            new_training_updates=1500, reused_layer_training_updates=0,
            longbench_status='not_evaluated', full8_status='not_evaluated', assembled_model=False)
        write_json(report_path, report)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', finished_unix=time.time())
        write_json(report_path, report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--preparation', type=Path)
    parser.add_argument('--base-root', type=Path); parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    prepare(args) if args.prepare else run(args.preparation)
