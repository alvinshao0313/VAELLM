"""Compare fixed-code down-only and gate+down fitting in the existing student context."""
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

from experiments.gate_down_student_context_probe.core import measure, ready, records, train
from experiments.gate_student_context_probe.data import SOURCE, SOURCE_ID, TARGET
from experiments.gate_student_context_probe.run import cache_records, merge, paired, read
from experiments.joint_reconstruction_probe.core import require
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _assert_state_equal
from experiments.rotation_factor_probe.capture import tensor_record
from experiments.rotation_factor_probe.run import load_archive
from train_utils.utils import configure_deterministic_mode, set_seed

ARMS = ('down_only', 'gate_down')
PROJECTIONS = ('gate_proj', 'down_proj')


def flat_source(state):
    result = {}
    for kind in PROJECTIONS:
        prefix = TARGET.replace('gate_proj', kind) + '.'
        local = {kind + '.' + k[len(prefix):]: tensor_record(v) for k, v in state.items() if k.startswith(prefix)}
        require(len(local) == 13, 'Each native projection must retain its complete thirteen-tensor state.')
        result.update(local)
    return result


def inputs(record):
    check_hashes(record['source_sha256']); check_hashes(record['input_sha256'])
    check_stat(record['source_weight_stat'])
    for item in record['hf_weight_files']:
        check_stat(item)


def prepare(args):
    require(args.base_root is not None and args.output_root is not None, 'Supply source evidence and a new output root.')
    base, output = args.base_root.resolve(), args.output_root.resolve()
    require(not output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Use fresh CPU-only preparation.')
    torch.set_num_threads(4)
    previous_dir = base / 'gate_student_context_probe'
    previous, report, review = (read(previous_dir / name) for name in ('preparation.json', 'run/report.json', 'review.json'))
    require(report['status'] == 'completed' and (previous_dir / 'run.exit').read_text().strip() == '0'
            and sha256(previous_dir / 'run/report.json') == '889c0c6e40ad9ae7f678a78a0962f6f6659aea7d5e1582021474e62212e39158'
            and review['status'] == 'verified' and review['report_sha256'] == sha256(previous_dir / 'run/report.json'),
            'Require the completed, reviewed original student-context comparison.')
    check_hashes(previous['source_sha256']); check_hashes(previous['input_sha256'])
    data = previous['data']
    manifest_path = previous_dir / 'run/student_capture/data_manifest.json'
    manifest = read(manifest_path)
    cache_path = previous_dir / 'run/student_capture/inputs.pt'
    require(manifest['status'] == 'captured' and manifest['checkpoint_id'] == SOURCE_ID
            and sha256(cache_path) == manifest['archive']['sha256']
            and sha256(manifest_path) == report['capture']['manifest_sha256'], 'Frozen real student activation cache differs.')
    cache = torch.load(cache_path, map_location='cpu', weights_only=True, mmap=True)
    require(cache_records(cache) == manifest['tensors'] == report['capture']['tensors']
            and cache['document_lengths'] == data['document_lengths']
            and cache['checkpoint_id'] == SOURCE_ID and cache['target'] == TARGET, 'Fixed 96 inputs or up/down context differs.')
    archive, _archive_manifest = load_archive(Path(data['archive_dir']))
    require(archive['checkpoint_id'] == SOURCE_ID and archive['layer_index'] == 18
            and all(TARGET.replace('gate_proj', kind) in archive['targets'] for kind in PROJECTIONS), 'Use the same original HF L18 targets.')
    meta = read(SOURCE / 'checkpoint_meta.json')
    require(meta['checkpoint_id'] == SOURCE_ID and data['source_checkpoint'] == str(SOURCE), 'Use fresh f927, not a previous local endpoint.')
    source = torch.load(SOURCE / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
    require(len(source) == 3423, 'Original persistent model inventory changed.')
    expected = flat_source(source)
    require({k.removeprefix('gate_proj.'): v for k, v in expected.items() if k.startswith('gate_proj.')}
            == data['expected_gate_state'], 'Initial gate differs from the old original-code control.')
    specs = {row['name']: row for row in meta['converted_modules']}
    require(specs[TARGET] == data['target_spec'] and specs[TARGET.replace('gate_proj', 'down_proj')] == data['down_spec'],
            'Original gate/down geometry, rotation or protection changed.')
    control = report['arms']['original_codes']
    control_path = Path(control['state_file'])
    require(sha256(control_path) == control['state_sha256'], 'The existing gate-only500 endpoint changed.')
    control_state = torch.load(control_path, map_location='cpu', weights_only=True, mmap=True)
    require({k: tensor_record(v) for k, v in control_state.items()} == control['training']['final_payload'],
            'The old gate-only endpoint no longer matches its actual training evidence.')
    full8_review_path = base / 'gate_student_context_full8_r1/review.json'
    full8_review = read(full8_review_path)
    require(full8_review['status'] == 'verified', 'The gate-only full8 interpretation must have a reviewed terminal.')
    sources, pins = dict(previous['source_sha256']), dict(previous['input_sha256'])
    for path in Path(__file__).resolve().parent.glob('*.py'):
        ast.parse(path.read_text()); merge(sources, {str(path): sha256(path)})
    for path in (previous_dir / 'preparation.json', previous_dir / 'run/report.json', previous_dir / 'review.json',
                 previous_dir / 'run.exit', manifest_path, cache_path, control_path, full8_review_path):
        merge(pins, {str(path): sha256(path)})
    record = {'status': 'cpu_ready', 'base_root': str(base), 'output_root': str(output), 'python': sys.executable,
        'source_checkpoint': str(SOURCE), 'source_checkpoint_id': SOURCE_ID, 'source_weight_stat': data['source_weight_stat'],
        'source_sha256': sources, 'input_sha256': pins, 'hf_weight_files': data['hf_weight_files'],
        'expected_initial_payload': expected, 'source_serialized_tensors': len(source), 'data': data,
        'cache_path': str(cache_path), 'cache_tensors': cache_records(cache), 'manifest_path': str(manifest_path),
        'archive_dir': data['archive_dir'], 'original_report': str(previous_dir / 'run/report.json'),
        'gate_only_control_path': str(control_path), 'gate_only_control_endpoint': control['endpoint'],
        'full8_evidence': str(full8_review_path), 'cuda_initialized': torch.cuda.is_initialized(),
        'protocol': {'source': 'fresh f927 L18 gate/down; fixed original four code banks',
            'arms': list(ARMS), 'control': 'Existing original-code gate-only500 endpoint; no retraining',
            'mechanism': 'Fresh joint gate+down two updates, then discarded', 'mechanism_updates': 2,
            'formal_updates_each': 500, 'schedule_horizon': 500, 'decoder_lr': 1e-5, 'decoder_betas': [.9, .95],
            'eps': 1e-8, 'weight_decay': 0, 'warmup': 0, 'wall_budget_seconds': 1200, 'kill_grace_seconds': 20,
            'precision': 'Native full-coordinate BF16 gate/down, FP32 decoder/Adam masters; no second inverse rotation',
            'training': 'All original train32 student-input tokens each update; HF_MLP(x_student) fixed conditional target; up fixed',
            'heldout': 'Old held32 and already-viewed replay32, endpoint-only; neither enters training or parameter choice',
            'search': 'None; all packed code banks, protection, rotation and nonselected decoder stay frozen',
            'boundary': 'Isolated local output fitting, no full-model export, downstream evaluation or production integration'},
        'command': [sys.executable, '-m', 'experiments.gate_down_student_context_probe.run', '--preparation', str(output / 'preparation.json')]}
    require(not record['cuda_initialized'], 'CPU preparation must not initialize CUDA.')
    inputs(record)
    output.mkdir(parents=True)
    write_json(output / 'preparation.json', record)
    print(json.dumps({'status': 'cpu_ready', 'preparation_sha256': sha256(output / 'preparation.json'),
        'source_pins': len(sources), 'input_pins': len(pins), 'tokens': {s: sum(v) for s, v in cache['document_lengths'].items()},
        'source_state_tensors': len(source), 'local_tensors': len(expected), 'cuda_initialized': False}), flush=True)


def fresh(initial, device, *, payload=None, mode=None):
    layers = deepcopy(initial)
    if payload is not None:
        require(mode in ('gate_only_control', *ARMS), 'Unknown saved local payload type.')
        selected = PROJECTIONS if mode == 'gate_down' else (('gate_proj',) if mode == 'gate_only_control' else ('down_proj',))
        for kind in selected:
            state = {key[len(kind) + 1:]: value for key, value in payload.items() if key.startswith(kind + '.')} if mode == 'gate_down' else payload
            loaded = layers[kind].load_state_dict(state, strict=True)
            require(not loaded.missing_keys and not loaded.unexpected_keys, 'Incomplete saved native local state.')
    for layer in layers.values():
        layer.to(device).eval().requires_grad_(False)
    ready(layers)
    return layers


def run(path):
    from train_utils.v6_model_loader import load_v6_model_checkpoint
    p = read(path)
    require(p['status'] == 'cpu_ready' and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Use the CPU preparation and one explicitly assigned GPU.')
    inputs(p)
    output = Path(p['output_root']) / 'run'
    require(not output.exists(), 'No overwrite, queue, resume or automatic retry.')
    output.mkdir()
    started = time.time(); deadline = started + p['protocol']['wall_budget_seconds']
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    configure_deterministic_mode(True); set_seed(31); torch.set_num_threads(4)
    report = {'status': 'running', 'pid': os.getpid(), 'argv': sys.argv,
        'process_start_ticks': int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'started_unix': started, 'deadline_unix': deadline, 'preparation_sha256': sha256(path),
        'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'protocol': p['protocol'], 'arms': {}}
    report_path = output / 'report.json'; write_json(report_path, report)
    try:
        old = read(p['original_report'])
        archive, _manifest = load_archive(Path(p['archive_dir']))
        cache = torch.load(p['cache_path'], map_location='cpu', weights_only=True, mmap=True)
        frozen_cache = cache_records(cache)
        require(frozen_cache == p['cache_tensors'], 'The original student activation cache changed.')
        model, meta, loaded = load_v6_model_checkpoint(p['source_checkpoint'], map_location='cpu', strict=True, expected_kind='final_model')
        require(meta['checkpoint_id'] == SOURCE_ID and not loaded.missing_keys and not loaded.unexpected_keys, 'Fresh f927 strict load failed.')
        model.eval().requires_grad_(False)
        source = torch.load(SOURCE / meta['state_dict_file'], map_location='cpu', weights_only=True, mmap=True)
        count = _assert_state_equal(model.state_dict(), source, 'Fresh complete f927 versus persistent source')
        require(count == p['source_serialized_tensors'] == 3423, 'Incomplete original full-model state check.')
        initial = {kind: deepcopy(model.get_submodule(TARGET.replace('gate_proj', kind))).cpu().eval().requires_grad_(False)
                   for kind in PROJECTIONS}
        ready(initial)
        require(records(initial) == p['expected_initial_payload'], 'Detached initial native gate/down state differs.')
        del model, source
        gc.collect(); torch.cuda.empty_cache()
        report['source_strict_tensors_compared'] = count
        layers = fresh(initial, device)
        with torch.no_grad():
            require(torch.equal(layers['down_proj']._decode_weight(torch.bfloat16).cpu(), cache['student_down_weight']),
                    'Actual native full down weight differs from the original frozen student cache.')
        report['before_original_f927'] = measure(layers, archive, cache, device)
        require({k: v for k, v in report['before_original_f927'].items() if k != 'down_original_coordinate_weight'}
                == old['before_original_codes'], 'Initial per-document f927 gate/MLP metrics do not replay the original arithmetic.')
        del layers; gc.collect(); torch.cuda.empty_cache()
        payload = torch.load(p['gate_only_control_path'], map_location='cpu', weights_only=True, mmap=True)
        layers = fresh(initial, device, payload=payload, mode='gate_only_control')
        report['gate_only_control_endpoint'] = measure(layers, archive, cache, device)
        require({k: v for k, v in report['gate_only_control_endpoint'].items() if k != 'down_original_coordinate_weight'}
                == p['gate_only_control_endpoint'], 'The saved gate-only500 control no longer replays its endpoint metrics.')
        del layers, payload; gc.collect(); torch.cuda.empty_cache()
        layers = fresh(initial, device)
        report['mechanism'] = train(layers, archive, cache, mode='gate_down', steps=2, deadline=deadline,
                                   observer=lambda row: logging.info('mechanism %s', row))
        require(report['mechanism']['initial_payload'] == p['expected_initial_payload'], 'Mechanism did not start from untouched source.')
        estimate = 1000 * max(row['seconds'] for row in report['mechanism']['trajectory']) + 60
        report['resource_projection'] = {'two_arm_seconds_from_slower_joint_step_plus60': estimate, 'remaining_seconds': deadline - time.time()}
        write_json(report_path, report)
        require(estimate < deadline - time.time(), 'Actual joint resource/speed cannot fit the approved budget; stop without extension.')
        del layers; gc.collect(); torch.cuda.empty_cache()
        for arm in ARMS:
            layers = fresh(initial, device)
            trained = train(layers, archive, cache, mode=arm, steps=500, deadline=deadline,
                            observer=lambda row: logging.info('%s %s', arm, row))
            require(trained['initial_payload'] == p['expected_initial_payload'], 'A formal arm consumed mechanism updates or a previous endpoint.')
            endpoint = measure(layers, archive, cache, device)
            payload = ({kind + '.' + key: value.detach().cpu() for kind, layer in layers.items() for key, value in layer.state_dict().items()}
                       if arm == 'gate_down' else {key: value.detach().cpu() for key, value in layers['down_proj'].state_dict().items()})
            destination = output / f'f927_l18_{arm}_decoder500.pt'
            torch.save(payload, destination)
            saved = torch.load(destination, map_location='cpu', weights_only=True, mmap=True)
            rebuilt = fresh(initial, device, payload=saved, mode=arm)
            require(records(rebuilt) == trained['final_payload'], 'Saved native states failed fresh strict reconstruction.')
            with torch.no_grad():
                require(all(torch.equal(rebuilt[k]._decode_weight(torch.bfloat16), layers[k]._decode_weight(torch.bfloat16))
                            for k in PROJECTIONS), 'Strict reconstruction changed a native BF16 full projection.')
            report['arms'][arm] = {'training': trained, 'endpoint': endpoint,
                'versus_original_f927': paired(report['before_original_f927'], endpoint),
                'versus_existing_gate_only500': paired(report['gate_only_control_endpoint'], endpoint),
                'state_file': str(destination), 'state_sha256': sha256(destination),
                'saved_tensor_count': len(saved), 'fresh_strict_state_and_native_decode_exact': True}
            write_json(report_path, report)
            del layers, rebuilt, saved, payload; gc.collect(); torch.cuda.empty_cache()
        require(records(initial) == p['expected_initial_payload'] and cache_records(cache) == frozen_cache,
                'Initial gate/down or fixed student context was mutated.')
        require(report['arms']['down_only']['training']['settings'] == report['arms']['gate_down']['training']['settings'],
                'Paired optimizer, target or population settings differ.')
        report['between_new_arms'] = paired(report['arms']['down_only']['endpoint'], report['arms']['gate_down']['endpoint'])
        inputs(p)
        report.update(status='completed', finished_unix=time.time(), source_and_context_unchanged=True)
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
