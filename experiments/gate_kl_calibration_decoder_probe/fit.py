"""Fit two fixed calibration128 code proposals with the original L18 decoder500 recipe."""
from __future__ import annotations

import argparse
import ast
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time

import torch

from experiments.gate_kl_calibration_evaluation.prepare import EXPECTED_XOR, replay_endpoint
from experiments.gate_local_delta_probe.run import BASE, NAME, ROOT, SOURCE_ID, assets, load_assets, read
from experiments.gate_student_context_probe.core import BANKS, measure, ready, records, train
from experiments.gate_student_context_probe.run import paired
from experiments.joint_reconstruction_probe.core import require
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.utils import configure_deterministic_mode, set_seed

ARMS = ('alpaca128', 'alpaca64_fineweb64')


def check(record):
    check_hashes(record['source_sha256']); check_hashes(record['input_sha256'])
    for item in record['protected_stats']: check_stat(item)


def chosen_state(layer, choice_path, arm):
    choices = torch.load(choice_path, map_location='cpu', weights_only=True)
    source = {NAME + '.' + key: value for key, value in layer.state_dict().items()}
    return replay_endpoint(source, choices, arm)


def prepare(output):
    require(not output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized(), 'Use a fresh CPU-only preparation directory.')
    torch.set_num_threads(4)
    paths = assets()
    prior, prior_review = read(paths['prior_report']), read(paths['prior_review'])
    old_prepared_path = BASE / 'gate_student_context_probe/preparation.json'
    old_prepared = read(old_prepared_path)
    require(prior['status'] == 'completed' and prior_review['status'] == 'verified'
            and prior_review['report_sha256'] == sha256(paths['prior_report'])
            and prior_review['preparation_sha256'] == prior['preparation_sha256'] == sha256(old_prepared_path)
            and Path(paths['prior_exit']).read_text().strip() == '0', 'The original decoder500 control must be verified.')
    control = prior['arms']['original_codes']
    require(control['training']['status'] == 'completed' and control['training']['steps'] == 500
            and control['training']['schedule_steps'] == 500 and control['training']['frozen_state_exact'],
            'Use the completed original-code500 control, without retraining it.')
    layer, archive, cache, _, manifest = load_assets(paths)
    initial = records(layer)
    require(initial == control['training']['initial_payload'] == manifest['expected_gate_state'],
            'The six initial decoder values must be fresh f927, not a fitted control endpoint.')
    for split in ('train', 'heldout', 'replay32'):
        require(tensor_record(cache['inputs'][split]) == manifest['tensors']['inputs'][split]
                and cache['document_lengths'][split] == manifest['document_lengths'][split],
                'A retained context tensor or document boundary changed.')
    archive_manifest = read(paths['archive_manifest'])
    require(sha256(paths['cache']) == manifest['archive']['sha256']
            and sha256(paths['archive']) == archive_manifest['inputs_sha256'], 'The retained input archives changed.')

    proto = BASE / 'gate_kl_calibration_probe'
    report, review, terminal = (read(proto / name) for name in ('run/report.json', 'review.json', 'launch.json'))
    require(report['status'] == terminal['status'] == 'completed' and terminal['exit_code'] == 0
            and all((proto / name).read_text().strip() == '0' for name in ('run.exit', 'outer.exit'))
            and review['status'] == 'verified'
            and review['report_sha256'] == terminal['report_sha256'] == sha256(proto / 'run/report.json')
            and review['prepared_sha256'] == sha256(proto / 'launch.prepared.json')
            and review['terminal_sha256'] == sha256(proto / 'launch.json')
            and report['source_checkpoint_id'] == review['source_checkpoint_id'] == SOURCE_ID,
            'Require the naturally completed, independently reviewed calibration proposals.')
    experiment = report['experiment']
    require(experiment['initial_state'] == initial and experiment['all13_final_bytes_exact'],
            'Calibration started with different decoder values or failed to restore the source.')
    arm_records = {}
    for arm in ARMS:
        saved, proof = report['saved_choices'][arm], experiment['endpoint_records'][arm]
        path = proto / 'run' / (arm + '_choices.pt')
        require(str(path) == saved['path'] and sha256(path) == saved['sha256'] and saved['roundtrip_exact']
                and proof['stage_xor'] == EXPECTED_XOR[arm] and proof['complete_bank_replay_exact']
                and proof['local11_noncode_bytes_exact'], 'The reviewed compact proposal differs.')
        endpoint = chosen_state(layer, path, arm)
        expected = {key: tensor_record(value) for key, value in endpoint.items()}
        require(expected == proof['endpoint_state'] == experiment['generations'][arm]['applied_state']['endpoint_state']
                and all(expected[key] == value for key, value in initial.items() if key not in BANKS),
                'Replay must change only the two banks, retaining all original decoder/rotation/protection values.')
        arm_records[arm] = {'choices_path': str(path), 'choices_sha256': saved['sha256'],
            'initial_state': expected, 'stage_xor': EXPECTED_XOR[arm]}

    sources = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.suffix == '.py' and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / 'result'):
                sources[str(path)] = sha256(path)
    ast.parse(Path(__file__).read_text()); sources[str(Path(__file__).resolve())] = sha256(__file__)
    for path, digest in sources.items():
        if path in old_prepared['source_sha256']:
            require(digest == old_prepared['source_sha256'][path], 'An applicable original decoder500 source changed: ' + path)
    evidence = {str(path): sha256(path) for key, path in paths.items() if key not in {'cache', 'archive'}}
    for path in (old_prepared_path, proto / 'run/report.json', proto / 'review.json', proto / 'launch.json',
                 proto / 'launch.prepared.json', proto / 'run.exit', proto / 'outer.exit',
                 *(Path(arm_records[arm]['choices_path']) for arm in ARMS)):
        evidence[str(path)] = sha256(path)
    record = {'status': 'cpu_ready', 'output_root': str(output), 'python': sys.executable, 'torch': torch.__version__,
        'source_sha256': sources, 'input_sha256': evidence,
        'protected_stats': [manifest['source_weight_stat'], file_stat(paths['cache']), file_stat(paths['archive'])],
        'paths': {key: str(value) for key, value in paths.items()}, 'source_checkpoint_id': SOURCE_ID,
        'original_state': initial, 'arms': arm_records, 'expected_training_settings': control['training']['settings'],
        'expected_persistent_bits': control['training']['persistent_bits'], 'original_codes500_endpoint': control['endpoint'],
        'cuda_initialized': False, 'queued': False, 'automatic_retry': False,
        'environment': {'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'},
        'protocol': {'arms': list(ARMS), 'steps_per_arm': 500, 'seed': 31,
            'initialization': 'Fresh original f927 decoder for each arm; only replay the existing fixed four-group choices. No fitted control weights, fresh search, recapture, or mechanism updates.',
            'updates': 'Unchanged core.train: six FP32 decoder masters and Adam, native BF16 decode/stage sum/MLP; both packed banks fixed throughout.',
            'target': 'Original train32/4289 student-prefix tokens and conditional HF_MLP(xs), fixed f927 up/down, LR1e-5/betas(.9,.95)/eps1e-8/wd0/linear500/warm0.',
            'observations': 'Only fixed endpoints use original heldout32/replay32. These are previously seen development documents, not new blind tests; no parameter/arm selection by these metrics.',
            'comparison': 'Reuse original-code500 local/full8 evidence. Decoder update budgets match, but the new proposals have extra128-document final-KL-gradient computation; fitting uses conditional MLP MSE, not final KL.',
            'save': 'One BF16 native thirteen-tensor gate state per arm; no optimizer, full model, downstream evaluation, or framework integration.',
            'wall_budget_seconds': 900, 'kill_grace_seconds': 20,
            'estimate': 'Observed same-recipe500 takes44.35–45.54s and about4.84GB peak allocated; two arms estimated2–3minutes, subject to actual measurement.'},
        'command': [sys.executable, '-m', 'experiments.gate_kl_calibration_decoder_probe.fit',
                    '--preparation', str(output / 'preparation.json')]}
    check(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'preparation.json', record)
    print(json.dumps({'status': record['status'], 'preparation_sha256': sha256(output / 'preparation.json'),
        'sources': len(sources), 'inputs': len(evidence), 'cuda_initialized': False}), flush=True)


def run(path):
    p = read(path); require(p['status'] == 'cpu_ready', 'Use immutable CPU preparation.'); check(p)
    require(sys.executable == p['python'] and torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Use the prepared bitvae interpreter and exactly one externally assigned GPU.')
    for key, value in p['environment'].items(): require(os.environ.get(key) == value, 'Explicit launch environment differs: ' + key)
    output = Path(p['output_root']) / 'run'; require(not output.exists(), 'No overwrite or automatic retry.'); output.mkdir()
    started = time.time()
    report = {'status': 'running', 'pid': os.getpid(), 'started_unix': started, 'argv': sys.argv,
        'process_start_ticks': int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'preparation_sha256': sha256(path),
        'source_checkpoint_id': SOURCE_ID, 'protocol': p['protocol'], 'arms': {}}
    write_json(output / 'report.json', report)
    try:
        torch.set_num_threads(4); torch.cuda.set_device(0); configure_deterministic_mode(True)
        for arm in ARMS:
            set_seed(31)
            layer, archive, cache, _, _ = load_assets(p['paths'])
            require(records(layer) == p['original_state'], 'An arm did not start from fresh f927.')
            initial = chosen_state(layer, p['arms'][arm]['choices_path'], arm)
            layer.load_state_dict(initial, strict=True); layer.to('cuda:0'); ready(layer)
            require(records(layer) == p['arms'][arm]['initial_state'], 'The prepared proposal/decoder initial state changed.')
            training = train(layer, archive, cache, steps=500, deadline=started + p['protocol']['wall_budget_seconds'],
                observer=lambda row: logging.info('%s decoder500 %s', arm, json.dumps(row)))
            require(training['initial_payload'] == p['arms'][arm]['initial_state']
                    and training['settings'] == p['expected_training_settings']
                    and training['persistent_bits'] == p['expected_persistent_bits']
                    and training['frozen_state_exact'] and training['frozen_objective_buffers_exact'],
                    'The matched recipe, fixed codes or source context changed.')
            endpoint = measure(layer, archive, cache, torch.device('cuda:0'))
            state_path = output / (arm + '_gate_state.pt')
            torch.save({key: value.detach().cpu() for key, value in layer.state_dict().items()}, state_path)
            saved = torch.load(state_path, map_location='cpu', weights_only=True, mmap=True)
            require({key: tensor_record(value) for key, value in saved.items()} == training['final_payload']
                    and all(torch.equal(saved[key], initial[key]) for key in BANKS),
                    'Saved native endpoint or fixed packed banks differ.')
            report['arms'][arm] = {'training': training, 'endpoint': endpoint,
                'versus_original_codes500': paired(p['original_codes500_endpoint'], endpoint),
                'stage_xor_vs_f927': p['arms'][arm]['stage_xor'],
                'saved': {'state_path': str(state_path), 'state_sha256': sha256(state_path),
                    'saved_state_exact': True, 'fixed_banks_exact': True,
                    'native_save_reload_evidence': 'Completed same-geometry group-accepted decoder500 and original-code500 strict native paths; no additional GPU reload.'}}
            write_json(output / 'report.json', report)
            del saved, initial, layer, archive, cache
            gc.collect(); torch.cuda.empty_cache()
        check(p)
        report.update(status='completed', finished_unix=time.time(), elapsed_seconds=time.time() - started)
        write_json(output / 'report.json', report)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', finished_unix=time.time())
        write_json(output / 'report.json', report); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--preparation', type=Path)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    if args.prepare:
        require(args.output_root is not None, '--prepare requires --output-root.'); prepare(args.output_root.resolve())
    else: run(args.preparation.resolve())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s'); main()
