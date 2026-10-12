"""Prepare/run a fixed sixteen-neighbor numerical diagnostic, with no training."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn

from experiments.gate_local_delta_probe.core import measure, select_candidates
from experiments.gate_student_context_probe.core import ready, records
from experiments.joint_reconstruction_probe.core import require
from experiments.longbench_v1.common import check_hashes, check_stat, file_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import tensor_record
from train_utils.checkpoint_v6 import _rebuild_converted_modules, refresh_vae_linear_runtime_after_state_load
from train_utils.utils import configure_deterministic_mode, set_seed

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'result/qwen3_8b_mmlu65_20261004'
SOURCE = ROOT / 'result/qwen3_8b_w25_20260930/e2e_best_combo_down0_joint_s2000/step_1200/final_model'
SOURCE_ID = 'f927ffda-dd5b-4d00-b63a-5e839a182d70'
NAME = 'model.layers.18.mlp.gate_proj'


def read(path):
    return json.loads(Path(path).read_text())


def assets():
    prior = BASE / 'gate_student_context_probe'
    capture = prior / 'run/student_capture'
    archive = BASE / 'rotation_factor_probe/middle_data/qwen'
    return {'prior_report': prior/'run/report.json', 'prior_review': prior/'review.json',
        'prior_exit': prior/'run.exit', 'capture_manifest': capture/'data_manifest.json',
        'cache': capture/'inputs.pt', 'archive_manifest': archive/'data_manifest.json',
        'archive': archive/'inputs.pt', 'choices': prior/'run/f927_l18_exact64_choices.pt',
        'source_meta': SOURCE/'checkpoint_meta.json',
        'cpu_formula_proof': prior/'local_support_cpu.json'}


def load_assets(paths):
    manifest, archive_manifest = read(paths['capture_manifest']), read(paths['archive_manifest'])
    meta = read(paths['source_meta'])
    require(meta['checkpoint_id'] == manifest['checkpoint_id'] == archive_manifest['checkpoint_id'] == SOURCE_ID,
            'Use the original f927 source, real student contexts and matched HF archive.')
    spec = next(s for s in meta['converted_modules'] if s['name'] == NAME)
    require(spec == manifest['target_spec'] and spec['weight_rotation']['block_size'] == 32,
            'Native target geometry differs from the captured projection.')
    state_path = SOURCE / meta['state_dict_file']
    check_stat(manifest['source_weight_stat'])
    require(state_path.resolve() == Path(manifest['source_weight_stat']['path']).resolve(), 'Wrong source state file.')
    whole = torch.load(state_path, map_location='cpu', weights_only=True, mmap=True)
    state = {k[len(NAME)+1:]: v for k, v in whole.items() if k.startswith(NAME+'.')}
    require({k: tensor_record(v) for k, v in state.items()} == manifest['expected_gate_state'] and len(state) == 13,
            'Read only the original thirteen gate tensors from the source mmap.')
    # A real one-projection holder is required by the native v6 reconstruction API;
    # it is never passed off as a complete causal model or used for a full forward.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        holder = nn.Module()
        holder.projection = nn.Linear(4096, 12288, bias=False, dtype=torch.bfloat16)
        local_spec = deepcopy(spec); local_spec['name'] = 'projection'
        _rebuild_converted_modules(holder, [local_spec])
        layer = holder.projection
    layer.load_state_dict(state, strict=True)
    refresh_vae_linear_runtime_after_state_load(holder)
    layer.eval().requires_grad_(False); ready(layer)
    require(records(layer) == manifest['expected_gate_state'] and layer._parallel_stage_restore_identity,
            'Native single-projection strict reconstruction changed source values/layout.')
    del whole, state, holder
    cache = torch.load(paths['cache'], map_location='cpu', weights_only=True, mmap=True)
    require(cache['checkpoint_id'] == SOURCE_ID and cache['target'] == NAME
            and tensor_record(cache['inputs']['train']) == manifest['tensors']['inputs']['train']
            and cache['document_lengths']['train'] == manifest['document_lengths']['train']
            and len(cache['document_lengths']['train']) == 32 and sum(cache['document_lengths']['train']) == 4289,
            'Retained training population differs.')
    for key in ('student_up_weight', 'student_down_weight'):
        require(tensor_record(cache[key]) == manifest['tensors'][key], 'Frozen student projection changed.')
    archive = torch.load(paths['archive'], map_location='cpu', weights_only=True, mmap=True)
    require(archive['model_type'] == 'qwen3' and archive['layer_index'] == 18
            and archive['checkpoint_id'] == SOURCE_ID, 'Wrong original-HF archive identity.')
    for name in (NAME, NAME.replace('gate_proj', 'down_proj')):
        require(tensor_record(archive['targets'][name]['reference_weight'])
                == archive_manifest['tensors'][name]['reference_weight'], 'Original-HF reference changed.')
    require(tensor_record(archive['up_proj_weight']) == archive_manifest['up_proj_weight'], 'Original-HF up changed.')
    choices = torch.load(paths['choices'], map_location='cpu', weights_only=True, mmap=True)
    candidates = select_candidates(choices)
    return layer, archive, cache, candidates, manifest


def prepare(output):
    require(not output.exists(), 'Use a fresh diagnostic output directory.')
    require(not torch.cuda.is_initialized(), 'CPU preparation must start without CUDA.')
    torch.set_num_threads(4)
    paths = assets()
    prior, review = read(paths['prior_report']), read(paths['prior_review'])
    require(prior['status'] == 'completed' and review['status'] == 'verified'
            and review['report_sha256'] == sha256(paths['prior_report'])
            and paths['prior_exit'].read_text().strip() == '0', 'The original search must have a verified natural terminal.')
    require(sha256(paths['choices']) == prior['choices_sha256'], 'The historical weight-search winners changed.')
    capture, archive = read(paths['capture_manifest']), read(paths['archive_manifest'])
    require(sha256(paths['cache']) == capture['archive']['sha256']
            and sha256(paths['archive']) == archive['inputs_sha256'], 'Retained input archives changed.')
    layer, _, cache, candidates, manifest = load_assets(paths)
    sources = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.suffix == '.py' and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'result'):
                sources[str(path)] = sha256(path)
    for path in Path(__file__).parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path.resolve())] = sha256(path)
    for path, digest in sources.items():
        if path in manifest['source_sha256']:
            require(digest == manifest['source_sha256'][path], 'An applicable original capture source changed.')
    evidence = {str(p): sha256(p) for key, p in paths.items() if key not in {'cache', 'archive'}}
    require(not torch.cuda.is_initialized(), 'Native CPU reconstruction initialized CUDA.')
    record = {'status': 'cpu_ready', 'output_root': str(output), 'python': sys.executable,
        'torch': torch.__version__, 'source_sha256': sources, 'input_sha256': evidence,
        'paths': {k: str(v) for k, v in paths.items()},
        'protected_stats': [manifest['source_weight_stat'], file_stat(paths['cache']), file_stat(paths['archive'])],
        'archive_sha256_verified': {'cache': capture['archive']['sha256'], 'archive': archive['inputs_sha256']},
        'initial_state': records(layer), 'candidates': candidates, 'train_input': tensor_record(cache['inputs']['train']),
        'cuda_initialized': False,
        'protocol': {'source_checkpoint_id': SOURCE_ID, 'target': NAME, 'candidate_count': 16,
            'selection': 'For each stage, take the eight midpoint rank quantiles among nonzero winners of the already completed original-f927 weight-only search; no output or heldout score enters selection.',
            'mutation': 'One genuine packed bit from the original state per candidate; restore immediately in finally; no optimizer, acceptance, accumulated flips or model saving.',
            'population': 'Only all4289 tokens from existing train32 student contexts. Heldout/replay are not scored or used for selection.',
            'target': 'Existing Objective: fixed original HF_MLP(x_student); candidate uses fixed decoded student up/down.',
            'precision': 'Unchanged native BF16 decode/sum/inverse/protection and BF16 MLP; local Gram arithmetic in FP64 plus a recorded FP32 version.',
            'local_bf16_recompute': 'Recompute only the affected32 gate rows with all4096 inputs; insert their BF16 activation into original full A and redo full-shaped native BF16 down GEMM.',
            'formula': 'Use native baseline E=Y_BF16-T_BF16 with local deltaA. This is a proxy because final BF16 down rounding is omitted. Also report full-native deltaA formula to isolate that error.',
            'judgment': 'Structural/frozen/restoration/finite checks must pass. Record every sign/order disagreement, false improvement and best-neighbor difference. No loose tolerance converts ranking failure to success; 16 measured neighbors cannot establish general quality benefit.',
            'wall_budget_seconds': 300, 'kill_grace_seconds': 20,
            'estimate': 'One A800; expected tens of seconds to a few minutes, reserve10GiB. No complete model loading/forward; actual time and peak measured.',
            'experience_boundary': 'Weight/SiLU improvement can reverse at multiplication by up; BF16 can independently reverse algebraic delta signs. Neither establishes the actual cause or a beneficial integrated algorithm.'},
        'command': [sys.executable, '-m', 'experiments.gate_local_delta_probe.run', '--preparation', str(output/'preparation.json')]}
    output.mkdir(parents=True)
    write_json(output/'preparation.json', record)
    print(json.dumps({'status':'cpu_ready', 'preparation_sha256':sha256(output/'preparation.json'),
        'sources':len(sources), 'inputs':len(evidence), 'candidate_vectors':[c['vector'] for c in candidates], 'cuda_initialized':False}))


def run(path):
    p = read(path)
    require(p['status'] == 'cpu_ready', 'Use the immutable CPU preparation.')
    check_hashes(p['source_sha256']); check_hashes(p['input_sha256'])
    for stat in p['protected_stats']: check_stat(stat)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one externally assigned GPU is required.')
    output = Path(p['output_root'])/'run'
    require(not output.exists(), 'No overwrite or automatic retry.')
    output.mkdir()
    started = time.time()
    report = {'status':'running', 'started_unix':started, 'pid':os.getpid(),
        'process_start_ticks':int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]),
        'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'), 'preparation_sha256':sha256(path), 'protocol':p['protocol']}
    write_json(output/'report.json',report)
    try:
        torch.set_num_threads(4); torch.cuda.set_device(0)
        configure_deterministic_mode(True); set_seed(31)
        layer, archive, cache, candidates, _ = load_assets(p['paths'])
        require(candidates == p['candidates'] and records(layer) == p['initial_state'], 'Prepared source/candidates changed.')
        layer.to('cuda:0'); ready(layer)
        with torch.autocast('cuda', enabled=False):
            result = measure(layer, archive, cache, candidates, deadline=started+p['protocol']['wall_budget_seconds'],
                observer=lambda row: logging.info('candidate %s', json.dumps(row)))
        check_hashes(p['source_sha256']); check_hashes(p['input_sha256'])
        for stat in p['protected_stats']: check_stat(stat)
        report.update(result, finished_unix=time.time(), elapsed_seconds=time.time()-started,
            outcome='Measurement completed; use recorded ranking disagreements to judge the proxy, not status alone.')
        write_json(output/'report.json',report)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',finished_unix=time.time())
        write_json(output/'report.json',report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true'); mode.add_argument('--preparation',type=Path)
    parser.add_argument('--output-root',type=Path)
    args = parser.parse_args()
    if args.prepare:
        require(args.output_root is not None, '--prepare requires --output-root.')
        prepare(args.output_root.resolve())
    else:
        run(args.preparation.resolve())


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    main()
