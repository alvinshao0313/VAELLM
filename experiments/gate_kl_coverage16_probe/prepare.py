"""CPU-only preparation of the fixed16-group versus reviewed4-group contrast."""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import sys

import torch

from experiments.gate_kl_calibration_probe.data import prepare_data
from experiments.gate_local_delta_probe.run import assets, load_assets
from experiments.gate_student_context_probe.core import records
from experiments.gate_kl_coverage16_probe.common import (
    ARM, BASE, CHUNK, GROUPS, ROOT, SOURCE, SOURCE_ID, TARGET, VECTORS, check, read,
)
from experiments.longbench_v1.common import check_stat, file_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and Path.cwd() == ROOT and not output.exists(), 'Prepare a fresh directory with CUDA hidden in the public root.')
    torch.set_num_threads(4)
    from experiments.gate_kl_coverage16_probe import core, run, evaluation
    old_dir = BASE / 'gate_kl_calibration_probe'
    old, history, review = (read(old_dir / name) for name in
                            ('launch.prepared.json', 'run/report.json', 'review.json'))
    require(history['status'] == 'completed' and review['status'] == 'verified'
            and review['report_sha256'] == sha256(old_dir / 'run/report.json')
            and history['source_checkpoint_id'] == SOURCE_ID
            and all((old_dir / name).read_text().strip() == '0' for name in ('run.exit', 'outer.exit')),
            'Require the completed reviewed original mixed128 proposal.')
    data, batches, _ = prepare_data()
    require(data == old['data'] and {name: {key: tensor_record(value) for key, value in batch.items()}
            for name, batch in batches.items()} == old['token_batches'], 'Original mixed128/data identities changed.')
    require(data['source_data']['checkpoint_id'] == SOURCE_ID
            and data['causal_tokens'][ARM] == 25718 and history['saved_model_or_gradient'] is False,
            'Use original25718-token mixed128; no saved gradient exists to reuse.')
    paths = assets()
    capture, archive = read(paths['capture_manifest']), read(paths['archive_manifest'])
    require(sha256(paths['cache']) == capture['archive']['sha256']
            and sha256(paths['archive']) == archive['inputs_sha256'], 'Retained student/HF archive file hash changed.')
    layer, _, cache, _, manifest = load_assets(paths)
    initial = records(layer)
    old_fit = read(BASE / 'gate_kl_calibration_decoder500/preparation.json')
    require(initial == old_fit['original_state'] == history['experiment']['initial_state'],
            'Start both code selection and fitting from the original f927 decoder.')
    for split in ('train', 'heldout', 'replay32'):
        require(tensor_record(cache['inputs'][split]) == manifest['tensors']['inputs'][split]
                and cache['document_lengths'][split] == manifest['document_lengths'][split], 'Cached context changed.')
    choice_path = Path(history['saved_choices'][ARM]['path'])
    require(sha256(choice_path) == history['saved_choices'][ARM]['sha256'], 'Historical four-group choices changed.')
    choices = torch.load(choice_path, map_location='cpu', weights_only=True)
    require(tuple(choices.shape) == (CHUNK,) and choices.dtype == torch.uint8
            and tensor_record(choices) == history['saved_choices'][ARM]['tensor'], 'Wrong historical choices layout.')
    old_teacher = Path(old['old_teacher_path'])
    require(sha256(old_teacher) == old['input_sha256'][str(old_teacher)], 'Historical teacher observation cache changed.')
    from experiments.gate_final_kl_accept_probe.run import validate_teacher_cache
    validate_teacher_cache(torch.load(old_teacher, map_location='cpu', weights_only=True, mmap=True),
                           batches, read(old['old_teacher_report']))
    plan = evaluation.cpu_plan(output)
    require(plan['source_checkpoint_id'] == SOURCE_ID, 'Full8 source does not match the proposal/fitting source.')
    sources = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / 'result'):
                sources[str(path)] = sha256(path)
    for path in Path(__file__).parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path.resolve())] = sha256(path)
    # Applicable older numerical/data source versions remain reviewable; no old record is rewritten.
    old_sources = {**old['source_sha256'], **old_fit['source_sha256']}
    differences = {path: {'historical': old_sources[path], 'current': digest} for path, digest in sources.items()
                   if path in old_sources and digest != old_sources[path]}
    inputs = dict(data['input_sha256'])
    inputs.update(plan['input_sha256'])
    for path in [old_dir / name for name in ('launch.prepared.json', 'run/report.json', 'review.json', 'run.exit', 'outer.exit')]:
        inputs[str(path)] = sha256(path)
    for path in [Path(value) for value in paths.values()] + [choice_path, old_teacher, Path(old['old_teacher_report']),
                 BASE / 'gate_kl_calibration_decoder500/preparation.json',
                 BASE / 'gate_final_kl_neighbor_accept_probe/run/proposal_generation.json',
                 BASE / 'gate_final_kl_neighbor_accept_probe/review.json']:
        inputs[str(path)] = sha256(path)
    protected = {item['path']: item for item in old['protected_stats']}
    for item in old_fit['protected_stats']:
        require(item['path'] not in protected or protected[item['path']] == item, 'Source/cache stat evidence disagrees.')
        protected[item['path']] = item
    for item in protected.values():
        check_stat(item)
    output.mkdir(parents=True)
    batch_path = output / 'fixed_batches.pt'
    torch.save(batches, batch_path)
    saved = torch.load(batch_path, map_location='cpu', weights_only=True)
    require({name: {key: tensor_record(value) for key, value in batch.items()} for name, batch in saved.items()}
            == old['token_batches'], 'Prepared batch serialization changed native tokens.')
    inputs[str(batch_path)] = sha256(batch_path)
    record = {'status': 'prepared_unbound', 'gpu': None, 'queued': False, 'automatic_retry': False,
        'cwd': str(ROOT), 'python_executable': sys.executable, 'output_dir': str(output),
        'source_checkpoint': str(SOURCE), 'source_checkpoint_id': SOURCE_ID,
        'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': list(protected.values()),
        'historical_source_differences': differences, 'data': data, 'token_batches': old['token_batches'],
        'batch_path': str(batch_path), 'paths': {key: str(value) for key, value in paths.items()},
        'old_teacher_path': str(old_teacher), 'old_teacher_report': old['old_teacher_report'],
        'reference_choices_path': str(choice_path), 'original_state': initial,
        'expected_training_settings': old_fit['expected_training_settings'],
        'expected_persistent_bits': old_fit['expected_persistent_bits'], 'evaluation': plan,
        'environment': {**old['environment'], 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'},
        'timeout_seconds': 1800, 'termination_grace_seconds': 20, 'compute_budget_seconds': 600,
        'protocol': {'source': SOURCE_ID, 'groups': list(range(GROUPS)), 'vectors': VECTORS,
            'enumeration_chunk_vectors': CHUNK, 'enumeration_chunks': 4, 'single_bit_neighbors': 64,
            'calibration_documents': 128, 'calibration_causal_tokens': 25718, 'max_length': 256,
            'gradient': 'Recompute actual fresh-f927 mixed128 BF16 leaf gradients, FP32 token-sum accumulation in original order.',
            'proposal': 'Same native BF16 stage-sum finite neighbors and FP64 rotated-gradient dot; strict-negative best, original0/first-bit ties; all16 groups applied together.',
            'single_variable': 'Only coverage0..3 to0..15. No greedy acceptance, observation selection, extra rounds or55a initialization.',
            'fit': 'Original train32/4289 student tokens; full conditional HF_MLP(xs), fixed f927 up/down. Six shared FP32 decoder/Adam masters, BF16 compute, LR1e-5, linear500, seed31.',
            'primary_control': 'Existing55a four-group mixed128 plus500; secondary original-code500 and f927.',
            'quality': 'Always native save and full8 after successful computation; no local/accuracy threshold. Full21LongBench is scheduled separately by root.',
            'budget': 'No added state or LoRA. Actual native integer bit budget must equal f927; expected2.4569887907608696bpw.'},
        'resources': {'measured_gradient_seconds': 26.899530172348022, 'measured_four_group_enumeration_seconds': .06360363960266113,
            'measured_384_group_enumeration_seconds': 6.053109169006348,
            'measured_fit500_seconds': 44.78368806297658, 'measured_proposal_peak_gib': 65.70871257781982,
            'measured_fit_peak_bytes': 4839857664, 'estimated_compute_seconds': [180, 360],
            'estimated_with_full8_seconds': [600, 1200], 'disk_reservation_gib': 6,
            'boundary': 'Estimates from same f927 paths; new16-group execution and final quality have not run.'},
        'cuda_initialized': torch.cuda.is_initialized(),
        'command': [sys.executable, '-m', 'experiments.gate_kl_coverage16_probe.run',
                    '--preparation', str(output / 'launch.bound.json')]}
    check(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    write_json(output / 'launch.prepared.json', record)
    print({'status': record['status'], 'prepared_sha256': sha256(output / 'launch.prepared.json'),
        'sources': len(sources), 'inputs': len(inputs), 'historical_source_differences': differences,
        'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=BASE / 'gate_kl_coverage16_decoder500')
    prepare(parser.parse_args().output_dir.resolve())
