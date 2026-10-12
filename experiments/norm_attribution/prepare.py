"""Prepare fixed norm attribution contracts on CPU; never reserve a GPU."""
import argparse
import json
from pathlib import Path
import sys

import torch

from experiments.longbench_v1.common import sha256, write_json
from experiments.longbench_candidate_probe.prepare import check_candidate_preparation
from experiments.mmlu_attribution.protocol import check_preparation as check_mmlu, require
from experiments.w25_eval.norm_state import norm_shapes
from .protocol import ARMS, check_control, check_preparation, norm_groups


def prepare(norm_preparation, mmlu_preparation, output):
    norm_preparation, mmlu_preparation, output = (Path(p).resolve() for p in
        (norm_preparation, mmlu_preparation, output))
    require(not output.exists() and not torch.cuda.is_available(), 'Use a new output and CPU-only preparation.')
    norm, retrieval, _manifest = check_candidate_preparation(norm_preparation)
    mmlu, _cases = check_mmlu(mmlu_preparation)
    candidate = Path(norm['config']['model']['checkpoint'])
    audit_path = candidate.parent / 'audit.json'
    audit = json.loads(audit_path.read_text())
    require(audit['status'] == 'verified' and audit['checkpoint_id'] == norm['config']['model']['checkpoint_id']
            and audit['source_checkpoint_id'] == mmlu['source_checkpoint_id']
            and audit['frozen_bytes_unchanged'] and audit['fresh_strict_reload_state_exact'], 'Norm export/source audit differs.')
    shapes = norm_shapes(norm['config']['model']['hf_path'])
    groups = norm_groups(shapes)
    base = torch.load(Path(mmlu['source_checkpoint']) / 'pytorch_model.bin', mmap=True, map_location='cpu', weights_only=True)
    deployed = torch.load(candidate / 'pytorch_model.bin', mmap=True, map_location='cpu', weights_only=True)
    native_path = Path(audit['training_checkpoint_dir']) / 'training_model_state.pt'
    require(sha256(native_path) == audit['selected_training_state_sha256'], 'Saved norm FP32 state changed.')
    selected = torch.load(native_path, mmap=True, map_location='cpu', weights_only=True)
    require(set(selected) == set(shapes), 'Native norm training inventory changed.')
    changes = {}
    for key, value in selected.items():
        require(list(value.shape) == shapes[key] and value.dtype == torch.float32
                and base[key].dtype == deployed[key].dtype == torch.bfloat16
                and torch.equal(value.to(torch.bfloat16), deployed[key]), 'Norm cast/shape mismatch: ' + key)
        changes[key] = {'shape': shapes[key], 'elements': value.numel(),
            'fp32_changed': int(torch.count_nonzero(value != base[key].float())),
            'bf16_changed': int(torch.count_nonzero(deployed[key] != base[key]))}
    require(sum(x['fp32_changed'] for x in changes.values()) == audit['changed_norm_fp32_elements']
            and sum(x['bf16_changed'] for x in changes.values()) == audit['changed_norm_bf16_elements'], 'Norm deltas disagree with audited export.')
    controls = {kind: str(mmlu_preparation.parent / 'arms' / kind) for kind in ('compressed', 'dense')}
    evidence = [norm_preparation, mmlu_preparation, audit_path, native_path]
    for directory in controls.values():
        report, files = check_control(directory, sha256(mmlu_preparation))
        require(report['non_target_state_and_identity_unchanged'], 'MMLU control state was not frozen.')
        evidence.extend(files)
    norm_control = norm_preparation.parent / 'run'
    report, files = check_control(norm_control, sha256(norm_preparation), retrieval=True)
    require(report['checkpoint_id'] == audit['checkpoint_id'] and report['frozen_state_and_identity_unchanged']
            and (norm_control.parent / 'run.exit').read_text().strip() == '0', 'Norm retrieval control was not completed.')
    evidence.extend([*files, norm_control.parent / 'run.exit'])
    paths = list(Path(__file__).parent.glob('*.py'))
    sources = {str(path.resolve()): sha256(path) for path in paths}
    sources.update(norm['source_sha256'])
    sources[str(Path(__file__).resolve().parents[1] / 'w25_eval/norm_state.py')] = sha256(
        Path(__file__).resolve().parents[1] / 'w25_eval/norm_state.py')
    record = {'schema': 'norm_attribution_v1', 'status': 'prepared_not_scheduled', 'gpu': None,
        'queued': False, 'automatic_retry': False, 'automatic_wallclock_stop': False,
        'arms': list(ARMS), 'norm_preparation': str(norm_preparation), 'mmlu_preparation': str(mmlu_preparation),
        'candidate_id': audit['checkpoint_id'], 'restoration_source_id': audit['source_checkpoint_id'],
        'restoration_source': mmlu['source_checkpoint'], 'norm_shapes': shapes, 'selected_norms': groups,
        'norm_changes': changes, 'mmlu_controls': controls, 'norm_retrieval_control': str(norm_control),
        'source_sha256': sources, 'evidence_sha256': {str(path): sha256(path) for path in evidence},
        'package_versions': {**norm['config']['package_versions'], **mmlu['package_versions']},
        'resource_estimate': {'gpus': 1, 'minutes': [10, 15], 'required_available_window_minutes': 20,
            'basis': 'Existing1824MMLU146s and fixed16norm86s; three fresh MMLU arms plus two retrieval arms.'},
        'scope': 'Conditional in-memory restoration of72 QK norms or73 other norms including final; not training, full MMLU, full LongBench or goal acceptance.',
        'production_selector': 'Existing none/final/all unchanged; no QK/other selector integrated.',
        'model_exported': False, 'goal_acceptance': False, 'cuda_initialized': False}
    output.mkdir(parents=True)
    record['arm_commands'] = [{'arm': arm, 'argv': [sys.executable, '-u', '-m', 'experiments.norm_attribution.run',
        '--preparation', str(output / 'preparation.json'), '--arm', arm, '--output', str(output / 'arms' / arm)]} for arm in ARMS]
    record['comparison_argv'] = [sys.executable, '-u', '-m', 'experiments.norm_attribution.compare',
        '--preparation', str(output / 'preparation.json'), '--output', str(output / 'comparison.json')]
    write_json(output / 'preparation.json', record)
    check_preparation(output / 'preparation.json')
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    print(json.dumps({'status': record['status'], 'sha256': sha256(output / 'preparation.json'),
        'norm_counts': {key: len(value) for key, value in groups.items()}, 'mmlu_arms': 3, 'retrieval_arms': 2}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('norm_preparation', 'mmlu_preparation', 'output'):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    prepare(**vars(parser.parse_args()))
