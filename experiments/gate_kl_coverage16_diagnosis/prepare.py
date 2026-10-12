"""Bind existing choices, local endpoints and unchanged data for a no-training diagnosis."""
import argparse
import ast
import os
from pathlib import Path
import sys

import torch

from experiments.gate_kl_coverage16_diagnosis.common import ROOT, BASE, SOURCE, SOURCE_ID, ARM, BANKS, POPULATIONS, read, check
from experiments.gate_kl_coverage16_diagnosis import export, kl, run
from experiments.gate_kl_calibration_probe.run import teacher_chunks
from experiments.gate_final_kl_accept_probe.run import validate_teacher_cache
from experiments.longbench_v1.common import check_stat, sha256, write_json
from experiments.rotation_factor_probe.capture import require, tensor_record


def prepare(output):
    require(not output.exists() and Path.cwd() == ROOT and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized(), 'Require a fresh CPU-only preparation in the public source root.')
    torch.set_num_threads(4)
    previous = BASE / 'gate_kl_coverage16_decoder500'
    old = read(previous / 'launch.prepared.json'); check(old)
    new_report = read(previous / 'run/report.json'); old_report = read(BASE / 'gate_kl_calibration_probe/run/report.json')
    fit_report_path = BASE / 'gate_kl_calibration_decoder500/run/report.json'; fit_report = read(fit_report_path)
    require(new_report['status'] == fit_report['status'] == old_report['status'] == 'completed'
            and new_report['source_checkpoint_id'] == fit_report['source_checkpoint_id'] == old_report['source_checkpoint_id'] == SOURCE_ID
            and read(previous / 'run.json')['status'] == 'completed'
            and (previous / 'run.exit').read_text().strip() == '0', 'Require the completed same-source proposal and fits.')
    batches = torch.load(old['batch_path'], map_location='cpu', weights_only=True)
    require({name: {key: tensor_record(value) for key, value in batch.items()} for name, batch in batches.items()}
            == old['token_batches'], 'Reuse the exact prepared native batches without retokenizing.')
    teacher_chunks(batches)
    validate_teacher_cache(torch.load(old['old_teacher_path'], map_location='cpu', weights_only=True, mmap=True),
                           batches, read(old['old_teacher_report']))
    original = old['original_state']
    require(original == new_report['proposal']['initial_state'] == old_report['experiment']['initial_state'],
            'The measured proposals must share the original f927 gate.')
    infos = {
        'four_decoder500': {'path': fit_report['arms'][ARM]['saved']['state_path'],
            'sha256': fit_report['arms'][ARM]['saved']['state_sha256'],
            'state': fit_report['arms'][ARM]['training']['final_payload'],
            'before_fit_observations': {name: old_report['experiment']['endpoints'][ARM][name] for name in POPULATIONS},
            'summary': BASE / 'gate_kl_calibration_decoder_full8/alpaca64_fineweb64/full8_summary.json'},
        'sixteen_decoder500': {'path': new_report['endpoint_path'], 'sha256': new_report['endpoint_sha256'],
            'state': new_report['endpoint_state'], 'before_fit_observations': new_report['proposal']['fixed_endpoint_observations'],
            'summary': previous / 'full8_summary.json'}}
    for arm, info in infos.items():
        require(sha256(info['path']) == info['sha256'], 'Existing local endpoint file changed: ' + arm)
        state = torch.load(info['path'], map_location='cpu', weights_only=True, mmap=True)
        require({key: tensor_record(value) for key, value in state.items()} == info['state']
                and len(state) == 13 and all(info['state'][key] == value for key, value in original.items()
                    if key not in BANKS and not key.startswith('_parallel_stage_decoder.')),
                'Reviewed local endpoint has an unexpected mutable tensor.')
        summary = read(info['summary'])
        require(summary['status'] == 'full8_evaluation_verified' and summary['source_checkpoint_id'] == SOURCE_ID
                and summary['endpoint_sha256'] == info['sha256'], 'Local patch is not the actual evaluated endpoint.')
        info['checkpoint_id'] = summary['checkpoint_id']; info['summary'] = str(info['summary'])
        del state
    require(infos['four_decoder500']['checkpoint_id'] == '55a3aeb5-9b37-43ae-9da9-6964c9d6abac'
            and infos['sixteen_decoder500']['checkpoint_id'] == '834bdf15-1e93-4088-8da3-02215d7e8f81',
            'Bind exactly55a and834b, not another fitted endpoint.')
    plan = export.cpu_plan(output)
    require(plan['initial_state'] == original and plan['applied_state'] == new_report['proposal']['applied_state']
            and plan['choices_sha256'] == new_report['choices_sha256'], 'Code-only export and KL refer to different proposals.')
    sources = dict(old['source_sha256'])
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / 'result'):
                digest = sha256(path)
                require(str(path) not in sources or sources[str(path)] == digest, 'A reused frozen implementation changed: ' + str(path))
                sources[str(path)] = digest
    for path in Path(__file__).parent.glob('*.py'):
        ast.parse(path.read_text()); sources[str(path.resolve())] = sha256(path)
    inputs = {**old['input_sha256'], **plan['input_sha256']}
    evidence = [previous / name for name in ('launch.prepared.json', 'run/report.json', 'run.json', 'run.exit', 'audit.json', 'analysis.json')]
    evidence += [fit_report_path, BASE / 'gate_kl_calibration_probe/run/report.json']
    for info in infos.values():
        evidence.extend([Path(info['path']), Path(info['summary'])])
    for path in evidence:
        inputs[str(path)] = sha256(path)
    for stat in old['protected_stats']:
        check_stat(stat)
    record = {'schema': 'gate_kl_coverage16_diagnosis_v1', 'status': 'prepared_unbound', 'gpu': None,
        'queued': False, 'automatic_retry': False, 'cwd': str(ROOT), 'python_executable': sys.executable,
        'output_dir': str(output), 'source_checkpoint': str(SOURCE), 'source_checkpoint_id': SOURCE_ID,
        'source_sha256': sources, 'input_sha256': inputs, 'protected_stats': old['protected_stats'],
        'environment': old['environment'], 'data': old['data'], 'token_batches': old['token_batches'],
        'batch_path': old['batch_path'], 'archive_path': old['paths']['archive'],
        'old_teacher_path': old['old_teacher_path'], 'old_teacher_report': old['old_teacher_report'],
        'original_state': original, 'kl_endpoints': infos, 'evaluation': plan,
        'kl_budget_seconds': 600, 'timeout_seconds': 1800, 'termination_grace_seconds': 20,
        'protocol': {'new_training': False, 'new_gradient_or_selection': False,
            'codes_only': 'Apply the already measured sixteen groups once to original f927 banks; all six original decoders and3410 outside tensors remain exact.',
            'kl_arms': ['four_decoder500', 'sixteen_decoder500'], 'kl_populations': list(POPULATIONS),
            'teacher': 'Reuse exact fixed batches and old32-domain teacher cache; recompute the same native256 teacher chunks with the existing teacher_pass/population mapping.',
            'student': 'One frozen f927 load, sequential actual55a/834b thirteen-tensor patches, native category group8 refresh; restore original13/cache/full logits. No duplicate8B fitted exports.',
            'kl': 'Original score_population: native BF16/FA2/full256/micro1; top100 plus outside-mass forward KL T1 and causal NLL, document-order FP64 reduction.',
            'comparisons': 'Postfit scores versus measured same-code prefit scores; unfitted16 full8 versus f927, unfitted4, fitted4, fitted16. No score selects/truncates a run.',
            'boundary': 'Fixed previously inspected local populations; no new validation claim, new algorithm, or nine-item acceptance. Full21 scheduling belongs to root.'},
        'resources': {'historical_teacher_seconds': new_report['teacher']['seconds'],
            'historical_native_prewarm_seconds': new_report['proposal']['warmup']['duration_sec'],
            'existing_proposal_peak_gib': new_report['proposal']['peak_allocated_bytes'] / 2**30,
            'estimated_kl_seconds': [120, 240], 'estimated_including_export_full8_seconds': [600, 1200],
            'disk_reservation_gib': 6, 'new_full_models': 1,
            'boundary': 'No gradient or optimizer; sequential teacher/student, then process-exit release before native export/full8. Exact new elapsed/peak remain unmeasured. Retain prior small13-tensor endpoints, not additional fitted8B models.'},
        'validation_reuse': {'new_gpu_short': False,
            'scope': 'Reuse measured four-group code-only deploy/full8 and sixteen-group bank replay, original native save/reload, fixed teacher/scorer/group8. Actual new3423-source union and scalar observations are checked in the requested run; no fabricated training report.'},
        'command': [sys.executable, '-u', '-m', 'experiments.gate_kl_coverage16_diagnosis.run',
                    '--preparation', str(output / 'launch.bound.json')], 'cuda_initialized': False}
    check(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    output.mkdir(parents=True); write_json(output / 'launch.prepared.json', record)
    script = '#!/usr/bin/env bash\nset -euo pipefail\n'
    for key, value in record['environment'].items():
        script += f'export {key}="{value}"\n'
    script += f'cd "{ROOT}"\npython -u -m experiments.gate_kl_coverage16_diagnosis.run --preparation "{output / "launch.bound.json"}"\n'
    (output / 'run.sh').write_text(script); (output / 'run.sh').chmod(0o755)
    print({'status': record['status'], 'prepared_sha256': sha256(output / 'launch.prepared.json'),
        'sources': len(sources), 'inputs': len(inputs), 'stat_count': len(record['protected_stats']),
        'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir', type=Path,
        default=BASE / 'gate_kl_coverage16_diagnosis')
    prepare(parser.parse_args().output_dir.resolve())
