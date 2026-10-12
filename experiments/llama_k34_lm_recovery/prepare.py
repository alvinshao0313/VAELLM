"""CPU binding of completed K34 backward proof to the unchanged native2000 recipe."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.general_lm_recovery.train import configuration, training_arguments
from experiments.llama_ce_reproduction.callback import stage_commands
from experiments.llama_k34_lm_validation.config import (
    CONTROL, OUTPUT as SHORT, PUBLIC, ROOT, R2, SOURCE, SOURCE_ID, check_inputs, read)
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.long_context_recovery.contract import require
from train_utils.runtime_snapshot import to_runtime_jsonable

OUTPUT = R2 / 'general_lm_llama_k34true_s2000'


def completed_proof():
    paths = [SHORT / p for p in ('launch.json', 'verification.json', 'training.json',
        'step_0004/audit.json', 'step_0004/final_model/export_record.json')]
    launch, proof, trained, audit, exported = map(read, paths)
    exits = {name: SHORT / f'{name}.exit' for name in ('training', 'export', 'audit', 'dispatcher', 'outer')}
    require(launch['status'] == proof['status'] == 'verified' and launch['dispatcher_exit_code'] == 0
            and all(p.read_text().strip() == '0' for p in exits.values())
            and trained['status'] == 'trained' and trained['training_result']['global_step'] == 4
            and trained['training_result']['round_base_checkpoint_id'] == SOURCE_ID
            and launch['source_checkpoint_id'] == proof['source_checkpoint_id'] == audit['source_checkpoint_id'] == SOURCE_ID
            and launch['source_checkpoint'] == str(SOURCE) and proof['native_vs_independent_tensors_compared'] == 2979,
            'Require the actual completed K34 four-step proof and all five terminal exits0.')
    check_hashes(proof['evidence_sha256'])
    require(sha256(SHORT / 'verification.json') == launch['verification_sha256']
            and proof['expected_inventory'] == launch['expected_inventory']
            and audit['status'] == 'verified' and audit['selected_decoder_tensor_count'] == 1344
            and audit['frozen_tensors_compared'] == 1635 and audit['code_bank_count'] == 448
            and audit['changed_code_bits'] == 0 and audit['packed_padding_unchanged']
            and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == 2979
            and audit['selected_decoder_matches_fp32_cast_exactly'] and audit['fresh_strict_reload_state_exact']
            and audit['frozen_bytes_unchanged'] and audit['rank_zero']
            and exported['source_global_step'] == 4 and exported['source_max_steps'] == 2000
            and exported['restored_decoder_tensors'] == 1344 and exported['optimizer_steps_executed'] == 0,
            'Incomplete original-scope/native independent export evidence.')
    observed = trained['observations']
    names = {f'model.layers.{layer}.mlp.gate_proj/stage{stage}' for layer in range(32) for stage in (0, 1)}
    for field in ('gate_gradients', 'gate_updates'):
        require([item['step'] for item in observed[field]] == [1, 2, 3, 4]
                and all(set(item['stages']) == names for item in observed[field]), 'Missing actual64-stage observations.')
    require(all(v['finite'] and v['l2'] > 0 and v['gradient_tensors'] == 6
                for item in observed['gate_gradients'] for v in item['stages'].values())
            and all(v['changed_fp32_elements_this_step'] > 0 for item in observed['gate_updates']
                    if item['step'] > 1 for v in item['stages'].values())
            and [x['step'] for x in observed['native_saves']] == [2, 4]
            and len(proof['batches']) == 4 and all(x['shape'] == [4, 1024] for x in proof['batches']),
            'Actual gradient/update/save or resource coverage differs; first warmup step may have zero update.')
    budget = audit['bit_budget']
    require(budget['user_linear_bits'] == 17326653440 and budget['packed_linear_bits'] == 18020286464
            and budget['code_padding_bits'] == 693633024 and budget['original_linear_weights'] == 6979321856
            and budget['budget_passed'] and budget['user_linear_bpw'] <= 2.5,
            'Keep logical and physical budgets separate; padding is immutable.')
    return {'status': 'verified', 'path': str(SHORT / 'verification.json'),
        'verification_sha256': sha256(SHORT / 'verification.json'),
        'evidence_sha256': {str(p): sha256(p) for p in [*paths, *exits.values()]},
        'actual_elapsed_seconds': launch['finished_unix'] - launch['started_unix'],
        'training_peak_allocated_bytes': proof['training_peak_allocated_bytes'],
        'training_peak_reserved_bytes': proof['training_peak_reserved_bytes'],
        'export_peak_allocated_bytes': proof['export_peak_allocated_bytes'],
        'scope': 'Existing native K34True FP32 packed backward/update and independent export; no quality claim and no short weights consumed.'}


def prepare():
    require(Path.cwd() == ROOT and Path(__file__).resolve().is_relative_to(ROOT)
            and os.environ.get('PYTHONPATH') == str(ROOT) and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized() and not OUTPUT.exists(), 'Require one new CPU-only S preparation.')
    proof = completed_proof()
    short, control = read(SHORT / 'launch.prepared.json'), read(CONTROL)
    check_inputs(short)
    record = deepcopy(short)
    for key in ('dispatch_argv', 'cpu_configuration', 'resource_cap_enforcement', 'scope', 'validation_reuse'):
        record.pop(key)
    source_summary = read(SOURCE.parent / 'full8_summary.json')
    require(source_summary['checkpoint_id'] == SOURCE_ID and control['status'] == 'completed'
            and control['source_checkpoint_id'] == 'e767c232-56fd-4b03-a4ae-dd0781b6e047', 'Wrong actual source or matched LM control.')
    record.update(schema='llama_k34true_all_decoder_lm_2000_v1', kind='formal_s2000', status='prepared_unbound',
        output_dir=str(OUTPUT), export_gpu=None,
        training_argv=training_arguments(SOURCE, short['architecture']['base_model_path'], OUTPUT, short['data']['train_file']),
        argv=[sys.executable, '-u', '-m', 'experiments.general_lm_recovery.train', '--launch_record', str(OUTPUT / 'launch.json')],
        schedule=deepcopy(control['schedule']), resource_budget_seconds=control['resource_budget_seconds'],
        protocol=deepcopy(control['protocol']), control_result_paths=[str(SOURCE.parent / f'{name}.json') for name in ('mmlu', 'seven')],
        control_mean_percent=source_summary['mean_percent'], resource_validation=proof,
        matched_recovery_control={'launch': str(CONTROL), 'source_checkpoint_id': control['source_checkpoint_id'],
            'steps': [str(CONTROL.parent / f'step_{s:04d}/full8_summary.json') for s in (400, 800, 1200, 1600, 2000)]},
        readiness={'cpu': 'pending', 'gpu': 'not_scheduled', 'k34_backward_resource': 'verified'},
        scientific_boundary='Fresh original98a8 K34True source; only source/output differ from original e767 FineWeb recipe. All1344 decoder tensors train;1635 frozen tensors including448 code banks, signs/protection/norm/head/embedding remain frozen. No LoRA or short-run weights.',
        comparison_boundary='Primary recovery gains are measured against original98a8. Same-recipe e7672000 trajectory is context, not a pure K-only or transpose-only effect:98a8 refreshed32 gate decoders/codes and geometry, while inheriting e767 other192.',
        validation_reuse='Completed64gate-stage K34 FP32 gradients/updates,2979 native/fresh export and allD resource evidence; retain the existing every400 synchronous two-GPU export/audit/full8 callback without another GPU short.')
    record['input_evidence_sha256'].update(proof['evidence_sha256'])
    for p in [SHORT / 'launch.prepared.json', *[Path(p) for p in record['control_result_paths']]]:
        record['input_evidence_sha256'][str(p)] = sha256(p)
    # The callback compares task protocol to the actual new starting model.
    for actual, old in zip(record['control_result_paths'], control['control_result_paths'], strict=True):
        aa, bb = (read(p)['evals']['lm_eval'] for p in (actual, old))
        require(all(aa[k] == bb[k] for k in ('configs', 'versions', 'n_shot', 'n_samples')), 'Full8 source protocols differ.')
    for p in [Path(__file__).resolve(), ROOT / 'experiments/general_lm_recovery/train.py']:
        ast.parse(p.read_text()); require(p.read_bytes() == (PUBLIC / p.relative_to(ROOT)).read_bytes(), 'Public/S entry differs.')
        record['source_sha256'][str(p)] = sha256(p)
    expected = list(control['training_argv'])
    for option, value in (('--student_checkpoint_dir', str(SOURCE)), ('--run_root_dir', str(OUTPUT))):
        expected[expected.index(option) + 1] = value
    require(record['training_argv'] == expected, 'Formal native argv differs beyond fresh source/output.')
    cfg, hf, args, names, _ = configuration(record)
    canonical = to_runtime_jsonable(cfg); reference = deepcopy(control['cpu_configuration']['canonical_config'])
    reference.update(student_checkpoint_dir=str(SOURCE), run_root_dir=str(OUTPUT))
    require(canonical == reference and len(names) == 224, 'Actual parser differs beyond source/output.')
    record['cpu_configuration'] = {'canonical_config': canonical, 'hf_args': to_runtime_jsonable(hf),
        'training_args': to_runtime_jsonable(args), 'canonical_difference_from_control': ['student_checkpoint_dir', 'run_root_dir']}
    record['stage_templates'] = [{'name': n, 'argv': a, 'allowed_exit_codes': e} for n, a, e in stage_commands(
        checkpoint=OUTPUT / 'NATIVE_RUN/trainer_state/checkpoint-400', source_dir=SOURCE, output_dir=OUTPUT / 'step_0400')]
    free = shutil.disk_usage(OUTPUT.parent).free
    require(free >= 35_000_000_000, 'Reserve35GB for five independent endpoints, native final and one step state.')
    record['estimate'] = {'training_hours_estimate': [10, 15], 'hard_seconds': record['resource_budget_seconds'],
        'storage_bytes_per_branch': 35000000000, 'disk_free_at_prepare': free,
        'measured_short_training_peak_allocated_bytes': proof['training_peak_allocated_bytes'],
        'measured_short_training_peak_reserved_bytes': proof['training_peak_reserved_bytes'],
        'boundary': 'Four real4x1024 updates measured resource feasibility; startup/export cost prevents precise long-run throughput extrapolation. Existing e767 full schedule supplies10–15h estimate; K34 may differ.'}
    check_inputs(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    record['readiness']['cpu'] = 'passed'
    OUTPUT.mkdir(); write_json(OUTPUT / 'launch.prepared.json', record)
    print({'status': record['status'], 'path': str(OUTPUT / 'launch.prepared.json'), 'sha256': sha256(OUTPUT / 'launch.prepared.json'),
        'sources': len(record['source_sha256']), 'inputs': len(record['input_evidence_sha256']), 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    prepare()
