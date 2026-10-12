"""Bind the completed mixed-stage proof to the original FineWeb2000 recipe."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.llama_s4_lm_recovery.callback import stage_commands
from experiments.llama_s4_lm_recovery.train import configuration, training_arguments
from experiments.llama_s4_lm_validation.config import CONTROL, OUTPUT as SHORT, check_inputs, read
from experiments.llama_s4_lm_validation.state import PUBLIC, ROOT, R2, SOURCE, SOURCE_ID, TOTAL_BITS
from experiments.longbench_v1.common import check_hashes, sha256, write_json
from experiments.long_context_recovery.contract import require
from train_utils.runtime_snapshot import to_runtime_jsonable

OUTPUT = R2 / 'general_lm_llama_gate_s4_s2000'
REVIEW_SHA = 'b4d29aa2f6eef40f07997d79922667375a852aa48c0eeea2898c53af4bce8007'
PROOF_SHA = 'b37265ffdfa73a807e73e465425441838c15439cd5794d4a3ae9c84a58f788e6'


def completed_proof():
    paths = [SHORT / p for p in ('launch.json', 'verification.json', 'training.json',
        'step_0004/audit.json', 'step_0004/final_model/export_record.json', 'independent_review.json')]
    launch, proof, trained, audit, exported, review = map(read, paths)
    exits = [SHORT / f'{name}.exit' for name in ('training', 'export', 'audit', 'dispatcher', 'outer')]
    require(sha256(SHORT / 'independent_review.json') == REVIEW_SHA
            and sha256(SHORT / 'verification.json') == PROOF_SHA == launch['verification_sha256']
            and review['status'] == 'independent_cpu_verified' and launch['status'] == proof['status'] == 'verified'
            and launch['dispatcher_exit_code'] == 0 and all(p.read_text().strip() == '0' for p in exits)
            and all(x['source_checkpoint_id'] == SOURCE_ID for x in (launch, proof, trained, audit, review))
            and launch['source_checkpoint'] == str(SOURCE) and review['verification_sha256'] == PROOF_SHA,
            'Require the actual independently verified13bd four-step run and all five terminal exits0.')
    check_hashes(proof['evidence_sha256'])
    inv = launch['expected_inventory']
    require(proof['expected_inventory'] == inv and inv['serialized_tensors'] == 3043
            and inv['selected_decoder_tensors'] == 1344 and inv['selected_decoder_elements'] == 4078592
            and inv['frozen_tensors'] == 1699 and inv['all_code_banks'] == 512
            and proof['native_vs_independent_tensors_compared'] == 3043
            and audit['status'] == 'verified' and audit['selected_decoder_tensor_count'] == 1344
            and audit['frozen_tensors_compared'] == 1699 and audit['code_bank_count'] == 512
            and audit['changed_code_bits'] == 0 and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == 3043
            and audit['selected_decoder_matches_fp32_cast_exactly'] and audit['fresh_strict_reload_state_exact']
            and audit['frozen_bytes_unchanged'] and audit['rank_zero'], 'Mixed-stage native/fresh inventory proof differs.')
    require(trained['status'] == 'trained' and trained['training_result']['global_step'] == 4
            and trained['training_result']['round_base_checkpoint_id'] == SOURCE_ID
            and exported['source_global_step'] == 4 and exported['source_max_steps'] == 2000
            and exported['restored_decoder_tensors'] == 1344 and exported['optimizer_steps_executed'] == 0
            and len(proof['batches']) == 4 and all(x['shape'] == [4, 1024] for x in proof['batches'])
            and review['observed_gate_stages'] == 128
            and [s['step'] for s in review['stage_update_summary']] == [1, 2, 3, 4]
            and all(s['finite_nonzero_gradient_stages'] == 128 for s in review['stage_update_summary'])
            and all(s['adjacent_fp32_changed_elements_min'] > 0 for s in review['stage_update_summary'] if s['step'] > 1),
            'Actual stage gradients/updates or native schedule/resource coverage differs.')
    budget = audit['bit_budget']
    require(budget['user_linear_bits'] == budget['packed_linear_bits'] == TOTAL_BITS
            and budget['code_padding_bits'] == 0 and budget['original_linear_weights'] == 6979321856
            and budget['budget_passed'], 'Original S4 budget differs.')
    return {'status': 'verified', 'path': str(SHORT / 'verification.json'), 'verification_sha256': PROOF_SHA,
        'independent_review_sha256': REVIEW_SHA, 'evidence_sha256': {str(p): sha256(p) for p in [*paths, *exits]},
        'actual_elapsed_seconds': launch['finished_unix'] - launch['started_unix'],
        'training_peak_allocated_bytes': proof['training_peak_allocated_bytes'],
        'training_peak_reserved_bytes': proof['training_peak_reserved_bytes'],
        'export_peak_allocated_bytes': proof['export_peak_allocated_bytes'],
        'scope': 'Actual128 gate-stage FP32 packed gradients/Adam and3043 native/independent state equality. No quality claim or short weights consumed.'}


def prepare():
    require(Path.cwd() == ROOT and Path(__file__).resolve().is_relative_to(ROOT)
            and os.environ.get('PYTHONPATH') == str(ROOT) and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized() and not OUTPUT.exists(), 'One new CPU-only S preparation is required.')
    proof = completed_proof()
    short, control = read(SHORT / 'launch.prepared.json'), read(CONTROL)
    check_inputs(short)
    record = deepcopy(short)
    for key in ('dispatch_argv', 'cpu_configuration', 'resource_cap_enforcement', 'validation_reuse'):
        record.pop(key)
    source_summary = read(SOURCE.parent / 'full8_summary.json')
    require(source_summary['checkpoint_id'] == SOURCE_ID and control['status'] == 'completed'
            and control['source_checkpoint_id'] == 'e767c232-56fd-4b03-a4ae-dd0781b6e047', 'Wrong source/control identity.')
    record.update(schema='llama_gate_s4_all_decoder_lm_2000_v1', kind='formal_s2000', status='prepared_unbound',
        output_dir=str(OUTPUT), export_gpu=None,
        training_argv=training_arguments(SOURCE, short['architecture']['base_model_path'], OUTPUT, short['data']['train_file']),
        argv=[sys.executable, '-u', '-m', 'experiments.llama_s4_lm_recovery.train', '--launch_record', str(OUTPUT / 'launch.json')],
        schedule=deepcopy(control['schedule']), resource_budget_seconds=control['resource_budget_seconds'],
        protocol=deepcopy(control['protocol']), control_result_paths=[str(SOURCE.parent / f'{name}.json') for name in ('mmlu', 'seven')],
        control_mean_percent=source_summary['mean_percent'], resource_validation=proof,
        matched_recovery_control={'launch': str(CONTROL), 'source_checkpoint_id': control['source_checkpoint_id'],
            'steps': [str(CONTROL.parent / f'step_{s:04d}/full8_summary.json') for s in (400, 800, 1200, 1600, 2000)]},
        readiness={'cpu': 'pending', 'gpu': 'not_scheduled', 'mixed_stage_backward_resource': 'verified'},
        scientific_boundary='Fresh original13bd source, unchanged2000/warm20/FineWeb/B4/max1024/LR3e-6. All1344 grouped decoder tensors train;1699 tensors including512 code banks, protection/signs/norm/head/embedding remain frozen. No LoRA or short weights.',
        comparison_boundary='Primary recovery gains versus original13bd. The e767 same-recipe trajectory is secondary context, not a pure residual-stage effect:13bd freshly re-encoded32 gate projections and retained192 e767 projections.',
        validation_reuse='Actual four-step mixed-stage backward/native save/export/audit proof. The existing synchronous callback still launches independent child processes then returns the same TrainerControl; only explicit state inspection and export/audit module routing use the validated S4 helpers. Native optimizer/scheduler/data and evaluation/return flow are unchanged; no repeated GPU short.')
    record['input_evidence_sha256'].update(proof['evidence_sha256'])
    for p in [SHORT / 'launch.prepared.json', *[Path(p) for p in record['control_result_paths']]]:
        record['input_evidence_sha256'][str(p)] = sha256(p)
    for actual, old in zip(record['control_result_paths'], control['control_result_paths'], strict=True):
        aa, bb = (read(p)['evals']['lm_eval'] for p in (actual, old))
        require(all(aa[k] == bb[k] for k in ('configs', 'versions', 'n_shot', 'n_samples')), 'Full8 protocols differ.')
    for p in Path(__file__).parent.glob('*.py'):
        ast.parse(p.read_text()); require(p.read_bytes() == (PUBLIC / p.relative_to(ROOT)).read_bytes(), 'Public/S new entry differs.')
        record['source_sha256'][str(p)] = sha256(p)
    expected = list(control['training_argv'])
    for option, value in (('--student_checkpoint_dir', str(SOURCE)), ('--run_root_dir', str(OUTPUT))):
        expected[expected.index(option) + 1] = value
    require(record['training_argv'] == expected, 'Formal argv differs beyond fresh source/output.')
    cfg, hf, args, names, _ = configuration(record)
    canonical = to_runtime_jsonable(cfg); reference = deepcopy(control['cpu_configuration']['canonical_config'])
    reference.update(student_checkpoint_dir=str(SOURCE), run_root_dir=str(OUTPUT))
    require(canonical == reference and len(names) == 224, 'Actual parser differs beyond source/output.')
    record['cpu_configuration'] = {'canonical_config': canonical, 'hf_args': to_runtime_jsonable(hf),
        'training_args': to_runtime_jsonable(args), 'canonical_difference_from_control': ['student_checkpoint_dir', 'run_root_dir']}
    record['stage_templates'] = [{'name': n, 'argv': a, 'allowed_exit_codes': e} for n, a, e in stage_commands(
        checkpoint=OUTPUT / 'NATIVE_RUN/trainer_state/checkpoint-400', source_dir=SOURCE, output_dir=OUTPUT / 'step_0400')]
    free = shutil.disk_usage(OUTPUT.parent).free
    require(free >= 35_000_000_000, 'Reserve35GB for five exports, native final and one native trainer checkpoint.')
    record['estimate'] = {'training_hours_estimate': [10, 15], 'hard_seconds': record['resource_budget_seconds'],
        'storage_bytes_per_branch': 35000000000, 'disk_free_at_prepare': free,
        'measured_short_training_peak_allocated_bytes': proof['training_peak_allocated_bytes'],
        'measured_short_training_peak_reserved_bytes': proof['training_peak_reserved_bytes'],
        'boundary': 'Four real updates verify resource feasibility; startup/export cost prevents precise throughput extrapolation. Original e7672000 supplies10–15h reference; mixed-stage training may differ.'}
    check_inputs(record); require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    record['readiness']['cpu'] = 'passed'
    OUTPUT.mkdir(); write_json(OUTPUT / 'launch.prepared.json', record)
    print({'status': record['status'], 'path': str(OUTPUT / 'launch.prepared.json'), 'sha256': sha256(OUTPUT / 'launch.prepared.json'),
        'sources': len(record['source_sha256']), 'inputs': len(record['input_evidence_sha256']), 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    prepare()
