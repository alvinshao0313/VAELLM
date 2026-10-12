"""CPU preparation for the fresh f927 versus43b0 QK-only source comparison."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys

import torch

from experiments.llama_ce_reproduction.callback import stage_commands
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require
from experiments.qwen_f927_qkonly_lm.train import (
    ROOT, R2, SOURCE, SOURCE_ID, CONTROL, OUTPUT, F927QKCallback, QKStepCallback,
    SAVE_STEPS, check_inputs, read, configuration, training_arguments)
from train_utils.runtime_snapshot import to_runtime_jsonable


def prepare():
    require(Path.cwd() == ROOT and Path(__file__).resolve().is_relative_to(ROOT)
            and os.environ.get('PYTHONPATH') == str(ROOT) and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and not torch.cuda.is_initialized() and not OUTPUT.exists(), 'Use CPU-only S and a fresh output.')
    control = read(CONTROL / 'launch.prepared.json')
    completed, trained = (read(CONTROL / n) for n in ('launch.json', 'training.json'))
    review = read(CONTROL / 'step_2000/independent_review.json')
    require(completed['status'] == trained['status'] == 'completed' and completed['exit_code'] == 0
            and trained['callback_completed_steps'] == SAVE_STEPS and trained['training_result']['global_step'] == 2000
            and trained['source_checkpoint_id'] == control['source_checkpoint_id'] == '43b0c5a2-4ef7-4a00-a6d1-ed887be18d01'
            and review['checkpoint_id'] == '228aa08f-9abf-4421-b907-073629e1045e'
            and review['source_checkpoint_id'] == control['source_checkpoint_id'], 'Require the completed matched source arm.')
    cached = R2 / 'qwen_qk_longalpaca8k_validate_s4'
    proof, bound, short = (read(cached / n) for n in ('verification.json', 'launch.json', 'training.json'))
    require(proof['status'] == bound['status'] == 'verified' and bound['dispatcher_exit_code'] == 0
            and proof['source_checkpoint_id'] == SOURCE_ID and proof['native_vs_independent_tensors_compared'] == 3423
            and short['selected_targets'] == control['selected_targets']
            and bound['expected_inventory'] == short['expected_inventory'] == control['expected_inventory'],
            'Reuse the actual f927 QK72/cache/independent-export proof, never its updated weights.')
    budget_path = R2 / 'nine_task_baseline_1200/acceptance.json'
    budget = read(budget_path)['bit_budget']
    require(budget['checkpoint_id'] == SOURCE_ID and budget['budget_passed']
            and all(budget[k] == control['source_bit_budget'][k] for k in
                    ('user_linear_bits', 'packed_linear_bits', 'original_linear_weights', 'components')),
            'Source storage differs from the matched arm.')
    record = deepcopy(control)
    for key in ('control_launch', 'resource_validation', 'added_native_entry_copy', 'matched_all_decoder_prepared'):
        record.pop(key, None)
    record.update(schema='qwen_f927_qkonly_lm_2000_v1', status='prepared_unbound', gpu=None, export_gpu=None,
        output_dir=str(OUTPUT), source_checkpoint=str(SOURCE), source_checkpoint_id=SOURCE_ID,
        source_weight_stat=bound['source_weight_stat'], source_bit_budget=budget,
        matched_qkonly_control=str(CONTROL), matched_all_decoder_output=str(R2 / 'general_lm_recovery_s2000/qwen3'),
        training_argv=training_arguments(OUTPUT), argv=[sys.executable, '-u', '-m',
            'experiments.qwen_f927_qkonly_lm.train', '--launch_record', str(OUTPUT / 'launch.json')],
        comparison_boundary='Fresh f927 versus43b0 QK-only: native training differs only in source/output. The source difference includes re-CAT QK codes/decoder and rotation; it is not a rotation-only comparison.',
        scientific_boundary='Eight downstream tasks are observations, with no standalone MMLU/eight-task quality threshold. Goal acceptance requires official full21 LongBench and the nine-item aggregate.',
        validation_reuse='Actual f927 QK/cache/save proof plus completed43b0 same-data/B4/1024/2000 five synchronous exports. No new GPU diagnostic or updated short-run weights.',
        readiness={'cpu': 'pending', 'gpu': 'not_scheduled', 'f927_qk_cache_export': 'verified_reuse',
                   'same_recipe_five_callbacks': 'completed_reuse'})
    # Keep only data/provenance inputs and the exact source/comparison execution evidence.
    evidence = [Path(p) for p in control['input_evidence_sha256'] if '/data/fineweb_edu_' in p]
    evidence += [CONTROL / n for n in ('launch.prepared.json', 'launch.json', 'training.json', 'actual_scope.json',
        'run.exit', 'outer.exit', 'step_2000/independent_review.json')]
    evidence += [cached / n for n in ('verification.json', 'launch.json', 'training.json',
        'training.exit', 'export.exit', 'audit.exit', 'verify.exit', 'dispatcher.exit', 'outer.exit')]
    evidence += [SOURCE / n for n in ('checkpoint_meta.json', 'config.json')]
    evidence += [Path(p) for p in record['control_result_paths']] + [budget_path]
    for path in evidence:
        if path.suffix == '.exit':
            require(path.read_text().strip() == '0', 'Required terminal stage failed: ' + str(path))
    record['input_evidence_sha256'] = {str(p): sha256(p) for p in evidence}
    for path in sorted(Path(__file__).parent.glob('*.py')):
        ast.parse(path.read_text())
        record['source_sha256'][str(path)] = sha256(path)
    require(F927QKCallback.on_save is QKStepCallback.on_save, 'Reuse the full native callback unchanged.')
    record['estimate'] = {'storage_bytes_per_branch': 35000000000, 'disk_free_at_prepare': shutil.disk_usage(R2).free,
        'matched_control_elapsed_seconds': completed['finished_unix'] - completed['started_unix'],
        'matched_control_peak_allocated_bytes': trained['peak_allocated_bytes'],
        'matched_control_peak_reserved_bytes': trained['peak_reserved_bytes'],
        'boundary': 'Completed43b0 runtime/peaks are measured references, not a bound on f927 throughput. Same persistent geometry, five4.63GB exports, one native final and save_total_limit1; 35GB retained branch reserve. Training/export GPUs must remain distinct.'}
    cfg, hf, args = configuration(record)
    record['cpu_configuration'] = {'canonical_config': to_runtime_jsonable(cfg), 'hf_args': to_runtime_jsonable(hf),
        'training_args': to_runtime_jsonable(args), 'canonical_difference_from_matched_control':
            ['student_checkpoint_dir', 'run_root_dir']}
    record['stage_templates'] = [{'name': n, 'argv': a, 'allowed_exit_codes': e}
        for n, a, e in stage_commands(checkpoint=OUTPUT / 'NATIVE_RUN/trainer_state/checkpoint-400',
            source_dir=SOURCE, output_dir=OUTPUT / 'step_0400')]
    check_inputs(record)
    require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA.')
    record['readiness']['cpu'] = 'passed'
    OUTPUT.mkdir()
    write_json(OUTPUT / 'launch.prepared.json', record)
    print({'status': 'prepared_unbound', 'path': str(OUTPUT / 'launch.prepared.json'),
        'sha256': sha256(OUTPUT / 'launch.prepared.json'), 'sources': len(record['source_sha256']),
        'inputs': len(record['input_evidence_sha256']), 'cuda_initialized': False}, flush=True)


if __name__ == '__main__':
    prepare()
