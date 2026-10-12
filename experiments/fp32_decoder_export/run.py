"""One bound, real QCE7200 FP32-storage export and full eight-task comparison.

The root supplies an existing launch_record only after native7200 and both BF16
control shards exist. This module never waits, queues, retries, or creates a
prepared record for a future checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import time
import traceback

from experiments.fp32_decoder_export.export import REPO, RUN_ROOT, SOURCE_FILES as EXPORT_SOURCES, inspect_scope
from experiments.longbench_v1.common import check_hashes, check_stat, sha256, write_json
from experiments.qwen3_w25.assemble_categories import _require

SOURCE_FILES = tuple(dict.fromkeys((*EXPORT_SOURCES,
    'experiments/fp32_decoder_export/run.py', 'experiments/w25_eval/acceptance.py',
    'experiments/w25_eval/model_inventory.py', 'experiments/qwen3_w25/summarize_results.py',
    'experiments/qwen3_w25/aux_step_export_callback.py',
    'experiments/qwen3_w25/step_export_callback.py', 'experiments/longbench_v1/common.py')))
SOURCE_ID = '552282f2-868a-4105-90df-46f3c6f6a818'


def read(path):
    return json.loads(Path(path).read_text())


def check(record):
    _require({str(REPO / name) for name in SOURCE_FILES} <= set(record['source_sha256']),
             'Bind this runner and its actual shared source dependencies.')
    check_hashes(record['source_sha256'])
    check_hashes(record['input_evidence_sha256'])
    for row in record['protected_weight_stats']:
        check_stat(row)


def stage_commands(*, checkpoint, bf16_checkpoint, output_dir):
    from experiments.qwen3_w25.aux_step_export_callback import stage_commands as original_commands
    output, native, control = (Path(p).resolve() for p in (output_dir, checkpoint, bf16_checkpoint))
    final = output / 'final_model'
    commands = [
        ('export', [sys.executable, '-m', 'experiments.fp32_decoder_export.export',
                    '--checkpoint_dir', str(native), '--output_dir', str(final)]),
        ('audit', [sys.executable, '-m', 'experiments.fp32_decoder_export.audit',
                   '--checkpoint_dir', str(final), '--training_checkpoint_dir', str(native),
                   '--bf16_checkpoint_dir', str(control), '--output', str(output / 'audit.json')]),
    ]
    # Reuse the actual fixed64/32 commands, excluding the historical69 summary.
    commands.extend((name, command) for name, command, _allowed in original_commands(
        checkpoint=native, source_dir=control, output_dir=output) if name in ('mmlu', 'seven'))
    return [{'name': name, 'argv': command} for name, command in commands]


def collect_auto_control(paths, architecture, *, checkpoint):
    """Read unchanged CE raw shards, proving their actual auto batch from logs."""
    from experiments.w25_eval.acceptance import TASK_METRICS, _group_agrees, _score, leaf_metrics
    _require(len(paths) == 2, 'Require both original control shards.')
    directory, meta = checkpoint
    scores, subjects, versions, sources = {}, {}, {}, []
    for path in paths:
        path = Path(path).resolve()
        payload = read(path)
        _require(Path(payload['checkpoint_dir']).resolve() == directory
                 and payload['checkpoint_id'] == meta['checkpoint_id']
                 and Path(payload['base_model_path']).resolve() == Path(architecture['base_model_path'])
                 and payload['checkpoint_loader'] == 'v6' and payload['strict_load'] is True
                 and payload.get('adapter_dir') is None and payload['eval_hif4_act'] is False
                 and payload['tokenizer_class'] == architecture['tokenizer_class']
                 and payload['tokenizer_is_fast'] is True, 'Original control model/tokenizer identity differs.')
        result = payload['evals']['lm_eval']
        _require(result['num_fewshot'] == 0 and 'limit' in result and result['limit'] is None
                 and result['batch_size'] == 'auto', 'This helper is only for original complete zero-shot auto control.')
        current, sub, batch = leaf_metrics(result['raw_results'], result['n_samples'], result['n_shot'], result['configs'])
        _group_agrees(current, result['raw_results'], result.get('group_results', {}))
        _require(not set(scores) & set(current) and len(result['tasks']) == len(current)
                 and set(result['tasks']) == set(result['task_metrics']) == set(result['task_metric_keys']) == set(current),
                 'Wrong or duplicate original control shard inventory.')
        for task, value in current.items():
            _require(result['task_metric_keys'][task] == TASK_METRICS[task]
                     and math.isclose(_score(result['task_metrics'][task], task), value, rel_tol=0, abs_tol=1e-12),
                     'Control summary differs from its original raw metrics.')
        log_path = path.with_suffix('.log')
        text = log_path.read_text().replace('\r', '\n')
        command = shlex.split(text.splitlines()[1])
        option = lambda name: command[command.index(name) + 1]
        _require('--strict' in command and option('--lm_batch_size') == 'auto'
                 and Path(option('--checkpoint_dir')).resolve() == directory
                 and Path(option('--output_json')).resolve() == path
                 and set(option('--tasks').split(',')) == set(current) and option('--num_fewshot') == '0'
                 and '--lm_limit' not in command
                 and re.findall(r'^Determined largest batch size: ([0-9]+)[ \t]*$', text, re.MULTILINE) == [str(batch)],
                 'Require the corresponding original log with one uniquely detected actual64/32 batch.')
        sources.append({'path': str(path), 'sha256': sha256(path), 'tasks': list(current),
            'leaf_count': len(result['n_samples']), 'sample_count': sum(v['effective'] for v in result['n_samples'].values()),
            'batch_size': batch, 'configured_batch_size': 'auto', 'format': 'cat_eval_v6',
            'actual_batch_evidence': {'path': str(log_path), 'sha256': sha256(log_path), 'unique_detected_batch_size': batch}})
        scores.update(current); subjects.update(sub)
        versions.update({task: result['versions'][task] for task in result['n_samples']})
    _require(set(scores) == set(TASK_METRICS) and len(subjects) == 57
             and sum(s['leaf_count'] for s in sources) == 64 and sum(s['sample_count'] for s in sources) == 24742,
             'Incomplete original control full8 coverage.')
    return {'task_metrics': scores, 'task_metric_keys': TASK_METRICS,
        'mean_percent': math.fsum(scores[t] for t in TASK_METRICS) * 100 / 8,
        'mmlu_subject_metrics': subjects, 'versions': versions, 'input_results': sources,
        'leaf_count': 64, 'sample_count': 24742}


def control_inputs(record):
    from experiments.w25_eval.acceptance import validate_checkpoint
    native = Path(record['training_checkpoint_dir']).resolve()
    meta, _state, _snapshot, source, base, architecture = inspect_scope(native)
    _require(base['checkpoint_id'] == record['source_checkpoint_id'] == SOURCE_ID
             and meta['checkpoint_id'] == record['native_checkpoint_id']
             and architecture == record['architecture'], 'Bound real7200 source/native architecture differs.')
    control = Path(record['bf16_checkpoint_dir']).resolve()
    _require(control == RUN_ROOT / 'step_7200/final_model', 'Require the original same7200 BF16 export.')
    paths = [control.parent / name for name in ('mmlu.json', 'seven.json')]
    required = [native / name for name in ('checkpoint_meta.json', 'trainer_state.json', 'training_model_state.pt')]
    required += [native.parent.parent / 'normalized_e2e_runtime_args.json', source / 'checkpoint_meta.json',
                 control / 'checkpoint_meta.json', control / 'export_record.json', control.parent / 'audit.json', *paths,
                 *(path.with_suffix('.log') for path in paths)]
    required += [control.parent / (name + '.exit') for name in ('export', 'audit', 'mmlu', 'seven')]
    _require({str(p) for p in required} <= set(record['input_evidence_sha256'])
             and all(p.is_file() for p in required), 'Actual native/control inputs must exist and be bound before launch.')
    index = read(Path(architecture['base_model_path']) / 'model.safetensors.index.json')
    weights = {str(source / 'pytorch_model.bin'), str(control / 'pytorch_model.bin')}
    weights.update(str(Path(architecture['base_model_path']) / name) for name in set(index['weight_map'].values()))
    _require(weights <= {row['path'] for row in record['protected_weight_stats']}, 'Bind actual source/control/HF weight identities.')
    for name in ('export', 'audit', 'mmlu', 'seven'):
        _require((control.parent / (name + '.exit')).read_text().strip() == '0', 'The BF16 control stage has not completed: ' + name)
    exported, audited = read(control / 'export_record.json'), read(control.parent / 'audit.json')
    control, control_meta, control_budget = validate_checkpoint(control, architecture)
    _require(exported['status'] == 'exported' and exported['source_global_step'] == 7200
             and Path(exported['source_checkpoint']).resolve() == native
             and exported['source_checkpoint_id'] == meta['checkpoint_id']
             and exported['checkpoint_id'] == control_meta['checkpoint_id'] == record['bf16_checkpoint_id'],
             'BF16 control is not an export of the identical native7200 state.')
    _require(audited['status'] == 'verified' and audited['checkpoint_id'] == control_meta['checkpoint_id']
             and audited['source_checkpoint_id'] == SOURCE_ID and audited['fresh_strict_reload_state_exact']
             and audited['frozen_bytes_unchanged'] and audited['changed_code_bits'] == 0
             and audited['reloaded_tensors_compared'] == 3423 and audited['rank_zero'], 'Original BF16 audit failed.')
    results = collect_auto_control(paths, architecture, checkpoint=(control, control_meta))
    return paths, results, control_budget


def summarize(record, control_paths, control_results, control_budget):
    from experiments.w25_eval.acceptance import collect_results, validate_checkpoint
    output = Path(record['output_dir']).resolve()
    model, meta, budget = validate_checkpoint(output / 'final_model', record['architecture'])
    audit = read(output / 'audit.json')
    _require(audit['status'] == 'verified' and audit['checkpoint_id'] == meta['checkpoint_id']
             and audit['bf16_checkpoint_id'] == record['bf16_checkpoint_id']
             and audit['selected_checkpoint_id'] == record['native_checkpoint_id']
             and audit['source_checkpoint_id'] == SOURCE_ID and audit['source_global_step'] == 7200
             and audit['fp32_native_decoder_tensors_exact'] == 1512
             and audit['frozen_source_tensors_exact'] == 1911 and audit['frozen_code_banks'] == 504
             and audit['fresh_strict_tensors_exact'] == 3423 and audit['same_native_bf16_cast_exact']
             and audit['persistent_state_after_inference_exact'] and not audit['optimizer_or_training_executed'],
             'The new precision export audit did not satisfy its real native/storage contract.')
    _require(budget == audit['bit_budget'] and control_budget == audit['bf16_bit_budget']
             and budget['budget_passed'] and budget['user_linear_bpw'] <= 2.5
             and budget['user_linear_bits'] == control_budget['user_linear_bits'] + 74133504
             and audit['added_storage_bits'] == 74133504 and audit['added_storage_bytes'] == 9266688,
             'Use actual FP32 storage with its measured integer increase, not an unchanged-budget assertion.')
    paths = [output / name for name in ('mmlu.json', 'seven.json')]
    current = collect_results(paths, record['architecture'], checkpoint=(model, meta))
    for actual_path, control_path in zip(paths, control_paths, strict=True):
        actual, reference = (read(p)['evals']['lm_eval'] for p in (actual_path, control_path))
        for field in ('configs', 'versions', 'n_shot', 'n_samples'):
            _require(actual[field] == reference[field], f'{actual_path.stem} actual/control {field} differs.')
    return {'status': 'full8_evaluation_verified', 'checkpoint_dir': str(model), 'checkpoint_id': meta['checkpoint_id'],
        'native_checkpoint_id': record['native_checkpoint_id'], 'source_global_step': 7200,
        'bf16_checkpoint_id': record['bf16_checkpoint_id'], **current,
        'bit_budget': budget, 'bf16_bit_budget': control_budget, 'audit': str(output / 'audit.json'),
        'audit_sha256': sha256(output / 'audit.json'), 'actual_configs_versions_shots_counts_equal_control': True,
        'control_results': control_results['input_results'], 'control_mean_percent': control_results['mean_percent'],
        'mean_delta_percentage_points': current['mean_percent'] - control_results['mean_percent'],
        'task_delta_percentage_points': {task: 100 * (value - control_results['task_metrics'][task])
                                         for task, value in current['task_metrics'].items()},
        'mmlu_65_passed': current['task_metrics']['mmlu'] >= .65,
        'longbench_status': 'not_evaluated', 'current_nine_item_goal_status': 'not_evaluated',
        'goal_acceptance': False,
        'scope': 'Same native7200 decoder values; FP32 storage versus BF16 storage using normal BF16/mixed-accumulation deployment. Full8 only; neither historical69 nor current nine-item acceptance.'}


def run(record_path):
    record_path = Path(record_path).resolve()
    record = read(record_path)
    output = Path(record['output_dir']).resolve()
    code = 1
    try:
        _require(record['status'] == 'bound_ready' and type(record['gpu']) is int
                 and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                 and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                 and Path.cwd().resolve() == Path(record['cwd']).resolve() == REPO
                 and not record['queued'] and not record['automatic_retry'], 'Use the root-bound one-shot environment.')
        _require(all(os.environ.get(k) == v for k, v in record['environment'].items())
                 and all(os.environ.get(k) == '1' for k in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'))
                 and os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True'
                 and not any(k in os.environ for k in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                 'Keep offline caches and the normal numerical/allocator policy.')
        _require(output.is_dir() and record_path.parent == output and not output.is_relative_to(RUN_ROOT)
                 and not any((output / name).exists() for name in ('final_model', 'audit.json', 'mmlu.json', 'seven.json',
                     'full8_summary.json', 'dispatcher.exit', 'export.log', 'audit.log', 'mmlu.log', 'seven.log', 'summary.exit')),
                 'Use a fresh isolated output; never overwrite a run or resume failed stages.')
        check(record)
        control_paths, control_results, control_budget = control_inputs(record)
        commands = stage_commands(checkpoint=record['training_checkpoint_dir'],
            bf16_checkpoint=record['bf16_checkpoint_dir'], output_dir=output)
        _require(commands == record['stage_commands'], 'Prepared actual commands changed.')
        record.update(status='running', worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            started_unix=time.time(), stages={}, actual_environment={k: os.environ.get(k) for k in record['environment']})
        write_json(record_path, record)
        from experiments.qwen3_w25.step_export_callback import run_logged_stage
        for row in commands:
            name = row['name']
            record['status'] = name
            record['stages'][name] = {'argv': row['argv'], 'started_unix': time.time()}
            write_json(record_path, record)
            try:
                run_logged_stage(row['argv'], name=name, output_dir=output, env=os.environ.copy(), allowed_exit_codes=(0,))
            finally:
                path = output / (name + '.exit')
                if path.exists():
                    value = path.read_text().strip()
                    record['stages'][name]['exit_code'] = int(value) if value.lstrip('-').isdigit() else value
                record['stages'][name]['finished_unix'] = time.time()
                write_json(record_path, record)
        record['status'] = 'summary'
        record['stages']['summary'] = {'started_unix': time.time(), 'entry': 'experiments.fp32_decoder_export.run.summarize'}
        write_json(record_path, record)
        summary_code = 1
        try:
            summary = summarize(record, control_paths, control_results, control_budget)
            check(record)
            write_json(output / 'full8_summary.json', summary)
            summary_code = 0
        finally:
            (output / 'summary.exit').write_text(str(summary_code) + '\n')
            record['stages']['summary'].update(exit_code=summary_code, finished_unix=time.time())
        record.update(status='completed', checkpoint_id=summary['checkpoint_id'], final_model=summary['checkpoint_dir'],
            full8_summary=str(output / 'full8_summary.json'), full8_summary_sha256=sha256(output / 'full8_summary.json'),
            mean_percent=summary['mean_percent'], mmlu_percent=100 * summary['task_metrics']['mmlu'],
            user_linear_bpw=summary['bit_budget']['user_linear_bpw'], longbench_status='not_evaluated',
            current_nine_item_goal_status='not_evaluated', current_nine_item_goal_passed=False)
        code = 0
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        print(record['traceback'], flush=True)
    finally:
        record.update(dispatcher_exit_code=code, exit_code=code, finished_unix=time.time())
        write_json(record_path, record)
        (output / 'dispatcher.exit').write_text(str(code) + '\n')
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch_record', required=True, type=Path)
    sys.exit(run(parser.parse_args().launch_record))
