"""Evaluate one reviewed sameKL500 endpoint, or compare the three completed arms on CPU."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

from experiments.full_rht_gate_pair.compare import delta, score
from experiments.gate_down_student_context_evaluation.run import compare_control
from experiments.gate_same_kl_decoder_evaluation.prepare import EXPECTED_XOR, budget_equal
from experiments.gate_student_context_evaluation.run import check, read
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require


def audit_contract(record, audit, meta):
    require(audit['status'] == 'verified' and audit['source_checkpoint_id'] == record['source_checkpoint_id']
            and audit['checkpoint_id'] == meta['checkpoint_id'] != record['source_checkpoint_id']
            and audit['endpoint_sha256'] == record['endpoint_sha256']
            and all(audit[key] == value for key, value in record['patch_contract'].items())
            and audit['patched_local_tensor_count'] == 13 and audit['frozen_outside_target_tensors'] == 3410
            and audit['selected_decoder_tensors'] == 6
            and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == 3423
            and audit['fresh_strict_reload_state_exact'] and audit['endpoint_state_exact']
            and audit['frozen_outside_target_exact'] and audit['native_full_weight_reload_exact']
            and audit['frozen_outside_target_packed_banks_exact']
            and audit['frozen_local_nonmutable_exact'] and audit['frozen_local_nonmutable_tensors'] == 5
            and audit['stage_changed_bits'] == EXPECTED_XOR[record['arm']]
            and audit['changed_code_bits'] == sum(EXPECTED_XOR[record['arm']])
            and audit['integer_budget_unchanged'] and audit['rank_zero'], 'Independent native deployment contract failed.')
    budget_equal(audit['bit_budget'], record['source_bit_budget'])


def run(record_path):
    record_path = Path(record_path).resolve(); record = read(record_path)
    output = Path(record['output_dir']).resolve(); code = 1
    try:
        require(record['status'] == 'bound_ready' and type(record['gpu']) is int
                and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                and Path.cwd().resolve() == Path(record['cwd']).resolve()
                and not record['queued'] and not record['automatic_retry'],
                'Use the explicitly bound GPU, repository and bitvae without retry or queue.')
        require(all(os.environ.get(key) == value for key, value in record['environment'].items())
                and all(os.environ.get(key) == '1' for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'))
                and os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True'
                and not any(key in os.environ for key in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                'Keep the prepared offline, allocator and numerical environment.')
        require(record['arm'] in EXPECTED_XOR and record['selected_projections'] == ['gate_proj']
                and record['timeout_seconds'] == 900 and record['termination_grace_seconds'] == 20,
                'Require one predeclared arm and its900+20 outer resource contract.')
        require(not any((output / name).exists() for name in
                ('final_model', 'audit.json', 'mmlu.json', 'seven.json', 'full8_summary.json', 'dispatcher.exit')),
                'Use a fresh output; never overwrite deployment or evaluation.')
        commands = record['stage_commands']
        require([row['name'] for row in commands] == ['deploy', 'mmlu', 'seven'], 'Unexpected stage inventory.')
        check(record)
        record.update(status='running', worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            started_unix=time.time(), stages={}, actual_environment={key: os.environ.get(key) for key in record['environment']})
        write_json(record_path, record)
        from experiments.qwen3_w25.step_export_callback import run_logged_stage
        for row in commands:
            name, command = row['name'], row['argv']
            require(Path(command[0]).resolve() == Path(sys.executable).resolve()
                    and '--limit' not in command and '--lm_limit' not in command,
                    'Do not change the interpreter or limit evaluation.')
            record['status'] = name; record['stages'][name] = {'argv': command, 'started_unix': time.time()}
            write_json(record_path, record)
            try:
                status = run_logged_stage(command, name=name, output_dir=output, env=os.environ.copy(), allowed_exit_codes=(0,))
            finally:
                exit_path = output / (name + '.exit')
                if exit_path.exists(): record['stages'][name]['exit_code'] = int(exit_path.read_text().strip())
                record['stages'][name]['finished_unix'] = time.time(); write_json(record_path, record)
            require(status == 0, 'Each prepared stage must complete successfully.')
        from experiments.w25_eval.acceptance import validate_checkpoint
        model, meta, budget = validate_checkpoint(output / 'final_model', record['architecture'])
        audit = read(output / 'audit.json'); audit_contract(record, audit, meta)
        budget_equal(budget, record['source_bit_budget'])
        paths = [output / name for name in ('mmlu.json', 'seven.json')]
        current, _ = score(paths, record['architecture'], model)
        source = Path(record['source_checkpoint']).resolve(); source_meta = read(source / 'checkpoint_meta.json')
        require(source_meta['checkpoint_id'] == record['source_checkpoint_id'], 'Original control identity changed.')
        control = compare_control(paths, record['control_result_paths'], record['architecture'], checkpoint=(source, source_meta))
        check(record)
        summary = {'status': 'full8_evaluation_verified', 'arm': record['arm'], 'checkpoint_dir': str(model),
            'source_checkpoint_id': record['source_checkpoint_id'], 'endpoint_sha256': record['endpoint_sha256'],
            **current, 'bit_budget': budget, 'audit': str(output / 'audit.json'), 'audit_sha256': sha256(output / 'audit.json'),
            'actual_changed_decoder_tensors': audit['actual_changed_decoder_tensors'],
            'actual_configs_versions_shots_counts_equal_source': True,
            'control_results': control['input_results'], 'control_mean_percent': control['mean_percent'],
            'mean_delta_percentage_points': current['mean_percent'] - control['mean_percent'],
            'task_delta_percentage_points': {task: 100 * (value - control['task_metrics'][task]) for task, value in current['task_metrics'].items()},
            'primary_control': record['primary_control'], 'paired_comparison': record['paired_comparison'],
            'longbench_status': 'not_evaluated', 'current_nine_item_goal_status': 'not_evaluated',
            'goal_acceptance': False, 'scope': record['protocol']['boundary']}
        write_json(output / 'full8_summary.json', summary)
        record.update(status='completed', checkpoint_id=meta['checkpoint_id'], final_model=str(model),
            mean_percent=current['mean_percent'], mmlu_percent=100 * current['task_metrics']['mmlu'],
            mmlu_correct=current['integer_counts']['mmlu']['correct'],
            full8_summary=str(output / 'full8_summary.json'), full8_summary_sha256=sha256(output / 'full8_summary.json'),
            audit_sha256=sha256(output / 'audit.json'), longbench_status='not_evaluated', current_nine_item_goal_status='not_evaluated')
        code = 0
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(record['traceback'], flush=True)
    finally:
        record.update(dispatcher_exit_code=code, exit_code=code, finished_unix=time.time()); write_json(record_path, record)
        (output / 'dispatcher.exit').write_text(str(code) + '\n')
    return code


def completed(directory):
    directory = Path(directory).resolve(); record = read(directory / 'launch.json')
    require(record['arm'] in EXPECTED_XOR and record['status'] == 'completed' and record['exit_code'] == 0
            and Path(record['output_dir']).resolve() == directory, 'Compare only a real completed arm.')
    for name in ('deploy', 'mmlu', 'seven', 'dispatcher', 'outer'):
        require((directory / (name + '.exit')).read_text().strip() == '0', 'Incomplete or failed stage: ' + name)
        if name in record['stages']: require(record['stages'][name]['exit_code'] == 0, 'Stage record differs: ' + name)
    check(record)
    model = directory / 'final_model'; meta = read(model / 'checkpoint_meta.json')
    audit = read(directory / 'audit.json'); audit_contract(record, audit, meta)
    result, shards = score([directory / name for name in ('mmlu.json', 'seven.json')], record['architecture'], model)
    summary = read(directory / 'full8_summary.json')
    require(record['checkpoint_id'] == summary['checkpoint_id'] == meta['checkpoint_id']
            and summary['status'] == 'full8_evaluation_verified'
            and summary['mean_percent'] == record['mean_percent'] == result['mean_percent']
            and summary['task_metrics'] == result['task_metrics'] and summary['integer_counts'] == result['integer_counts']
            and record['full8_summary_sha256'] == sha256(directory / 'full8_summary.json')
            and record['audit_sha256'] == summary['audit_sha256'] == sha256(directory / 'audit.json'),
            'Stored complete result or audit differs from actual native raw output.')
    evidence = [directory / name for name in ('launch.prepared.json', 'launch.bound.json', 'launch.json', 'audit.json',
        'full8_summary.json', 'mmlu.json', 'seven.json', 'deploy.exit', 'mmlu.exit', 'seven.exit', 'dispatcher.exit', 'outer.exit')]
    evidence.append(model / 'checkpoint_meta.json')
    return record, result, shards, {str(path): sha256(path) for path in evidence}


def compare(directories, output):
    import torch
    output = Path(output).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
            and not output.exists() and not output.is_symlink(), 'Use CPU-only comparison with a fresh output.')
    arms, records, raw, evidence = {}, {}, {}, {}
    for directory in directories:
        record, result, shards, files = completed(directory); label = record['arm']
        require(label not in arms, 'Duplicate comparison arm.')
        arms[label], records[label], raw[label] = result, record, shards; evidence.update(files)
    require(set(arms) == set(EXPECTED_XOR), 'Compare all three completed endpoints, without selecting by loss or quality.')
    reference = records['original']; original = arms['original']
    for label, record in records.items():
        for key in ('source_checkpoint', 'source_checkpoint_id', 'source_sha256', 'source_weight_stat', 'hf_weight_stats',
                    'architecture', 'source_native_policy', 'environment', 'protocol', 'training_contract', 'control_result_paths'):
            require(record[key] == reference[key], 'SameKL500 pairing differs: ' + label + '/' + key)
        budget_equal(record['source_bit_budget'], reference['source_bit_budget'])
        for actual, control in zip(raw[label], raw['original'], strict=True):
            for key in ('configs', 'versions', 'n_shot', 'n_samples'):
                require(actual[key] == control[key], 'Paired raw protocol differs: ' + label + '/' + key)
    source, _ = score([Path(path) for path in reference['control_result_paths']], reference['architecture'], Path(reference['source_checkpoint']))
    for path in reference['control_result_paths']: evidence[path] = sha256(path)
    source_meta = Path(reference['source_checkpoint']) / 'checkpoint_meta.json'; evidence[str(source_meta)] = sha256(source_meta)
    report = {'status': 'verified_three_arm_comparison', 'primary_control': 'original_same_final_KL500',
        'primary_control_checkpoint_id': original['checkpoint_id'], 'arms': arms, 'source': source,
        'new_codes_minus_original_sameKL500': {arm: delta(arms[arm], original) for arm in EXPECTED_XOR if arm != 'original'},
        'secondary_vs_fresh_f927': {arm: delta(result, source) for arm, result in arms.items()},
        'paired_training_contract': reference['training_contract'], 'same_raw_protocol': True,
        'leaf_count': 64, 'sample_count': 24742, 'mmlu_subjects': 57, 'mmlu_samples': 14042,
        'actual_batches': {'mmlu': 64, 'seven': 32}, 'fewshot': 0, 'limit': None,
        'evidence_sha256': evidence, 'comparison_source_sha256': sha256(Path(__file__)), 'cuda_initialized': False,
        'quality_boundary': 'All three reviewed500 endpoints are evaluated. Valid negative results succeed. Old conditional-MLP500 is a different objective and is not the matched control.',
        'longbench_status': 'not_evaluated', 'nine_item_goal_status': 'not_evaluated'}
    require(not torch.cuda.is_initialized(), 'Comparison initialized CUDA.'); write_json(output, report)
    print({'status': report['status'], 'output': str(output), 'sha256': sha256(output)}, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--launch_record', type=Path); mode.add_argument('--compare', nargs=3, type=Path)
    parser.add_argument('--output', type=Path); args = parser.parse_args()
    if args.compare:
        if args.output is None: parser.error('--compare requires --output')
        compare(args.compare, args.output)
    else:
        if args.output is not None: parser.error('--output is only for --compare')
        sys.exit(run(args.launch_record))
