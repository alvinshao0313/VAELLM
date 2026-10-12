"""Deploy and evaluate the completed four-gate fixed-code endpoint once."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

from experiments.gate_student_context_evaluation.run import check, read
from experiments.gate_down_student_context_evaluation.run import compare_control
from experiments.longbench_v1.common import sha256, write_json
from experiments.long_context_recovery.contract import require


TARGETS = [f'model.layers.{layer}.mlp.gate_proj' for layer in (0, 9, 18, 27)]


def run(record_path):
    record_path = Path(record_path).resolve()
    record = read(record_path)
    output = Path(record['output_dir']).resolve()
    code = 1
    try:
        require(record['status'] == 'bound_ready' and type(record['gpu']) is int
                and os.environ.get('CUDA_VISIBLE_DEVICES') == str(record['gpu'])
                and Path(sys.executable).resolve() == Path(record['python_executable']).resolve()
                and Path.cwd().resolve() == Path(record['cwd']).resolve()
                and not record['queued'] and not record['automatic_retry'],
                'Use the bound GPU, repository and bitvae interpreter without retry or queue.')
        require(all(os.environ.get(key) == value for key, value in record['environment'].items())
                and all(os.environ.get(key) == '1' for key in
                        ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'))
                and os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True'
                and not any(key in os.environ for key in ('FLASH_ATTENTION_DETERMINISTIC', 'NVIDIA_TF32_OVERRIDE')),
                'Keep the prepared offline, allocator and numerical environment.')
        require(record['arm'] == 'four_gates' and record['selected_targets'] == TARGETS
                and record['patched_local_tensor_count'] == 52
                and record['frozen_outside_target_tensors'] == 3371
                and record['selected_decoder_tensors'] == 24,
                'Only the four preselected complete native gates are mutable.')
        require(record_path.parent == output and not any((output / name).exists() for name in
                        ('final_model', 'audit.json', 'mmlu.json', 'seven.json', 'full8_summary.json', 'dispatcher.exit')),
                'Use a fresh output; never overwrite deployment or evaluation.')
        commands = record['stage_commands']
        require([row['name'] for row in commands] == ['deploy', 'mmlu', 'seven'], 'Unexpected stage inventory.')
        check(record)
        record.update(status='running', worker_pid=os.getpid(),
            worker_start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            started_unix=time.time(), stages={},
            actual_environment={key: os.environ.get(key) for key in record['environment']})
        write_json(record_path, record)
        from experiments.qwen3_w25.step_export_callback import run_logged_stage
        for row in commands:
            name, command = row['name'], row['argv']
            require(Path(command[0]).resolve() == Path(sys.executable).resolve()
                    and '--limit' not in command and '--lm_limit' not in command,
                    'Do not change the interpreter or limit evaluation.')
            record['status'] = name
            record['stages'][name] = {'argv': command, 'started_unix': time.time()}
            write_json(record_path, record)
            try:
                status = run_logged_stage(command, name=name, output_dir=output,
                                          env=os.environ.copy(), allowed_exit_codes=(0,))
            finally:
                exit_path = output / (name + '.exit')
                if exit_path.exists():
                    value = exit_path.read_text().strip()
                    record['stages'][name]['exit_code'] = int(value) if value.lstrip('-').isdigit() else value
                record['stages'][name]['finished_unix'] = time.time()
                write_json(record_path, record)
            require(status == 0, 'Each prepared stage must finish successfully.')
        from experiments.w25_eval.acceptance import collect_results, validate_checkpoint
        from experiments.gate_student_context_evaluation.prepare import budget_equal
        model, meta, budget = validate_checkpoint(output / 'final_model', record['architecture'])
        audit = read(output / 'audit.json')
        require(audit['status'] == 'verified' and audit['source_checkpoint_id'] == record['source_checkpoint_id']
                and audit['checkpoint_id'] == meta['checkpoint_id'] != record['source_checkpoint_id']
                and audit['endpoint_sha256'] == record['endpoint_sha256']
                and audit['selected_targets'] == TARGETS
                and audit['patched_local_tensor_count'] == 52
                and audit['frozen_outside_target_tensors'] == 3371
                and audit['selected_decoder_tensors'] == 24
                and audit['serialized_tensors_compared'] == audit['reloaded_tensors_compared'] == 3423
                and audit['fresh_strict_reload_state_exact'] and audit['endpoint_state_exact']
                and audit['frozen_outside_target_exact'] and audit['native_full_weight_reload_exact']
                and audit['source_packed_bank_count'] == 504 and audit['all_source_packed_banks_exact']
                and audit['changed_code_bits'] == 0 and audit['integer_budget_unchanged'] and audit['rank_zero'],
                'Independent deployment audit failed.')
        budget_equal(budget, record['source_bit_budget'])
        paths = [output / name for name in ('mmlu.json', 'seven.json')]
        current = collect_results(paths, record['architecture'], checkpoint=(model, meta))
        require(current['leaf_count'] == 64 and current['sample_count'] == 24742, 'Incomplete eight-task evaluation.')
        source = Path(record['source_checkpoint']).resolve()
        source_meta = read(source / 'checkpoint_meta.json')
        require(source_meta['checkpoint_id'] == record['source_checkpoint_id'], 'Original control identity changed.')
        control = compare_control(paths, record['control_result_paths'], record['architecture'], checkpoint=(source, source_meta))
        gate_control = compare_control(paths, record['gate_only_control_results'], record['architecture'])
        check(record)
        summary = {'status': 'full8_evaluation_verified', 'arm': record['arm'], 'checkpoint_dir': str(model),
            'checkpoint_id': meta['checkpoint_id'], 'source_checkpoint_id': record['source_checkpoint_id'],
            'endpoint_sha256': record['endpoint_sha256'], 'selected_targets': TARGETS,
            **current, 'bit_budget': budget, 'audit': str(output / 'audit.json'),
            'audit_sha256': sha256(output / 'audit.json'), 'actual_configs_versions_shots_counts_equal_both_controls': True,
            'control_results': control['input_results'], 'control_mean_percent': control['mean_percent'],
            'mean_delta_percentage_points': current['mean_percent'] - control['mean_percent'],
            'task_delta_percentage_points': {task: 100 * (value - control['task_metrics'][task])
                                             for task, value in current['task_metrics'].items()},
            'gate_only_control_results': gate_control['input_results'], 'gate_only_control_mean_percent': gate_control['mean_percent'],
            'mean_delta_vs_gate_only_percentage_points': current['mean_percent'] - gate_control['mean_percent'],
            'task_delta_vs_gate_only_percentage_points': {task: 100 * (value - gate_control['task_metrics'][task])
                                                         for task, value in current['task_metrics'].items()},
            'longbench_status': 'not_evaluated', 'current_nine_item_goal_status': 'not_evaluated',
            'goal_acceptance': False, 'scope': record['protocol']['boundary']}
        write_json(output / 'full8_summary.json', summary)
        record.update(status='completed', checkpoint_id=meta['checkpoint_id'], final_model=str(model),
            mean_percent=current['mean_percent'], mmlu_percent=100 * current['task_metrics']['mmlu'],
            full8_summary=str(output / 'full8_summary.json'), full8_summary_sha256=sha256(output / 'full8_summary.json'),
            audit_sha256=sha256(output / 'audit.json'), longbench_status='not_evaluated',
            current_nine_item_goal_status='not_evaluated', current_nine_item_goal_passed=False)
        code = 0
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
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
