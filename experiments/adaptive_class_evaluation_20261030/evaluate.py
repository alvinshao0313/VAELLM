"""One completed whole-class endpoint: complete full8, full21 and current acceptance."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.adaptive_class_evaluation_20261030 import bindings as b
from experiments.adaptive_class_evaluation_20261030.summarize import full8_summary
from experiments.longbench_v1.data import load_archive
from experiments.longbench_v1.protocol import DATASETS, EXPECTED_COUNTS


def canonical():
    return b.module(b.ROOT / 'experiments/cat_aligned_evaluation_20261028/evaluate.py', '_adaptive_canonical_partitions')


def make_plan(config, config_path, binding, output, gpus, pins):
    api = canonical()
    units = []
    for index, (gpu, tasks) in enumerate(zip(gpus, api.task_groups(gpus), strict=True)):
        group = output / 'longbench/groups' / f'part_{index + 1:02d}'
        argv = b.command('predict',
            '--config', str(config_path), '--output', str(group), '--datasets', *tasks,
            '--max-runtime-seconds', '86400')
        units.append(dict(output_dir=str(group), datasets=tasks, expected_samples=sum(EXPECTED_COUNTS[task] for task in tasks),
            native_reports=[], outer_exit=None, launch_records=[], terminal_attempts=[], gpu=gpu, argv=argv,
            native_runtime_seconds=86400, outer_timeout_seconds=86580, termination_grace_seconds=20))
    merged = output / 'longbench/merged'
    nine = b.command('summarize',
        '--checkpoint', binding['checkpoint_dir'], '--pipeline-report', binding['pipeline_report'],
        '--compressed-results', str(output / 'full8/mmlu.json'), str(output / 'full8/seven.json'),
        '--compressed-longbench-scores', str(merged / 'scores.json'), '--output', str(merged / 'nine_task_acceptance.json'))
    template = b.read(b.TEMPLATE)
    return dict(schema='fixed_full21_merge_plan_v1', status='draft_awaiting_explicit_terminal_binding',
        checkpoint_id=binding['checkpoint_id'], checkpoint_format=b.FORMAT, expected_tasks=list(DATASETS), expected_samples=4750,
        config_path=str(config_path), config_sha256=b.sha256(config_path), protocol=config['protocol'], execution_units=units,
        task_file_sources={task: str(Path(unit['output_dir']) / 'predictions' / (task + '.jsonl')) for unit in units for task in unit['datasets']},
        pins={**pins, str(config_path): b.sha256(config_path)}, merged_output=str(merged), scoring_cwd=str(b.ROOT),
        python_executable=str(b.PYTHON), nine_task_argv=nine,
        scoring_environment=dict(PYTHONPATH=str(b.ROOT), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'),
        acceptance=dict(max_linear_bpw=2.6, max_lora_rank=0, maximum_absolute_drop_points=.9, strict_score_drop_bound=True),
        nine_task_exit_semantics=template['nine_task_exit_semantics'], automatic_retry=False, automatic_resume=False,
        partial_scoring=False, boundary='Actual mixed v2 same-UUID complete full8+21; no dense export or v6 native-audit substitution.')


def evaluate(args):
    b.cpu_environment()
    binding, meta = b.source(args.checkpoint, args.pipeline_report)
    pins = b.source_pins(binding)
    config = b.make_config(binding, pins)
    manifest, rows = load_archive(config['archive']['path'], check_token_file=False)
    b.require(len(rows) == 4750 and manifest['data']['counts'] == EXPECTED_COUNTS
        and all(row['input_tokens'] + row['max_new_tokens'] <= 32768 for row in rows), 'Canonical21/4750 archive/window differs.')
    if args.check_only:
        print(json.dumps(dict(status='actual_mixed_endpoint_bound', checkpoint_id=meta['checkpoint_id'],
            checkpoint_format=b.FORMAT, bit_budget=binding['bit_budget'])), flush=True)
        return binding
    output = args.output.resolve()
    b.require(not os.path.lexists(output) and output.is_relative_to(b.TOPIC)
        and not output.is_relative_to(Path(binding['checkpoint_dir'])), 'Use a new evaluation directory under the topic result root.')
    resources = canonical().check_resources(args.gpus)
    b.check_hashes(pins)
    output.mkdir(parents=True)
    report_path = output / 'evaluation.json'
    record = dict(status='running', **binding, source_sha256=pins, gpus=args.gpus, pid=os.getpid(),
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        started_unix=time.time(), argv=getattr(b, 'CLI_ARGUMENTS', sys.argv), initial_GPU_resource_check=resources,
        automatic_retry=False, automatic_resume=False, stages={})
    b.write_json(report_path, record)
    api = b.module(b.HELPERS, '_adaptive_existing_full9_execution')
    api.S = b.ROOT  # Start the new module in this workspace; its package selects its own exact native runtime.
    runner = b.module(b.ROOT / 'experiments/qwen3_w25/step_export_callback.py', '_adaptive_logged_stage').run_logged_stage

    def stage(name, argv, gpu='', seconds=1800):
        command = ['timeout', '--signal=TERM', '--kill-after=20s', str(seconds) + 's', *argv]
        record['stages'][name] = dict(command=command, gpu=gpu, started_unix=time.time())
        b.write_json(report_path, record)
        try:
            return runner(command, name=name, output_dir=output, env=api.environment(b.ROOT, gpu))
        finally:
            exit_path = output / (name + '.exit')
            record['stages'][name].update(finished_unix=time.time(), exit_code=exit_path.read_text().strip() if exit_path.exists() else None)
            b.write_json(report_path, record)

    try:
        full8 = output / 'full8'
        full8.mkdir()
        full8_config = full8 / 'config.json'
        b.write_json(full8_config, config)
        for shard in ('mmlu', 'seven'):
            stage(shard, b.command('full8', '--config', str(full8_config), '--shard', shard,
                '--output', str(full8 / (shard + '.json'))), args.gpus[0], 3600)
        paths = [full8 / (shard + '.json') for shard in ('mmlu', 'seven')]
        summary = full8_summary(config, paths, full8 / 'full8_summary.json')
        pins.update(summary['sample_artifacts']['sha256'])
        pins.update({str(path): b.sha256(path) for path in (*paths, full8 / 'full8_summary.json')})
        record.update(full8_mean_percent=summary['mean_percent'], full8_sample_count=24742, full8_leaf_tasks=64)
        lb = output / 'longbench'
        lb.mkdir()
        config_path, plan_path = lb / 'config.json', lb / 'merge_plan.json'
        b.write_json(config_path, config)
        plan = make_plan(config, config_path, binding, output, args.gpus, pins)
        plan['actual_GPU_resource_check'] = canonical().check_resources(args.gpus)
        b.write_json(plan_path, plan)
        b.check_hashes(plan['pins']); b.check_config(config)
        processes = []
        try:
            for unit in plan['execution_units']:
                Path(unit['output_dir']).mkdir(parents=True)
                processes.append(api.start_group(unit, config, plan['config_sha256']))
        finally:
            codes = [process.wait() for process in processes]
            record['longbench_outer_exit_codes'] = codes
            b.write_json(report_path, record)
        b.require(len(codes) == len(args.gpus) and all(code == 0 for code in codes), 'A partition failed; no retry/partial scoring.')
        for index, unit in enumerate(plan['execution_units']):
            group = Path(unit['output_dir'])
            reports = list((group / 'runs').glob('*.json'))
            b.require(len(reports) == 1, 'One fresh actual inference attempt per partition is required.')
            actual = b.read(reports[0])
            audit = actual['load_audit']
            b.require(actual['status'] == 'partition_complete' and actual['exit_code'] == 0
                and actual['already_present'] == 0 and actual['new_predictions'] == unit['expected_samples']
                and audit['checkpoint_format'] == b.FORMAT and audit['persistent_tensors'] == len(meta['tensor_table'])
                and audit['checkpoint_files_sha256'] == binding['checkpoint_files_sha256'], 'Incomplete actual mixed partition output.')
            stage(f'bind_{index}', [str(b.PYTHON), str(api.BINDER), '--plan', str(plan_path), '--unit-index', str(index),
                '--native-report', str(reports[0]), '--outer-launch', str(group / 'outer.launch.json'),
                '--outer-exit', str(group / 'outer.exit'), '--outer-log', str(group / 'outer.log')])
        plan = b.read(plan_path)
        b.require(sum(unit['completed_samples_at_last_terminal'] for unit in plan['execution_units']) == 4750
            and not plan['pending_output_directories_at_last_terminal'], 'Complete official21 coverage is required.')
        b.check_hashes(plan['pins']); b.check_config(config)
        plan['status'] = 'prepared_not_scored'
        b.write_json(plan_path, plan)
        stage('merge_nine', b.command('merge', '--plan', str(plan_path)))
        merged = Path(plan['merged_output'])
        nine = b.read(merged / 'nine_task_acceptance.json')
        code = int((merged / 'nine_task.exit').read_text().strip())
        b.require(code in (0, 1) and nine['quality_target_evaluated'] and nine['checkpoint_id'] == meta['checkpoint_id']
            and nine['protocol']['strict_score_drop_bound'] and nine['protocol']['maximum_absolute_drop_points'] == .9
            and nine['protocol']['max_linear_bpw'] == 2.6, 'Require complete current-profile same-UUID evidence.')
        b.check_hashes(plan['pins']); b.check_config(config)
        record.update(status='complete_full8_full21_current_goal', longbench_plan=str(plan_path),
            longbench_sample_count=4750, longbench_scores=str(merged / 'scores.json'),
            acceptance_report=str(merged / 'nine_task_acceptance.json'), acceptance_exit_code=code,
            score=nine['compressed_nine_task_score'], passed=nine['passed'])
        b.write_json(output / 'current_goal_result.json', record)
    except BaseException as error:
        record.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        b.write_json(report_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--pipeline-report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpus', required=True, type=canonical().gpu_list)
    parser.add_argument('--check-only', action='store_true')
    evaluate(parser.parse_args())


if __name__ == '__main__':
    main()
