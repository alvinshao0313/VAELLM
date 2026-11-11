"""Use the established two-card LongBench partition, terminal binder and merger."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from bindings import HERE, PYTHON, ROOT, boundary, environment, module, process_event, s
from experiments.longbench_v1.protocol import DATASETS, EXPECTED_COUNTS

HELPERS = ROOT / 'result/qwen3_8b_mmlu65_20261004/qwen_aux_retention32_batch_20261011/evaluate.py'
TERMINAL = ROOT / 'result/qwen3_8b_mmlu65_20261004/llama_fullqk_qkonly2000_longbench21'
PARTITIONS = ROOT / 'experiments/cat_aligned_evaluation_20261028/evaluate.py'


def canonical():
    result = module(PARTITIONS, '_progressive_original_partitions')
    result.b.require = s.require
    return result


def make_plan(request, request_path, checkpoint, config, config_path):
    output = Path(request['output_dir'])
    base = output / 'nodes/down_proj/longbench'
    units = []
    for index, (gpu, tasks) in enumerate(zip(('7', '6'), canonical().task_groups(('7', '6')), strict=True)):
        group = base / 'groups' / f'part_{index + 1:02d}'
        argv = [str(PYTHON), '-B', '-u', str(HERE / 'evaluation.py'),
            '--request', str(request_path), '--component', 'predict',
            '--config', str(config_path), '--output', str(group), '--datasets', *tasks]
        units.append(dict(output_dir=str(group), datasets=tasks,
            expected_samples=sum(EXPECTED_COUNTS[task] for task in tasks), native_reports=[], outer_exit=None,
            launch_records=[], terminal_attempts=[], gpu=gpu, argv=argv, native_runtime_seconds=86400,
            outer_timeout_seconds=86580, termination_grace_seconds=20))
    merged = base / 'merged'
    return dict(schema='fixed_full21_merge_plan_v1', status='draft_awaiting_explicit_terminal_binding',
        checkpoint_id=checkpoint['checkpoint_id'], expected_tasks=list(DATASETS), expected_samples=4750,
        config_path=str(config_path), config_sha256=s.sha(config_path), protocol=config['protocol'],
        execution_units=units, task_file_sources={task: str(Path(unit['output_dir']) / 'predictions' / (task + '.jsonl'))
            for unit in units for task in unit['datasets']},
        pins={**request['source_sha256'], str(config_path): s.sha(config_path)},
        merged_output=str(merged), scoring_cwd=str(ROOT), python_executable=str(PYTHON),
        nine_task_argv=[str(PYTHON), '-B', '-u', str(HERE / 'run.py'), '--request', str(request_path),
            '--summarize-output', str(merged / 'nine_task_acceptance.json')],
        automatic_retry=False, automatic_resume=False, partial_scoring=False,
        boundary='Same native S4 UUID/config. Original task partition, terminal binding, merge and unrounded scoring. '
            'Fixed4 research parent; final <=2.6bit acceptance is not evaluated.')


def cpu_stage(request, name, argv):
    output = Path(request['output_dir'])
    env = environment('')
    env['PYTHONPATH'] = str(ROOT)
    with (output / (name + '.log')).open('x', encoding='utf-8') as log:
        result = subprocess.run(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, check=False)
    (output / (name + '.exit')).write_text(str(result.returncode) + '\n', encoding='utf-8')
    s.require(result.returncode == 0, name + ' failed; no retry or partial scoring.')


def execute(request_path):
    request = s.read(request_path)
    output = Path(request['output_dir'])
    s.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU controller only.')
    from experiments.cmcc_runtime_20261030.sequence import wait_identity
    running = s.read(output / 'run.json')['processes']['fit']
    observed = wait_identity(running['pid'], running['start_ticks'])
    fitted = s.read(output / 'fit.json')
    s.require(fitted['status'] == 'complete_bf16_s4_no_recovery_fit' and fitted['exit_code'] == 0,
        'Fit must actually exit successfully before GPU6 is reused.')
    checkpoint = fitted['boundaries'][-1]
    s.require(checkpoint['category'] == 'down_proj' and boundary(checkpoint['path'])['checkpoint_id'] == checkpoint['checkpoint_id'],
        'Use the exact final cumulative S4 node already evaluated by MMLU.')
    from evaluation import native_config
    from experiments.longbench_v1.models import check_config
    base = output / 'nodes/down_proj/longbench'
    base.mkdir()
    config = native_config(request, checkpoint['path'], short=False)
    config_path = base / 'config.json'
    s.write(config_path, config)
    check_config(config)
    plan = make_plan(request, request_path, checkpoint, config, config_path)
    plan['fit_exit_observed'] = observed
    plan['initial_resource_check'] = canonical().check_resources(['6', '7'])
    plan_path = base / 'merge_plan.json'
    s.write(plan_path, plan)
    api = module(HELPERS, '_progressive_existing_LB_execution')
    api.S, api.PYTHON = ROOT, PYTHON
    processes = []
    try:
        unit = plan['execution_units'][0]
        Path(unit['output_dir']).mkdir(parents=True)
        processes.append(api.start_group(unit, config, plan['config_sha256']))
        from run import stage
        stage(request_path, 'final_seven', '--component', 'seven', '--checkpoint', checkpoint['path'],
            '--output', str(output / 'nodes/down_proj/seven.json'), gpu=6)
        unit = plan['execution_units'][1]
        Path(unit['output_dir']).mkdir(parents=True)
        processes.append(api.start_group(unit, config, plan['config_sha256']))
    finally:
        codes = [process.wait() for process in processes]
        s.write(base / 'execution.json', dict(**process_event(os.getpid()), exit_codes=codes,
            completed_unix=time.time(), fit_exit_observed=observed))
    s.require(len(codes) == 2 and all(code == 0 for code in codes), 'A real assigned LB partition failed.')
    for index, unit in enumerate(plan['execution_units']):
        group = Path(unit['output_dir'])
        reports = list((group / 'runs').glob('*.json'))
        s.require(len(reports) == 1, 'One actual fresh attempt per partition is required.')
        report = s.read(reports[0])
        s.require(report['status'] == 'partition_complete' and report['exit_code'] == 0
            and report['already_present'] == 0 and report['new_predictions'] == unit['expected_samples']
            and report['load_audit']['checkpoint_id'] == checkpoint['checkpoint_id'],
            'Real partition terminal/UUID/coverage differs.')
        cpu_stage(request, f'bind_longbench_{index}', [str(PYTHON), '-B', str(TERMINAL / 'bind_terminal.py'),
            '--plan', str(plan_path), '--unit-index', str(index), '--native-report', str(reports[0]),
            '--outer-launch', str(group / 'outer.launch.json'), '--outer-exit', str(group / 'outer.exit'),
            '--outer-log', str(group / 'outer.log')])
    plan = s.read(plan_path)
    s.require(sum(unit['completed_samples_at_last_terminal'] for unit in plan['execution_units']) == 4750
        and not plan['pending_output_directories_at_last_terminal'], 'Complete21 coverage is required.')
    plan['status'] = 'prepared_not_scored'
    s.write(plan_path, plan)
    cpu_stage(request, 'merge_longbench_nine', [str(PYTHON), '-B', str(TERMINAL / 'merge_full21.py'),
        '--plan', str(plan_path)])
    final = Path(plan['merged_output']) / 'nine_task_acceptance.json'
    result = s.read(final)
    s.require(result['status'] == 'complete_four_stage_research_parent'
        and result['checkpoint_id'] == checkpoint['checkpoint_id']
        and result['final_lowbit_acceptance_evaluated'] is False,
        'Expected complete same-UUID S4 research-parent scores.')
    (output / 'result.json').symlink_to(final.relative_to(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    args = parser.parse_args()
    execute(args.request.resolve(strict=True))

