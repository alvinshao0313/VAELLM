"""One isolated native-resume completion of the retired D16/VAE-Muon full21."""
import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
TOPIC = REPO / 'result/qwen3_8b_mmlu65_20261004'
ORIGINAL = TOPIC / 'down_d16_vae_muon_cmcc_20261103'
EVALUATION = ORIGINAL / 'evaluation'
OUTPUT = TOPIC / 'down_d16_muon_terminal_completion_cmcc_20261104'
CHECKPOINT_ID = 'ce4759de-af8a-46d5-a487-1425c43ae566'
BINDER = TOPIC / 'llama_fullqk_qkonly2000_longbench21/bind_terminal.py'
MODULE = 'experiments.down_d16_muon_terminal_completion_cmcc_20261104.complete'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def retired(pid, ticks):
    path = Path('/proc') / str(pid) / 'stat'
    if not path.exists():
        return True
    fields = path.read_text().rsplit(')', 1)[1].split()
    return fields[0] == 'Z' or int(fields[19]) != ticks


def command(component, *arguments):
    return [sys.executable, '-B', '-u', '-m', MODULE, '--component', component, *map(str, arguments)]


def setup():
    from experiments.down_d16_vae_muon_cmcc_20261103 import evaluate as frozen
    from experiments.adaptive_class_evaluation_20261030 import bindings as b
    # Only source/loader dispatch is reused; each new subprocess has its own outer limit.
    frozen.setup(time.time() + 3600)
    b.command = command
    return b


def stage(name, argv, record, seconds=1800):
    from experiments.cmcc_runtime_20261030 import sequence as original
    log, exit_path = OUTPUT / (name + '.log'), OUTPUT / (name + '.exit')
    actual = ['timeout', '--signal=TERM', '--kill-after=20s', str(seconds) + 's', *argv]
    with log.open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(actual, cwd=REPO, env={**os.environ, **original.environment('')},
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        row = dict(argv=actual, pid=process.pid, start_ticks=original.proc_start(process.pid),
            started_unix=time.time(), log=str(log), exit_file=str(exit_path))
        record.setdefault('stages', {})[name] = row
        write(OUTPUT / 'run.json', record)
        code = process.wait()
    exit_path.write_text(str(code) + '\n', encoding='utf-8')
    row.update(exit_code=code, finished_unix=time.time())
    write(OUTPUT / 'run.json', record)
    require(code == 0, 'Finite stage failed; no retry: ' + name)


def prepare():
    b = setup()
    b.cpu_environment()
    require(not OUTPUT.exists(), 'Never overwrite/repeat this isolated completion.')
    run, report = read(ORIGINAL / 'formal_run.json'), read(EVALUATION / 'evaluation.json')
    plan = read(EVALUATION / 'longbench/merge_plan.json')
    require(retired(run['pid'], run['start_ticks']) and retired(report['pid'], report['start_ticks']),
        'The original sequence/evaluator must naturally retire before preparation.')
    require((ORIGINAL / 'formal_sequence.exit').is_file() and (ORIGINAL / 'formal_full9.exit').is_file()
        and run['status'] == 'failed' and run['exit_code'] == 1, 'Require the unchanged actual original failure terminal.')
    require(plan['checkpoint_id'] == report['checkpoint_id'] == CHECKPOINT_ID
        and plan['status'] == 'draft_awaiting_explicit_terminal_binding'
        and len(plan['execution_units']) == 1 and not Path(plan['merged_output']).exists(),
        'Only the original unbound single-partition same-UUID full21 is eligible.')
    old_group = Path(plan['execution_units'][0]['output_dir'])
    paths = list((old_group / 'runs').glob('*.json'))
    require(len(paths) == 1, 'Require exactly one original native prediction attempt.')
    native, launch = read(paths[0]), read(old_group / 'outer.launch.json')
    require(retired(native['pid'], native['start_tick']) and retired(launch['pid'], launch['start_ticks'])
        and (old_group / 'outer.exit').read_text().strip() == '0'
        and native['status'] in ('partial_runtime_budget', 'partition_complete') and native['exit_code'] == 0
        and native['already_present'] == 0 and native['input_pins_unchanged_after'],
        'Require a naturally exited original native terminal; retain hard failures without resume.')
    binding, meta = b.source(report['checkpoint_dir'], report['pipeline_report'])
    require(binding['checkpoint_id'] == CHECKPOINT_ID and binding['checkpoint_files_sha256'] == report['checkpoint_files_sha256']
        == native['load_audit']['checkpoint_files_sha256'] and native['load_audit']['persistent_tensors'] == len(meta['tensor_table']),
        'Original final payload/loading identity changed.')
    b.check_hashes(plan['pins'])
    config = read(plan['config_path'])
    b.check_config(config)
    counts, prediction_hashes, timings = {}, {}, []
    for task, source in plan['task_file_sources'].items():
        data = Path(source).read_bytes()
        require(not data or data.endswith(b'\n'), 'Original prediction has an incomplete JSONL row: ' + task)
        rows = [json.loads(line) for line in data.splitlines()]
        counts[task], prediction_hashes[source] = len(rows), sha(source)
        require(len({row['_id'] for row in rows}) == len(rows)
            and [row['row_index'] for row in rows] == list(range(len(rows))), 'Duplicate/non-prefix original IDs: ' + task)
        if task == 'repobench-p':
            timings = [row['elapsed_seconds'] for row in rows]
    present = sum(counts.values())
    require(present == native['new_predictions'] and 0 < present <= 4750, 'Original report/prediction count differs.')
    from experiments.longbench_v1.protocol import EXPECTED_COUNTS
    missing = {task: EXPECTED_COUNTS[task] - count for task, count in counts.items() if count != EXPECTED_COUNTS[task]}
    require(not missing or (set(missing) == {'repobench-p'} and missing['repobench-p'] > 0),
        'Unexpected incomplete task inventory; report before extending scope.')
    p95 = sorted(timings)[math.ceil(.95 * len(timings)) - 1]
    remaining = 4750 - present
    native_budget = math.ceil(remaining * p95) + 180 if remaining else 0
    protected = dict(plan['pins'])
    for path in (ORIGINAL / 'formal_run.json', ORIGINAL / 'formal_sequence.exit', ORIGINAL / 'formal_full9.exit',
                 ORIGINAL / 'formal_full9.log', EVALUATION / 'evaluation.json', EVALUATION / 'longbench/merge_plan.json',
                 old_group / 'outer.launch.json', old_group / 'outer.log', old_group / 'outer.exit', paths[0]):
        protected[str(path)] = sha(path)
    protected.update(prediction_hashes)
    OUTPUT.mkdir()
    group = OUTPUT / 'longbench/groups/part_01'
    (group / 'predictions').mkdir(parents=True)
    (group / 'runs').mkdir()
    for task, source in plan['task_file_sources'].items():
        shutil.copyfile(source, group / 'predictions' / (task + '.jsonl'))
    for name in ('config.json',):
        shutil.copyfile(old_group / name, group / name)
    copied_native = group / 'runs' / paths[0].name
    shutil.copyfile(paths[0], copied_native)
    for name in ('outer.launch.json', 'outer.log', 'outer.exit'):
        shutil.copyfile(old_group / name, group / ('original.' + name))
    config_path = OUTPUT / 'longbench/config.json'
    shutil.copyfile(plan['config_path'], config_path)
    new = deepcopy(plan)
    new.update(config_path=str(config_path), merged_output=str(OUTPUT / 'longbench/merged'))
    new['pins'].update({str(config_path): sha(config_path), str(Path(__file__).resolve()): sha(__file__)})
    unit = new['execution_units'][0]
    unit.update(output_dir=str(group), native_reports=[], outer_exit=None, launch_records=[], terminal_attempts=[])
    new['task_file_sources'] = {task: str(group / 'predictions' / (task + '.jsonl')) for task in counts}
    merged = Path(new['merged_output'])
    new['nine_task_argv'] = command('summarize', '--checkpoint', report['checkpoint_dir'], '--pipeline-report', report['pipeline_report'],
        '--compressed-results', EVALUATION / 'full8/mmlu.json', EVALUATION / 'full8/seven.json',
        '--compressed-longbench-scores', merged / 'scores.json', '--output', merged / 'nine_task_acceptance.json')
    new['terminal_completion'] = dict(original_plan=str(EVALUATION / 'longbench/merge_plan.json'),
        original_plan_sha256=sha(EVALUATION / 'longbench/merge_plan.json'), preserved_original_sha256=protected,
        original_prediction_bytes_copied=True, original_terminal_files_copied_byte_exact=True,
        boundary='Original failed controller/deadline retained. Copied terminal evidence retains original argv/identities; only its verified prefix is continued in this isolated group with the existing runner resume=True.')
    plan_path = OUTPUT / 'longbench/merge_plan.json'
    write(plan_path, new)
    record = dict(status='preparing_native_resume', checkpoint_id=CHECKPOINT_ID, started_unix=time.time(),
        pid=os.getpid(), start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        argv=[sys.executable, *sys.argv], original_status=run['status'], original_exit_code=run['exit_code'],
        original_full9_exit_code=int((ORIGINAL / 'formal_full9.exit').read_text()), original_native_status=native['status'],
        present=present, remaining=remaining, counts=counts, missing=missing,
        repobench_seconds_p95=p95, proposed_native_runtime_seconds=native_budget,
        proposed_outer_timeout_seconds=native_budget + 180 if remaining else 0,
        budget_basis='ceil(actual_missing * observed_original_repobench_p95) +180s loading margin; outer adds180s.',
        original_inputs_sha256=protected, source_sha256={str(Path(__file__).resolve()): sha(__file__)},
        automatic_retry=False, automatic_resume=False, training_rerun=False, completed_ID_regeneration=False, stages={})
    write(OUTPUT / 'run.json', record)
    stage('bind_original', [sys.executable, '-B', '-u', str(BINDER), '--plan', str(plan_path), '--unit-index', '0',
        '--native-report', str(copied_native), '--outer-launch', str(group / 'original.outer.launch.json'),
        '--outer-exit', str(group / 'original.outer.exit'), '--outer-log', str(group / 'original.outer.log')], record)
    b.check_hashes(protected)
    record.update(status='prepared_original_prefix_verified', prepare_finished_unix=time.time())
    write(OUTPUT / 'run.json', record)
    print(json.dumps({key: record[key] for key in ('status', 'present', 'remaining', 'missing', 'repobench_seconds_p95',
        'proposed_native_runtime_seconds', 'proposed_outer_timeout_seconds')}), flush=True)


def predict(args):
    b = setup()
    from experiments.adaptive_class_evaluation_20261030.loading import load_model
    from experiments.longbench_v1 import runner
    runner.check_config, runner.load_model = b.check_config, load_model
    result = runner.run(args.config, args.output, datasets=args.datasets, resume=True,
                        max_runtime_seconds=args.max_runtime_seconds)
    require(result['status'] == 'partition_complete' and result['exit_code'] == 0,
        'The one finite native resume did not complete; no retry or partial scoring.')
    print(json.dumps({key: result[key] for key in ('status', 'new_predictions', 'already_present', 'duration_seconds')}), flush=True)


def resume():
    b = setup()
    b.cpu_environment()
    record = read(OUTPUT / 'run.json')
    require(record['status'] == 'prepared_original_prefix_verified', 'Never repeat the one finite resume.')
    b.check_hashes(record['original_inputs_sha256'])
    if record['remaining'] == 0:
        return finish()
    native_seconds, outer_seconds = record['proposed_native_runtime_seconds'], record['proposed_outer_timeout_seconds']
    require(0 < native_seconds < outer_seconds <= 1800, 'The formula exceeds the authorized30min GPU ceiling.')
    resources = b.module(REPO / 'experiments/cat_aligned_evaluation_20261028/evaluate.py',
        '_d16_muon_completion_resources').check_resources(['3'])
    plan = read(OUTPUT / 'longbench/merge_plan.json')
    unit = plan['execution_units'][0]
    group = Path(unit['output_dir'])
    require(unit['completed_samples_at_last_terminal'] == record['present'], 'Bound prefix count changed.')
    argv = command('predict', '--config', plan['config_path'], '--output', group,
        '--datasets', *unit['datasets'], '--max-runtime-seconds', native_seconds)
    original_launch = read(group / 'original.outer.launch.json')
    environment = {**os.environ, **original_launch['environment'], 'CUDA_VISIBLE_DEVICES': '3'}
    shell = 'exit_path=$1\nshift\n"$@"\nstatus=$?\nprintf "%s\\n" "$status" > "$exit_path"\nexit "$status"\n'
    actual = ['/bin/bash', '-c', shell, 'longbench-completion-once', str(group / 'outer.exit'),
        'timeout', '--signal=TERM', '--kill-after=20s', str(outer_seconds) + 's', *argv]
    with (group / 'outer.log').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(argv=actual, cwd=str(REPO), CUDA_VISIBLE_DEVICES='3')) + '\n')
        stream.flush()
        process = subprocess.Popen(actual, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    from experiments.cmcc_runtime_20261030 import sequence as original
    write(group / 'outer.launch.json', dict(pid=process.pid, start_ticks=original.proc_start(process.pid),
        argv=actual, native_argv=argv, cwd=str(REPO), gpu='3', started_unix=time.time(),
        checkpoint_id=CHECKPOINT_ID, config_sha256=plan['config_sha256'], expected_already_present=record['present'],
        native_runtime_seconds=native_seconds, outer_timeout_seconds=outer_seconds, termination_grace_seconds=20,
        environment={**original_launch['environment'], 'CUDA_VISIBLE_DEVICES': '3'}))
    record.update(status='native_resume_running', native_resume_argv=argv, initial_GPU_resource_check=resources,
        native_resume_outer_pid=process.pid, native_resume_outer_start_ticks=original.proc_start(process.pid),
        native_resume_started_unix=time.time(), GPU_total_seconds_ceiling=1800)
    write(OUTPUT / 'run.json', record)
    code = process.wait()
    record.update(status='native_resume_exited', native_resume_exit_code=code,
        native_resume_finished_unix=time.time())
    write(OUTPUT / 'run.json', record)
    require(code == 0 and (group / 'outer.exit').read_text().strip() == '0',
        'The one finite resume failed; preserve evidence without retry or scoring.')
    return finish()


def finish():
    b = setup()
    b.cpu_environment()
    from experiments.cmcc_runtime_20261030 import sequence as original
    record = read(OUTPUT / 'run.json')
    require(record['status'] in ('prepared_original_prefix_verified', 'native_resume_exited'), 'Completion not ready.')
    b.check_hashes(record['original_inputs_sha256'])
    plan_path = OUTPUT / 'longbench/merge_plan.json'
    plan = read(plan_path)
    group = Path(plan['execution_units'][0]['output_dir'])
    if record['remaining']:
        reports = sorted((group / 'runs').glob('*.json'))
        old = set(plan['execution_units'][0]['native_reports'])
        new = [path for path in reports if str(path) not in old]
        require(len(new) == 1, 'Require exactly one actual native resume attempt.')
        stage('bind_resume', [sys.executable, '-B', '-u', str(BINDER), '--plan', str(plan_path), '--unit-index', '0',
            '--native-report', str(new[0]), '--outer-launch', str(group / 'outer.launch.json'),
            '--outer-exit', str(group / 'outer.exit'), '--outer-log', str(group / 'outer.log')], record)
    plan = read(plan_path)
    require(not plan['pending_output_directories_at_last_terminal']
        and sum(unit['completed_samples_at_last_terminal'] for unit in plan['execution_units']) == 4750,
        'Only complete canonical21/4750 can be scored.')
    plan['status'] = 'prepared_not_scored'
    write(plan_path, plan)
    stage('merge_nine', command('merge', '--plan', plan_path), record)
    merged = Path(plan['merged_output'])
    acceptance = original.validate_acceptance(merged / 'nine_task_acceptance.json', merged / 'nine_task.exit', CHECKPOINT_ID)
    try:
        original.validate_acceptance(merged / 'nine_task_acceptance.json', merged / 'nine_task.exit', 'foreign-UUID-must-fail')
    except ValueError:
        record['foreign_UUID_rejected'] = True
    else:
        raise ValueError('Original acceptance validator accepted a foreign UUID.')
    b.check_hashes(record['original_inputs_sha256'])
    terminal = read(EVALUATION / 'evaluation.json')
    terminal.update(status='complete_full8_full21_current_goal', longbench_plan=str(plan_path), longbench_sample_count=4750,
        longbench_scores=str(merged / 'scores.json'), acceptance_report=str(merged / 'nine_task_acceptance.json'),
        acceptance_exit_code=acceptance['acceptance_exit'], score=acceptance['nine_task_score'], passed=acceptance['passed'],
        terminal_completion=dict(original_evaluation_report=str(EVALUATION / 'evaluation.json'),
            original_record_unchanged=True, original_status=record['original_status'],
            original_full9_exit_code=record['original_full9_exit_code'], reused_original_predictions=record['present'],
            new_predictions=record['remaining'], native_resume_semantics='unchanged runner.run(resume=True)',
            training_rerun=False, completed_ID_regeneration=False), finished_unix=time.time())
    write(OUTPUT / 'evaluation.json', terminal)
    validation = original.validate_evaluation(OUTPUT, CHECKPOINT_ID)
    record.update(status='completed_isolated_native_resume_full9', exit_code=0, complete_nine=validation,
        original_inputs_unchanged=True, canonical_binder_merger_unchanged=True,
        finished_unix=time.time(), elapsed_seconds=time.time() - record['started_unix'])
    write(OUTPUT / 'run.json', record)
    print(json.dumps(validation, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'resume', 'finish'))
    parser.add_argument('--component', choices=('predict', 'merge', 'summarize'))
    parser.add_argument('--config')
    parser.add_argument('--output')
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--max-runtime-seconds', type=float)
    args, remaining = parser.parse_known_args()
    require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae', 'Use caller-activated bitvae.')
    if args.component in ('merge', 'summarize'):
        setup()
        component = importlib.import_module('experiments.adaptive_class_evaluation_20261030.' + args.component)
        # These consumers keep their original argument parsing and exit semantics.
        sys.argv = [sys.argv[0], *remaining]
        if args.output is not None:
            sys.argv.extend(['--output', args.output])
        return component.main()
    if args.component == 'predict':
        require(not remaining and args.phase is None, 'Unexpected native resume arguments.')
        return predict(args)
    require(not remaining and args.phase and not args.component, 'Select one finite preparation/completion phase.')
    return {'prepare': prepare, 'resume': resume, 'finish': finish}[args.phase]()


if __name__ == '__main__':
    raise SystemExit(main())
