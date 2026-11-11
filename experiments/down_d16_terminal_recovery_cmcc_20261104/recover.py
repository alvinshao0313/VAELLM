"""Close the retired D16 evaluation on CPU, preserving its timed-out controller."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
TOPIC = REPO / 'result/qwen3_8b_mmlu65_20261004'
ORIGINAL = TOPIC / 'down_geometry_baseline_cmcc_20261103/allocator_repair/d16'
EVALUATION = ORIGINAL / 'final_full9'
OUTPUT = TOPIC / 'down_d16_terminal_recovery_cmcc_20261104'
CHECKPOINT_ID = '9b43d213-7147-4939-b04a-4a8c6b2b2300'


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
    return not path.exists() or int(path.read_text().rsplit(')', 1)[1].split()[19]) != ticks


def inspect():
    require(not (OUTPUT / 'diagnosis.json').exists(), 'Do not overwrite the original diagnosis.')
    started = time.time()
    run, report = read(ORIGINAL / 'run.json'), read(EVALUATION / 'evaluation.json')
    plan = read(EVALUATION / 'longbench/merge_plan.json')
    config = read(plan['config_path'])
    require(run['status'] == 'failed' and run['exit_code'] == 1
        and run['stages']['final_full9']['exit_code'] == 124
        and (ORIGINAL / 'run.exit').read_text().strip() == '1'
        and (ORIGINAL / 'final_full9.exit').read_text().strip() == '124', 'Keep the actual outer timeout failure.')
    require(retired(run['pid'], run['start_ticks']) and retired(report['pid'], report['start_ticks']),
        'The original controller/evaluator must have retired.')
    require(plan['checkpoint_id'] == report['checkpoint_id'] == config['model']['checkpoint_id'] == CHECKPOINT_ID
        and plan['status'] == 'draft_awaiting_explicit_terminal_binding'
        and not Path(plan['merged_output']).exists(), 'Only the actual unbound, unmerged D16 endpoint is eligible.')
    require(len(plan['execution_units']) == 1, 'This repair is only for the original single-partition D16 evaluation.')
    unit = plan['execution_units'][0]
    group = Path(unit['output_dir'])
    reports = list((group / 'runs').glob('*.json'))
    require(len(reports) == 1, 'Require exactly one genuine prediction attempt.')
    native, launch = read(reports[0]), read(group / 'outer.launch.json')
    require((group / 'outer.exit').read_text().strip() == '0' and native['exit_code'] == 0
        and native['status'] == 'partition_complete' and native['already_present'] == 0
        and native['new_predictions'] == unit['expected_samples'] == 4750
        and native['input_pins_unchanged_after'] is True, 'Require the complete naturally exited detached prediction.')
    require(retired(native['pid'], native['start_tick']) and retired(launch['pid'], launch['start_ticks']),
        'The detached predictor/launcher must have retired.')
    require(native['load_audit']['checkpoint_id'] == CHECKPOINT_ID
        and native['load_audit']['checkpoint_files_sha256'] == report['checkpoint_files_sha256']
        and native['config_sha256'] == launch['config_sha256'] == plan['config_sha256']
        and sha(plan['config_path']) == sha(group / 'config.json') == plan['config_sha256'],
        'The actual prediction UUID, payload or config binding differs.')
    archive = read(config['archive']['path'])
    counts, predictions = {}, {}
    for task, source in plan['task_file_sources'].items():
        path = Path(source)
        data = path.read_bytes()
        require(data.endswith(b'\n'), 'A prediction file ends in a partial JSON row: ' + task)
        rows = [json.loads(line) for line in data.splitlines()]
        require(len(rows) == archive['data']['counts'][task], 'Incomplete task predictions: ' + task)
        counts[task], predictions[str(path)] = len(rows), sha(path)
    require(len(counts) == 21 and sum(counts.values()) == 4750, 'Require all21/4750 raw predictions.')
    full8 = read(EVALUATION / 'full8/full8_summary.json')
    require(full8['checkpoint_id'] == CHECKPOINT_ID and full8['leaf_count'] == 64
        and full8['sample_count'] == 24742, 'Require the original full8 terminal coverage.')
    pins = dict(plan['pins'])
    for mapping in (run['source_sha256'], report['source_sha256'], config['model']['mixed_runtime']['source_sha256'],
                    full8['sample_artifacts']['sha256']):
        for path, digest in mapping.items():
            require(path not in pins or pins[path] == digest, 'Original source bindings conflict: ' + path)
            pins[path] = digest
    for path, digest in pins.items():
        require(sha(path) == digest, 'Original pinned input/source changed: ' + path)
    preserved = {str(path): sha(path) for path in (ORIGINAL / 'run.json', ORIGINAL / 'run.exit',
        ORIGINAL / 'final_full9.exit', ORIGINAL / 'final_full9.log', EVALUATION / 'evaluation.json',
        EVALUATION / 'longbench/merge_plan.json', group / 'outer.log', group / 'outer.exit',
        group / 'outer.launch.json', reports[0])}
    preserved.update(pins)
    preserved.update(predictions)
    active = read(TOPIC / 'down_d16_vae_muon_cmcc_20261103/prepared.json')
    protected = {path: digest for path, digest in active['source_sha256'].items()
                 if 'down_geometry_baseline_cmcc_20261103' in path or 'down_geometry_runtime_repair_cmcc_20261103' in path}
    require(all(sha(path) == digest for path, digest in protected.items()), 'The active combination baseline dependencies changed.')
    diagnosis = dict(status='complete_detached_predictions_CPU_closure_possible', checkpoint_id=CHECKPOINT_ID,
        original_run=str(ORIGINAL / 'run.json'), original_status=run['status'], original_exit_code=run['exit_code'],
        outer_full9_exit_code=124, evaluator_recorded_status=report['status'],
        controller_retired=True, evaluator_retired=True, native_retired=True, launcher_retired=True,
        original_full9_started_unix=run['stages']['final_full9']['started_unix'],
        original_full9_timeout_unix=run['stages']['final_full9']['finished_unix'],
        native_report=str(reports[0]), native_exit_code=0, outer_partition_exit_code=0,
        native_started_unix=native['started_unix'], native_finished_unix=native['finished_unix'],
        native_duration_seconds=native['duration_seconds'], original_native_runtime_seconds=native['max_runtime_seconds'],
        full8_leaf_count=64, full8_sample_count=24742, full8_mean_percent=full8['mean_percent'],
        longbench_task_counts=counts, longbench_sample_count=4750,
        source_pin_count=len(pins), active_original_dependencies=protected,
        preserved_inputs_sha256=preserved, source_sha256={str(Path(__file__).resolve()): sha(__file__)},
        boundary='Prediction coverage and unchanged source are verified. Canonical per-row binding, scoring and nine-task merge remain to run on CPU.',
        started_unix=started, finished_unix=time.time())
    write(OUTPUT / 'diagnosis.json', diagnosis)
    print(json.dumps({key: value for key, value in diagnosis.items()
        if key not in ('preserved_inputs_sha256', 'source_sha256')}, ensure_ascii=False), flush=True)


def complete():
    from experiments.cmcc_runtime_20261030 import sequence as original
    from experiments.down_geometry_baseline_cmcc_20261103 import evaluate as frozen
    from experiments.adaptive_class_evaluation_20261030 import bindings as b
    from experiments.adaptive_class_evaluation_20261030.summarize import full8_summary
    # This is the original explicit allocator/path dispatch, with no source edit.
    from experiments import down_geometry_runtime_repair_cmcc_20261103 as dispatch

    original.cpu_environment()
    require(not (OUTPUT / 'run.json').exists(), 'Never repeat or overwrite a terminal recovery attempt.')
    diagnosis = read(OUTPUT / 'diagnosis.json')
    require(diagnosis['status'] == 'complete_detached_predictions_CPU_closure_possible'
        and diagnosis['checkpoint_id'] == CHECKPOINT_ID, 'Require the completed original terminal diagnosis.')
    original.check_pins(diagnosis['preserved_inputs_sha256'])
    frozen.setup()
    old_report = read(EVALUATION / 'evaluation.json')
    binding, meta = b.source(old_report['checkpoint_dir'], old_report['pipeline_report'])
    require(binding['checkpoint_id'] == CHECKPOINT_ID
        and binding['checkpoint_files_sha256'] == old_report['checkpoint_files_sha256'],
        'The original final endpoint payload/source binding changed.')
    config = read(EVALUATION / 'full8/config.json')
    b.check_config(config)
    record = dict(status='running_CPU_terminal_recovery', checkpoint_id=CHECKPOINT_ID,
        pid=os.getpid(), start_ticks=original.proc_start(os.getpid()), started_unix=time.time(),
        argv=[sys.executable, *sys.argv], stages={}, diagnosis=str(OUTPUT / 'diagnosis.json'),
        original_controller_status='failed', original_controller_exit_code=1, original_full9_exit_code=124,
        original_evaluation_report=str(EVALUATION / 'evaluation.json'),
        original_evaluation_report_sha256=sha(EVALUATION / 'evaluation.json'),
        source_sha256={str(Path(__file__).resolve()): sha(__file__)},
        GPU_inference_rerun=False, training_rerun=False, CUDA_initialized=b.torch.cuda.is_initialized(),
        automatic_retry=False, automatic_resume=False)
    write(OUTPUT / 'run.json', record)

    def stage(name, argv):
        log, exit_path = OUTPUT / (name + '.log'), OUTPUT / (name + '.exit')
        with log.open('x', encoding='utf-8') as stream:
            process = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=20s', '1800s', *argv],
                cwd=REPO, env={**os.environ, **original.environment('')}, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT)
            row = dict(argv=argv, pid=process.pid, start_ticks=original.proc_start(process.pid),
                started_unix=time.time(), log=str(log))
            record['stages'][name] = row
            write(OUTPUT / 'run.json', record)
            code = process.wait()
        exit_path.write_text(str(code) + '\n', encoding='utf-8')
        row.update(exit_code=code, finished_unix=time.time())
        write(OUTPUT / 'run.json', record)
        require(code == 0, 'CPU terminal stage failed; no retry: ' + name)

    try:
        paths = [EVALUATION / 'full8' / (shard + '.json') for shard in ('mmlu', 'seven')]
        summary = full8_summary(config, paths, OUTPUT / 'full8_summary.json')
        require(read(OUTPUT / 'full8_summary.json') == read(EVALUATION / 'full8/full8_summary.json'),
            'The canonical CPU revalidation must exactly reproduce the original full8 summary.')
        plan = deepcopy(read(EVALUATION / 'longbench/merge_plan.json'))
        plan['merged_output'] = str(OUTPUT / 'longbench/merged')
        argv = plan['nine_task_argv']
        argv[argv.index('--compressed-longbench-scores') + 1] = str(Path(plan['merged_output']) / 'scores.json')
        argv[argv.index('--output') + 1] = str(Path(plan['merged_output']) / 'nine_task_acceptance.json')
        plan['terminal_recovery'] = dict(original_plan=str(EVALUATION / 'longbench/merge_plan.json'),
            original_plan_sha256=sha(EVALUATION / 'longbench/merge_plan.json'),
            diagnosis=str(OUTPUT / 'diagnosis.json'), original_outer_exit=124,
            boundary='Only merge destination and its score/acceptance paths changed; original configs, assignments, predictions, protocol and inference argv retained.')
        plan['pins'].update(record['source_sha256'])
        (OUTPUT / 'longbench').mkdir()
        plan_path = OUTPUT / 'longbench/merge_plan.json'
        write(plan_path, plan)
        for index, unit in enumerate(plan['execution_units']):
            group = Path(unit['output_dir'])
            native_path = Path(diagnosis['native_report'])
            native = read(native_path)
            require(native['load_audit']['persistent_tensors'] == len(meta['tensor_table'])
                and native['load_audit']['checkpoint_files_sha256'] == binding['checkpoint_files_sha256'],
                'Require the original full mixed tensor inventory and payload loading audit.')
            stage('bind_' + str(index), [sys.executable, '-B', '-u',
                str(TOPIC / 'llama_fullqk_qkonly2000_longbench21/bind_terminal.py'),
                '--plan', str(plan_path), '--unit-index', str(index), '--native-report', str(native_path),
                '--outer-launch', str(group / 'outer.launch.json'), '--outer-exit', str(group / 'outer.exit'),
                '--outer-log', str(group / 'outer.log')])
        plan = read(plan_path)
        require(not plan['pending_output_directories_at_last_terminal']
            and sum(unit['completed_samples_at_last_terminal'] for unit in plan['execution_units']) == 4750,
            'The unchanged canonical binder must verify every row/input slice before scoring.')
        plan['status'] = 'prepared_not_scored'
        write(plan_path, plan)
        stage('merge_nine', dispatch.command('evaluate', '--component', 'merge', '--plan', plan_path))
        merged = Path(plan['merged_output'])
        acceptance = original.validate_acceptance(merged / 'nine_task_acceptance.json', merged / 'nine_task.exit', CHECKPOINT_ID)
        try:
            original.validate_acceptance(merged / 'nine_task_acceptance.json', merged / 'nine_task.exit', 'foreign-UUID-must-fail')
        except ValueError:
            record['foreign_UUID_rejected'] = True
        else:
            raise ValueError('The original validator accepted a foreign UUID.')
        original.check_pins(diagnosis['preserved_inputs_sha256'])
        original.check_pins(diagnosis['active_original_dependencies'])
        require(not b.torch.cuda.is_initialized(), 'This recovery must remain CPU only.')
        terminal = dict(old_report)
        terminal.update(status='complete_full8_full21_current_goal', longbench_plan=str(plan_path),
            longbench_sample_count=4750, longbench_scores=str(merged / 'scores.json'),
            acceptance_report=str(merged / 'nine_task_acceptance.json'),
            acceptance_exit_code=acceptance['acceptance_exit'], score=acceptance['nine_task_score'],
            passed=acceptance['passed'], started_unix=record['started_unix'], finished_unix=time.time(),
            pid=os.getpid(), start_ticks=record['start_ticks'], argv=record['argv'], stages=record['stages'],
            terminal_recovery=dict(original_evaluation_report=str(EVALUATION / 'evaluation.json'),
                diagnosis=str(OUTPUT / 'diagnosis.json'), original_record_unchanged=True,
                original_controller_status='failed', original_full9_exit_code=124,
                CPU_seconds_scope='Only terminal binding/scoring/merge; original GPU inference timings remain in diagnosis/native reports.',
                original_full9_started_unix=diagnosis['original_full9_started_unix'],
                original_native_finished_unix=diagnosis['native_finished_unix'],
                CUDA_initialized=False, GPU_inference_rerun=False, training_rerun=False))
        write(OUTPUT / 'evaluation.json', terminal)
        validation = original.validate_evaluation(OUTPUT, CHECKPOINT_ID)
        record.update(status='completed_CPU_terminal_recovery', exit_code=0, complete_nine=validation,
            full8_summary_exact_original=True, canonical_binder_and_merger_unchanged=True,
            original_inputs_unchanged=True, active_dependencies_unchanged=True,
            finished_unix=time.time(), elapsed_seconds=time.time() - record['started_unix'])
        print(json.dumps(validation, ensure_ascii=False), flush=True)
    except BaseException:
        record.update(status='failed_CPU_terminal_recovery', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record.update(finished_unix=time.time(), elapsed_seconds=time.time() - record['started_unix'])
        write(OUTPUT / 'run.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('inspect', 'complete'))
    args = parser.parse_args()
    require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        and Path(sys.executable).resolve() == Path('/home/cmcc/miniconda3/envs/bitvae/bin/python').resolve(),
        'Use caller-activated bitvae with CUDA hidden.')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.phase == 'inspect':
        inspect()
    else:
        complete()
