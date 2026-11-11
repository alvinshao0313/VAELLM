"""Execute the fixed cmcc short -> original formal pair -> complete-nine DAG once."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import select
import subprocess
import sys
import threading
import time
import traceback

from .paths import ARTIFACTS, MODELS, PYTHON, REPO, TOPIC

ROOT = ARTIFACTS / 'adaptive_class_cmcc_20261030'
H192 = ARTIFACTS / 'g09_hotpot_coverage_cmcc_20261030'
ARMS = ('dynamic', 'rice')
TASKS = ('boolq', 'rte', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa', 'piqa', 'mmlu', 'longbench_v1')
OLD_MUON_SCORE = 66.06887949148307
BASELINE_SCORE = 68.04714491790848
OLD_MUON = TOPIC / 'overnight_goal_20261029/evaluation/Muon/transfer_acceptance.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def proc_start(pid):
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except FileNotFoundError:
        return None


def pidfd_open(pid):
    # This server's bitvae Python omits os.pidfd_open. Its x86_64 Linux header
    # defines __NR_pidfd_open=434; use that same kernel API directly.
    require(sys.platform == 'linux' and platform.machine() == 'x86_64', 'This fixed runner requires the current Linux x86_64 server.')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    descriptor = int(libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0)))
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), pid)
    return descriptor


def wait_identity(pid, start_ticks):
    """Wait on a Linux kernel process handle, never on a recycled numeric PID."""
    actual = proc_start(pid)
    if actual is None:
        return dict(pid=pid, start_ticks=start_ticks, state='already_absent')
    require(actual == start_ticks, f'PID {pid} was reused; refuse to attach.')
    try:
        descriptor = pidfd_open(pid)
    except ProcessLookupError:
        return dict(pid=pid, start_ticks=start_ticks, state='exited_before_pidfd_open')
    try:
        actual = proc_start(pid)
        require(actual in (None, start_ticks), f'PID {pid} changed while opening pidfd.')
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        poller.poll()  # Blocking kernel wait; no log/status polling or timeout retry.
    finally:
        os.close(descriptor)
    return dict(pid=pid, start_ticks=start_ticks, state='pidfd_exit_observed')


def wait_external(row):
    observed = [wait_identity(row['worker_pid'], row['worker_start_ticks'])]
    if row.get('shell_start_ticks') is not None:
        observed.append(wait_identity(row['shell_pid'], row['shell_start_ticks']))
    require(Path(row['exit_file']).read_text().strip() == '0', 'Original short did not exit0: ' + row['arm'])
    report = read(row['report'])
    require(report['pid'] == row['worker_pid'] and report['arm'] == row['arm'], 'Original short terminal identity differs.')
    return observed


def check_pins(pins):
    for path, digest in pins.items():
        require(sha(path) == digest, 'A pinned input/source changed: ' + path)


def cpu_environment():
    require(Path(sys.executable).resolve() == PYTHON.resolve() and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Activate bitvae in the caller and hide CUDA for the CPU driver.')
    descriptor = pidfd_open(os.getpid())
    os.close(descriptor)


def environment(gpu):
    return dict(CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')


def check_resources(gpus):
    devices = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                             check=True, capture_output=True, text=True)
    found = {line.split(',')[0].strip(): line.split(',')[1].strip() for line in devices.stdout.splitlines()}
    require(set(gpus) <= set(found), 'Assigned GPU missing.')
    selected = {gpu: found[gpu] for gpu in gpus}
    active = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True)
    blockers = [line for line in active.stdout.splitlines() if line.split(',')[0].strip() in selected.values()]
    require(not blockers, 'Assigned GPUs occupied; no retry/wait: ' + repr(blockers))
    return dict(checked_unix=time.time(), gpu_uuids=selected, blockers=blockers)


def launch_child(row, on_change):
    """One launch and one wait; callers validate real output after this returns."""
    log, exit_path = Path(row['log']), Path(row['exit_file'])
    require(not log.exists() and not exit_path.exists(), 'Never relaunch/overwrite a recorded stage: ' + row['name'])
    record = dict(name=row['name'], argv=row['argv'], environment=row['environment'], cwd=str(REPO),
                  log=str(log), exit_file=str(exit_path), started_unix=time.time(), status='starting')
    with log.open('xb') as stream:
        process = subprocess.Popen(row['argv'], cwd=REPO, env={**os.environ, **row['environment']},
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        record.update(pid=process.pid, start_ticks=proc_start(process.pid), status='running')
        on_change(record.copy())
        code = process.wait()
    exit_path.write_text(str(code) + '\n', encoding='utf-8')
    record.update(exit_code=code, finished_unix=time.time(), status='exited')
    record['elapsed_seconds'] = record['finished_unix'] - record['started_unix']
    on_change(record.copy())
    require(code == 0, row['name'] + ' exited ' + str(code) + '; dependent stages stopped.')
    return record


def validate_acceptance(path, exit_file, checkpoint_id):
    """Accept valid not_met results, reject incomplete/invalid/foreign evidence."""
    path = Path(path)
    report = read(path)
    code = int(Path(exit_file).read_text().strip())
    require(code in (0, 1) and report['status'] in ('passed', 'not_met')
        and report['quality_target_evaluated'] is True and report['checkpoint_id'] == checkpoint_id
        and code == (0 if report['passed'] else 1), 'Require a complete, valid same-UUID nine-term result.')
    protocol = report['protocol']
    require(protocol['max_linear_bpw'] == 2.6 and protocol['maximum_absolute_drop_points'] == .9
        and protocol['strict_score_drop_bound'] and protocol['max_lora_rank'] == 0,
        'The current 2.6/rank0/strict<0.9 acceptance profile changed.')
    scores, baseline = report['compressed_scores'], report['baseline_scores']
    require(set(scores) == set(baseline) == set(TASKS)
        and all(math.isfinite(value) and 0 <= value <= 100 for value in (*scores.values(), *baseline.values())),
        'Require all nine finite 0..100 task scores.')
    mean, dense_mean = math.fsum(scores.values()) / 9, math.fsum(baseline.values()) / 9
    require(abs(mean - report['compressed_nine_task_score']) < 1e-10
        and abs(dense_mean - report['baseline_nine_task_score']) < 1e-10
        and abs(dense_mean - BASELINE_SCORE) < 1e-12, 'Reported means or fixed BF16 baseline differ.')
    downstream = report['downstream_compressed']
    require(downstream['leaf_count'] == 64 and downstream['sample_count'] == 24742,
        'Require complete full8 64 leaves/24742 samples.')
    budget = report['bit_budget']
    require(budget['budget_passed'] and budget['user_linear_bpw'] <= 2.6
        and budget.get('max_lora_rank', 0) == 0, 'Actual deployed storage/rank exceeds this fixed profile.')
    return dict(checkpoint_id=checkpoint_id, acceptance_report=str(path), acceptance_sha256=sha(path),
        acceptance_exit=code, quality_status=report['status'], passed=report['passed'], task_scores=scores,
        full8_mean_percent=math.fsum(scores[task] for task in TASKS[:-1]) / 8,
        longbench_mean_percent=scores['longbench_v1'], nine_task_score=mean,
        delta_vs_bf16={task: scores[task] - baseline[task] for task in TASKS},
        nine_delta_vs_bf16=mean - dense_mean,
        bit_budget={key: budget[key] for key in ('user_linear_bpw', 'packed_linear_bpw', 'user_linear_bits',
                    'packed_linear_bits', 'original_linear_weights', 'full_model_tensor_bytes', 'complete_directory_bytes') if key in budget})


def validate_evaluation(output, expected_id):
    output = Path(output)
    report = read(output / 'evaluation.json')
    require(report['status'] in ('complete_full8_full21_current_goal', 'complete_full8_full21_nine')
        and report['checkpoint_id'] == expected_id and report['full8_leaf_tasks'] == 64
        and report['full8_sample_count'] == 24742 and report['longbench_sample_count'] == 4750,
        'Evaluation terminal report is not complete/same-UUID.')
    plan = read(report['longbench_plan'])
    require(plan['checkpoint_id'] == expected_id and plan['expected_samples'] == 4750
        and len(plan['expected_tasks']) == 21 and not plan['pending_output_directories_at_last_terminal']
        and sum(unit['completed_samples_at_last_terminal'] for unit in plan['execution_units']) == 4750,
        'LongBench terminal coverage is incomplete.')
    check_pins(plan['pins'])
    merged = Path(plan['merged_output'])
    scores = read(merged / 'scores.json')
    require(scores['status'] == 'complete' and scores['sample_count'] == 4750 and len(scores['tasks']) == 21
        and scores['model']['checkpoint_id'] == expected_id, 'Require actual same-UUID complete21/4750 scores.')
    summary = validate_acceptance(merged / 'nine_task_acceptance.json', merged / 'nine_task.exit', expected_id)
    summary.update(evaluation_report=str(output / 'evaluation.json'),
        evaluation_seconds=report['finished_unix'] - report['started_unix'])
    return summary


def validate_formal(row):
    # Reuse the actual consumer's complete payload SHA/strict reload/inventory/ledger checks.
    from experiments.adaptive_class_evaluation_20261030 import bindings as b
    b.BASE = MODELS / 'Qwen/Qwen3-8B'
    binding, _meta = b.source(Path(row['output']) / 'final_model', Path(row['output']) / 'report.json')
    report = read(row['report'])
    require(report['arm'] == row['name'] and [stage['category'] for stage in report['stages']]
        == ['gate_proj', 'up_proj', 'down_proj'], 'Formal category/arm sequence differs.')
    for stage in report['stages']:
        require(stage['recovery']['actual_steps'] == 500 and stage['recovery']['frozen_persistent_fields_exact'],
            'Formal recovery did not complete the original500 updates.')
    for stage in report['stages'][:-1]:
        fit = stage['next_live_category_fit']
        require(fit['actual_stages'] == 2 and fit['actual_updates_per_stage'] == fit['schedule_steps'] == 10000
            and fit['fresh_VAE_initialization_seed'] == 31 and len(fit['actual_optimizer_initializations']) == 2,
            'Formal live up/down is not the fixed two-stage10K fitter.')
    check_pins(report['pipeline_source_sha256'])
    return binding, report


def prepare(output, handoff_path):
    cpu_environment()
    output = output.resolve()
    require(output.is_relative_to(TOPIC) and not os.path.lexists(output), 'Use a fresh topic result directory.')
    short_launch = ROOT / 'short_trl01_launch.json'
    shorts = read(short_launch)['branches']
    require([row['arm'] for row in shorts] == list(ARMS), 'Use the fixed current short pair.')
    for row in shorts:
        require(row['gpu'] == ('0' if row['arm'] == 'dynamic' else '1'), 'Original short GPU identity changed.')
        actual_shell = proc_start(row['shell_pid'])
        require(actual_shell in (None, row['shell_start_ticks']), 'Original short shell PID reused.')
        actual = proc_start(row['worker_pid'])
        require(actual in (None, row['worker_start_ticks']), 'Original short PID reused.')
    handoff = read(handoff_path)
    require(handoff['status'] == 'verified_h192_continuation_ready' and handoff['gpus'] == ['5', '6', '7']
        and handoff['checkpoint_id'] == '74329fcf-c762-4498-9f41-218df9645b3f', 'Use the validated fixed H192 continuation handoff.')
    argv = handoff['argv']
    require(argv[:4] == [str(PYTHON), '-B', '-u', '-m']
        and argv[4] == 'experiments.g09_hotpot_coverage_20261030.cmcc_continue'
        and argv[argv.index('--gpus') + 1] == '5,6,7', 'Unexpected H192 continuation command.')
    paths = set(Path(__file__).parent.glob('*.py'))
    paths.update((REPO / 'experiments/adaptive_class_evaluation_20261030').glob('*.py'))
    paths.update(REPO / 'experiments/adaptive_class_20261030' / name
        for name in ('run_pipeline.py', 'continuation.py', 'initialization.py'))
    paths.update((short_launch, handoff_path, OLD_MUON, ROOT / 'evaluation_support/prepared.json', ROOT / 'prepared_v2/prepared.json'))
    pins = {str(path.resolve()): sha(path) for path in paths}
    for path in (ROOT / 'prepared_v2/prepared.json', ROOT / 'evaluation_support/prepared.json'):
        record = read(path)
        for key in ('original_source_files_sha256', 'prepared_files_sha256', 'source_sha256'):
            pins.update(record.get(key, {}))
    pins.update(handoff['source_sha256'])
    check_pins(pins)
    control = read(OLD_MUON)
    require(control['quality_target_evaluated'] and control['compressed_nine_task_score'] == OLD_MUON_SCORE
        and control['baseline_nine_task_score'] == BASELINE_SCORE, 'Fixed original Muon comparison source changed.')
    output.mkdir(parents=True)
    tag = output.name
    formal = {}
    evaluations = {}
    for arm, gpu, eval_gpus in (('dynamic', '0', '0,2,3'), ('rice', '1', '1,4')):
        target = ROOT / (tag + '_' + arm)
        result = TOPIC / 'adaptive_class_evaluation_20261030' / (tag + '_' + arm)
        require(not os.path.lexists(target) and not os.path.lexists(result), 'Formal/evaluation output already exists.')
        command = [str(PYTHON), '-B', '-u', '-m', 'experiments.cmcc_runtime_20261030.adaptive_pipeline',
            '--prepared', str(ROOT / 'prepared_v2'), '--arm', arm, '--artifacts',
            *(str(ARTIFACTS / 'adaptive_class_20261030' / ('formal_gpu' + str(index))) for index in (1, 2, 3)),
            '--output', str(target)]
        formal[arm] = dict(name=arm, argv=command, environment=environment(gpu), gpus=[gpu], output=str(target),
            report=str(target / 'report.json'), log=str(output / (arm + '_formal.log')), exit_file=str(output / (arm + '_formal.exit')))
        evaluations[arm] = dict(name=arm + '_evaluation', argv=[str(PYTHON), '-B', '-u', '-m',
            'experiments.cmcc_runtime_20261030.adaptive_evaluation', '--support', str(ROOT / 'evaluation_support'),
            '--checkpoint', str(target / 'final_model'), '--pipeline-report', str(target / 'report.json'),
            '--output', str(result), '--gpus', eval_gpus], environment=environment(''), gpus=eval_gpus.split(','),
            output=str(result), log=str(output / (arm + '_evaluation.log')), exit_file=str(output / (arm + '_evaluation.exit')))
    h192 = dict(name='h192_continuation', argv=argv, environment=environment(''), gpus=['5', '6', '7'],
        output=handoff['output'], log=str(output / 'h192_continuation.log'), exit_file=str(output / 'h192_continuation.exit'),
        checkpoint_id=handoff['checkpoint_id'])
    plan = dict(schema='cmcc_fixed_sequence_20261030_v1', created_unix=time.time(), output=str(output),
        source_sha256=pins, short=shorts, formal=formal, evaluation=evaluations, h192=h192,
        old_muon_control=dict(report=str(OLD_MUON), sha256=sha(OLD_MUON), checkpoint_id=control['checkpoint_id'],
            nine_task_score=control['compressed_nine_task_score'], task_scores=control['compressed_scores']),
        lock_file=str(ROOT / 'sequence.lock'), automatic_retry=False, automatic_resume=False,
        boundary='Only the fixed current short -> original500/live up/down2x10K+500 pair -> full8/full21, plus H192 reuse-full8 continuation. No search or quality-based scheduling.')
    write(output / 'plan.json', plan)
    print(json.dumps(dict(status='prepared_fixed_sequence', plan=str(output / 'plan.json')), ensure_ascii=False), flush=True)


class Sequence:
    def __init__(self, plan):
        self.plan = plan
        self.output = Path(plan['output'])
        self.mutex = threading.Lock()
        self.state = dict(schema='cmcc_fixed_sequence_status_v1', status='running', pid=os.getpid(),
            start_ticks=proc_start(os.getpid()), started_unix=time.time(), argv=sys.argv,
            python=sys.executable, conda_environment=os.environ.get('CONDA_DEFAULT_ENV'),
            environment=environment(''), source_sha256=plan['source_sha256'], branches={}, stages={})

    def update(self, section, name, record):
        with self.mutex:
            self.state[section][name] = record
            write(self.output / 'status.json', self.state)

    def child(self, row):
        check_pins(self.plan['source_sha256'])
        resources = check_resources(row['gpus'])
        return launch_child(row, lambda record: self.update('stages', row['name'],
            dict(**record, initial_resource_check=resources)))

    def arm(self, arm):
        self.child(self.plan['formal'][arm])
        binding, report = validate_formal(self.plan['formal'][arm])
        self.child(self.plan['evaluation'][arm])
        result = validate_evaluation(self.plan['evaluation'][arm]['output'], binding['checkpoint_id'])
        result['formal_seconds'] = report['elapsed_seconds']
        return result

    def adaptive(self):
        from .verify_adaptive_short import verify
        with ThreadPoolExecutor(max_workers=2) as pool:
            waits = list(pool.map(wait_external, self.plan['short']))
        write(self.output / 'short_wait.json', waits)
        pair = verify(*(row['report'] for row in self.plan['short']))
        write(self.output / 'short_pair_verification.json', pair)
        self.update('branches', 'adaptive', dict(status='short_pair_verified'))
        with ThreadPoolExecutor(max_workers=2) as pool:
            arms = {arm: pool.submit(self.arm, arm) for arm in ARMS}
            results, failures = {}, {}
            names = {future: arm for arm, future in arms.items()}
            for future in as_completed(names):
                arm = names[future]
                try:
                    results[arm] = future.result()
                except BaseException:
                    failures[arm] = traceback.format_exc()
                self.update('branches', 'adaptive', dict(status='running', results=results.copy(), errors=failures.copy()))
        record = dict(status='complete' if not failures else 'technical_failure', results=results, errors=failures)
        if set(results) == set(ARMS):
            left, right = results['dynamic'], results['rice']
            record['dynamic_minus_rice'] = dict(task_scores={task: left['task_scores'][task] - right['task_scores'][task] for task in TASKS},
                nine_task_score=left['nine_task_score'] - right['nine_task_score'],
                logical_bpw=left['bit_budget']['user_linear_bpw'] - right['bit_budget']['user_linear_bpw'],
                formal_seconds=left['formal_seconds'] - right['formal_seconds'])
        return record

    def h192(self):
        row = self.plan['h192']
        self.child(row)
        result = validate_evaluation(row['output'], row['checkpoint_id'])
        control = read(OLD_MUON)
        require(control['quality_target_evaluated'] and control['compressed_nine_task_score'] == OLD_MUON_SCORE
            and control['checkpoint_id'] == self.plan['old_muon_control']['checkpoint_id']
            and control['baseline_nine_task_score'] == BASELINE_SCORE and set(control['compressed_scores']) == set(TASKS),
            'The fixed original Muon complete-nine control changed.')
        result['nine_delta_vs_old_muon'] = result['nine_task_score'] - OLD_MUON_SCORE
        result['task_delta_vs_old_muon'] = {task: result['task_scores'][task] - control['compressed_scores'][task] for task in TASKS}
        result['old_muon_nine_task_score'] = OLD_MUON_SCORE
        result['old_muon_report'] = str(OLD_MUON)
        result['old_muon_checkpoint_id'] = control['checkpoint_id']
        result['old_muon_report_sha256'] = self.plan['old_muon_control']['sha256']
        return dict(status='complete', result=result)

    def run(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = dict(adaptive=pool.submit(self.adaptive), h192=pool.submit(self.h192))
            names = {future: branch for branch, future in futures.items()}
            for future in as_completed(names):
                branch = names[future]
                try:
                    record = future.result()
                except BaseException:
                    record = dict(status='technical_failure', traceback=traceback.format_exc())
                self.update('branches', branch, record)
        check_pins(self.plan['source_sha256'])
        self.state.update(status='complete' if all(row['status'] == 'complete' for row in self.state['branches'].values())
                          else 'technical_failure', finished_unix=time.time())
        write(self.output / 'result.json', self.state)
        write(self.output / 'status.json', self.state)
        return 0 if self.state['status'] == 'complete' else 1


def run(plan_path):
    cpu_environment()
    plan_path = plan_path.resolve()
    plan = read(plan_path)
    require(plan['schema'] == 'cmcc_fixed_sequence_20261030_v1' and Path(plan['output']) == plan_path.parent,
        'Use this prepared fixed sequence plan.')
    descriptor = os.open(plan['lock_file'], os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (plan_path.parent / 'status.json').exists(), 'This plan already ran; no resume/retry.')
        check_pins(plan['source_sha256'])
        runner = Sequence(plan)
        write(plan_path.parent / 'metadata.json', dict(**runner.state,
            plan_sha256=sha(plan_path), package_versions={name: importlib.metadata.version(name)
                for name in ('torch', 'transformers', 'accelerate', 'trl', 'peft', 'datasets', 'lm_eval')}))
        write(plan_path.parent / 'status.json', runner.state)
        try:
            code = runner.run()
        except BaseException:
            runner.state.update(status='technical_failure', finished_unix=time.time(), traceback=traceback.format_exc())
            write(plan_path.parent / 'result.json', runner.state)
            write(plan_path.parent / 'status.json', runner.state)
            code = 1
        (plan_path.parent / 'sequence.exit').write_text(str(code) + '\n', encoding='utf-8')
        return code
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--prepare', type=Path)
    choice.add_argument('--plan', type=Path)
    parser.add_argument('--h192-handoff', type=Path)
    args = parser.parse_args()
    if args.prepare:
        require(args.h192_handoff is not None, 'Prepare requires the validated H192 continuation handoff.')
        prepare(args.prepare, args.h192_handoff.resolve())
        return 0
    require(args.h192_handoff is None, 'The runtime command consumes only its fixed prepared plan.')
    return run(args.plan)


if __name__ == '__main__':
    raise SystemExit(main())
