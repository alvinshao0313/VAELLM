"""One fixed GPU2 chain, one4h absolute deadline and declared sub-budgets."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments.cmcc_runtime_20261030.paths import ARTIFACTS, PYTHON, REPO, TOPIC
from experiments.night_sequence_20261026.queue_wait import free_gpu, process_start, require, sha256, write_json

NAME = 'down_tail24_full_cmcc_20261104'
LIMITS = dict(validation=1800, fit_route_readout=7200, evaluation=3600, capture=1800)


def terminate_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    require(Path(sys.executable).resolve() == PYTHON.resolve() and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae'
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '2' and not root.exists()
        and os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True'
        and root.is_relative_to((ARTIFACTS / NAME).resolve()), 'Use caller bitvae, GPU2 and one independent new formal output.')
    prepared_path = TOPIC / NAME / 'prepared.json'
    prepared = json.loads(prepared_path.read_text(encoding='utf-8'))
    require(prepared['status'] == 'CPU_full_bridge_verified' and prepared['exit_code'] == 0, 'Require completed CPU bridge evidence.')
    for path, digest in prepared['source_sha256'].items():
        require(sha256(Path(path)) == digest, 'Prepared source changed: ' + path)
    resource = free_gpu(2)
    root.mkdir(parents=True)
    started = time.time()
    deadline = started + 14400
    report = dict(status='running', pid=os.getpid(), start_ticks=process_start(os.getpid()),
        command=[sys.executable, *sys.argv], started_unix=started, deadline_unix=deadline,
        total_wall_limit_seconds=14400, subbudget_limits_seconds=LIMITS, spent_seconds={key: 0. for key in LIMITS},
        GPU=resource, source_prepared=str(prepared_path), source_prepared_sha256=sha256(prepared_path),
        source_sha256=prepared['source_sha256'], retries=0, stages=[],
        runtime_environment={name: os.environ.get(name) for name in ('CONDA_DEFAULT_ENV', 'CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF')},
        accounting='Group local readout conservatively charged inside fit/route2h; capture alone capped. No budget transferred to new tasks.')
    output = root / 'sequence.json'
    write_json(output, report)
    base = [str(PYTHON), '-B', '-u', '-m', 'experiments.' + NAME + '.run', '--root', str(root), '--deadline-unix', str(deadline)]
    units = [('capture', 'capture', base + ['--component', 'capture']),
        ('control_bridge_save_fresh', 'validation', base + ['--component', 'control'])]
    units += [(f'group_{index:02d}', 'fit_route_readout', base + ['--component', 'group', '--group-index', str(index)]) for index in range(12)]
    units.append(('candidate_bridge_save_fresh', 'validation', base + ['--component', 'candidate']))
    for arm in ('control', 'k24'):
        units.append((arm + '_full8_setup', 'evaluation', base + ['--component', 'full8_setup', '--arm', arm]))
        for component in ('mmlu', 'seven', 'summary'):
            command = [str(PYTHON), '-B', '-u', '-m', 'experiments.' + NAME + '.evaluate',
                '--component', component, '--root', str(root), '--arm', arm, '--deadline-unix', str(deadline)]
            units.append((arm + '_full8_' + component, 'evaluation', command))
    units.append(('complete_pair_summary', 'evaluation', base + ['--component', 'summary']))
    process = None
    try:
        for name, category, command in units:
            allowance = min(deadline - time.time(), LIMITS[category] - report['spent_seconds'][category])
            require(allowance > 0, 'Common deadline or declared sub-budget exhausted before ' + name)
            row = dict(name=name, category=category, command=command, started_unix=time.time(),
                maximum_remaining_seconds=allowance, common_deadline_unix=deadline, log=str(root / (name + '.log')), status='running')
            report['stages'].append(row)
            with (root / (name + '.log')).open('x') as log:
                process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                row.update(pid=process.pid, start_ticks=process_start(process.pid))
                write_json(output, report)
                try:
                    code = process.wait(timeout=allowance)
                except subprocess.TimeoutExpired:
                    terminate_group(process.pid)
                    process.wait()
                    code = 124
                    row['whole_process_group_deadline_enforced'] = True
            if code != 0:
                # A failed leader may already be gone while its nested GPU child lives.
                terminate_group(process.pid)
            row.update(exit_code=code, returncode=process.returncode, finished_unix=time.time(),
                elapsed_seconds=time.time() - row['started_unix'], status='completed' if code == 0 else 'failed')
            report['spent_seconds'][category] += row['elapsed_seconds']
            write_json(output, report)
            require(code == 0, 'Technical failure stops dependent execution without retry: ' + name)
        result = json.loads((root / 'result.json').read_text(encoding='utf-8'))
        require(result['status'] == 'completed_full36_raw_phase' and result['exit_code'] == 0, 'Require the complete valid raw-stage result.')
        report.update(status='completed_full36_raw_phase', exit_code=0,
            candidate_beats_both_raw_controls=result['candidate_beats_both_raw_controls'], no_automatic_suffix=True)
    except BaseException as error:
        if process is not None:
            terminate_group(process.pid)
            process.wait()
        report.update(status='failed_stop_no_followup', exit_code=1, error=repr(error))
        raise
    finally:
        report.update(finished_unix=time.time(), elapsed_seconds=time.time() - started)
        write_json(output, report)
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
