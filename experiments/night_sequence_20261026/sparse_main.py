"""Finite C then G raw abs005/Rice7 candidate entry for the fixed night queue."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback

import sparse_common as c


def stage(name, command, *, gpu='', seconds=3600, directory, record=None, report_path=None):
    directory = Path(directory)
    # Inner quality timers stay in the outer full9 process group, so its8h cap releases the GPU too.
    timer_mode = ['--foreground'] if name in ('mmlu', 'seven', 'longbench') else []
    wrapped = ['timeout', *timer_mode, '--signal=TERM', '--kill-after=20s', str(seconds) + 's', *command]
    item = dict(command=wrapped, gpu=gpu, started_unix=time.time(), timeout_seconds=seconds)
    if record is not None:
        record['stages'][name] = item
        c.write(report_path, record)
    with (directory / (name + '.log')).open('x', encoding='utf-8') as log:
        log.write(str(wrapped) + '\n')
        log.flush()
        result = subprocess.run(wrapped, cwd=c.SNAPSHOT, env=c.environment(gpu), stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    (directory / (name + '.exit')).write_text(str(result.returncode) + '\n', encoding='utf-8')
    item.update(exit_code=result.returncode, finished_unix=time.time())
    if record is not None:
        c.write(report_path, record)
    c.require(result.returncode == 0, f'{name} exited {result.returncode}; see {directory / (name + ".log")}')
    return result.returncode


def run(profile, gpu, output, artifact):
    c.cpu_only()
    c.require(gpu == 4 and output == c.OUTPUT / profile and artifact == c.ARTIFACT / profile,
              'Only the fixedGPU4 C/G candidate paths are authorized for this entry.')
    c.require(not (output / 'pipeline.json').exists(), 'One fresh execution only; no retry.')
    if not (output / 'prepared.json').exists():
        import sparse_build
        sparse_build.prepare(profile, output, artifact)
    prepared = c.check_prepared(output / 'prepared.json')
    c.require(prepared['profile'] == profile, 'Wrong profile preparation.')
    record = dict(status='running', profile=profile, gpu=gpu, source_checkpoint_id=prepared['source_checkpoint_id'],
        started_unix=time.time(), stages={}, automatic_retry=False, automatic_resume=False,
        fixed_density=.005, full9_timeout_seconds=28800, build_timeout_seconds=5400,
        export_timeout_seconds=2400, independent_consumer_timeout_seconds=1800)
    report_path = output / 'pipeline.json'
    c.write(report_path, record)
    def invoke(name, argv, *, gpu='', seconds=3600, directory=output):
        return stage(name, argv, gpu=gpu, seconds=seconds, directory=directory,
                     record=record, report_path=report_path)
    try:
        occupied = subprocess.check_output(['nvidia-smi', '--id=4', '--query-compute-apps=pid',
                                            '--format=csv,noheader,nounits'], text=True)
        c.require(not occupied.strip(), 'Assigned GPU4 is occupied: ' + occupied.strip())
        common = ['--prepared', str(output / 'prepared.json'), '--artifact-root', str(artifact)]
        invoke('build', [str(c.PYTHON), '-B', '-u', str(c.HERE / 'sparse_build.py'), 'build', *common], gpu=gpu, seconds=5400)
        invoke('export', [str(c.PYTHON), '-B', '-u', str(c.HERE / 'sparse_build.py'), 'export', *common], seconds=2400)
        audit = c.read(output / 'audit.json')
        c.require(audit['status'] in ('verified', 'over_budget') and audit['actual_native_save']
                  and audit['independent_fresh_strict'] and audit['serialized_tensors_compared']
                  == audit['fresh_strict_tensors_compared'] == 4287, 'Complete actual native/fresh accounting is required.')
        record.update(final_model=audit['checkpoint_dir'], checkpoint_id=audit['checkpoint_id'], bit_budget=audit['bit_budget'],
            complete_construction=True, complete_native_save=True, complete_fresh_strict=True,
            complete_actual_accounting=True, native_audit=str(output / 'audit.json'))
        if audit['status'] == 'over_budget':
            c.require(not audit['bit_budget']['budget_passed'] and not audit['bit_budget']['contract_violations']
                      and audit['bit_budget']['user_linear_bits'] * 5 > audit['bit_budget']['original_linear_weights'] * 13,
                      'Only the measured fixed-density storage overflow is a valid over_budget outcome.')
            record.update(status='over_budget', quality_evaluation_started=False)
        else:
            checkpoint = Path(audit['checkpoint_dir'])
            invoke('consumer', [str(c.PYTHON), '-B', '-u', str(c.HERE / 'sparse_runtime.py'), 'verify',
                '--checkpoint', str(checkpoint), '--output', str(output / 'gpu_validation.json')], gpu=gpu, seconds=1800)
            invoke('full9', [str(c.PYTHON), '-B', '-u', str(c.HERE / 'sparse_main.py'), 'evaluate',
                '--profile', profile, '--gpu', str(gpu), '--output', str(output), '--artifact-root', str(artifact)], seconds=28800)
            evaluation = c.read(output / 'full9/evaluation.json')
            c.require(evaluation['status'] == 'complete_full9' and evaluation['checkpoint_id'] == audit['checkpoint_id'],
                      'Require the actual complete same-UUID full9 terminal report.')
            record.update(status='complete_full9', quality_evaluation_started=True,
                          evaluation=str(output / 'full9/evaluation.json'), nine_task_report=evaluation['nine_task_report'],
                          nine_task_score=evaluation['nine_task_score'], passed=evaluation['passed'])
    except BaseException as error:
        record.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        c.write(report_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'evaluate'])
    parser.add_argument('--profile', choices=['C', 'G'], required=True)
    parser.add_argument('--gpu', type=int, choices=[4], default=4)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    args = parser.parse_args()
    output, artifact = args.output.resolve(), args.artifact_root.resolve()
    if args.command == 'prepare':
        from sparse_build import prepare
        result = prepare(args.profile, output, artifact)
    elif args.command == 'run':
        result = run(args.profile, args.gpu, output, artifact)
    else:
        from sparse_eval import evaluate
        c.require(output == c.OUTPUT / args.profile and artifact == c.ARTIFACT / args.profile,
                  'Evaluate the fixed profile output only.')
        result = evaluate(artifact / 'final_model', output / 'full9', args.gpu, stage)
    print(dict(status=result['status']), flush=True)


if __name__ == '__main__':
    main()
