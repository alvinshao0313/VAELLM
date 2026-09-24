"""Run recovery followed by downstream evaluation, without retries.

Arguments not owned here are passed unchanged to the existing recovery CLI.
Every invocation uses a fresh run directory and initial checkpoint load.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu-memory-gib', type=float, required=True)
    parser.add_argument('--evaluate-baseline', action='store_true')
    parser.add_argument('--eval-limit', type=int)
    parser.add_argument('--purpose', choices=('formal', 'short_validation'), default='formal')
    args, recovery_args = parser.parse_known_args()
    if args.purpose == 'formal' and args.eval_limit is not None:
        parser.error('Formal evaluation must not have a sample limit.')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = str(Path(args.checkpoint).resolve())
    recovery = [sys.executable, '-u', '-m', 'experiments.liftquant_recovery.recover',
                '--checkpoint', checkpoint, '--output', str(output / 'recovery'),
                '--gpu-memory-gib', str(args.gpu_memory_gib), *recovery_args]
    evaluation = [sys.executable, '-u', '-m', 'experiments.liftquant_recovery.evaluate_pair',
                  '--a', checkpoint, '--b', str(output / 'recovery' / 'recovered_model'),
                  '--output', str(output / 'evaluation'), '--residency', 'resident',
                  '--gpu-memory-gib', str(args.gpu_memory_gib)]
    if not args.evaluate_baseline:
        evaluation += ['--only', 'B']
    if args.eval_limit is not None:
        evaluation += ['--limit', str(args.eval_limit)]
    run = dict(status='running', stage='recovery', purpose=args.purpose,
               started_unix=time.time(), pid=os.getpid(), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
               initial_checkpoint=checkpoint, recovery_command=recovery, evaluation_command=evaluation,
               baseline_evaluated_here=args.evaluate_baseline)
    path = output / 'run.json'
    def save():
        path.write_text(json.dumps(run, indent=2) + '\n')
    save()
    try:
        print('Starting recovery: ' + ' '.join(recovery), flush=True)
        subprocess.run(recovery, check=True)
        summary = json.loads((output / 'recovery' / 'summary.json').read_text())
        if summary['status'] != 'PASS' or not summary['strict_native_reload'] or not summary['frozen_state_unchanged']:
            raise RuntimeError('Recovery did not satisfy its saved-state contracts.')
        run.update(stage='evaluation')
        save()
        print('Starting downstream evaluation: ' + ' '.join(evaluation), flush=True)
        subprocess.run(evaluation, check=True)
        evaluation_summary = json.loads((output / 'evaluation' / 'summary.json').read_text())
        if evaluation_summary['status'] != 'PASS':
            raise RuntimeError('Evaluation did not pass.')
        run.update(status='completed', stage='complete', exit_code=0, finished_unix=time.time())
        save()
    except BaseException as error:
        run.update(status='failed', error=f'{type(error).__name__}: {error}',
                   exit_code=getattr(error, 'returncode', 1), finished_unix=time.time())
        save()
        raise


if __name__ == '__main__':
    main()
