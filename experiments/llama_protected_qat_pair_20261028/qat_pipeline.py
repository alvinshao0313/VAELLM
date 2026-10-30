"""One finite fixed64/full8/full21/nine chain per arm, with no retry or gate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from qat_common import HERE, CODE, PUBLIC, S, DATA


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def ticks(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('D', 'DQ'), required=True)
    parser.add_argument('--gpu', type=int, choices=range(8), required=True)
    args = parser.parse_args()
    output = HERE / args.arm
    output.mkdir(exist_ok=True)
    path = output / 'pipeline.json'
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu) or path.exists():
        raise ValueError('Use root-assigned physical GPU and a fresh fixed arm chain.')
    prepared = read(HERE / 'prepared.json')
    started = time.monotonic(); deadline = started + prepared['budgets']['full_chain_seconds_per_arm']
    state = {'status': 'running', 'arm': args.arm, 'gpu': args.gpu, 'source_checkpoint_id': prepared['source_checkpoint_id'],
        'wrapper_pid': os.getpid(), 'wrapper_start_ticks': ticks(os.getpid()), 'actual_argv': sys.argv,
        'started_unix': time.time(), 'stages': {}, 'automatic_retry': False, 'automatic_resume': False,
        'full_chain_seconds': prepared['budgets']['full_chain_seconds_per_arm'], 'full9_predeclared': True, 'H50_gate': False}

    def stage(name, argv, timeout, *, public=False, cpu=False, allowed=(0,)):
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError('Fixed24h arm chain budget expired; no automatic extension.')
        actual_timeout = min(timeout, remaining)
        actual = ['timeout', '--signal=TERM', '--kill-after=20s', str(actual_timeout) + 's', *argv]
        cwd = PUBLIC if public else S
        env = os.environ.copy(); env['PYTHONPATH'] = str(cwd)
        if cpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        item = {'argv': actual, 'cwd': str(cwd), 'PYTHONPATH': str(cwd), 'CUDA_VISIBLE_DEVICES': env['CUDA_VISIBLE_DEVICES'],
            'started_unix': time.time(), 'timeout_seconds': actual_timeout, 'log': str(output / (name + '.log'))}
        state['stages'][name] = item; state['active_stage'] = name; write(path, state)
        print(f'STAGE_BEGIN {name} {actual}', flush=True)
        with (output / (name + '.log')).open('x') as stream:
            child = subprocess.Popen(actual, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            item.update(pid=child.pid, start_ticks=ticks(child.pid)); write(path, state)
            code = child.wait()
        item.update(exit_code=code, finished_unix=time.time(), elapsed_seconds=time.time() - item['started_unix'])
        write(path, state)
        print(f'STAGE_END {name} exit={code}', flush=True)
        if code not in allowed:
            raise RuntimeError(f'{name} exited {code}; no automatic retry')

    try:
        stage('train_export', [sys.executable, '-B', '-u', str(CODE / 'qat_run.py'), 'formal64', '--arm', args.arm, '--gpu', str(args.gpu)], 43800)
        final = DATA / args.arm / 'formal64/native_model'
        for name, tasks, batch in (('mmlu', 'mmlu', '64'), ('seven', 'boolq,rte,winogrande,arc_easy,arc_challenge,openbookqa,piqa', '32')):
            stage(name, [sys.executable, '-B', '-u', str(PUBLIC / 'result/qwen3_8b_mmlu65_20261004/lm_eval_sample_artifacts_20261010/cat_eval.py'), '--checkpoint_dir', str(final), '--strict',
                '--eval_lm_eval', '--eval_device', 'cuda:0', '--tasks', tasks, '--num_fewshot', '0', '--lm_batch_size', batch,
                '--tokenizer_use_fast', 'true', '--prewarm_group_size', '8', '--eval_log_dir', str(output / (name + '_logs')),
                '--output_json', str(output / (name + '.json'))], 7200)
        stage('full8_summary', [sys.executable, '-B', '-u', str(CODE / 'qat_evaluation.py'), 'full8_summary', '--arm', args.arm], 600, cpu=True)
        stage('longbench_prepare', [sys.executable, '-B', '-u', str(CODE / 'qat_evaluation.py'), 'prepare', '--arm', args.arm], 600, cpu=True)
        stage('longbench', [sys.executable, '-B', '-u', str(CODE / 'qat_longbench.py'), '--arm', args.arm, '--gpu', str(args.gpu), '--config',
            str(output / 'longbench_config.json'), '--output', str(output / 'longbench'), '--max-runtime-seconds', '32400'], 32460)
        reports = list((output / 'longbench/runs').glob('*.json'))
        if len(reports) != 1:
            raise ValueError('Require exactly one native full21 execution report.')
        terminal = read(reports[0])
        if terminal['status'] != 'partition_complete' or terminal['exit_code'] != 0 or terminal['new_predictions'] != 4750 or terminal['already_present'] != 0:
            raise ValueError('Full21 is incomplete; retain prefix without scoring or automatic continuation.')
        dense = Path(prepared['full9']['dense'])
        baseline8 = [row['path'] for row in read(dense / 'score_verification.json')['baseline_eight_raw_inputs']]
        manifest = read(output / 'longbench_config.json')['archive']['path']
        stage('longbench_score', [sys.executable, '-B', '-u', '-m', 'experiments.longbench_v1.score', '--manifest', manifest,
            '--prediction-dir', str(output / 'longbench'), '--output', str(output / 'longbench/scores.json')], 900, public=True, cpu=True)
        stage('nine', [sys.executable, '-B', '-u', str(CODE / 'qat_nine.py'), '--base_model_path',
            '/home/star/models/meta-llama/Llama-3.1-8B-Instruct', '--checkpoint_dir', str(final), '--baseline_results', *baseline8,
            '--compressed_results', str(output / 'mmlu.json'), str(output / 'seven.json'), '--longbench_manifest', manifest,
            '--baseline_longbench_scores', str(dense / 'scores.json'), '--compressed_longbench_scores', str(output / 'longbench/scores.json'),
            '--output', str(output / 'nine_task_acceptance.json')], 900, public=True, cpu=True, allowed=(0, 1))
        verdict = read(output / 'nine_task_acceptance.json')
        if verdict.get('schema') != 'llama_protected_qat_nine_task_acceptance_v1' or not verdict.get('quality_target_evaluated'):
            raise ValueError('PUBLIC nine-item evidence is invalid.')
        state.update(status='complete', checkpoint_id=read(final / 'checkpoint_meta.json')['checkpoint_id'], nine_task_result=verdict)
        code = 0
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        print(state['traceback'], flush=True); code = 1
    finally:
        state.update(exit_code=code, finished_unix=time.time(), elapsed_seconds=time.monotonic() - started)
        write(path, state)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
