"""One common prefix, five fixed continuations, then each complete nine-task evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARMS = ('baseline', 'risk', 'matched', 'half_lr', 'freeze_codes')
DATA = Path('/data/star/VAELLM/qwen_deoscillation_20261026')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pins():
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(HERE.glob('*.py'))}


def check_pins(expected):
    require(pins() == expected, 'The finite experiment source changed after launch.')


def gpu_free(gpu):
    rows = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                          check=True, text=True, capture_output=True).stdout.splitlines()
    devices = {int(row.split(',')[0]): row.split(',')[1].strip() for row in rows}
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
                               check=True, text=True, capture_output=True).stdout.splitlines()
    require(gpu in devices and not any(row.split(',')[0].strip() == devices[gpu] for row in processes),
            f'Assigned GPU {gpu} is occupied; no wait, retry or interruption of other tasks.')


def environment(gpu=''):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT),
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', FLASH_ATTENTION_DETERMINISTIC='1',
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    return env


def stage(name, argv, output, chain, *, gpu='', seconds):
    if gpu != '':
        gpu_free(int(gpu))
    check_pins(chain['source_sha256'])
    command = ['timeout', '--signal=TERM', '--kill-after=20s', str(seconds) + 's', *argv]
    path = output / (name + '.log')
    row = {'argv': command, 'gpu': gpu, 'log': str(path), 'started_unix': time.time()}
    with path.open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment(gpu), stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT)
        row['pid'] = process.pid
        row['start_ticks'] = int(Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
        chain['stages'][name] = row
        write(output / 'chain.json', chain)
        code = process.wait()
    row.update(exit_code=code, finished_unix=time.time())
    write(output / 'chain.json', chain)
    require(code == 0, f'{name} failed with exit {code}; see {path}. No automatic retry.')
    check_pins(chain['source_sha256'])


def export_evaluate(endpoint, gpu, output, chain, tag):
    stage('export_' + tag, [sys.executable, '-B', str(HERE / 'export.py'), '--endpoint', str(endpoint),
        '--output', str(DATA / tag / 'final_model')], output, chain, seconds=1800)
    gpu_free(gpu)
    stage('full9_' + tag, [sys.executable, '-B', str(HERE / 'evaluate_full9.py'),
        '--endpoint', str(endpoint), '--gpu', str(gpu)], output, chain, seconds=95000)


def run(args):
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / 'chain.json').exists(), 'Never overwrite a finite chain.')
    chain = {'status': 'running', 'started_unix': time.time(), 'pid': os.getpid(),
             'source_sha256': pins(), 'stages': {}, 'automatic_retry': False, 'resource_polling': False,
             'arm': args.arm, 'GPU5_reserved_for_existing_CAT': True}
    try:
        if args.arm:
            require(args.anchor is not None and args.gpu == ARMS.index(args.arm), 'Use the fixed arm/GPU mapping.')
            stage('train', [sys.executable, '-B', str(HERE / 'run.py'), '--phase', 'arm', '--arm', args.arm,
                '--anchor', str(args.anchor.resolve()), '--output-root', str(output / 'training'),
                '--max-seconds', '14400'], output, chain, gpu=args.gpu, seconds=14580)
            report = read(output / 'training/report.json')
            require(report['status'] == 'training_complete' and report['completed_updates'] == 256,
                    'Only a real completed continuation may be exported.')
            export_evaluate(output / 'training/step_0256', args.gpu, output, chain, args.arm)
        else:
            require(args.validation is not None, 'The common prefix requires the completed real validation.')
            validation = args.validation.resolve()
            report = read(validation / 'report.json')
            proof = read(validation / 'fresh_gpu.json')
            audit = read(validation / 'step_0002/native_audit.json')
            require(report['status'] == 'short_passed' and report['completed_updates'] == 2
                and proof['status'] == 'verified' and proof['target_deployment_equal']
                and proof['checkpoint_id'] == audit['checkpoint_id'] and audit['status'] == 'verified',
                'Real long-input updates, live state preservation, export and fresh consumption must pass first.')
            require(all(chain['source_sha256'][row['path']] == row['sha256'] for row in report['source_pins']),
                    'The validated training implementation changed.')
            chain['validation'] = str(validation)
            stage('common', [sys.executable, '-B', str(HERE / 'run.py'), '--phase', 'anchor',
                '--output-root', str(output / 'common'), '--max-seconds', '21600'],
                output, chain, gpu=0, seconds=21780)
            common = read(output / 'common/report.json')
            if common['status'] == 'anchor_ready':
                anchor = Path(common['shared_anchor']['path'])
                processes = []
                for gpu, arm in enumerate(ARMS):
                    gpu_free(gpu)
                    destination = output / 'arms' / arm
                    destination.mkdir(parents=True, exist_ok=False)
                    argv = [sys.executable, '-B', str(HERE / 'run_chain.py'), '--output-root', str(destination),
                            '--arm', arm, '--anchor', str(anchor), '--gpu', str(gpu)]
                    with (destination / 'driver.log').open('x') as stream:
                        child = subprocess.Popen(argv, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL,
                                                 stdout=stream, stderr=subprocess.STDOUT)
                    processes.append((arm, child))
                    chain['stages'][arm] = {'pid': child.pid, 'argv': argv, 'gpu': gpu,
                                          'output': str(destination), 'started_unix': time.time()}
                    write(output / 'chain.json', chain)
                codes = {arm: child.wait() for arm, child in processes}
                chain['arm_exit_codes'] = codes
                require(all(code == 0 for code in codes.values()), 'A fixed arm failed; retain its evidence without retry.')
            else:
                require(common['status'] == 'training_complete' and common['completed_updates'] == 256
                        and common['only_baseline_continues'], 'Unexpected common trajectory state.')
                chain['comparison_not_triggered'] = common['comparison_not_triggered']
                export_evaluate(output / 'common/step_0256', 0, output, chain, 'baseline_only')
        chain['status'] = 'complete'
    except BaseException as error:
        chain.update(status='failed', error=repr(error))
        raise
    finally:
        chain['finished_unix'] = time.time()
        write(output / 'chain.json', chain)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--anchor', type=Path)
    parser.add_argument('--gpu', type=int, choices=range(5))
    run(parser.parse_args())
