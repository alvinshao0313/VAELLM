"""One finite GPU-0 continuation: real-shape validation, then the fixed down pair."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from experiments.night_sequence_20261026.queue_wait import (
    free_gpu,
    process_start,
    require,
    run_stage,
    sha256,
    wait_for_process,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path('/data/star/VAELLM/adaptive_vector_rvq_20261027')
PREDECESSOR = {'pid': 4172937, 'start_ticks': 238351746}
CORE = (
    'litebsq/autoencoder.py', 'litebsq/parallel_layers.py',
    'litebsq/llm_vae.py', 'litebsq/vae_args.py', 'litebsq/bsq.py',
    'litebsq/misc.py', 'litebsq/bitpack.py',
    'litebsq/protected_channel_quant.py', 'rotation/weight_preprocess.py',
    'train_utils/cat_train_data.py', 'train_utils/cat_train_pipeline.py',
    'experiments/night_sequence_20261026/queue_wait.py',
)


def check_result(path: Path) -> dict:
    report = json.loads(path.read_text(encoding='utf-8'))
    require(report['status'] == 'completed', f'Incomplete experiment: {path}')
    for name in ('budget', 'freshdecode', 'freeze'):
        require(report[name]['passed'] is True, f'{name} failed: {path}')
    require(report['allfinite'] is True, f'Nonfinite experiment: {path}')
    return {key: report[key] for key in ('status', 'budget', 'freshdecode', 'freeze', 'allfinite')}


def prepare() -> None:
    require(Path(sys.executable).resolve() == Path('/home/star/miniconda3/envs/bitvae/bin/python').resolve(),
            'Use the confirmed bitvae Python.')
    path = OUTPUT / 'queue_prepared.json'
    require(not path.exists(), 'Do not overwrite the prepared continuation.')
    cpu = OUTPUT / 'cpu_contract' / 'report.json'
    checked = check_result(cpu)
    manifest = OUTPUT / 'prepared' / 'prepared.json'
    require(manifest.is_file(), 'The real input preparation is missing.')
    files = sorted((ROOT / 'experiments/adaptive_vector_rvq').glob('*.py'))
    files.extend(ROOT / relative for relative in CORE)
    files.extend((manifest, cpu))
    require(all(file.is_file() for file in files), 'A required source file is absent.')
    require(process_start(PREDECESSOR['pid']) in (None, PREDECESSOR['start_ticks']),
            'The predecessor PID was reused.')
    write_json(path, {
        'status': 'prepared_not_started', 'predecessor': PREDECESSOR, 'gpu': 0,
        'cpu_validation': checked,
        'bindings': [{'path': str(file), 'sha256': sha256(file)} for file in files],
        'commands': [
            [sys.executable, '-m', 'experiments.adaptive_vector_rvq.run',
             '--prepared-dir', str(OUTPUT / 'prepared'), '--output-dir', str(OUTPUT / 'gpu_contract'),
             '--steps', '2', '--layers', '0'],
            [sys.executable, '-m', 'experiments.adaptive_vector_rvq.run',
             '--prepared-dir', str(OUTPUT / 'prepared'), '--output-dir', str(OUTPUT / 'formal'),
             '--steps', '2000', '--layers', '0,18,35'],
        ],
    })


def run() -> None:
    prepared = json.loads((OUTPUT / 'queue_prepared.json').read_text(encoding='utf-8'))
    path = OUTPUT / 'queue.json'
    require(not path.exists(), 'This finite continuation has already been started.')
    require(prepared['status'] == 'prepared_not_started' and prepared['predecessor'] == PREDECESSOR
            and prepared['gpu'] == 0, 'Continuation identity differs.')
    report = {'status': 'waiting_for_predecessor_exit', 'gpu': 0, 'predecessor': PREDECESSOR,
              'pid': os.getpid(), 'start_ticks': process_start(os.getpid()), 'stages': {}}
    write_json(path, report)
    try:
        wait_for_process(PREDECESSOR['pid'], PREDECESSOR['start_ticks'])
        for binding in prepared['bindings']:
            require(sha256(Path(binding['path'])) == binding['sha256'],
                    'A bound source/input changed: ' + binding['path'])
        report['resource'] = free_gpu(0)
        report['status'] = 'running'
        write_json(path, report)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4',
                           MKL_NUM_THREADS='4', PYTHONPATH=str(ROOT),
                           CUBLAS_WORKSPACE_CONFIG=':4096:8',
                           PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        for label, command in zip(('gpu_contract', 'formal'), prepared['commands'], strict=True):
            run_stage(command, cwd=ROOT, environment=environment, log=OUTPUT / f'{label}.log',
                      report=report, report_path=path, name=label)
            report['stages'][label]['validation'] = check_result(OUTPUT / label / 'report.json')
            write_json(path, report)
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_json(path, report)
        raise
    write_json(path, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run'))
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else run)()


if __name__ == '__main__':
    main()
