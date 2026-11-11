"""One fixed arm: fit, fresh strict readback and complete same-node MMLU/readout."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

from . import bindings as b


def run(arm):
    b.require(Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae', 'Caller must activate bitvae.')
    settings = b.ARMS[arm]
    prepared = b.read(b.RESULT / 'prepared.json')
    b.require(prepared['status'] == 'cpu_verified', 'CPU preparation is incomplete.')
    b.check_pins(prepared['source_sha256'])
    output = b.OUTPUT / arm
    report_path = output / 'run.json'
    b.require(not report_path.exists(), 'Never repeat an existing arm.')
    resources = b.free_gpu(settings['gpu'])
    record = dict(status='running', arm=arm, settings=settings, pid=os.getpid(),
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        started_unix=time.time(), command=[sys.executable, *sys.argv], source_sha256=prepared['source_sha256'],
        resource_check=resources, automatic_retry=False, stages={})
    b.write(report_path, record)

    def stage(name, component, arguments, seconds, *, cpu=False, result_dir=output):
        command = ['timeout', '--signal=TERM', '--kill-after=20s', str(seconds)+'s', str(b.PYTHON),
            '-B', '-u', '-m', 'experiments.cat_geometry_cohort_cmcc_20261102.'+component, *map(str, arguments)]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='' if cpu else str(settings['gpu']), PYTHONHASHSEED='31',
            HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8')
        result_dir.mkdir(parents=True, exist_ok=True)
        with (result_dir / (name+'.log')).open('x', encoding='utf-8') as log:
            log.write('CUDA_VISIBLE_DEVICES='+env['CUDA_VISIBLE_DEVICES']+'\n'+shlex.join(command)+'\n')
            log.flush()
            process = subprocess.Popen(command, cwd=b.REPO, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT)
            record['stages'][name] = dict(command=command, pid=process.pid, started_unix=time.time(),
                CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'], log=str(result_dir / (name+'.log')))
            b.write(report_path, record)
            code = process.wait()
        (result_dir / (name+'.exit')).write_text(str(code)+'\n', encoding='utf-8')
        record['stages'][name].update(exit_code=code, finished_unix=time.time())
        b.write(report_path, record)
        b.require(code == 0, 'Fixed dependency '+name+' failed; no retry or dependent continuation.')

    try:
        stage('short_fit', 'fit', ['--arm', arm, '--phase', 'short'], 1800)
        short = b.read(output / 'short/report.json')
        stage('short_strict_reload', 'prepare', ['--verify', short['raw2']['path'], '--expected', output / 'short/report.json',
            '--output', output / 'short/strict_reload.json', '--arm', arm], 1800, cpu=True)
        stage('short_readout', 'evaluate', ['--arm', arm, '--component', 'readout', '--phase', 'short',
            '--output', output / 'short/readout.json'], 1800)
        record['new_shape_validation'] = dict(status='passed', short_fit=str(output / 'short/report.json'),
            strict_reload=str(output / 'short/strict_reload.json'), group8_readout=str(output / 'short/readout.json'),
            formal_initialization='Fresh original-parent fork, never short weights.')
        b.write(report_path, record)
        stage('fit', 'fit', ['--arm', arm, '--phase', 'formal'], 5400)
        fitted = b.read(output / 'formal/report.json')
        stage('strict_reload', 'prepare', ['--verify', fitted['raw2']['path'], '--expected', output / 'formal/report.json',
            '--output', output / 'formal/strict_reload.json', '--arm', arm], 1800, cpu=True)
        stage('mmlu', 'evaluate', ['--arm', arm, '--component', 'mmlu', '--output', output / 'mmlu.json'], 5400)
        stage('readout', 'evaluate', ['--arm', arm, '--component', 'readout', '--output', output / 'readout.json'], 1800)
        record.update(status='completed_fit_fresh_full_mmlu_readout', exit_code=0,
            raw2=fitted['raw2'], boundary='Stage mechanism screening only; no automatic winner or full suffix.')
    except BaseException:
        record.update(status='failed', exit_code=1, error=traceback.format_exc())
        raise
    finally:
        record['finished_unix'] = time.time()
        b.write(report_path, record)
        (output / 'run.exit').write_text(str(record['exit_code'])+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=tuple(b.ARMS), required=True)
    run(parser.parse_args().arm)
