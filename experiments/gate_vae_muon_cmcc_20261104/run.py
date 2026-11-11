"""Finite gate optimizer sequence; short, formal and missing control-seven only."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import bindings as b


def command(component, *arguments):
    return [str(b.PYTHON), '-B', '-u', '-m', __package__+'.'+component, *map(str,arguments)]


def execute(phase):
    b.prepared()
    b.require(os.environ.get('CONDA_DEFAULT_ENV') == 'bitvae' and Path(sys.executable).resolve() == b.PYTHON.resolve()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Use a CPU bitvae controller with assigned GPU workers.')
    root = b.RESULT / 'control' if phase == 'control' else b.OUTPUT / 'muon'
    root.mkdir(parents=True, exist_ok=True)
    path = root / (phase + '_run.json')
    b.require(not path.exists(), 'Never repeat an existing finite sequence.')
    gpu = 2 if phase == 'control' else 0
    resource = b.base.free_gpu(gpu)
    if phase == 'formal':
        short = b.read(root / 'short_run.json')
        b.require(short['status'] == 'completed_functional_short' and short['exit_code'] == 0,
            'Require both real short stages and independent full-state/BF16-logit fresh before fresh formal fitting.')
    record = dict(status='running', phase=phase, contract=b.CONTRACT, resource_check=resource,
        pid=os.getpid(), start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]),
        started_unix=time.time(), argv=[sys.executable,*sys.argv], stages={},
        source_sha256=b.read(b.RESULT / 'prepared.json')['source_sha256'], automatic_retry=False,
        automatic_recovery_or_suffix=False)
    b.write(path, record)

    def stage(name, component, arguments, seconds, *, cpu=False, deadline=None):
        b.base.check_pins(record['source_sha256'])
        if deadline is not None:
            seconds = min(seconds, int(deadline-time.time()))
        b.require(seconds > 0, 'The fixed phase budget has expired; no continuation.')
        argv = ['timeout','--signal=TERM','--kill-after=20s',str(seconds)+'s',*command(component,*arguments)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if cpu else str(gpu), PYTHONHASHSEED='31',
            PYTHONPATH=str(b.REPO), PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
            OPENBLAS_NUM_THREADS='8', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        log = root / (name+'.log')
        with log.open('x',encoding='utf-8') as stream:
            worker = subprocess.Popen(argv,cwd=b.REPO,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            record['stages'][name] = dict(argv=argv,pid=worker.pid,started_unix=time.time(),
                CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'],log=str(log))
            b.write(path,record)
            code = worker.wait()
        record['stages'][name].update(exit_code=code,finished_unix=time.time())
        (root / (name+'.exit')).write_text(str(code)+'\n',encoding='utf-8')
        b.write(path,record)
        b.require(code == 0, 'A fixed dependency failed; no retry/continuation: '+name)

    try:
        if phase == 'short':
            deadline = time.time()+1800
            stage('short_fit','fit',['--phase','short'],1800,deadline=deadline)
            stage('short_strict_reload','evaluate',['--component','fresh_cpu','--phase','short'],1800,cpu=True,deadline=deadline)
            stage('short_deployment_fresh','evaluate',['--component','fresh_gpu','--phase','short'],1800,deadline=deadline)
            record.update(status='completed_functional_short',exit_code=0,
                boundary='Two-stage linear2 functional short; formal restarts from genuine after-o and linear10000.')
        elif phase == 'control':
            deadline = time.time()+3600
            stage('seven','evaluate',['--component','seven','--arm','control'],3600,deadline=deadline)
            stage('full8','evaluate',['--component','full8','--arm','control'],3600,cpu=True,deadline=deadline)
            record.update(status='completed_missing_control_seven_and_full8',exit_code=0,
                reused_MMLU=str(b.CONTROL_ROOT / 'mmlu.json'),reused_readout=str(b.CONTROL_ROOT / 'readout.json'))
        else:
            stage('fit','fit',['--phase','formal'],5400)
            deadline = time.time()+1800
            stage('strict_reload','evaluate',['--component','fresh_cpu'],1800,cpu=True,deadline=deadline)
            stage('deployment_fresh','evaluate',['--component','fresh_gpu'],1800,deadline=deadline)
            deadline = time.time()+5400
            stage('mmlu','evaluate',['--component','mmlu'],5400,deadline=deadline)
            stage('seven','evaluate',['--component','seven'],5400,deadline=deadline)
            stage('full8','evaluate',['--component','full8'],5400,cpu=True,deadline=deadline)
            stage('readout','evaluate',['--component','readout'],1800)
            control = b.read(b.RESULT / 'control/full8_summary.json')
            candidate = b.read(root / 'full8_summary.json')
            reference = b.read(b.CONTROL_ROOT / 'readout.json')
            readout = b.read(root / 'readout.json')
            b.require(readout['fixed_train_input']['input_ids_sha256'] == reference['fixed_train_input']['input_ids_sha256'],
                'The paired deployment readout must use the same original train8x256 tokens.')
            record.update(status='completed_gate_Muon_fit_fresh_full8_readout',exit_code=0,
                checkpoint_id=candidate['checkpoint_id'],full8_mean_percent=candidate['mean_percent'],
                full8_delta_vs_control=candidate['mean_percent']-control['mean_percent'],
                task_deltas_vs_control={task:(score-control['task_metrics'][task])*100 for task,score in candidate['task_metrics'].items()},
                deployment_NMSE=dict(control=reference['deployed_native_BF16'],candidate=readout['deployed_native_BF16']),
                boundary='One partial-stage point only. Root reviews full8/knowledge/error/cost tradeoffs; no500/up/down/full9 or additional grids.')
    except BaseException:
        record.update(status='failed',exit_code=1,error=traceback.format_exc())
        raise
    finally:
        record.update(finished_unix=time.time(),elapsed_seconds=time.time()-record['started_unix'])
        b.write(path,record)
        (root / (phase+'_run.exit')).write_text(str(record['exit_code'])+'\n',encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('short','formal','control'))
    execute(parser.parse_args().phase)
