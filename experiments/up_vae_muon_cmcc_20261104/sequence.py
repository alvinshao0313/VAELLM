"""Fixed up arm chains, one common deadline each, no retry or recovery suffix."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback

from . import bindings as b
from experiments.cmcc_runtime_20261030 import sequence as mature


def execute(arm):
    b.cpu_environment()
    prepared = b.read(b.RESULT / 'prepared.json');b.check_pins(prepared['source_sha256'])
    gpu, hours = (b.CONTROL_GPU,1.5) if arm == 'control' else (b.ARMS[arm]['gpu'],4)
    output = b.RESULT / arm;output.mkdir(parents=True,exist_ok=True)
    path = output / 'run.json'
    with (output / 'sequence.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        b.require(not path.exists(), 'Do not start or retry an arm twice.')
        started = time.time();deadline = started + hours * 3600
        record = dict(status='running',arm=arm,pid=os.getpid(),start_ticks=mature.proc_start(os.getpid()),
            started_unix=started,absolute_deadline_unix=deadline,upper_GPU_hours=hours,gpu=gpu,
            contract=b.CONTRACT,source_sha256=prepared['source_sha256'],stages={},automatic_retry=False,automatic_recovery_or_full9=False)
        b.write(path,record)
        def stage(name,module,arguments,cap,*,cpu=False):
            b.check_pins(prepared['source_sha256'])
            remaining = min(cap,int(deadline-time.time())-20)
            b.require(remaining > 0, 'The common arm deadline is exhausted; no new dependency starts.')
            resources = mature.check_resources([str(gpu)])
            argv = ['timeout','--signal=TERM','--kill-after=20s',str(remaining)+'s',str(b.PYTHON),'-B','-u','-m',
                'experiments.up_vae_muon_cmcc_20261104.'+module,*map(str,arguments)]
            env = dict(os.environ,CUDA_VISIBLE_DEVICES='' if cpu else str(gpu),PYTHONHASHSEED='31',PYTHONDONTWRITEBYTECODE='1',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',NUMEXPR_NUM_THREADS='8',CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='offline')
            log = output / (name+'.log')
            with log.open('x',encoding='utf-8') as stream:
                process = subprocess.Popen(argv,cwd=b.REPO,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                row = dict(command=argv,pid=process.pid,start_ticks=mature.proc_start(process.pid),gpu=gpu,
                    resource_check=resources,process_group_id=process.pid,started_unix=time.time(),absolute_deadline_unix=deadline,log=str(log))
                record['stages'][name] = row;record['current_stage'] = name;b.write(path,record)
                try:
                    code = process.wait()
                finally:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait()
            row.update(exit_code=code,finished_unix=time.time());b.write(path,record)
            b.require(code == 0, 'Actual arm stage failed/timed out; no retry or dependent continuation: ' + name)
        try:
            if arm != 'control':
                stage('short','run',['--arm',arm,'--phase','short'],1800)
                proof = b.read(b.OUTPUT / arm / 'short/report.json')
                b.require(proof['status'] == 'completed_up_short' and proof['exit_code'] == 0
                    and proof['fit']['technical_original_scheduler'] == [dict(actual_updates=2,schedule_max=10000)]*2
                    and proof['fit']['all_frozen_prefix_fields_exact'] and proof['fit']['protection_and_rotation_fields_exact_to_control'],
                    'Require both actual formal-shape short stages and independent fresh before fresh formal fitting.')
                stage('formal','run',['--arm',arm,'--phase','formal'],5400)
            stage('raw_full8','evaluate',['--component','raw_full8','--arm',arm],5400,cpu=True)
            stage('raw_readout','evaluate',['--component','readout','--arm',arm],1800)
            record.update(status='completed_up_raw_full8_and_readout',exit_code=0)
        except BaseException:
            record.update(status='failed',exit_code=1,error=traceback.format_exc());raise
        finally:
            record.update(finished_unix=time.time(),elapsed_seconds=time.time()-started);b.write(path,record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('muon','control','d16','control_then_d16'),required=True)
    args = parser.parse_args()
    if args.arm in ('muon','control','d16'):execute(args.arm)
    else:execute('control');execute('d16')
