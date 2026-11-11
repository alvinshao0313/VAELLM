"""Restore the original allocator environment without changing frozen down code."""
import os

# The established C0 launcher sets this before importing PyTorch. Preserve that
# runtime condition for the same B64/B32 evaluation protocol.
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

import argparse
from copy import deepcopy
import fcntl
import importlib
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback
import select
import signal

from experiments.down_geometry_baseline_cmcc_20261103 import bindings as s
from experiments.cmcc_runtime_20261030 import sequence as mature

FROZEN_RESULT=s.RESULT
REPAIR_RESULT=FROZEN_RESULT / 'allocator_repair'
ENTRY='experiments.down_geometry_runtime_repair_cmcc_20261103'
FROZEN_PINS=s.pins
s.RESULT=REPAIR_RESULT
s.pins=lambda:{**FROZEN_PINS(),str(Path(__file__).resolve()):s.sha(__file__)}


def prepare():
    s.cpu_environment()
    s.require(not (REPAIR_RESULT / 'prepared.json').exists(),'Never replace a repair binding.')
    original=s.read(FROZEN_RESULT / 'prepared.json')
    s.check_pins(original['source_sha256'])
    failed=s.read(FROZEN_RESULT / 'transpose_false/run.json')
    log=FROZEN_RESULT / 'control/raw_full8/mmlu.log'
    s.require(failed['status'] == 'failed' and failed['current_stage'] == 'control_raw_full8'
        and failed['stages']['control_raw_full8']['exit_code'] == 1
        and 'torch.OutOfMemoryError: CUDA out of memory.' in log.read_text(encoding='utf-8'),
        'Repair only the actual documented evaluation allocator omission, with no fit retry.')
    previous=s.read(s.C0.parent / 'single_20261030_v3/plan.json')
    s.require(previous['evaluation']['environment']['PYTORCH_CUDA_ALLOC_CONF'] == os.environ['PYTORCH_CUDA_ALLOC_CONF']
        == 'expandable_segments:True','Keep the actual original C0 evaluation allocator environment.')
    from experiments.down_geometry_baseline_cmcc_20261103 import evaluate,run,recover
    record=deepcopy(original)
    record.update(source_sha256=s.pins(),runtime_repair=dict(kind='restore_original_allocator_environment',
        original_prepared=str(FROZEN_RESULT / 'prepared.json'),original_prepared_sha256=s.sha(FROZEN_RESULT / 'prepared.json'),
        original_source_sha256=original['source_sha256'],original_failed_run=str(FROZEN_RESULT / 'transpose_false/run.json'),
        original_failed_log=str(log),original_failed_log_sha256=s.sha(log),
        original_C0_plan=str(s.C0.parent / 'single_20261030_v3/plan.json'),
        allocator=os.environ['PYTORCH_CUDA_ALLOC_CONF'],CUDA_initialized=s.torch.cuda.is_initialized(),
        frozen_source_files_unchanged=True,protocol_changed=False,automatic_retry=False,
        boundary='One explicit technical repair. Existing GPU0 fit remains untouched; its raw model is reused only after its original controller exits.'))
    s.write(REPAIR_RESULT / 'prepared.json',record)
    print(dict(status='cpu_verified_allocator_repair',path=str(REPAIR_RESULT / 'prepared.json')),flush=True)


def command(target,*args):
    return [str(s.PYTHON),'-B','-u','-m',ENTRY,'--target',target,*map(str,args)]


def sequence(arm):
    s.cpu_environment()
    prepared=s.read(REPAIR_RESULT / 'prepared.json')
    s.check_pins(prepared['source_sha256'])
    original=s.read(FROZEN_RESULT / (arm+'/run.json'))
    if arm == 'd16':
        handoff=s.read(REPAIR_RESULT / 'handoff.json')
        s.require(handoff['status'] == 'fit_completed_and_fresh_verified'
            and handoff['controller']['pid'] == original['pid']
            and mature.proc_start(original['pid']) is None,
            'Never attach a second controller before the exact old controller exits and its retained fit verifies.')
        fitted=s.read(s.OUTPUT / 'd16/formal/report.json')
        s.require(fitted['status'] == 'completed_formal_raw_down' and fitted['exit_code'] == 0,
            'Reuse only the actual completed D16 fit, never repeat it.')
    else:
        s.require(original['status'] == 'failed' and original['exit_code'] == 1,
            'The failed direction controller must already have exited before explicit technical repair.')
    output=REPAIR_RESULT / arm
    output.mkdir(parents=True,exist_ok=True)
    path=output / 'run.json'
    with (output / 'sequence.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        s.require(not path.exists(),'No repeated repair sequence.')
        record=dict(status='running',arm=arm,pid=os.getpid(),start_ticks=mature.proc_start(os.getpid()),
            started_unix=time.time(),command=[sys.executable,*sys.argv],stages={},source_sha256=prepared['source_sha256'],
            runtime_repair=prepared['runtime_repair'],automatic_retry=False,automatic_resume=False)
        s.write(path,record)

        def stage(name,target,args,gpu,seconds,*,cpu=False):
            s.check_pins(prepared['source_sha256'])
            resources=mature.check_resources([str(gpu)])
            argv=['timeout','--signal=TERM','--kill-after=20s',str(seconds)+'s',*command(target,*args)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if cpu else str(gpu),PYTHONHASHSEED='31',
                PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',NUMEXPR_NUM_THREADS='8',
                CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='offline',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            log=output / (name+'.log')
            with log.open('x',encoding='utf-8') as stream:
                process=subprocess.Popen(argv,cwd=s.REPO,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                row=dict(command=argv,pid=process.pid,start_ticks=mature.proc_start(process.pid),resource_check=resources,
                    gpu=gpu,allocator=env['PYTORCH_CUDA_ALLOC_CONF'],started_unix=time.time(),log=str(log))
                record['stages'][name]=row
                record['current_stage']=name
                s.write(path,record)
                code=process.wait()
            row.update(exit_code=code,finished_unix=time.time())
            (output / (name+'.exit')).write_text(str(code)+'\n',encoding='utf-8')
            s.write(path,record)
            s.require(code == 0,'Explicit repair stage failed; no retry or continuation: '+name)

        try:
            gpu=s.ARMS[arm]['gpu']
            if arm == 'transpose_false':
                stage('control_raw_full8','evaluate',['--component','raw_full8','--arm','control'],gpu,5400,cpu=True)
                stage('control_raw_readout','evaluate',['--component','readout','--arm','control'],gpu,1800)
                stage('fit','run',['--arm',arm,'--phase','formal'],gpu,5400)
            stage('raw_full8','evaluate',['--component','raw_full8','--arm',arm],gpu,5400,cpu=True)
            stage('raw_readout','evaluate',['--component','readout','--arm',arm],gpu,1800)
            if arm == 'd16':
                stage('down500','recover',[],gpu,43200)
                stage('final_full9','evaluate',['--checkpoint',s.OUTPUT / 'd16/recovery/final_model',
                    '--pipeline-report',s.OUTPUT / 'd16/recovery/report.json','--output',REPAIR_RESULT / 'd16/final_full9','--gpus','0'],gpu,28800,cpu=True)
                record['complete_nine']=mature.validate_evaluation(REPAIR_RESULT / 'd16/final_full9',
                    s.read(s.OUTPUT / 'd16/recovery/report.json')['final_checkpoint_id'])
            record.update(status='completed_d16_baseline_full9' if arm == 'd16' else 'completed_raw_direction_comparison',exit_code=0)
        except BaseException:
            record.update(status='failed',exit_code=1,error=traceback.format_exc())
            raise
        finally:
            record.update(finished_unix=time.time(),elapsed_seconds=time.time()-record['started_unix'])
            s.write(path,record)
            (output / 'run.exit').write_text(str(record['exit_code'])+'\n',encoding='utf-8')


def process(pid):
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,state=fields[0],ppid=int(fields[1]),pgid=int(fields[2]),sid=int(fields[3]),
        tpgid=int(fields[5]),start_ticks=int(fields[19]),
        command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode())


def handoff():
    """Own exact process handles before interrupting only the new CPU controller."""
    s.cpu_environment()
    prepared=s.read(REPAIR_RESULT / 'prepared.json')
    s.check_pins(prepared['source_sha256'])
    path=REPAIR_RESULT / 'handoff.json'
    s.require(not path.exists(),'Do not repeat a controller handoff.')
    original=s.read(FROZEN_RESULT / 'd16/run.json')
    fit=s.read(s.OUTPUT / 'd16/formal/report.json')
    s.require(original['status'] == fit['status'] == 'running' and original['current_stage'] == 'fit',
        'Handoff applies only to the actual new live D16 fit stage.')
    controller=process(original['pid'])
    timeout=process(original['stages']['fit']['pid'])
    worker=process(fit['pid'])
    s.require(controller['start_ticks'] == original['start_ticks']
        and timeout['start_ticks'] == original['stages']['fit']['start_ticks']
        and worker['start_ticks'] == fit['start_ticks']
        and timeout['ppid'] == controller['pid'] and worker['ppid'] == timeout['pid']
        and timeout['pgid'] == worker['pgid'] == timeout['pid']
        and timeout['pgid'] != controller['tpgid'] == controller['pgid']
        and controller['sid'] == timeout['sid'] == worker['sid']
        and Path(f'/proc/{worker["pid"]}/fd/0').resolve() == Path('/dev/null')
        and Path(f'/proc/{timeout["pid"]}/fd/0').resolve() == Path('/dev/null'),
        'Require the audited separate timeout/worker group, no controlling input, and exact current identities.')
    handles={name:mature.pidfd_open(row['pid']) for name,row in
        (('controller',controller),('timeout',timeout),('worker',worker))}
    record=dict(status='pidfds_owned_before_controller_signal',pid=os.getpid(),start_ticks=mature.proc_start(os.getpid()),
        started_unix=time.time(),command=[sys.executable,*sys.argv],controller=controller,timeout=timeout,worker=worker,
        source_sha256=prepared['source_sha256'],runtime_repair=prepared['runtime_repair'],
        controller_signal='SIGTERM single pidfd only',training_worker_signal=None,timeout_signal=None,
        reason='Restore the established allocator for future children before the original controller starts the known failing raw evaluation.',
        scope='Root authorized interrupting this newly started, undelivered CPU controller only. Retain its real fitting worker and timeout.')
    s.write(path,record)
    try:
        for name,row in (('controller',controller),('timeout',timeout),('worker',worker)):
            s.require(mature.proc_start(row['pid']) == row['start_ticks'],'Identity changed after opening '+name+' pidfd.')
        libc=mature.ctypes.CDLL(None,use_errno=True)
        libc.syscall.restype=mature.ctypes.c_long
        result=libc.syscall(mature.ctypes.c_long(424),mature.ctypes.c_int(handles['controller']),
            mature.ctypes.c_int(signal.SIGTERM),mature.ctypes.c_void_p(0),mature.ctypes.c_uint(0))
        if result != 0:
            error=mature.ctypes.get_errno()
            raise OSError(error,os.strerror(error))
        poller=select.poll()
        poller.register(handles['controller'],select.POLLIN)
        s.require(poller.poll(10000),'The exact controller did not exit after its authorized single-process SIGTERM.')
        record.update(status='controller_exited_fit_continues',controller_exit_observed_unix=time.time(),
            surviving_timeout=process(timeout['pid']),surviving_worker=process(worker['pid']))
        s.require(record['surviving_worker']['start_ticks'] == worker['start_ticks']
            and record['surviving_timeout']['start_ticks'] == timeout['start_ticks']
            and record['surviving_worker']['state'] not in ('Z','X'),
            'The retained actual fitter/timeout must continue after the CPU controller exits.')
        s.write(path,record)
        for name in ('worker','timeout'):
            poller=select.poll()
            poller.register(handles[name],select.POLLIN)
            poller.poll()
        fitted=s.read(s.OUTPUT / 'd16/formal/report.json')
        strict=s.read(s.OUTPUT / 'd16/formal/strict_reload.json')
        s.require(fitted['status'] == 'completed_formal_raw_down' and fitted['exit_code'] == 0
            and fitted['pid'] == worker['pid'] and fitted['start_ticks'] == worker['start_ticks']
            and fitted['source_sha256'] == prepared['runtime_repair']['original_source_sha256']
            and fitted['raw_checkpoint_id'] == strict['checkpoint_id']
            and strict['status'] == 'strict_full_state_and_BF16_logits_verified' and strict['exit_code'] == 0
            and strict['same_input_logits_exact'] and strict['actual_budget']['budget_passed'],
            'Process exit alone is insufficient: require this same completed real fit, actual budget and independent fresh.')
        record.update(status='fit_completed_and_fresh_verified',fit_checkpoint_id=fitted['raw_checkpoint_id'],
            fit_self_reported_exit_code=fitted['exit_code'],raw_report=str(s.OUTPUT / 'd16/formal/report.json'),
            strict_report=str(s.OUTPUT / 'd16/formal/strict_reload.json'),finished_fit_wait_unix=time.time(),
            timeout_OS_exit_code=None,OS_exit_boundary='Non-parent pidfd observes exit only; no fabricated waitpid exit status.')
        s.write(path,record)
    except BaseException:
        record.update(status='failed',error=traceback.format_exc(),finished_unix=time.time())
        s.write(path,record)
        raise
    finally:
        for descriptor in handles.values():
            os.close(descriptor)
    sequence('d16')


def evaluate(arguments):
    from experiments.down_geometry_baseline_cmcc_20261103 import evaluate as frozen
    from experiments.adaptive_class_evaluation_20261030 import bindings as b
    parser=argparse.ArgumentParser()
    parser.add_argument('--component',default='evaluate',choices=('evaluate','full8','predict','summarize','merge','raw_full8','readout'))
    parser.add_argument('--arm',choices=('control',*s.ARMS))
    args,remaining=parser.parse_known_args(arguments)
    frozen.setup()
    b.command=lambda component,*items:command('evaluate','--component',component,*items)
    b.CLI_ARGUMENTS=list(sys.argv)
    s.torch.set_num_threads(8)
    if args.component == 'raw_full8':
        frozen.raw_full8(args.arm)
    elif args.component == 'readout':
        frozen.readout(args.arm)
    else:
        sys.argv=[sys.argv[0],*remaining]
        return importlib.import_module('experiments.adaptive_class_evaluation_20261030.'+args.component).main()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--target',required=True,choices=('prepare','sequence','handoff','run','recover','evaluate'))
    args,remaining=parser.parse_known_args()
    if args.target == 'prepare':
        prepare()
    elif args.target == 'handoff':
        handoff()
    elif args.target == 'sequence':
        parser=argparse.ArgumentParser()
        parser.add_argument('--arm',required=True,choices=tuple(s.ARMS))
        sequence(parser.parse_args(remaining).arm)
    elif args.target == 'evaluate':
        raise SystemExit(evaluate(remaining))
    else:
        sys.argv=[sys.argv[0],*remaining]
        runpy.run_module('experiments.down_geometry_baseline_cmcc_20261103.'+args.target,run_name='__main__')
