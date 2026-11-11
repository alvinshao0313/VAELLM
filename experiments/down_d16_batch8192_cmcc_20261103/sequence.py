"""One reviewed fit/fresh -> full8 -> fixed train readout chain, then stop."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import bindings as b
from experiments.cmcc_runtime_20261030 import sequence as mature


def compare():
    report = b.read(b.OUTPUT / b.ARM / 'formal/report.json')
    full8 = b.read(b.RESULT / b.ARM / 'raw_full8/full8_summary.json')
    readout = b.read(b.RESULT / b.ARM / 'raw_readout.json')
    control, control_readout = b.read(b.CONTROL_FULL8), b.read(b.CONTROL_READOUT)
    identity = report['raw_checkpoint_id']
    b.require(report['status'] == 'completed_formal_raw_down' and report['exit_code'] == 0
        and full8['checkpoint_id'] == readout['checkpoint_id'] == identity
        and full8['leaf_count'] == 64 and full8['sample_count'] == 24742
        and readout['input_ids_sha256'] == control_readout['input_ids_sha256'] == b.INPUT_SHA,
        'Require complete same-UUID fit,64-leaf24742 full8 and the bound train readout.')
    for key in ('user_linear_bits','packed_linear_bits','full_model_tensor_bytes'):
        b.require(report['raw_budget'][key] == b.read(b.CONTROL_REPORT)['raw_budget'][key],
            'Changing fit batch must preserve the actual complete deployment layout/cost.')
    return dict(status='completed_raw_batch_comparison', checkpoint_id=identity,
        control_checkpoint_id=control['checkpoint_id'], mean_percent=full8['mean_percent'],
        control_mean_percent=control['mean_percent'], mean_delta_pp=full8['mean_percent']-control['mean_percent'],
        per_task_delta_pp={name:100*(value-control['task_metrics'][name])
            for name,value in full8['task_metrics'].items()},
        deployment_readout={key:dict(candidate=readout['deployment'][key], control=control_readout['deployment'][key],
            relative_change=readout['deployment'][key]/control_readout['deployment'][key]-1)
            for key in ('weight_MSE','weight_NMSE','output_MSE','output_NMSE')},
        raw_budget=report['raw_budget'], fit_elapsed_seconds=report['elapsed_seconds'],
        fit_peak_allocated=report['peak_allocated'], fit_peak_reserved=report['peak_reserved'],
        full8_elapsed_seconds=b.read(b.RESULT / b.ARM / 'raw_full8/run.json')['elapsed_seconds'],
        readout_elapsed_seconds=readout['elapsed_seconds'],
        full8_path=str(b.RESULT / b.ARM / 'raw_full8/full8_summary.json'),
        readout_path=str(b.RESULT / b.ARM / 'raw_readout.json'),
        boundary='Single seed fixed-step batch/exposure/pooled-entropy recipe study. '
            'Stop regardless of quality; no500,full9,retry,grid or automatic adoption.')


def execute():
    b.cpu_environment()
    prepared = b.read(b.RESULT / 'prepared.json')
    b.check_pins(prepared['source_sha256'])
    b.require(prepared['status'] == 'cpu_verified' and prepared['exit_code'] == 0,
        'CPU source/configuration/batch/dtype/init evidence must pass.')
    result = b.RESULT / b.ARM
    result.mkdir(parents=True, exist_ok=True)
    path = result / 'run.json'
    with (result / 'sequence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        b.require(not path.exists(), 'Never duplicate the one-shot experiment.')
        record = dict(status='running', pid=os.getpid(), start_ticks=mature.proc_start(os.getpid()),
            started_unix=time.time(), command=[sys.executable,*sys.argv], stages={},
            source_sha256=prepared['source_sha256'], source_binding=prepared['source_binding'],
            automatic_retry=False, automatic_resume=False, GPU=4,
            fixed_stage_order=['fit','raw_full8','raw_readout'], maximum_stage_seconds=[5400,5400,1800])
        b.write(path,record)

        def stage(name,module,args,seconds,cpu=False):
            b.check_pins(prepared['source_sha256'])
            resources = mature.check_resources(['4'])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if cpu else '4', PYTHONHASHSEED='31',
                PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                OPENBLAS_NUM_THREADS='8', NUMEXPR_NUM_THREADS='8', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                WANDB_MODE='offline', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            command = ['timeout','--signal=TERM','--kill-after=20s',str(seconds)+'s',
                str(b.PYTHON),'-B','-u','-m','experiments.down_d16_batch8192_cmcc_20261103.'+module,*args]
            log = result / (name+'.log')
            with log.open('x',encoding='utf-8') as stream:
                process = subprocess.Popen(command,cwd=b.REPO,env=env,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                row = dict(command=command,pid=process.pid,start_ticks=mature.proc_start(process.pid),
                    resource_check=resources,gpu=4,started_unix=time.time(),log=str(log),
                    allocator=env['PYTORCH_CUDA_ALLOC_CONF'],status='running')
                record['stages'][name] = row
                record['current_stage'] = name
                b.write(path,record)
                code = process.wait()
            row.update(status='exited',exit_code=code,finished_unix=time.time(),
                elapsed_seconds=time.time()-row['started_unix'])
            (result / (name+'.exit')).write_text(str(code)+'\n',encoding='utf-8')
            b.write(path,record)
            b.require(code == 0,'Actual stage failed; stop dependencies without retry: '+name)

        try:
            stage('fit','run',['--arm',b.ARM,'--phase','formal'],5400)
            stage('raw_full8','evaluate',['--component','raw_full8','--arm',b.ARM],5400,cpu=True)
            stage('raw_readout','evaluate',['--component','readout','--arm',b.ARM],1800)
            comparison = compare()
            b.write(result / 'comparison.json', comparison)
            record.update(status='completed_fit_fresh_full8_readout',exit_code=0,
                checkpoint_id=comparison['checkpoint_id'],comparison=str(result / 'comparison.json'))
        except BaseException:
            record.update(status='failed',exit_code=1,error=traceback.format_exc())
            raise
        finally:
            record.update(finished_unix=time.time(),elapsed_seconds=time.time()-record['started_unix'])
            b.write(path,record)
            (result / 'run.exit').write_text(str(record['exit_code'])+'\n',encoding='utf-8')


if __name__ == '__main__':
    execute()
