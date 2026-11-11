"""The two reviewed one-shot chains; fixed dependencies, no retry or reservation."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import bindings as b
from experiments.cmcc_runtime_20261030 import sequence as mature


def execute(arm):
    b.cpu_environment()
    prepared=b.read(b.RESULT / 'prepared.json')
    b.check_pins(prepared['source_sha256'])
    short=b.read(b.OUTPUT / 'd16/short/report.json')
    b.require(short['status'] == 'completed_new_path_short' and short['exit_code'] == 0
        and short['source_sha256'] == prepared['source_sha256'] and short['fit']['protection_and_rotation_fields_exact_to_control']
        and short['fit']['all_frozen_prefix_fields_exact'],'The one new formal-shape path validation must pass first.')
    result=b.RESULT / arm
    result.mkdir(parents=True,exist_ok=True)
    path=result / 'run.json'
    with (result / 'sequence.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        b.require(not path.exists(),'Never duplicate an existing one-shot chain.')
        record=dict(status='running',arm=arm,pid=os.getpid(),started_unix=time.time(),
            start_ticks=mature.proc_start(os.getpid()),command=[sys.executable,*sys.argv],
            source_sha256=prepared['source_sha256'],source_binding=prepared['source_binding'],stages={},
            automatic_retry=False,automatic_resume=False,full_candidate='d16',
            boundary='D16 is the predesignated unique full candidate; raw scores are descriptive, no automatic branch selection.')
        b.write(path,record)

        def stage(name,module,args,gpu,seconds):
            b.check_pins(prepared['source_sha256'])
            # CPU controllers call resource validation only at actual stage boundaries.
            resources=mature.check_resources([str(gpu)])
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu) if module != 'evaluate' or args[:2] != ['--component','raw_full8'] else '',
                PYTHONHASHSEED='31',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
                TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',
                NUMEXPR_NUM_THREADS='8',CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='offline')
            if name == 'final_full9':
                env['CUDA_VISIBLE_DEVICES']=''
            command=['timeout','--signal=TERM','--kill-after=20s',str(seconds)+'s',str(b.PYTHON),'-B','-u','-m',
                'experiments.down_geometry_baseline_cmcc_20261103.'+module,*map(str,args)]
            log=result / (name+'.log')
            with log.open('x',encoding='utf-8') as stream:
                process=subprocess.Popen(command,cwd=b.REPO,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                row=dict(command=command,pid=process.pid,start_ticks=mature.proc_start(process.pid),
                    gpu=gpu,resource_check=resources,started_unix=time.time(),log=str(log),status='running')
                record['stages'][name]=row
                record['current_stage']=name
                b.write(path,record)
                code=process.wait()
            row.update(status='exited',exit_code=code,finished_unix=time.time(),elapsed_seconds=time.time()-row['started_unix'])
            (result / (name+'.exit')).write_text(str(code)+'\n',encoding='utf-8')
            b.write(path,record)
            b.require(code == 0,'Actual stage failed; stop dependencies without retry: '+name)

        try:
            gpu=b.ARMS[arm]['gpu']
            if arm == 'transpose_false':
                stage('control_raw_full8','evaluate',['--component','raw_full8','--arm','control'],gpu,5400)
                stage('control_raw_readout','evaluate',['--component','readout','--arm','control'],gpu,1800)
            stage('fit','run',['--arm',arm,'--phase','formal'],gpu,5400)
            stage('raw_full8','evaluate',['--component','raw_full8','--arm',arm],gpu,5400)
            stage('raw_readout','evaluate',['--component','readout','--arm',arm],gpu,1800)
            if arm == 'd16':
                stage('down500','recover',[],gpu,43200)
                stage('final_full9','evaluate',['--checkpoint',b.OUTPUT / 'd16/recovery/final_model',
                    '--pipeline-report',b.OUTPUT / 'd16/recovery/report.json','--output',b.RESULT / 'd16/final_full9','--gpus','0'],gpu,28800)
                final=mature.validate_evaluation(b.RESULT / 'd16/final_full9',b.read(b.OUTPUT / 'd16/recovery/report.json')['final_checkpoint_id'])
                record['complete_nine']=final
            record.update(status='completed_d16_baseline_full9' if arm == 'd16' else 'completed_raw_direction_comparison',exit_code=0)
        except BaseException:
            record.update(status='failed',exit_code=1,error=traceback.format_exc())
            raise
        finally:
            record.update(finished_unix=time.time(),elapsed_seconds=time.time()-record['started_unix'])
            b.write(path,record)
            (result / 'run.exit').write_text(str(record['exit_code'])+'\n',encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=tuple(b.ARMS),required=True)
    execute(parser.parse_args().arm)
