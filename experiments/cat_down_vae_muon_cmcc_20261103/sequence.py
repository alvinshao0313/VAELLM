"""One fixed formal VAE optimizer point, fresh, full8 and train readout only."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from . import bindings as b
from experiments.cmcc_runtime_20261030 import sequence as mature


def execute():
    b.cpu_environment()
    prepared=b.read(b.RESULT / 'prepared.json')
    b.check_pins(prepared['source_sha256'])
    proof=b.read(b.OUTPUT / 'short/report.json')
    b.require(proof['status']=='completed_VAE_Muon_short' and proof['exit_code']==0
        and hashlib.sha256(json.dumps(proof['source_sha256'],sort_keys=True).encode()).hexdigest()==prepared['applicable_GPU_short_source_digest']
        and proof['fit']['canonical_original_initialization_matches']==[True,True]
        and all(len(row['updates'])==2 and row['calibration']['relative_error']<=.01 for row in proof['optimizer_stages']),
        'Both real formal-shape stages and BF16 amplitude/initialization contracts must pass first.')
    path=b.RESULT / 'run.json'
    with (b.RESULT / 'sequence.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        b.require(not path.exists(),'Never launch this fixed point twice.')
        record=dict(status='running',pid=os.getpid(),start_ticks=mature.proc_start(os.getpid()),started_unix=time.time(),
            command=[sys.executable,*sys.argv],contract=b.CONTRACT,source_sha256=prepared['source_sha256'],stages={},
            automatic_retry=False,automatic_recovery_or_full9=False)
        b.write(path,record)

        def stage(name,module,args,seconds,*,cpu=False):
            b.check_pins(prepared['source_sha256'])
            resources=mature.check_resources([str(b.GPU)])
            argv=['timeout','--signal=TERM','--kill-after=20s',str(seconds)+'s',str(b.PYTHON),'-B','-u','-m',
                'experiments.cat_down_vae_muon_cmcc_20261103.'+module,*map(str,args)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if cpu else str(b.GPU),PYTHONHASHSEED='31',
                PYTHONDONTWRITEBYTECODE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',NUMEXPR_NUM_THREADS='8',
                CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='offline')
            log=b.RESULT / (name+'.log')
            with log.open('x',encoding='utf-8') as stream:
                process=subprocess.Popen(argv,cwd=b.REPO,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                row=dict(command=argv,pid=process.pid,start_ticks=mature.proc_start(process.pid),gpu=b.GPU,
                    allocator=env['PYTORCH_CUDA_ALLOC_CONF'],resource_check=resources,started_unix=time.time(),log=str(log))
                record['stages'][name]=row;record['current_stage']=name;b.write(path,record)
                code=process.wait()
            row.update(exit_code=code,finished_unix=time.time())
            (b.RESULT / (name+'.exit')).write_text(str(code)+'\n',encoding='utf-8')
            b.write(path,record)
            b.require(code==0,'Actual fixed stage failed; no retry or dependent continuation: '+name)

        try:
            stage('fit','run',['--phase','formal'],5400)
            stage('raw_full8','evaluate',['--component','raw_full8'],5400,cpu=True)
            stage('raw_readout','evaluate',['--component','readout'],1800)
            record.update(status='completed_VAE_Muon_raw_eight_and_readout',exit_code=0)
        except BaseException:
            record.update(status='failed',exit_code=1,error=traceback.format_exc())
            raise
        finally:
            record.update(finished_unix=time.time(),elapsed_seconds=time.time()-record['started_unix'])
            b.write(path,record)
            (b.RESULT / 'run.exit').write_text(str(record['exit_code'])+'\n',encoding='utf-8')


if __name__=='__main__':execute()
