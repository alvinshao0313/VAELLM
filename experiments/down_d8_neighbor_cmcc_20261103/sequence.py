"""Fixed short or formal/raw8/train readout on GPU1 only; no recovery suffix."""
import argparse
import fcntl
import os
import sys
import time
import traceback

from . import bindings as b
from .run import verified_short


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('short','formal'),required=True)
    phase=parser.parse_args().phase
    b.cpu_environment()
    prepared=b.prepared()
    review=b.require_core_review()
    if phase=='formal':
        verified_short()
    path=b.RESULT/(phase+'_run.json')
    with (b.RESULT/'sequence.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b.require(not path.exists(),'Never retry or overwrite this fixed phase.')
        resource=b.executor.check_resources(['1'])
        requests=[('fit','run',['--phase',phase],1800 if phase=='short' else 5400,False)]
        if phase=='formal':
            requests.extend([('raw_full8','evaluate',['--component','raw_full8'],5400,True),
                ('raw_readout','evaluate',['--component','readout'],1800,False)])
        record=dict(schema='D8_fixed_raw_sequence_v1',status='running',phase=phase,pid=os.getpid(),
            start_ticks=b.executor.proc_start(os.getpid()),started_unix=time.time(),command=[sys.executable,*sys.argv],
            source_sha256=prepared['source_sha256'],contract=b.CONTRACT,effective_consumer=b.consumer.contract(),
            core_review=review,initial_resources=resource,stages={},automatic_retry=False,automatic_resume=False,
            recovery_or_full9_budget=0)
        def changed(row):
            record['stages'][row['name']]=row
            record.update(current_stage=row['name'],status=row['name']+'_'+row['status'])
            b.write(path,record)
        try:
            b.write(path,record)
            for name,module,args,seconds,cpu in requests:
                b.check_pins(prepared['source_sha256'])
                b.executor.check_resources(['1'])
                env={**b.executor.environment('' if cpu else '1'),'PYTHONHASHSEED':'31',
                    'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','NUMEXPR_NUM_THREADS':'8',
                    'CUBLAS_WORKSPACE_CONFIG':':4096:8','WANDB_MODE':'offline'}
                row=dict(name=name,argv=['timeout','--signal=TERM','--kill-after=20s',str(seconds)+'s',
                    str(b.PYTHON),'-B','-u','-m','experiments.down_d8_neighbor_cmcc_20261103.'+module,*args],
                    environment=env,gpus=['1'],log=str(b.RESULT/(phase+'_'+name+'.log')),
                    exit_file=str(b.RESULT/(phase+'_'+name+'.exit')))
                b.executor.launch_child(row,changed)
            if phase=='short':
                record['short_evidence']=verified_short()
                record.update(status='completed_actual_D8_short_and_fresh',exit_code=0)
            else:
                formal=b.read(b.OUTPUT/'formal/report.json')
                summary=b.read(b.RESULT/'raw_full8/full8_summary.json')
                readout=b.read(b.RESULT/'raw_readout.json')
                control=b.read(prepared['D16_control']['readout'])
                b.require(summary['checkpoint_id']==readout['checkpoint_id']==formal['raw_checkpoint_id']
                    and summary['leaf_count']==64 and summary['sample_count']==24742,
                    'Complete actual same-UUID raw8 and fixed train readout are required before ending the neighbor.')
                delta=summary['mean_percent']-prepared['D16_control']['mean_percent']
                ratio=readout['deployment']['output_NMSE']/control['deployment']['output_NMSE']-1
                record.update(status='completed_D8_raw_eight_and_readout',exit_code=0,checkpoint_id=formal['raw_checkpoint_id'],
                    full8=summary['mean_percent'],delta_full8_vs_D16=delta,relative_output_NMSE_vs_D16=ratio,
                    fixed_point_disposition='end_this_neighbor_no_further_D_shrink' if delta<=0 and ratio>=0 else
                        'record_single_seed_tradeoff_only_no_automatic_suffix',measured_budget=formal['raw_budget'])
        except BaseException:
            record.update(status='failed',exit_code=1,error=traceback.format_exc())
            raise
        finally:
            record.update(finished_unix=time.time(),elapsed_seconds=time.time()-record['started_unix'])
            b.write(path,record)
            (b.RESULT/(phase+'_sequence.exit')).write_text(str(record['exit_code'])+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
