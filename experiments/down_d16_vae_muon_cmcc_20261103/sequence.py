"""One disconnect-safe short or fit/fresh -> own Adam500/fresh -> full9 sequence."""
import argparse
import fcntl
import os
import sys
import time
import traceback
from pathlib import Path

from . import bindings as b
from .run import verified_short


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=('short','formal'),required=True)
    phase=parser.parse_args().phase
    b.cpu_environment();p=b.prepared()
    if phase=='formal':verified_short()
    path=b.RESULT/(phase+'_run.json')
    with (b.RESULT/'sequence.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b.require(not path.exists(),'Never repeat/overwrite this fixed finite phase.')
        resource=b.executor.check_resources(['1'])
        requests=[('fit','run',['--phase',phase],1800 if phase=='short' else 5400,False)]
        if phase=='formal':
            requests.extend([('Adam500','recover',[],43200,False),('full9','evaluate',
                ['--checkpoint',str(b.OUTPUT/'recovery/final_model'),'--pipeline-report',str(b.OUTPUT/'recovery/report.json'),
                 '--output',str(b.RESULT/'evaluation'),'--gpus','1'],28800,True)])
        record=dict(schema='D16_VAE_Muon_fixed_sequence_v1',status='running',phase=phase,pid=os.getpid(),
            start_ticks=b.executor.proc_start(os.getpid()),started_unix=time.time(),command=[sys.executable,*sys.argv],
            source_sha256=p['source_sha256'],prepared_sha256=b.sha(b.PREPARED),contract=b.CONTRACT,
            source_binding=p['source_binding'],controls=p['controls'],initial_resources=resource,stages={},
            automatic_retry=False,automatic_resume=False,quality_cancellation=False,extra_budget_after_full9=0)
        def changed(row):
            record['stages'][row['name']]=row
            record.update(current_stage=row['name'],status=row['name']+'_'+row['status']);b.write(path,record)
        try:
            b.write(path,record)
            for name,module,args,seconds,cpu in requests:
                b.check_pins(p['source_sha256']);b.executor.check_resources(['1'])
                arguments=list(args)
                if name=='full9':arguments.extend(['--deadline-unix',str(time.time()+seconds)])
                row=dict(name=name,argv=['timeout','--signal=TERM','--kill-after=20s',str(seconds)+'s',
                    str(b.PYTHON),'-B','-u','-m','experiments.down_d16_vae_muon_cmcc_20261103.'+module,*arguments],
                    environment={**b.executor.environment('' if cpu else '1'),'PYTHONHASHSEED':'31',
                        'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','NUMEXPR_NUM_THREADS':'8',
                        'CUBLAS_WORKSPACE_CONFIG':':4096:8','WANDB_MODE':'offline'},gpus=['1'],
                    log=str(b.RESULT/(phase+'_'+name+'.log')),exit_file=str(b.RESULT/(phase+'_'+name+'.exit')))
                b.executor.launch_child(row,changed)
            if phase=='short':
                record.update(status='completed_D16_Muon_actual_short_and_fresh',short_evidence=verified_short(),exit_code=0)
            else:
                final=b.read(b.OUTPUT/'recovery/report.json')
                quality=b.executor.validate_evaluation(b.RESULT/'evaluation',final['final_checkpoint_id'])
                controls={};pending=[]
                for name,control_path in p['controls']['acceptance_paths'].items():
                    control_path=Path(control_path);exit_path=control_path.with_name('nine_task.exit')
                    if control_path.is_file() and exit_path.is_file():
                        controls[name]=b.executor.validate_acceptance(control_path,exit_path,p['controls']['final_ids'][name])
                    else:pending.append(name)
                record.update(status='completed_D16_Muon_own_Adam500_complete_nine',exit_code=0,quality=quality,
                    validated_controls=controls,comparison_pending_controls=pending)
                if not pending:
                    best=max(item['nine_task_score'] for item in controls.values())
                    record.update(delta_vs_best_of_three=quality['nine_task_score']-best,
                        candidate_disposition='single_seed_combination_candidate' if quality['nine_task_score']>best else 'end_fixed_point',
                        scope='One same-parent recipe interaction only, not additive old gains, a robust winner or seven-category/BF16 transfer verification.')
        except BaseException:
            record.update(status='failed',exit_code=1,error=traceback.format_exc());raise
        finally:
            record.update(finished_unix=time.time(),elapsed_seconds=time.time()-record['started_unix']);b.write(path,record)
            (b.RESULT/(phase+'_sequence.exit')).write_text(str(record['exit_code'])+'\n',encoding='utf-8')


if __name__=='__main__':main()
