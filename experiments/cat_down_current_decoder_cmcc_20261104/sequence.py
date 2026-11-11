"""Exactly one original500/fresh -> same-final-UUID full9, without retry or reservation."""
import fcntl
import json
import os
import sys
import time
import traceback

from . import bindings as b


def requests():
    package='experiments.cat_down_current_decoder_cmcc_20261104.'
    return [dict(name='down500',argv=[str(b.PYTHON),'-B','-u','-m',package+'run'],
        environment={**b.executor.environment('0'),'PYTHONHASHSEED':'31'},gpus=['0'],
        log=str(b.RESULT/'down500.log'),exit_file=str(b.RESULT/'down500.exit')),
        dict(name='full9',argv=[str(b.PYTHON),'-B','-u','-m',package+'evaluate',
            '--checkpoint',str(b.OUTPUT/'formal/final_model'),'--pipeline-report',str(b.OUTPUT/'formal/report.json'),
            '--output',str(b.RESULT/'evaluation'),'--gpus','0'],environment=b.executor.environment(''),gpus=['0'],
            log=str(b.RESULT/'full9.log'),exit_file=str(b.RESULT/'full9.exit'))]


def main():
    b.cpu_environment()
    prepared=b.prepared()
    path=b.RESULT/'run.json'
    with (b.RESULT/'sequence.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b.require(not path.exists(),'Never repeat this approved one-shot closure.')
        stages=requests()
        resource=b.executor.check_resources(['0'])
        record=dict(schema='current36_down_original_Adam500_sequence_v1',status='starting',pid=os.getpid(),
            start_ticks=b.executor.proc_start(os.getpid()),command=[sys.executable,*sys.argv],started_unix=time.time(),
            contract=b.CONTRACT,source_sha256=prepared['source_sha256'],source_binding=prepared['source_binding'],
            prepared_sha256=b.sha(b.PREPARED),initial_resources=resource,stages={},requests=stages,
            automatic_retry=False,automatic_resume=False,budget=dict(down500_fresh_seconds=28800,full9_seconds=36000),
            quality_cancellation=False,extra_budget_after_closure=0)
        def changed(row):
            record['stages'][row['name']]=row
            record.update(status=row['name']+'_'+row['status'],current_stage=row['name'])
            b.write(path,record)
        try:
            b.write(path,record)
            for request in stages:
                b.check_pins(prepared['source_sha256'])
                b.executor.check_resources(request['gpus'])
                seconds=28800 if request['name']=='down500' else 36000
                arguments=request['argv']
                if request['name']=='full9':
                    arguments=[*arguments,'--deadline-unix',str(time.time()+seconds-30)]
                bounded={**request,'argv':['timeout','--signal=TERM','--kill-after=20s',str(seconds)+'s',*arguments]}
                b.executor.launch_child(bounded,changed)
            report=b.read(b.OUTPUT/'formal/report.json')
            quality=b.executor.validate_evaluation(b.RESULT/'evaluation',report['final_checkpoint_id'])
            control=b.read(b.CONTROL_ACCEPTANCE)
            delta=quality['nine_task_score']-control['compressed_nine_task_score']
            record.update(status='completed_current36_down_original_Adam500_full9',exit_code=0,quality=quality,
                delta_vs_same_raw_all252_AdamW500=dict(nine=delta,
                    task_scores={name:value-control['compressed_scores'][name] for name,value in quality['task_scores'].items()}),
                fixed_point_disposition='end_fixed_point' if delta<=0 else 'single_seed_candidate_only_no_automatic_extension')
        except BaseException:
            record.update(status='failed',exit_code=1,error=traceback.format_exc())
            raise
        finally:
            record.update(finished_unix=time.time(),elapsed_seconds=time.time()-record['started_unix'])
            b.write(path,record)
            (b.RESULT/'sequence.exit').write_text(str(record['exit_code'])+'\n',encoding='utf-8')
        print(json.dumps(dict(status=record['status'])),flush=True)


if __name__=='__main__':
    main()
