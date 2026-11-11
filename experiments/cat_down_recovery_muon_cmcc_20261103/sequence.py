"""Fixed engineering2/fresh or approved fresh original500/fresh/full9 on GPU3."""
import argparse
import json
import os
import sys
import time
import traceback

from . import bindings as b
from .run import verified_short


def requests(phase):
    package='experiments.cat_down_recovery_muon_cmcc_20261103.'
    def request(name,arguments,gpu='',devices=()):
        return dict(name=name,argv=[str(b.PYTHON),'-B','-u','-m',*arguments],
            environment={**b.executor.environment(gpu),'PYTHONHASHSEED':'31'},gpus=list(devices),
            log=str(b.RESULT/(name+'.log')),exit_file=str(b.RESULT/(name+'.exit')))
    return [request(phase,[package+'run','--phase',phase,'--timeout-seconds','1800' if phase=='short' else '43200'],'3',('3',)),
        request(phase+'_fresh',[package+'run','--fresh',str(b.OUTPUT/phase/'final_model'),
            '--evidence',str(b.OUTPUT/phase/'same_input_deployment.pt')],'3',('3',))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('short','formal'),required=True)
    args=parser.parse_args()
    b.cpu_environment()
    b.torch.set_num_threads(4)
    prepared=b.prepared()
    if args.phase=='formal':
        verified_short()
    plan_path=b.RESULT/(args.phase+'_plan.json')
    run_path=b.RESULT/(args.phase+'_run.json')
    b.require(not plan_path.exists() and not run_path.exists(),'No automatic retry, resume or overwritten sequence.')
    resource=b.executor.check_resources(['3'])
    stages=requests(args.phase)
    if args.phase=='formal':
        stages.append(dict(name='full9',argv=[str(b.PYTHON),'-B','-u','-m',
            'experiments.cat_down_recovery_muon_cmcc_20261103.evaluate','--checkpoint',str(b.OUTPUT/'formal/final_model'),
            '--pipeline-report',str(b.OUTPUT/'formal/report.json'),'--output',str(b.RESULT/'evaluation'),'--gpus','3'],
            environment=b.executor.environment(''),gpus=[],log=str(b.RESULT/'full9.log'),exit_file=str(b.RESULT/'full9.exit')))
    plan=dict(schema='C0_down_category_Muon_single_sequence_v1',phase=args.phase,created_unix=time.time(),
        prepared_sha256=b.sha(b.PREPARED),contract=b.CONTRACT,source_sha256=prepared['source_sha256'],
        stages=stages,initial_resources=resource,automatic_retry=False,automatic_resume=False,
        budget=dict(short_seconds=1800,formal_seconds=43200,complete_nine_hours=8),
        completion='Predesignated original raw C0 fresh500/fresh/same-UUID complete-nine; designated AdamW500 control reused. Last-category evidence only.')
    b.write(plan_path,plan)
    record=dict(schema=plan['schema'],phase=args.phase,status='starting',pid=os.getpid(),start_ticks=b.executor.proc_start(os.getpid()),
        argv=sys.argv,python=sys.executable,started_unix=time.time(),plan_sha256=b.sha(plan_path),stages={})
    def changed(row):
        record['stages'][row['name']]=row
        record['status']=row['name']+'_'+row['status']
        b.write(run_path,record)
    try:
        b.write(run_path,record)
        recovery_deadline=time.monotonic()+(1800 if args.phase=='short' else 43200)
        for request in stages:
            b.check_pins(plan['source_sha256'])
            if request['gpus']:
                b.executor.check_resources(request['gpus'])
            remaining=28800 if request['name']=='full9' else int(recovery_deadline-time.monotonic())
            b.require(remaining>0,'Fixed recovery+fresh phase budget exhausted; stop without launching a dependent stage.')
            arguments=request['argv']
            if request['name']=='full9':
                arguments=[*arguments,'--deadline-unix',str(time.time()+remaining)]
            bounded={**request,'argv':['timeout','--signal=TERM','--kill-after=20s',str(remaining)+'s',*arguments]}
            b.executor.launch_child(bounded,changed)
        if args.phase=='short':
            record['short_lifecycle']=verified_short()
        else:
            report=b.read(b.OUTPUT/'formal/report.json')
            quality=b.executor.validate_evaluation(b.RESULT/'evaluation',report['final_checkpoint_id'])
            control=b.read(b.CONTROL_ACCEPTANCE)
            record.update(quality=quality,delta_vs_original_AdamW500=dict(
                nine=quality['nine_task_score']-control['compressed_nine_task_score'],
                task_scores={name:value-control['compressed_scores'][name] for name,value in quality['task_scores'].items()}),
                optimizer_scope='Only original last down category500, not all seven categories.')
        record.update(status='completed_'+args.phase+'_sequence',exit_code=0)
    except BaseException:
        record.update(status='failed',exit_code=1,error=traceback.format_exc())
        raise
    finally:
        record['finished_unix']=time.time()
        b.write(run_path,record)
        (b.RESULT/(args.phase+'_sequence.exit')).write_text(str(record['exit_code'])+'\n',encoding='utf-8')
    print(json.dumps(dict(status=record['status'],phase=args.phase)),flush=True)


if __name__=='__main__':
    main()
