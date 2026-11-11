"""Finite validation or one fixed64 arm; never wait for the old WebMed batch."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback
from . import bindings as b

MODULE='experiments.s4_recovery_screen_20261105'


def stage(root,name,arguments,gpu,report):
    root=Path(root)
    with (root/(name+'.log')).open('x',encoding='utf-8') as log:
        argv=[str(b.PYTHON),'-B','-u','-m',MODULE+'.'+arguments[0],*arguments[1:]]
        process=subprocess.Popen(argv,cwd=b.ROOT,env=b.environment(gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        report['processes'][name]=dict(command=argv,gpu=gpu,**b.parent.process_event(process.pid));b.write(root/'run.json',report)
        code=process.wait()
    report['processes'][name]['exit_code']=code;b.write(root/'run.json',report)
    (root/(name+'.exit')).write_text(str(code)+'\n')
    b.require(code==0,name+' failed; no automatic retry or dependent execution.')


def execute(validation,arm):
    b.initialize();b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Controller must hide CUDA.')
    plan=b.read(b.RUN/'plan.json');b.parent.s.check_pins(plan['source_sha256'])
    tag='validation' if validation else arm;request_path=Path(plan['requests'][tag]['path'])
    b.require(b.binding(request_path)==plan['requests'][tag],'Prepared actual request changed.')
    request=b.read(request_path);root=b.RUN/tag;gpu=request['gpu']
    b.require(not (root/'run.json').exists(),'Never overwrite a running or ended scope attempt.')
    dependency=None
    if not validation:
        b.require(request['stop_step']==64 and request['groups']==list(range(128)),'This controller only launches fixed64.')
        dependency=b.dependency(arm)
        proof=b.read(b.RUN/'validation/result.json')
        reuse=b.read(plan['validation_reuse']['path'])
        previous,current=proof['source_sha256'],plan['source_sha256']
        changed={path:dict(previous=previous[path],current=current[path]) for path in previous if previous[path]!=current[path]}
        b.require(proof['status']=='verified_attention_scope_cross_process_2plus1'
            and b.binding(plan['validation_reuse']['path'])==plan['validation_reuse']
            and reuse['validation']==b.binding(b.RUN/'validation/result.json')
            and set(previous)==set(current) and set(changed)=={str(b.HERE/'bindings.py'),str(b.HERE/'run.py')}
            and changed==reuse['changed_source_sha256'],
            'Reuse only the original2+1 with these audited attention-scheduling controller changes.')
    report=dict(status='running_attention_validation' if validation else 'running_fixed64_scope_arm',
        **b.parent.process_event(os.getpid()),request=b.binding(request_path),dependency=dependency,
        source_sha256=plan['source_sha256'],schedule_update=plan.get('validation_reuse'),processes={})
    began=time.monotonic();code=1
    try:
        if validation:
            first=root/'first';last=root/'resumed'
            stage(root,'first_two',['train','--request',str(request_path),'--output',str(first),'--stop-after','2'],gpu,report)
            stage(root,'resume_one',['train','--request',str(request_path),'--output',str(last),
                '--resume',str(first/'training_state.pt')],gpu,report)
        else:
            last=root/'training'
            stage(root,'training',['train','--request',str(request_path),'--output',str(last)],gpu,report)
        stage(root,'export',['export','--request',str(request_path),'--state',str(last/'training_state.pt'),'--output',str(root/'post')],gpu,report)
        checkpoint=root/'post/model';evaluation=root/'post/evaluation'
        stage(root,'fresh',['evaluate','--request',str(request_path),'--checkpoint',str(checkpoint),
            '--output',str(evaluation),'--component','fresh'],gpu,report)
        fresh=b.read(evaluation/'fresh.json');b.require(fresh['full_state_exact'] and fresh['BF16_forward_finite'],'Native fresh failed.')
        if validation:
            restored=b.read(last/'resume_verified.json');training=b.read(last/'training.json')
            rows=[b.read(first/'updates'/f'step_{step:06d}.json') for step in (1,2)]+[b.read(last/'updates/step_000003.json')]
            b.require([r['group_index'] for r in rows]==[2,9,10] and training['executed_updates']==1
                and training['completed_updates']==3 and restored['saved_updates']==2 and restored['next_group_index']==10,
                'Actual cross-process save2/resume1 sequence changed.')
            b.require(rows[0]['exposure']['groups'][0]['QA_max_length']==request['required_shapes']['QA_peak']
                and rows[1]['exposure']['groups'][0]['M_max_length']==request['required_shapes']['M_peak'],'Actual real peak shapes were not executed.')
            b.require(all(sum(r['layer_strata_displacement'][key]['BF16_changed'] for r in rows)>0
                for key in ('early','middle','late')),'Real attention BF16 changes must survive in every layer stratum.')
            result=dict(status='verified_attention_scope_cross_process_2plus1',source_sha256=plan['source_sha256'],
                groups=[2,9,10],actual_peak_QA=request['required_shapes']['QA_peak'],actual_peak_M=request['required_shapes']['M_peak'],resume=restored,fresh=fresh,
                first_frozen_exact=b.read(first/'training.json')['full_frozen_complement_exact'],
                resumed_frozen_exact=training['full_frozen_complement_exact'],
                scope_output_difference=b.read(first/'scope_initial_output.json'),
                all252_S4=True,active_projections=144,active_leaves=864,
                peak_allocated=max(r['memory']['peak_allocated'] for r in rows),seconds=time.monotonic()-began)
            b.write(root/'result.json',result)
        else:
            for component in ('mmlu','seven'):
                stage(root,component,['evaluate','--request',str(request_path),'--checkpoint',str(checkpoint),
                    '--output',str(evaluation),'--component',component],gpu,report)
            from .result import summarize
            summarize(request_path,root)
        report.update(status='complete_scope_validation' if validation else 'complete_scope64_eight',exit_code=0);code=0
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-began,exit_code=code);b.write(root/'run.json',report)
        (root/'run.exit').write_text(str(code)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--validate',action='store_true')
    group.add_argument('--arm',choices=list(b.ARMS));args=parser.parse_args();execute(args.validate,args.arm)
