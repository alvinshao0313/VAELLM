"""One finite source_weight64 chain; reuse verified QK GPU evidence without a new smoke run."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback
from . import bindings as b

MODULE='experiments.s4_qk_source_weights_20261107'


def stage(root,name,arguments,gpu,report):
    with (root/(name+'.log')).open('x',encoding='utf-8') as log:
        argv=[str(b.PYTHON),'-B','-u','-m',MODULE+'.'+arguments[0],*arguments[1:]]
        process=subprocess.Popen(argv,cwd=b.ROOT,env=b.environment(gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        report['processes'][name]=dict(command=argv,gpu=gpu,**b.parent.process_event(process.pid));b.write(root/'run.json',report)
        code=process.wait()
    report['processes'][name]['exit_code']=code;b.write(root/'run.json',report)
    (root/(name+'.exit')).write_text(str(code)+'\n');b.require(code==0,name+' failed; no automatic retry or dependent execution.')


def execute(arm):
    b.initialize();b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Controller must hide CUDA.')
    plan=b.read(b.RUN/'plan.json');b.parent.s.check_pins(plan['source_sha256'])
    request_path=Path(plan['requests'][arm]['path']);b.require(b.binding(request_path)==plan['requests'][arm],'Actual request changed.')
    request=b.read(request_path);root=b.RUN/arm;gpu=request['gpu'];proof=b.read(b.RUN/'CPU_validation.json')
    b.require(not (root/'run.json').exists(),'Never overwrite an existing source-weight attempt.')
    b.require(b.binding(b.RUN/'CPU_validation.json')==plan['CPU_validation']
        and proof['status']=='CPU_QK_source_weights_fresh64_verified' and proof['all4224_targets_exact']
        and proof['source_sha256']==plan['source_sha256'] and request['stop_step']==64
        and request['groups']==list(range(128)) and request['contract']==b.contract(arm),
        'Formal source_weight64 requires matching complete CPU proof and fixed contract.')
    dependency=b.dependency(arm);report=dict(status='running_QK_source_weight_fresh64_eight',
        **b.parent.process_event(os.getpid()),request=b.binding(request_path),teacher_alpha=request['teacher_alpha'],source_weights=request['source_weights'],
        dependency=dependency,source_sha256=plan['source_sha256'],processes={})
    began=time.monotonic();code=1
    try:
        last=root/'training'
        stage(root,'training',['train','--request',str(request_path),'--output',str(last)],gpu,report)
        stage(root,'export',['export','--request',str(request_path),'--state',str(last/'training_state.pt'),'--output',str(root/'post')],gpu,report)
        checkpoint=root/'post/model';evaluation=root/'post/evaluation'
        stage(root,'fresh',['evaluate','--request',str(request_path),'--checkpoint',str(checkpoint),
            '--output',str(evaluation),'--component','fresh'],gpu,report)
        fresh=b.read(evaluation/'fresh.json')
        b.require(fresh['full_state_exact'] and fresh['persistent_fields']==3927 and fresh['BF16_forward_finite'],'Native fresh failed.')
        for component in ('mmlu','seven'):
            stage(root,component,['evaluate','--request',str(request_path),'--checkpoint',str(checkpoint),
                '--output',str(evaluation),'--component',component],gpu,report)
        from .result import summarize
        summarize(request_path,root);report.update(status='complete_QK_source_weight64_eight');code=0
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-began,exit_code=code);b.write(root/'run.json',report)
        (root/'run.exit').write_text(str(code)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=list(b.ARMS),required=True)
    args=parser.parse_args();execute(args.arm)
