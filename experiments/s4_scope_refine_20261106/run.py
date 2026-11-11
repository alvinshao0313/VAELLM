"""Run one new exact-scope validation or its independently approved fresh64 arm."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback
from . import bindings as b

MODULE='experiments.s4_scope_refine_20261106'


def stage(root,name,arguments,gpu,report):
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
    tag='validation_'+arm if validation else arm;request_path=Path(plan['requests'][tag]['path'])
    b.require(b.binding(request_path)==plan['requests'][tag],'Prepared actual request changed.')
    request=b.read(request_path);root=b.RUN/tag;gpu=request['gpu']
    b.require(not (root/'run.json').exists(),'Never overwrite a running or ended exact-scope attempt.')
    dependency=None
    if not validation:
        b.require(request['stop_step']==64 and request['groups']==list(range(128)),'Formal controller only launches fresh64.')
        dependency=b.dependency(arm);proof=b.read(b.RUN/('validation_'+arm)/'result.json')
        b.require(proof['status']=='verified_exact_scope_real_two_export_fresh' and proof['arm']==arm
            and proof['source_sha256']==plan['source_sha256'] and proof['full_frozen_complement_exact']
            and proof['active_projections']==72 and proof['active_leaves']==432,
            'This exact scope needs its own real peak-shape update/export/fresh evidence.')
    report=dict(status='running_exact_scope_validation' if validation else 'running_fresh64_exact_scope_arm',
        **b.parent.process_event(os.getpid()),request=b.binding(request_path),dependency=dependency,
        source_sha256=plan['source_sha256'],processes={})
    began=time.monotonic();code=1
    try:
        last=root/'training'
        stage(root,'training',['train','--request',str(request_path),'--output',str(last)],gpu,report)
        stage(root,'export',['export','--request',str(request_path),'--state',str(last/'training_state.pt'),'--output',str(root/'post')],gpu,report)
        checkpoint=root/'post/model';evaluation=root/'post/evaluation'
        stage(root,'fresh',['evaluate','--request',str(request_path),'--checkpoint',str(checkpoint),
            '--output',str(evaluation),'--component','fresh'],gpu,report)
        fresh=b.read(evaluation/'fresh.json');b.require(fresh['full_state_exact'] and fresh['BF16_forward_finite'],'Native fresh failed.')
        if validation:
            training=b.read(last/'training.json');exported=b.read(root/'post/export.json')
            rows=[b.read(last/'updates'/f'step_{step:06d}.json') for step in (1,2)]
            b.require([r['group_index'] for r in rows]==[2,9] and training['executed_updates']==training['completed_updates']==2,
                'Actual real two peak-group updates changed.')
            b.require(rows[0]['exposure']['groups'][0]['QA_max_length']==request['required_shapes']['QA_peak']
                and rows[1]['exposure']['groups'][0]['M_max_length']==request['required_shapes']['M_peak'],'Actual peak shapes not executed.')
            b.require(all(r['layer_strata_displacement'][key]['BF16_changed']>0 for r in rows
                for key in ('early','middle','late')),'Selected BF16 changes must survive in every layer stratum.')
            result=dict(status='verified_exact_scope_real_two_export_fresh',arm=arm,source_sha256=plan['source_sha256'],
                groups=[2,9],actual_peak_QA=request['required_shapes']['QA_peak'],actual_peak_M=request['required_shapes']['M_peak'],
                fresh=fresh,full_frozen_complement_exact=training['full_frozen_complement_exact'],
                independent_export_named_state_exact=exported['actual_FP32_masters_Adam_RNG_cursor_exact'],
                scope_output_difference=b.read(last/'scope_initial_output.json'),reference_resume_validation=plan['reference_resume_validation'],
                all252_S4=True,active_projections=72,active_leaves=432,
                peak_allocated=max(r['memory']['peak_allocated'] for r in rows),seconds=time.monotonic()-began)
            b.write(root/'result.json',result)
        else:
            for component in ('mmlu','seven'):
                stage(root,component,['evaluate','--request',str(request_path),'--checkpoint',str(checkpoint),
                    '--output',str(evaluation),'--component',component],gpu,report)
            from .result import summarize
            summarize(request_path,root)
        report.update(status='complete_exact_scope_validation' if validation else 'complete_exact_scope64_eight');code=0
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-began,exit_code=code);b.write(root/'run.json',report)
        (root/'run.exit').write_text(str(code)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=list(b.ARMS),required=True);parser.add_argument('--validate',action='store_true')
    args=parser.parse_args();execute(args.validate,args.arm)
