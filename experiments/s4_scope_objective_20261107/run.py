"""One finite validation or formal chain, with no polling, retry or successor."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback
from . import bindings as b

MODULE='experiments.s4_scope_objective_20261107'


def stage(root,name,arguments,gpu,report):
    with (root/(name+'.log')).open('x',encoding='utf-8') as log:
        argv=[str(b.PYTHON),'-B','-u','-m',MODULE+'.'+arguments[0],*arguments[1:]]
        process=subprocess.Popen(argv,cwd=b.ROOT,env=b.environment(gpu),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        report['processes'][name]=dict(command=argv,gpu=gpu,**b.parent.process_event(process.pid));b.write(root/'run.json',report)
        code=process.wait()
    report['processes'][name]['exit_code']=code;b.write(root/'run.json',report)
    (root/(name+'.exit')).write_text(str(code)+'\n')
    b.require(code==0,name+' failed; no automatic retry or dependent execution.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=list(b.ARMS),required=True)
    parser.add_argument('--validation',action='store_true');args=parser.parse_args()
    b.initialize();b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Hide controller CUDA.')
    arm=args.arm;tag='validation_'+arm if args.validation else arm;root=b.RUN/tag;request_path=root/'request.json'
    request=b.read(request_path);b.check_request(request)
    prep=b.read(b.RUN/'CPU_validation.json')
    b.require(b.binding(request_path)==prep['request_bindings'][tag] and not (root/'run.json').exists(),
        'Fresh bound request required; never overwrite an attempt.')
    if not args.validation and arm!='A':
        proof=b.read(b.RUN/('validation_'+arm)/'result.json')
        b.require(proof['status']=='verified_new_path' and proof['source_sha256']==request['source_sha256'],
            'Applicable successful GPU validation required.')
    report=dict(status='running_validation' if args.validation else 'running_four_arm128_chain',
        **b.parent.process_event(os.getpid()),arm=arm,request=b.binding(request_path),gpu=request['recipe']['gpu'],processes={})
    started=time.monotonic();code=1
    b.write(root/'run.json',report)
    try:
        if args.validation:
            if arm in ('C','D'):
                stage(root,'train_one',['train','--request',str(request_path),'--output',str(root/'first'),'--stop-after','1'],report['gpu'],report)
                stage(root,'resume_one',['train','--request',str(request_path),'--output',str(root/'resumed'),
                    '--stop-after','2','--resume',str(root/'first/training_state.pt')],report['gpu'],report)
                trained=root/'resumed';steps=[(root/'first',1),(trained,2)]
            else:
                stage(root,'train_one',['train','--request',str(request_path),'--output',str(root/'first'),'--stop-after','1'],report['gpu'],report)
                trained=root/'first';steps=[(trained,1)]
            stage(root,'export',['export','--request',str(request_path),'--state',str(trained/'training_state.pt'),
                '--output',str(root/'post')],report['gpu'],report)
            stage(root,'fresh',['evaluate','--request',str(request_path),'--checkpoint',str(root/'post/model'),
                '--output',str(root/'post/evaluation'),'--component','fresh'],report['gpu'],report)
            rows=[b.read(directory/'updates'/f'step_{step:06d}.json') for directory,step in steps]
            fresh=b.read(root/'post/evaluation/fresh.json')
            b.require(fresh['full_state_exact'] and fresh['BF16_forward_finite'],'Fresh strict/finite validation failed.')
            if arm in ('C','D'):
                restored=b.read(trained/'resume_verified.json')
                b.require(restored['saved_updates']==1 and restored['next_group_index']==9
                    and rows[0]['exposure']['groups'][0]['QA_max_length']==10593
                    and rows[1]['exposure']['groups'][0]['M_max_length']==969,'Real peak/resume coverage failed.')
            else:restored=None
            proof=dict(status='verified_new_path',arm=arm,source_sha256=request['source_sha256'],
                fresh=fresh,resume=restored,rows=[dict(step=r['step'],group=r['group_index'],motion=r['motion'],
                    protection=r['protection'],gradient_squared_by_stratum=r['gradient_squared_by_stratum'],memory=r['memory']) for r in rows],
                initial_difference=b.read(root/'first/initial_output.json'),
                frozen_exact=all(b.read(directory/'training.json')['full_frozen_complement_exact'] for directory,_ in steps),
                seconds=time.monotonic()-started)
            b.write(root/'result.json',proof)
        else:
            previous=None
            for step in (64,128):
                directory=root/f'step_{step:06d}';directory.mkdir(exist_ok=False)
                arguments=['train','--request',str(request_path),'--output',str(directory/'training'),'--stop-after',str(step)]
                if previous:arguments.extend(['--resume',str(previous)])
                stage(root,'train_'+str(step),arguments,report['gpu'],report)
                previous=directory/'training/training_state.pt'
                stage(root,'export_'+str(step),['export','--request',str(request_path),'--state',str(previous),
                    '--output',str(directory/'post')],report['gpu'],report)
                for component in ('fresh','mmlu','seven'):
                    stage(root,component+'_'+str(step),['evaluate','--request',str(request_path),
                        '--checkpoint',str(directory/'post/model'),'--output',str(directory/'post/evaluation'),
                        '--component',component],report['gpu'],report)
                from .result import summarize
                summarize(request_path,directory,step)
                report.update(completed_endpoint=step,next='exact_resume_to128' if step==64 else None)
                b.write(root/'run.json',report)
        code=0;report['status']='complete_validation' if args.validation else 'complete128_and_both_eight_endpoints'
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report.update(exit_code=code,elapsed_seconds=time.monotonic()-started)
        b.write(root/'run.json',report);(root/'run.exit').write_text(str(code)+'\n')


if __name__=='__main__':main()
