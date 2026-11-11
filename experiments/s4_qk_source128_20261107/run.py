"""One authorized continuation, independent native export/fresh, then complete eight."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback
from . import bindings as b

MODULE='experiments.s4_qk_source128_20261107'


def stage(root,name,arguments,report):
    with (root/(name+'.log')).open('x',encoding='utf-8') as log:
        argv=[str(b.PYTHON),'-B','-u','-m',MODULE+'.'+arguments[0],*arguments[1:]]
        process=subprocess.Popen(argv,cwd=b.ROOT,env=b.environment(report['gpu']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        report['processes'][name]=dict(command=argv,gpu=report['gpu'],**b.parent.process_event(process.pid));b.write(root/'run.json',report)
        code=process.wait()
    report['processes'][name]['exit_code']=code;b.write(root/'run.json',report)
    (root/(name+'.exit')).write_text(str(code)+'\n')
    b.require(code==0,name+' failed; no automatic retry or dependent execution.')


def main():
    b.initialize();b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Controller must hide CUDA.')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=list(b.ARMS),required=True)
    args=parser.parse_args();root=b.RUN/args.arm
    request_path=root/'request.json';request=b.read(request_path);b.check_request(request)
    proof=b.read(b.RUN/'CPU_validation.json')['arms'][args.arm]
    b.require(proof['request']==b.binding(request_path) and proof['original_identity_preserved'], 'CPU preparation binding changed.')
    b.require(not (root/'run.json').exists(),'Never overwrite a running or ended continuation attempt.')
    report=dict(status='running_QK64_to128_and_eight',**b.parent.process_event(os.getpid()),
        request=b.binding(request_path),CPU_validation=b.binding(b.RUN/'CPU_validation.json'),
        gpu=request['continuation']['gpu'],processes={})
    began=time.monotonic();code=1
    try:
        stage(root,'training',['train','--request',str(request_path),'--output',str(root/'training')],report)
        stage(root,'export',['export','--request',str(request_path),'--state',str(root/'training/training_state.pt'),
            '--output',str(root/'post')],report)
        checkpoint=root/'post/model';evaluation=root/'post/evaluation'
        for component in ('fresh','mmlu','seven'):
            stage(root,component,['evaluate','--request',str(request_path),'--checkpoint',str(checkpoint),
                '--output',str(evaluation),'--component',component],report)
            if component=='fresh':
                fresh=b.read(evaluation/'fresh.json');b.require(fresh['full_state_exact'] and fresh['BF16_forward_finite'],'Native fresh failed.')
        from .result import summarize
        summarize(request_path,root)
        report.update(status='complete_QK128_eight',exit_code=0);code=0
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-began,exit_code=code);b.write(root/'run.json',report)
        (root/'run.exit').write_text(str(code)+'\n')


if __name__=='__main__':main()
