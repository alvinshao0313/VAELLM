"""One bounded merge/export/fresh/complete-eight chain on one assigned GPU."""
import argparse
import os
import subprocess
import time
import traceback
from . import bindings as b

MODULE='experiments.s4_model_merge_20261108'


def stage(root,name,arguments,gpu,report):
    with (root/(name+'.log')).open('x',encoding='utf-8') as stream:
        argv=[str(b.PYTHON),'-B','-u','-m',MODULE+'.'+arguments[0],*arguments[1:]]
        child=subprocess.Popen(argv,cwd=b.ROOT,env=b.environment(gpu),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        report['processes'][name]=dict(command=argv,gpu=gpu,**b.parent.process_event(child.pid));b.write(root/'run.json',report)
        code=child.wait()
    report['processes'][name]['exit_code']=code;b.write(root/'run.json',report)
    (root/(name+'.exit')).write_text(str(code)+'\n')
    b.require(code==0,name+' failed; no automatic retry or dependent evaluation.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=list(b.ARMS),required=True)
    args=parser.parse_args();b.initialize();b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Controller CUDA must be hidden.')
    root=b.RUN/args.arm;request_path=root/'request.json';request=b.read(request_path);b.check(request)
    cpu=b.read(b.RUN/'CPU_validation.json')
    b.require(cpu['status']=='verified_aligned_donors_endpoints_and_three_native_merges'
        and b.binding(request_path)==cpu['request_bindings'][args.arm] and not (root/'run.json').exists(),
        'Fresh CPU-bound merge chain required.')
    report=dict(status='running_native_merge_and_eight',**b.parent.process_event(os.getpid()),gpu=request['recipe']['gpu'],
        request=b.binding(request_path),processes={});start=time.monotonic();code=1;b.write(root/'run.json',report)
    try:
        stage(root,'merge',['export','--request',str(request_path),'--output',str(root/'post')],report['gpu'],report)
        for part in ('fresh','mmlu','seven'):
            stage(root,part,['evaluate','--request',str(request_path),'--checkpoint',str(root/'post/model'),
                '--output',str(root/'post/evaluation'),'--component',part],report['gpu'],report)
            if part=='fresh':
                fresh=b.read(root/'post/evaluation/fresh.json')
                b.require(fresh['full_state_exact'] and fresh['BF16_forward_finite'],'Fresh/finite merge check failed.')
        from .result import summarize
        summarize(request_path,root);code=0;report['status']='complete_native_merge_and_eight'
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report.update(exit_code=code,elapsed_seconds=time.monotonic()-start);b.write(root/'run.json',report)
        (root/'run.exit').write_text(str(code)+'\n')


if __name__=='__main__':main()
