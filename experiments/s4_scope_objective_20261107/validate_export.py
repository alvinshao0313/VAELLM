"""Finish the already successful real training checks after a loader-only fix."""
import argparse
import gc
import os
from pathlib import Path
import time
import traceback
from . import bindings as b,state
from .run import stage

PREVIOUS=b.OUTPUT/'trial_20261108_four_arms128_v2'


def training_evidence(arm):
    previous=PREVIOUS/('validation_'+arm)
    request=b.read(previous/'request.json');current=b.read(b.RUN/('validation_'+arm)/'request.json')
    b.require({k:v for k,v in request.items() if k!='source_sha256'}==
        {k:v for k,v in current.items() if k!='source_sha256'},'Original learning request semantics changed.')
    allowed={str(b.HERE/name) for name in ('bindings.py','run.py','export.py')};changed={}
    for name,digest in request['source_sha256'].items():
        actual=b.sha(name)
        if actual==digest:continue
        b.require(name in allowed and b.sha(PREVIOUS/'export_failure_source'/Path(name).name)==digest,
            'A learning/data/state/evaluation operator changed; real validation cannot be reused: '+name)
        changed[name]=dict(previous=digest,current=actual)
    b.require(set(changed)==allowed,'Only the three audited routing/visibility files may change.')
    before=(PREVIOUS/'export_failure_source/bindings.py').read_text()
    b.require(before.replace("trial_20261108_four_arms128_v2","trial_20261108_four_arms128_v3")==
        (b.HERE/'bindings.py').read_text(),'Bindings change must only relocate the trial.')
    last=previous/('resumed' if arm in ('C','D') else 'first')
    training=b.read(last/'training.json')
    b.require(training['exit_code']==0 and training['completed_updates']==(2 if arm in ('C','D') else 1)
        and training['full_frozen_complement_exact'],'Completed exact original validation updates required.')
    return previous,request,last,changed,current


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=['B','C','D'],required=True)
    parser.add_argument('--component',choices=['export']);args=parser.parse_args();b.initialize()
    previous,request,last,changed,current=training_evidence(args.arm);b.check_request(current)
    root=b.RUN/('validation_'+args.arm)
    if args.component=='export':
        resource=b.resources(current['recipe']['gpu'])
        saved=state.check_payload(b.torch.load(last/'training_state.pt',map_location='cpu',weights_only=False),request)
        report=state.export(request,saved,root/'post')
        report.update(training_state=b.binding(last/'training_state.pt'),historical_training_request=b.binding(previous/'request.json'),
            actual_export_source_sha256=current['source_sha256'],audited_loader_only_changes=changed,
            model_weights_on_CPU=True,assigned_GPU_visible_for_FA2_loader=resource)
        b.write(root/'post/export.json',report);print(dict(status='native_export_complete',arm=args.arm),flush=True)
        return
    b.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (root/'run.json').exists(),'Fresh completion controller required.')
    report=dict(status='running_export_completion',**b.parent.process_event(os.getpid()),processes={},
        original_training_root=str(previous),actual_request=b.binding(root/'request.json'),gpu=current['recipe']['gpu'])
    started=time.monotonic();code=1;b.write(root/'run.json',report)
    try:
        stage(root,'export',['validate_export','--arm',args.arm,'--component','export'],report['gpu'],report)
        stage(root,'fresh',['evaluate','--request',str(root/'request.json'),'--checkpoint',str(root/'post/model'),
            '--output',str(root/'post/evaluation'),'--component','fresh'],report['gpu'],report)
        fresh=b.read(root/'post/evaluation/fresh.json')
        b.require(fresh['full_state_exact'] and fresh['BF16_forward_finite'],'Actual independent fresh check failed.')
        steps=[(previous/'first',1),(last,2)] if args.arm in ('C','D') else [(last,1)]
        rows=[b.read(path/'updates'/f'step_{step:06d}.json') for path,step in steps]
        restored=b.read(last/'resume_verified.json') if args.arm in ('C','D') else None
        if restored:
            b.require(restored['saved_updates']==1 and restored['next_group_index']==9
                and rows[0]['exposure']['groups'][0]['QA_max_length']==10593
                and rows[1]['exposure']['groups'][0]['M_max_length']==969,'Peak/resume coverage failed.')
        b.write(root/'result.json',dict(status='verified_new_path',arm=args.arm,source_sha256=current['source_sha256'],
            original_training_source_sha256=request['source_sha256'],unchanged_learning_operator_hashes=True,
            audited_loader_only_changes=changed,original_request=b.binding(previous/'request.json'),
            original_training_reports=[b.binding(path/'training.json') for path,_ in steps],
            fresh=fresh,resume=restored,initial_difference=b.read(previous/'first/initial_output.json'),
            rows=[dict(step=r['step'],group=r['group_index'],motion=r['motion'],protection=r['protection'],
                gradient_squared_by_stratum=r['gradient_squared_by_stratum'],memory=r['memory']) for r in rows],
            frozen_exact=True,seconds=time.monotonic()-started,
            reuse_boundary='Actual original updates and cross-process exact resume; only CUDA visibility/routing of independent export changed'))
        code=0;report['status']='complete_validation_with_reused_training'
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report.update(exit_code=code,elapsed_seconds=time.monotonic()-started);b.write(root/'run.json',report)
        (root/'run.exit').write_text(str(code)+'\n')


if __name__=='__main__':main()
