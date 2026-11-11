"""Fixed-kappa numeric tolerance, immutable context and exact fee inheritance."""
import argparse
from copy import deepcopy
from pathlib import Path
import tempfile
import torch
from . import io
from .identity import source_pins
from .optimizer import RecoveryOptimizer
from .processes import Budget


def check(root):
    torch.manual_seed(31)
    initial=torch.randn(16,24);gradient=torch.randn_like(initial)
    context=dict(raw_id='same_raw',data='same_first_batch',seed=31)
    def construct(grad,fixed=None,parameter=None,ctx=None):
        p=torch.nn.Parameter(initial.clone() if parameter is None else parameter.clone());p.grad=grad.clone()
        group=dict(params=[p],names=['lora.B'],group_name='lora',kind='muon',stages=[1],reference_lr=1e-4,
            multiplier=1.,lr=1e-4,weight_decay=0.,muon_kappa=None,calibration=None)
        return RecoveryOptimizer([group],fixed,context=context if ctx is None else ctx)
    original=construct(gradient);original.step();cal=deepcopy(original.param_groups[0]['calibration'])
    assert cal['context']==context
    fixed={'lora':cal}
    adjacent=gradient.clone()
    adjacent[0,0]=torch.nextafter(adjacent[0,0],torch.tensor(float('inf')))
    numerical=construct(adjacent,fixed);numerical.step()
    assert numerical.param_groups[0]['muon_kappa']==cal['kappa']
    assert not numerical.param_groups[0]['fixed_calibration_check']['exact_gradient']
    state=numerical.state_dict();loaded=construct(gradient,fixed);loaded.load_state_dict(state)
    assert loaded.param_groups[0]['calibration']==cal and loaded.param_groups[0]['muon_kappa']==cal['kappa']
    rejected=[]
    changed_gradient=gradient.clone();changed_gradient[:12]=0
    for name,optimizer in [('parameter',construct(gradient,fixed,parameter=initial+.01)),
        ('context',construct(gradient,fixed,ctx=dict(context,seed=32))),
        ('gradient_norm',construct(changed_gradient,fixed))]:
        try:optimizer.step()
        except ValueError:rejected.append(name)
        else:raise AssertionError('Invalid calibration accepted: '+name)
    auth=dict(budget_by_component=dict(technical=4,training=10),total_GPU_hours=14,gpus=[4,5,6,7],maximum_trajectories=2)
    b=Budget(root/'old.json',auth,[dict(started_epoch=0,ended_epoch=10)],[])
    b.reserve({'new/train':dict(component='training',seconds=300),'old/failed':dict(component='training',seconds=300)},counts=dict(trajectories=2))
    request=root/'request.json';io.write(request,{'fixture':True})
    for i,key in enumerate(('new/train','old/failed')):
        b.start(key,gpu=4,start=100+i*100,deadline=400+i*100,command=[],log=root/'log',request=request)
        b.finish(key,end=120+i*100,exit_code=1,error='fixture')
    continued=Budget(root/'next.json',auth,b.value['technical_windows']+[dict(started_epoch=300,ended_epoch=320)],[],
        previous=b.value,retry_tasks=['new/train'])
    assert continued.value['tasks']['old/failed']['status']=='failed' and continued.value['tasks']['new/train']['status']=='reserved'
    assert abs(continued.value['total_actual_GPU_hours']-b.value['total_actual_GPU_hours']-20/3600)<1e-12
    assert continued.value['counter_reserved']==b.value['counter_reserved']
    return dict(fixed_kappa_exact=True,normal_numeric_difference_accepted=True,invalid_rejected=rejected,
        original_failed_windows_preserved=True,new_technical_windows_charged_once=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='SC_CAT_kappa_CPU_') as tmp:result=check(Path(tmp))
    io.write(a.output,dict(result,status='CPU_verified',source_pins=source_pins(),formal_eligible=False))


if __name__=='__main__':main()
