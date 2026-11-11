"""One scope, one named update sequence, and an explicit bounded resume entry."""
import argparse
from collections import defaultdict
from pathlib import Path
import math
import os
import random
import time
import traceback
import numpy as np
from . import bindings as b,core,state


def build(request):
    model,meta,budget=state.load(b.SOURCE)
    random.seed(137);np.random.seed(137);b.torch.manual_seed(137)
    model.to('cuda:0');b.restore_rope_alias(model)
    active,warm=core.install(model,request['arm'])
    b.require(set(active)==set(request['exact_scope_parameter_names']),
        'Actual model selection differs from CPU-pinned exact QK/VO names.')
    optimizer=b.torch.optim.AdamW(list(active.values()),lr=5e-7,betas=(.9,.95),eps=1e-8,weight_decay=0.,foreach=False)
    b.require(not optimizer.state,'Fresh Adam must have no inherited state.')
    return model,active,optimizer,warm,budget


def strata(name):
    layer=int(name.split('.')[2])
    return 'early' if layer<12 else 'middle' if layer<24 else 'late'


def update(model,active,optimizer,dataset,targets,index,step,observe_motion):
    optimizer.zero_grad(set_to_none=True)
    previous={n:p.detach().clone() for n,p in active.items()} if observe_motion else None
    b.torch.cuda.synchronize();b.torch.cuda.reset_peak_memory_stats(0);started=time.monotonic()
    losses,ids=core.training.losses(model,dataset,targets,index,True)
    gradients=core.training.gradient_contract(model,active)
    layer_grad=defaultdict(float)
    for n,p in active.items():layer_grad[strata(n)]+=float(p.grad.double().square().sum())
    b.require(all(value>0 and math.isfinite(value) for value in layer_grad.values()),'Early/middle/late active gradients must be finite nonzero.')
    norm=float(b.torch.nn.utils.clip_grad_norm_(list(active.values()),1.5,error_if_nonfinite=True,foreach=False))
    b.require(norm>0 and math.isfinite(norm),'Finite nonzero joint gradient required.')
    lr=5e-7*(1-(step-1)/500);optimizer.param_groups[0]['lr']=lr;optimizer.step()
    for p in active.values():
        item=optimizer.state[p]
        b.require(p.dtype==item['exp_avg'].dtype==item['exp_avg_sq'].dtype==b.torch.float32
            and int(item['step'])==step and b.torch.isfinite(p).all() and b.torch.isfinite(item['exp_avg']).all()
            and b.torch.isfinite(item['exp_avg_sq']).all(),'Actual master/Adam dtype, clock, or finite contract failed.')
    motion=None
    if previous is not None:
        motion={key:dict(FP32_changed=0,BF16_changed=0) for key in ('early','middle','late')}
        for n,p in active.items():
            row=motion[strata(n)];row['FP32_changed']+=int((p.detach()!=previous[n]).sum())
            row['BF16_changed']+=int((p.detach().bfloat16()!=previous[n].bfloat16()).sum())
        b.require(all(row['FP32_changed']>0 and row['BF16_changed']>0 for row in motion.values()),'Each layer stratum must actually update FP32 masters and BF16 deployment.')
        del previous
    b.torch.cuda.synchronize()
    return dict(step=step,group_index=index,lr=lr,losses=losses,weighted_objective=sum(losses.values()),
        gradient_norm_before_clip=norm,gradients=gradients,layer_strata_gradient_squared=dict(layer_grad),
        layer_strata_displacement=motion,case_ids=ids,exposure=core.data.exposure(dataset,targets,[index]),
        seconds=time.monotonic()-started,memory=b.memory())


def execute(request_path,output,resume=None,stop_after=None):
    request=b.read(request_path);b.initialize();b.parent.s.check_pins(request['source_sha256'])
    b.require(request['parent']==b.source() and request['contract']==b.CONTRACT
        and request['learning_contract']==b.LEARNING_CONTRACT,'Request source/contract changed.')
    resource=b.resources(request['gpu']);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    limit=request['stop_step'] if stop_after is None else stop_after
    b.require(0<limit<=len(request['groups']) and (request['validation_only'] or limit==64
        or (resume is not None and stop_after is not None and 64<limit<=128)),
        'Only an explicit resume can extend the stop step beyond the current fixed64 batch.')
    started=time.monotonic();code=1
    report=dict(status='running_fixed_scope_updates',**b.parent.process_event(os.getpid()),request=b.binding(request_path),resources=resource)
    b.write(output/'training.json',report)
    try:
        dataset,targets,evidence=core.load_data()
        b.require(evidence['actual_input_target_records_sha256']==request['data_digest'],'Static data/target digest changed.')
        model,active,optimizer,warm,budget=build(request)
        frozen=b.old.old_life.frozen_proof(model,set(active))
        cursor=0
        if resume:
            saved=state.restore(resume,model,active,optimizer,request);cursor=saved['next_cursor']
            b.require(limit>=cursor,'Explicit resume cannot move the completed cursor backwards.')
            b.write(output/'resume_verified.json',dict(status='cross_process_exact_masters_Adam_RNG_cursor',
                state=b.binding(resume),saved_updates=cursor,next_group_index=saved['next_group_index'],parameter_names_exact=True,
                masters_exact=True,complete_Adam_exact=True,RNG_exact=True,frozen_complement_exact=True))
        elif request['validation_only']:
            fixture=b.torch.load(request['fixture']['path'],map_location='cpu',weights_only=True)
            # Record initial kernel differences against the matched all64 path; no exact-output gate.
            native,_,_=state.load(b.SOURCE);native.to('cuda:0');b.restore_rope_alias(native)
            from experiments.s4_recovery_screen_20261105 import core as reference_core
            reference_active,_=reference_core.install(native,'all')
            reference=b.old.run_short.probe(native,fixture)
            del native,reference_active
            import gc
            gc.collect();b.torch.cuda.empty_cache()
            actual=b.old.run_short.probe(model,fixture)
            b.write(output/'scope_initial_output.json',b.old.run_short.comparison(reference,actual))
        if not request['validation_only'] and cursor==0:
            heldout=b.torch.load(request['heldout']['path'],map_location='cpu',weights_only=True)
            b.write(output/'diagnostic_000000.json',core.panel(model,dataset,targets,heldout))
        rows=[]
        for offset in range(cursor,limit):
            row=update(model,active,optimizer,dataset,targets,request['groups'][offset],offset+1,
                request['validation_only'] or offset==0)
            b.write(output/'updates'/f'step_{offset+1:06d}.json',row);rows.append(row)
            print({k:row[k] for k in ('step','group_index','lr','weighted_objective','seconds','gradients','layer_strata_displacement')},flush=True)
        optimizer.zero_grad(set_to_none=True)
        b.require(b.old.old_life.frozen_proof(model,set(active))==frozen,'Training changed frozenMLP/code/protection/rotation/backbone.')
        if not request['validation_only']:
            heldout=b.torch.load(request['heldout']['path'],map_location='cpu',weights_only=True)
            b.write(output/f'diagnostic_{limit:06d}.json',core.panel(model,dataset,targets,heldout))
        training_state=state.save(model,active,optimizer,request,limit,output/'training_state.pt')
        report.update(status='complete_actual_named_scope_training_state',completed_updates=limit,executed_updates=len(rows),
            group_indices=request['groups'][cursor:limit],training_state=training_state,full_frozen_complement_exact=True,
            active_leaves=len(active),active_projections=b.ARMS[request['arm']]['projections'],warmup=warm,bit_budget=budget,
            resume=b.binding(resume) if resume else None,mean_update_seconds=sum(row['seconds'] for row in rows)/len(rows) if rows else None)
        code=0
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report.update(exit_code=code,elapsed_seconds=time.monotonic()-started)
        b.write(output/'training.json',report);(output/'training.exit').write_text(str(code)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',type=Path);parser.add_argument('--stop-after',type=int)
    args=parser.parse_args();execute(args.request,args.output,args.resume,args.stop_after)
