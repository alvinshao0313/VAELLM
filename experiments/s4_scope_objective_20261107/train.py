"""Bounded named updates with exact external-master resume and original precision."""
import argparse
from collections import defaultdict
import math
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
from . import bindings as b,core,state,objective


def build(request):
    model,meta,budget=state.load(b.SOURCE)
    random.seed(137);np.random.seed(137);b.torch.manual_seed(137)
    model.to('cuda:0');b.restore_rope_alias(model)
    active,controller,warm=core.install(model,request['arm'])
    b.require(list(active)==request['parameter_names'],'Actual parameter order differs from CPU binding.')
    decoder=[p for n,p in active.items() if '._parallel_stage_decoder.' in n]
    groups=[dict(params=decoder,lr=5e-7,initial_recipe_lr=5e-7,name='decoder')]
    if controller:
        lr=1e-3 if controller.mode=='scale' else .03
        groups.append(dict(params=list(controller.parameters.values()),lr=lr,initial_recipe_lr=lr,name=controller.mode))
    optimizer=b.torch.optim.AdamW(groups,betas=(.9,.95),eps=1e-8,weight_decay=0.,foreach=False)
    b.require(not optimizer.state,'Fresh optimizer must not inherit state.')
    return model,active,optimizer,controller,warm,budget


def update(model,active,optimizer,controller,dataset,targets,raw,index,step,arm,observe):
    optimizer.zero_grad(set_to_none=True)
    previous={n:p.detach().clone() for n,p in active.items()} if observe else None
    b.torch.cuda.synchronize();b.torch.cuda.reset_peak_memory_stats(0);started=time.monotonic()
    values,ids,details=objective.losses(model,dataset,targets,raw,index,arm,True)
    norm_groups=defaultdict(float);strata=defaultdict(float)
    for name,p in active.items():
        b.require(p.grad is not None and p.grad.dtype==b.torch.float32 and b.torch.isfinite(p.grad).all(),
            'Missing/nonfinite active gradient: '+name)
        value=float(p.grad.double().square().sum());layer=int(name.split('.')[2])
        key='early' if layer<12 else 'middle' if layer<24 else 'late'
        kind='decoder' if '._parallel_stage_decoder.' in name else 'protection'
        norm_groups[kind]+=value;strata[kind+'/'+key]+=value
    b.require(all(v>0 and math.isfinite(v) for v in strata.values())
        and all(p.grad is None for n,p in model.named_parameters() if n not in active),
        'Active strata must have effective gradients; frozen parameters must not.')
    norm=float(b.torch.nn.utils.clip_grad_norm_(list(active.values()),1.5,error_if_nonfinite=True,foreach=False))
    for group in optimizer.param_groups:group['lr']=group['initial_recipe_lr']*(1-(step-1)/500)
    optimizer.step()
    for p in active.values():
        item=optimizer.state[p]
        b.require(p.dtype==item['exp_avg'].dtype==item['exp_avg_sq'].dtype==b.torch.float32
            and int(item['step'])==step and all(b.torch.isfinite(v).all() for v in (p,item['exp_avg'],item['exp_avg_sq'])),
            'Actual master/Adam dtype, step or finite contract failed.')
    motion=None
    if previous is not None:
        motion={}
        for name,p in active.items():
            layer=int(name.split('.')[2]);key=('decoder/' if '._parallel_stage_decoder.' in name else 'protection/')+(
                'early' if layer<12 else 'middle' if layer<24 else 'late')
            row=motion.setdefault(key,dict(FP32_changed=0,BF16_changed=0))
            row['FP32_changed']+=int(b.torch.count_nonzero(p.detach()!=previous[name]))
            row['BF16_changed']+=int(b.torch.count_nonzero(p.detach().bfloat16()!=previous[name].bfloat16()))
        b.require(all(row['FP32_changed']>0 for row in motion.values()),'Active strata must really update.')
        del previous
    protection=controller.statistics(step) if controller else None
    b.torch.cuda.synchronize()
    return dict(step=step,group_index=index,learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},
        losses=values,weighted_objective=math.fsum(values.values()),knowledge=details,case_ids=ids,
        gradient_norm_before_clip=norm,gradient_squared_by_kind=dict(norm_groups),gradient_squared_by_stratum=dict(strata),
        motion=motion,protection=protection,exposure=core.data.exposure(dataset,targets,[index]),
        seconds=time.monotonic()-started,memory=b.memory())


def execute(request_path,output,limit,resume):
    b.initialize();request=b.read(request_path);b.check_request(request);resource=b.resources(request['recipe']['gpu'])
    b.require(limit in ((1,2) if request['validation_only'] else (64,128))
        and limit<=len(request['groups']),'Only declared endpoints may execute.')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.monotonic();code=1
    report=dict(status='running_named_S4_updates',**b.parent.process_event(os.getpid()),request=b.binding(request_path),resource=resource)
    b.write(output/'training.json',report)
    try:
        dataset,targets,raw,evidence=core.load_data()
        b.require(evidence['actual_input_target_records_sha256']==request['data_digest'],'Data/target identity changed.')
        model,active,optimizer,controller,warm,budget=build(request)
        mutable=core.mutable_names(active,request['arm']);frozen=b.old.old_life.frozen_proof(model,mutable)
        cursor=0
        if resume:
            saved=state.restore(resume,model,active,optimizer,controller,request);cursor=saved['next_cursor']
            b.require(cursor<limit,'Resume must execute the next real update.')
            b.write(output/'resume_verified.json',dict(status='exact_named_masters_Adam_RNG_cursor',
                saved_updates=cursor,next_group_index=saved['next_group_index'],masters_exact=True,Adam_exact=True,RNG_exact=True,
                frozen_complement_exact=True,parameter_order_exact=True,external_protection_restored=bool(controller),
                state=b.binding(resume)))
        if request['validation_only'] and cursor==0:
            fixture=b.torch.load(request['fixture']['path'],map_location='cpu',weights_only=True)
            initial=b.old.run_short.probe(model,fixture)
            # Evaluate the same native BF16 parent before any actual update.
            native,_,_=state.load(b.SOURCE);native.to('cuda:0');b.restore_rope_alias(native)
            reference_active,_=core.original.install(native,'all')
            reference=b.old.run_short.probe(native,fixture)
            b.write(output/'initial_output.json',b.old.run_short.comparison(reference,initial))
            del native,reference_active
            import gc
            gc.collect();b.torch.cuda.empty_cache()
        if not request['validation_only']:
            heldout=b.torch.load(request['heldout']['path'],map_location='cpu',weights_only=True)
            b.write(output/f'diagnostic_{cursor:06d}.json',core.panel(model,dataset,targets,heldout,cursor))
        rows=[]
        for offset in range(cursor,limit):
            row=update(model,active,optimizer,controller,dataset,targets,raw,request['groups'][offset],offset+1,
                request['arm'],request['validation_only'] or offset==0)
            b.write(output/'updates'/f'step_{offset+1:06d}.json',row);rows.append(row)
            print({k:row[k] for k in ('step','group_index','learning_rates','weighted_objective','seconds','memory','protection')},flush=True)
            if offset+1 in (32,96):b.write(output/f'progress_{offset+1:06d}.json',row)
        optimizer.zero_grad(set_to_none=True)
        b.require(b.old.old_life.frozen_proof(model,mutable)==frozen,'Training changed frozen fields.')
        if not request['validation_only']:
            b.write(output/f'diagnostic_{limit:06d}.json',core.panel(model,dataset,targets,heldout,limit))
        saved=state.save(model,active,optimizer,controller,request,limit,output/'training_state.pt',frozen)
        report.update(status='complete_named_S4_training_state',completed_updates=limit,executed_updates=len(rows),
            groups=request['groups'][cursor:limit],training_state=saved,active_leaves=len(active),warmup=warm,bit_budget=budget,
            full_frozen_complement_exact=True,mean_update_seconds=math.fsum(r['seconds'] for r in rows)/len(rows),
            peak_allocated=max(r['memory']['peak_allocated'] for r in rows),resume=b.binding(resume) if resume else None)
        code=0
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report.update(exit_code=code,elapsed_seconds=time.monotonic()-started);b.write(output/'training.json',report)
        (output/'training.exit').write_text(str(code)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stop-after',type=int,required=True)
    parser.add_argument('--resume',type=Path);args=parser.parse_args()
    execute(args.request,args.output,args.stop_after,args.resume)
